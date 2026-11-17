"""Benchmark Team_A models (ONNX FP32/INT8 & Orange Pi 5 RKNN 6-Head NPU+RGA) on Roboflow `cctv-naxyo`."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import PersonDetector

try:
    from src.npu_detector import NPUDetector
except Exception:
    NPUDetector = None  # Allows running ONNX-only evaluation on non-RK3588 hosts


def _box_iou_matrix(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    a1 = np.maximum(0.0, (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1]))[:, None]
    a2 = np.maximum(0.0, (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1]))[None, :]
    return inter / np.maximum(a1 + a2 - inter, 1e-6)


def load_unified_cctv_dataset(root: Path) -> tuple[COCO, list[dict[str, Any]], dict[str, list[int]], dict[int, np.ndarray]]:
    """Load train/valid/test splits of cctv-naxyo into one unified non-overlapping COCO GT object."""
    unified_images = []
    unified_anns = []
    split_img_ids: dict[str, list[int]] = {"train": [], "valid": [], "test": [], "valid_test": [], "all": []}
    gt_op_map: dict[int, np.ndarray] = {}

    next_img_id = 1
    next_ann_id = 1
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]

    items_meta: list[dict[str, Any]] = []
    for sp in ("train", "valid", "test"):
        ann_path = root / sp / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        raw = json.loads(ann_path.read_text(encoding="utf-8"))
        anns_by_local: dict[int, list[dict[str, Any]]] = {}
        for a in raw["annotations"]:
            if a.get("category_id") == 1:
                anns_by_local.setdefault(int(a["image_id"]), []).append(a)

        for im in sorted(raw["images"], key=lambda x: x["id"]):
            gid = next_img_id
            next_img_id += 1
            unified_images.append({
                "id": gid,
                "file_name": f"{sp}/{im['file_name']}",
                "width": int(im["width"]),
                "height": int(im["height"]),
            })
            img_path = root / sp / im["file_name"]
            items_meta.append({"gid": gid, "split": sp, "path": img_path})
            split_img_ids[sp].append(gid)
            split_img_ids["all"].append(gid)
            if sp in ("valid", "test"):
                split_img_ids["valid_test"].append(gid)

            gt_boxes = []
            for a in anns_by_local.get(int(im["id"]), []):
                bx = [float(v) for v in a["bbox"]]
                area = float(a.get("area", bx[2] * bx[3]))
                unified_anns.append({
                    "id": next_ann_id,
                    "image_id": gid,
                    "category_id": 1,
                    "bbox": bx,
                    "area": area,
                    "iscrowd": int(a.get("iscrowd", 0)),
                    "segmentation": [],
                })
                next_ann_id += 1
                if int(a.get("iscrowd", 0)) == 0:
                    gt_boxes.append([bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]])
            gt_op_map[gid] = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "Roboflow cctv-naxyo v2 unified benchmark"},
        "images": unified_images,
        "annotations": unified_anns,
        "categories": categories,
    }
    coco_gt.createIndex()
    return coco_gt, items_meta, split_img_ids, gt_op_map


def run_cocoeval_subset(coco_gt: COCO, coco_dt: COCO | None, img_ids: list[int]) -> dict[str, float]:
    """Compute COCOeval metrics over an explicit `img_ids` subset."""
    if coco_dt is None or not img_ids:
        return {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "AR_100")}
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = [1]
    ev.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {
        "mAP50_95": round(float(s[0]), 5),
        "mAP50": round(float(s[1]), 5),
        "mAP75": round(float(s[2]), 5),
        "AP_small": round(float(s[3]), 5),
        "AP_medium": round(float(s[4]), 5),
        "AP_large": round(float(s[5]), 5),
        "AR_100": round(float(s[8]), 5),
    }


def compute_op_metrics(
    preds_by_gid: dict[int, tuple[np.ndarray, np.ndarray]],
    gt_op_map: dict[int, np.ndarray],
    img_ids: list[int],
    conf_op: float = 0.25,
    iou_op: float = 0.45,
) -> dict[str, Any]:
    """Compute Operating-Point Precision / Recall / F1 and Negative-Image FP boxes."""
    tp = fp = fn = neg_fp = neg_imgs = total_preds = 0
    for gid in img_ids:
        bx_all, sc_all = preds_by_gid.get(gid, (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)))
        mask = sc_all >= conf_op
        if np.any(mask):
            bx = bx_all[mask]
            sc = sc_all[mask]
            xywh = np.stack([bx[:, 0], bx[:, 1], np.maximum(0.0, bx[:, 2] - bx[:, 0]), np.maximum(0.0, bx[:, 3] - bx[:, 1])], axis=1)
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc.tolist(), float(conf_op), float(iou_op))
            bx_op = bx[np.asarray(keep, dtype=np.int64).reshape(-1)] if len(keep) > 0 else np.zeros((0, 4), dtype=np.float32)
        else:
            bx_op = np.zeros((0, 4), dtype=np.float32)

        total_preds += len(bx_op)
        g = gt_op_map[gid]
        if len(g) == 0:
            if len(bx_op) > 0:
                neg_fp += len(bx_op)
                neg_imgs += 1
            fp += len(bx_op)
        elif len(bx_op) == 0:
            fn += len(g)
        else:
            ious = _box_iou_matrix(bx_op, g)
            used = set()
            for pi in range(len(bx_op)):
                bg = int(np.argmax(ious[pi]))
                if ious[pi, bg] >= 0.50 and bg not in used:
                    used.add(bg)
                    tp += 1
                else:
                    fp += 1
            fn += len(g) - len(used)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "op_precision": round(prec, 4),
        "op_recall": round(rec, 4),
        "op_f1": round(f1, 4),
        "op_tp": tp,
        "op_fp": fp,
        "op_fn": fn,
        "op_preds": total_preds,
        "neg_fp_boxes": neg_fp,
        "neg_fp_imgs": neg_imgs,
    }


def compute_box_iou_parity(
    ref_preds: dict[int, tuple[np.ndarray, np.ndarray]],
    cand_preds: dict[int, tuple[np.ndarray, np.ndarray]],
    img_ids: list[int],
    conf_op: float = 0.25,
) -> tuple[float, float]:
    """Compute Mean IoU and Mean |Delta score| between quantized predictions and FP32 reference boxes."""
    matched_ious: list[float] = []
    score_diffs: list[float] = []
    for gid in img_ids:
        rb, rs = ref_preds.get(gid, (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)))
        cb, cs = cand_preds.get(gid, (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)))
        rm = rs >= conf_op
        cm = cs >= conf_op
        if not np.any(rm) or not np.any(cm):
            continue
        rb_f, rs_f = rb[rm], rs[rm]
        cb_f, cs_f = cb[cm], cs[cm]
        ious = _box_iou_matrix(rb_f, cb_f)
        for ri in range(len(rb_f)):
            cj = int(np.argmax(ious[ri]))
            if ious[ri, cj] >= 0.1:
                matched_ious.append(float(ious[ri, cj]))
                score_diffs.append(float(abs(rs_f[ri] - cs_f[cj])))
    mean_iou = round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0
    mean_ds = round(float(np.mean(score_diffs)), 5) if score_diffs else 0.0
    return mean_iou, mean_ds


def evaluate_variant(
    label: str,
    detector: Any,
    items_meta: list[dict[str, Any]],
    coco_gt: COCO,
    split_img_ids: dict[str, list[int]],
    gt_op_map: dict[int, np.ndarray],
    fp32_ref: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Run 1 inference pass over all 3,737 cctv-naxyo images and report train/valid/test/valid+test/all splits."""
    coco_res: list[dict[str, Any]] = []
    preds_by_gid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    lat_ms: list[float] = []
    rga_hits = 0

    first_img = cv2.imread(str(items_meta[0]["path"]), cv2.IMREAD_COLOR)
    detector.detect(first_img)

    t0 = time.perf_counter()
    for idx_i, item in enumerate(items_meta):
        if (idx_i + 1) % 1000 == 0:
            print(f"  [{label}] {idx_i + 1}/{len(items_meta)} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
        gid = item["gid"]
        img = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        dets, tm = detector.detect(img)
        lat_ms.append(tm["total_ms"])
        if tm.get("rga_used"):
            rga_hits += 1

        if dets:
            bx = np.array([d["bbox"] for d in dets], dtype=np.float32)
            sc = np.array([d["score"] for d in dets], dtype=np.float32)
            preds_by_gid[gid] = (bx, sc)
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                coco_res.append({
                    "image_id": int(gid),
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)],
                    "score": round(d["score"], 5),
                })
        else:
            preds_by_gid[gid] = (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))

    wall_s = time.perf_counter() - t0
    coco_dt = coco_gt.loadRes(coco_res) if coco_res else None
    mean_iou_all, mean_ds_all = (1.0, 0.0) if fp32_ref is None else compute_box_iou_parity(fp32_ref, preds_by_gid, split_img_ids["all"])

    rows: list[dict[str, Any]] = []
    for sp_name in ("all", "valid_test", "valid", "test", "train"):
        ids = split_img_ids[sp_name]
        c_met = run_cocoeval_subset(coco_gt, coco_dt, ids)
        op_25 = compute_op_metrics(preds_by_gid, gt_op_map, ids, conf_op=0.25, iou_op=0.45)
        op_15 = compute_op_metrics(preds_by_gid, gt_op_map, ids, conf_op=0.15, iou_op=0.45)
        miou, mds = (1.0, 0.0) if fp32_ref is None else (
            (mean_iou_all, mean_ds_all) if sp_name == "all" else compute_box_iou_parity(fp32_ref, preds_by_gid, ids)
        )
        row = {
            "variant": label,
            "split": sp_name,
            "n_imgs": len(ids),
            "eval_wall_s": round(wall_s, 2),
            "mean_e2e_ms": round(float(np.mean(lat_ms)), 3),
            "rga_hits": f"{rga_hits}/{len(items_meta)}",
            **c_met,
            **op_25,
            "op_f1_at_015": op_15["op_f1"],
            "op_prec_at_015": op_15["op_precision"],
            "op_rec_at_015": op_15["op_recall"],
            "mean_iou_vs_fp32": miou,
            "mean_dscore_vs_fp32": mds,
        }
        rows.append(row)
        if sp_name in ("all", "valid", "test"):
            print(
                f"  [CCTV {sp_name:5s}] {label:32s} | mAP50={row['mAP50']:.5f} mAP50-95={row['mAP50_95']:.5f} "
                f"F1@0.25={row['op_f1']:.4f} (P={row['op_precision']:.3f} R={row['op_recall']:.3f}) F1@0.15={row['op_f1_at_015']:.4f} IoU={miou:.4f}"
            )
    return rows, preds_by_gid


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Team_A on Roboflow cctv-naxyo v2.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "datasets" / "cctv-naxyo")
    parser.add_argument("--bench-loops", type=int, default=5)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    coco_gt, items_meta, split_img_ids, gt_op_map = load_unified_cctv_dataset(args.data_dir)
    print(f"[CCTV-NAXYO] Loaded {len(items_meta)} images: " + ", ".join(f"{k}={len(v)}" for k, v in split_img_ids.items()))

    has_npu = NPUDetector is not None and Path("/dev/rga").exists() and (REPO_ROOT / "cpp" / "libyolo26_rk3588.so").exists()
    out_dir = REPO_ROOT / "results" / ("cctv_naxyo_orangepi5" if has_npu else "cctv_naxyo_local")
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_npu:
        subprocess.run(["sudo", "-n", "bash", str(REPO_ROOT / "scripts" / "tune_orangepi5.sh"), "performance"], check=False)

    all_rows: list[dict[str, Any]] = []
    fp32_ref: dict[int, tuple[np.ndarray, np.ndarray]] | None = None

    # 1. ONNX Variants
    onnx_specs = [
        ("ONNX_FP32_Baseline", REPO_ROOT / "models" / "model_fp32.onnx", False, has_npu),
        ("ONNX_INT8_Static_Before", REPO_ROOT / "models" / "model_int8_static.onnx", False, False),
        ("ONNX_INT8_Static_FastCPU", REPO_ROOT / "models" / "model_int8_static.onnx", True, has_npu),
        ("ONNX_INT8_Dynamic", REPO_ROOT / "models" / "model_int8_dynamic.onnx", True, has_npu),
    ]
    if not has_npu:
        onnx_specs.append(("ONNX_FP16", REPO_ROOT / "models" / "model_fp16.onnx", False, False))

    for lbl, mp, fast_m, pin_a in onnx_specs:
        if not mp.exists():
            continue
        if not pin_a and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(range(os.cpu_count() or 8)))
        det = PersonDetector(mp, conf_threshold=0.001, iou_threshold=0.65, max_det=300, num_threads=4, fast_mode=fast_m, pin_a76=pin_a)
        rows, preds = evaluate_variant(lbl, det, items_meta, coco_gt, split_img_ids, gt_op_map, fp32_ref)
        if fp32_ref is None:
            fp32_ref = preds
        all_rows.extend(rows)

    # 2. RKNN NPU Variants (when running on Orange Pi 5)
    rknn_dir = REPO_ROOT / "models" / "rknn"
    if has_npu:
        npu_specs = [
            ("RKNN_FP16_Uncut_NPU", rknn_dir / "model_fp16_uncut.rknn", False, False, False),
            ("RKNN_INT8_Uncut_NPU", rknn_dir / "model_int8_uncut_normal.rknn", False, False, False),
            ("RKNN_INT8_6Head_Normal_Q11", rknn_dir / "model_int8_6head_normal.rknn", True, True, False),
            ("RKNN_INT8_6Head_Normal_RGA_HW", rknn_dir / "model_int8_6head_normal.rknn", True, True, True),
            ("RKNN_INT8_6Head_MMSE_RGA_HW", rknn_dir / "model_int8_6head_mmse.rknn", True, True, True),
            ("RKNN_INT8_6Head_KL_RGA_HW", rknn_dir / "model_int8_6head_kl.rknn", True, True, True),
            ("RKNN_INT8_6Head_Hybrid_RGA_HW", rknn_dir / "model_int8_6head_hybrid.rknn", True, True, True),
            ("RKNN_INT8_6Head_Normal_SC_RGA_HW", rknn_dir / "model_int8_6head_normal_sc.rknn", True, True, True),
        ]
        for lbl, rp, zc, pt, rga in npu_specs:
            if not rp.exists():
                continue
            det = NPUDetector(rp, conf_threshold=0.001, iou_threshold=0.65, max_det=300, n_workers=1, zero_copy=zc, pass_through_out=pt, use_rga=rga, pin_a76=True)
            rows, _ = evaluate_variant(lbl, det, items_meta, coco_gt, split_img_ids, gt_op_map, fp32_ref)
            det.close()
            all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_dir / "cctv_naxyo_accuracy.csv", index=False)
    (out_dir / "cctv_naxyo_accuracy.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    # 3. Real 640x640 CCTV Latency/FPS Before-vs-After Benchmark on Orange Pi 5
    if has_npu and not args.skip_latency:
        pool = [cv2.imread(str(it["path"]), cv2.IMREAD_COLOR) for it in items_meta[:100]]
        lat_rows: list[dict[str, Any]] = []

        def _bench_onnx(lbl: str, mp: Path, gov: str, fast_m: bool, pin_a: bool) -> None:
            subprocess.run(["sudo", "-n", "bash", str(REPO_ROOT / "scripts" / "tune_orangepi5.sh"), gov], check=False, stdout=subprocess.DEVNULL)
            if not pin_a and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(range(os.cpu_count() or 8)))
            d = PersonDetector(mp, conf_threshold=0.25, iou_threshold=0.45, num_threads=4, fast_mode=fast_m, pin_a76=pin_a)
            for _ in range(5):
                d.detect(pool[0])
            pr, inf, ps, tot = [], [], [], []
            t0 = time.perf_counter()
            for _ in range(2):
                for im in pool:
                    _, tm = d.detect(im)
                    pr.append(tm["preprocess_ms"]); inf.append(tm["inference_ms"]); ps.append(tm["postprocess_ms"]); tot.append(tm["total_ms"])
            wall_ms = (time.perf_counter() - t0) * 1000.0
            arr = np.array(tot, dtype=np.float64)
            row = {
                "config": lbl, "model": mp.name, "frames": len(tot), "wall_ms": round(wall_ms, 2),
                "fps": round(1000.0 * len(tot) / wall_ms, 2),
                "pre_ms": round(float(np.mean(pr)), 3), "inf_ms": round(float(np.mean(inf)), 3), "post_ms": round(float(np.mean(ps)), 3),
                "mean_total_ms": round(float(np.mean(arr)), 3), "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3), "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "rga_active": False,
            }
            lat_rows.append(row)
            print(f"[CCTV-LAT] {lbl:40s} | FPS={row['fps']:7.2f} | total={row['mean_total_ms']:6.3f}ms (pre={row['pre_ms']:5.2f} inf={row['inf_ms']:5.2f} post={row['post_ms']:5.2f})")

        _bench_onnx("1_BEFORE_ONNX_FP32_ondemand", REPO_ROOT / "models" / "model_fp32.onnx", "ondemand", False, False)
        _bench_onnx("2_BEFORE_ONNX_INT8_Static_ondemand", REPO_ROOT / "models" / "model_int8_static.onnx", "ondemand", False, False)
        _bench_onnx("3_CPU_ONNX_INT8_FastBlob_PinnedA76", REPO_ROOT / "models" / "model_int8_static.onnx", "performance", True, True)

        subprocess.run(["sudo", "-n", "bash", str(REPO_ROOT / "scripts" / "tune_orangepi5.sh"), "performance"], check=False, stdout=subprocess.DEVNULL)
        best_rk = rknn_dir / "model_int8_6head_normal.rknn"
        sc_rk = rknn_dir / "model_int8_6head_normal_sc.rknn"
        for lbl, rp, nw, zc, pt, rga, lp in [
            ("4_NPU_FP16_Uncut_Copy_1Core", rknn_dir / "model_fp16_uncut.rknn", 1, False, False, False, args.bench_loops),
            ("5_NPU_INT8_6Head_Copy_1Core", best_rk, 1, False, False, False, args.bench_loops),
            ("6_NPU_INT8_6Head_ZeroCopy_Q11_1Core", best_rk, 1, True, True, False, args.bench_loops * 2),
            ("7_NPU_INT8_6Head_ZeroCopy_RGA_1Core", best_rk, 1, True, True, True, args.bench_loops * 2),
            ("8_NPU_INT8_6Head_ZeroCopy_Q11_3Workers", best_rk, 3, True, True, False, args.bench_loops * 3),
            ("9_NPU_INT8_6Head_ZeroCopy_RGA_6Workers", best_rk, 6, True, True, True, args.bench_loops * 4),
            ("10_NPU_INT8_6Head_SCMode_RGA_6Workers", sc_rk, 6, True, True, True, args.bench_loops * 4),
        ]:
            if not rp.exists():
                continue
            dn = NPUDetector(rp, conf_threshold=0.25, iou_threshold=0.45, n_workers=nw, zero_copy=zc, pass_through_out=pt, use_rga=rga, pin_a76=True)
            res = dn.bench_prepacked_multicore(pool, loops=lp, pin_a76=True)
            _, tc = dn.detect(pool[0])
            dn.close()
            row = {
                "config": lbl, "model": rp.name, "frames": res["total_frames"], "wall_ms": res["wall_ms"], "fps": res["fps"],
                "pre_ms": res["mean_pre_ms"], "inf_ms": res["mean_npu_ms"], "post_ms": res["mean_post_ms"],
                "mean_total_ms": round(res["mean_pre_ms"] + res["mean_npu_ms"] + res["mean_post_ms"], 3),
                "p50_ms": res["p50_ms"], "p95_ms": res["p95_ms"], "p99_ms": res["p99_ms"],
                "rga_active": bool(tc["rga_used"]),
            }
            lat_rows.append(row)
            print(f"[CCTV-LAT] {lbl:40s} | FPS={row['fps']:7.2f} | step={row['mean_total_ms']:6.3f}ms (pre={row['pre_ms']:5.2f} npu={row['inf_ms']:5.2f} post={row['post_ms']:5.2f}) | rga={row['rga_active']}")

        pd.DataFrame(lat_rows).to_csv(out_dir / "cctv_naxyo_latency_fps.csv", index=False)
        (out_dir / "cctv_naxyo_latency_fps.json").write_text(json.dumps(lat_rows, indent=2), encoding="utf-8")
        subprocess.run(["sudo", "-n", "bash", str(REPO_ROOT / "scripts" / "tune_orangepi5.sh"), "ondemand"], check=False, stdout=subprocess.DEVNULL)

    print(f"[DONE] Saved cctv-naxyo benchmark outputs in {out_dir}")


if __name__ == "__main__":
    main()
