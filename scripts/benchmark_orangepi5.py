"""Complete Before-vs-After Accuracy (5,000 COCO val2017) & Edge Latency/FPS Benchmark on Orange Pi 5."""

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

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import PersonDetector
from src.npu_detector import NPUDetector


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


def run_cocoeval_val2017(coco_gt: COCO, coco_results: list[dict], img_ids: list[int]) -> dict[str, float]:
    """Compute official COCOeval person metrics cleanly."""
    if not coco_results:
        return {k: 0.0 for k in ("mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large", "AR_100")}
    dt = coco_gt.loadRes(coco_results)
    ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.catIds = [1]
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
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


def evaluate_val2017_accuracy(label: str, detector, val_items: list[tuple[int, Path]], coco_gt: COCO, gt_op_map: dict[int, np.ndarray]) -> dict:
    """Evaluate exact 5,000-image COCO val2017 accuracy (`conf=0.001, iou=0.65`) and OpPoint P/R/F1 (`0.25/0.45`)."""
    coco_res: list[dict] = []
    tp = fp = fn = neg_fp = neg_imgs = total_op = 0
    lat_ms: list[float] = []
    img_ids = [iid for iid, _ in val_items]

    # Warmup once
    first_img = cv2.imread(str(val_items[0][1]), cv2.IMREAD_COLOR)
    detector.detect(first_img)

    t0 = time.perf_counter()
    for idx_i, (iid, p) in enumerate(val_items):
        if (idx_i + 1) % 1000 == 0:
            print(f"  [{label}] {idx_i + 1}/{len(val_items)} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        dets, tm = detector.detect(img)
        lat_ms.append(tm["total_ms"])

        op_boxes = []
        op_scores = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            sc = d["score"]
            coco_res.append({
                "image_id": int(iid),
                "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)],
                "score": round(sc, 5),
            })
            if sc >= 0.25:
                op_boxes.append([x1, y1, x2, y2])
                op_scores.append(sc)

        if op_boxes:
            bx_np = np.array(op_boxes, dtype=np.float32)
            sc_np = np.array(op_scores, dtype=np.float32)
            xywh = np.stack([bx_np[:, 0], bx_np[:, 1], np.maximum(0.0, bx_np[:, 2] - bx_np[:, 0]), np.maximum(0.0, bx_np[:, 3] - bx_np[:, 1])], axis=1)
            keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc_np.tolist(), 0.25, 0.45)
            bx_op = bx_np[np.asarray(keep, dtype=np.int64).reshape(-1)] if len(keep) > 0 else np.zeros((0, 4), dtype=np.float32)
        else:
            bx_op = np.zeros((0, 4), dtype=np.float32)

        total_op += len(bx_op)
        g = gt_op_map[iid]
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

    wall_s = time.perf_counter() - t0
    metrics = run_cocoeval_val2017(coco_gt, coco_res, img_ids)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    row = {
        "variant": label,
        "val_imgs": len(val_items),
        "eval_wall_s": round(wall_s, 2),
        "mean_e2e_ms": round(float(np.mean(lat_ms)), 2),
        **metrics,
        "op_precision": round(prec, 4),
        "op_recall": round(rec, 4),
        "op_f1": round(f1, 4),
        "op_preds": total_op,
        "neg_fp_boxes": neg_fp,
        "neg_fp_imgs": neg_imgs,
    }
    print(f"[VAL2017-ACC] {label:32s} | mAP50={row['mAP50']:.5f} mAP50-95={row['mAP50_95']:.5f} AP_s={row['AP_small']:.5f} F1={row['op_f1']:.4f} NegFP={neg_fp} ({wall_s:.1f}s)")
    return row


def bench_cpu_onnx_preloaded(label: str, model_path: Path, bgr_images: list[np.ndarray], loops: int, fast_mode: bool, pin_a76: bool, num_threads: int) -> dict:
    """Measure exact per-stage latency & FPS on preloaded BGR frames for CPU ONNX Runtime."""
    if not pin_a76 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(os.cpu_count() or 8)))
    det = PersonDetector(model_path, conf_threshold=0.25, iou_threshold=0.45, num_threads=num_threads, fast_mode=fast_mode, pin_a76=pin_a76)
    for _ in range(5):
        det.detect(bgr_images[0])

    pre_l, inf_l, post_l, tot_l = [], [], [], []
    t0 = time.perf_counter()
    for _ in range(loops):
        for img in bgr_images:
            _, tm = det.detect(img)
            pre_l.append(tm["preprocess_ms"])
            inf_l.append(tm["inference_ms"])
            post_l.append(tm["postprocess_ms"])
            tot_l.append(tm["total_ms"])
    wall_ms = (time.perf_counter() - t0) * 1000.0
    n_frames = len(bgr_images) * loops
    arr = np.array(tot_l, dtype=np.float64)
    row = {
        "config": label,
        "model": model_path.name,
        "size_mb": round(model_path.stat().st_size / 1048576, 2),
        "frames": n_frames,
        "wall_ms": round(wall_ms, 2),
        "fps": round(1000.0 * n_frames / max(1e-6, wall_ms), 2),
        "pre_ms": round(float(np.mean(pre_l)), 3),
        "inf_ms": round(float(np.mean(inf_l)), 3),
        "post_ms": round(float(np.mean(post_l)), 3),
        "mean_total_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }
    print(f"[BENCH] {label:42s} | FPS={row['fps']:7.2f} | total={row['mean_total_ms']:6.3f}ms (pre={row['pre_ms']:5.2f} inf={row['inf_ms']:5.2f} post={row['post_ms']:5.2f}) | p95={row['p95_ms']:6.3f}ms")
    return row


def bench_npu_preloaded(
    label: str,
    rknn_path: Path,
    bgr_images: list[np.ndarray],
    loops: int,
    n_workers: int,
    zero_copy: bool,
    pass_through_out: bool,
    use_rga: bool,
    enable_sram: bool = False,
) -> dict:
    """Measure exact per-stage latency & FPS on preloaded BGR frames using C++ NPU engine."""
    det = NPUDetector(
        rknn_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        n_workers=n_workers,
        core_mask_mode=1,
        zero_copy=zero_copy,
        pass_through_out=pass_through_out,
        use_rga=use_rga,
        enable_sram=enable_sram,
        pin_a76=True,
    )
    info = det.info()
    res = det.bench_prepacked_multicore(bgr_images, loops=loops, pin_a76=True)
    # Also check RGA usage on 1 sample if RGA requested
    rga_ok = False
    if use_rga:
        _, t1 = det.detect(bgr_images[0], worker_id=0)
        rga_ok = bool(t1["rga_used"])
    det.close()
    mean_tot = round(res["mean_pre_ms"] + res["mean_npu_ms"] + res["mean_post_ms"], 3)
    row = {
        "config": label,
        "model": rknn_path.name,
        "size_mb": round(rknn_path.stat().st_size / 1048576, 2),
        "frames": res["total_frames"],
        "wall_ms": res["wall_ms"],
        "fps": res["fps"],
        "pre_ms": res["mean_pre_ms"],
        "inf_ms": res["mean_npu_ms"],
        "post_ms": res["mean_post_ms"],
        "mean_total_ms": mean_tot,
        "p50_ms": res["p50_ms"],
        "p95_ms": res["p95_ms"],
        "p99_ms": res["p99_ms"],
        "dma_mb": round(info["total_dma_bytes"] / 1048576, 2),
        "rga_active": rga_ok if use_rga else False,
    }
    print(f"[BENCH] {label:42s} | FPS={row['fps']:7.2f} | step={mean_tot:6.3f}ms (pre={row['pre_ms']:5.2f} npu={row['inf_ms']:5.2f} post={row['post_ms']:5.2f}) | p95={row['p95_ms']:6.3f}ms")
    return row


def set_governor(mode: str) -> None:
    subprocess.run(["sudo", "-n", "bash", str(REPO_ROOT / "scripts" / "tune_orangepi5.sh"), mode], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Team_A on Orange Pi 5 before and after optimizations.")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-onnx-val", action="store_true")
    parser.add_argument("--bench-loops", type=int, default=3)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "orangepi5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 200 real calibration BGR images into RAM for zero-I/O latency/FPS benchmarking
    calib_paths = sorted((REPO_ROOT / "calibration_data").glob("*.jpg"))
    bgr_pool = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in calib_paths]
    print(f"[SETUP] Preloaded {len(bgr_pool)} calibration BGR frames into RAM for latency/FPS benchmarking.")

    rknn_dir = REPO_ROOT / "models" / "rknn"
    cand_6head = [
        ("RKNN_INT8_6Head_Normal", rknn_dir / "model_int8_6head_normal.rknn"),
        ("RKNN_INT8_6Head_MMSE", rknn_dir / "model_int8_6head_mmse.rknn"),
        ("RKNN_INT8_6Head_KL", rknn_dir / "model_int8_6head_kl.rknn"),
        ("RKNN_INT8_6Head_Hybrid", rknn_dir / "model_int8_6head_hybrid.rknn"),
        ("RKNN_INT8_6Head_Normal_SC", rknn_dir / "model_int8_6head_normal_sc.rknn"),
    ]
    cand_6head = [(n, p) for n, p in cand_6head if p.exists()]

    # 2. Full 5,000-image COCO val2017 Accuracy Parity Verification (run on performance governor for speed)
    if not args.skip_accuracy:
        set_governor("performance")
        ann_path = REPO_ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"
        val_dir = REPO_ROOT / "datasets" / "coco" / "val2017"
        coco_gt = COCO(str(ann_path))
        val_items = [(im["id"], val_dir / im["file_name"]) for im in sorted(coco_gt.dataset["images"], key=lambda x: x["id"])]
        gt_op_map = {}
        for iid, _ in val_items:
            anns = [a for a in coco_gt.imgToAnns[iid] if a["category_id"] == 1 and a.get("iscrowd", 0) == 0]
            gt_op_map[iid] = (
                np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32)
                if anns
                else np.zeros((0, 4), dtype=np.float32)
            )

        acc_rows: list[dict] = []
        if not args.skip_onnx_val:
            for lbl, mp, fmode, pin in [
                ("ONNX_FP32_Baseline", REPO_ROOT / "models" / "model_fp32.onnx", False, True),
                ("ONNX_INT8_Static_Before", REPO_ROOT / "models" / "model_int8_static.onnx", False, True),
                ("ONNX_INT8_Static_FastCPU", REPO_ROOT / "models" / "model_int8_static.onnx", True, True),
            ]:
                det = PersonDetector(mp, conf_threshold=0.001, iou_threshold=0.65, max_det=300, num_threads=4, fast_mode=fmode, pin_a76=pin)
                acc_rows.append(evaluate_val2017_accuracy(lbl, det, val_items, coco_gt, gt_op_map))

        if (rknn_dir / "model_fp16_uncut.rknn").exists():
            d_fp16 = NPUDetector(rknn_dir / "model_fp16_uncut.rknn", conf_threshold=0.001, iou_threshold=0.65, max_det=300, n_workers=1, zero_copy=False, pass_through_out=False, use_rga=False)
            acc_rows.append(evaluate_val2017_accuracy("RKNN_FP16_Uncut_NPU", d_fp16, val_items, coco_gt, gt_op_map))
            d_fp16.close()

        if (rknn_dir / "model_int8_uncut_normal.rknn").exists():
            d_i8u = NPUDetector(rknn_dir / "model_int8_uncut_normal.rknn", conf_threshold=0.001, iou_threshold=0.65, max_det=300, n_workers=1, zero_copy=False, pass_through_out=False, use_rga=False)
            acc_rows.append(evaluate_val2017_accuracy("RKNN_INT8_Uncut_NPU", d_i8u, val_items, coco_gt, gt_op_map))
            d_i8u.close()

        best_6h_lbl, best_6h_rp, best_6h_sc = None, None, -1.0
        for lbl, rp in cand_6head:
            d_npu = NPUDetector(rp, conf_threshold=0.001, iou_threshold=0.65, max_det=300, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=False)
            r = evaluate_val2017_accuracy(lbl + "_ZeroCopy_NC1", d_npu, val_items, coco_gt, gt_op_map)
            acc_rows.append(r)
            d_npu.close()
            if (r["mAP50"] + r["mAP50_95"]) > best_6h_sc:
                best_6h_sc = r["mAP50"] + r["mAP50_95"]
                best_6h_lbl, best_6h_rp = lbl, rp

        if best_6h_rp is not None:
            # Verify RGA hardware preproc accuracy on winning 6-head INT8 model
            d_rga = NPUDetector(best_6h_rp, conf_threshold=0.001, iou_threshold=0.65, max_det=300, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=True)
            acc_rows.append(evaluate_val2017_accuracy(best_6h_lbl + "_RGA_HW", d_rga, val_items, coco_gt, gt_op_map))
            d_rga.close()

        df_acc = pd.DataFrame(acc_rows)
        df_acc.to_csv(out_dir / "val2017_accuracy_parity.csv", index=False)
        (out_dir / "val2017_accuracy_parity.json").write_text(json.dumps(acc_rows, indent=2), encoding="utf-8")

    # Pick highest-mAP50+mAP50_95 6-head INT8 model for latency/FPS ladder
    best_rknn = cand_6head[0][1]
    acc_json = out_dir / "val2017_accuracy_parity.json"
    if acc_json.exists():
        rows = json.loads(acc_json.read_text(encoding="utf-8"))
        best_rknn_name = None
        best_score = -1.0
        for lbl, rp in cand_6head:
            match = [r for r in rows if r["variant"].startswith(lbl) and "RGA" not in r["variant"]]
            if match:
                sc = match[0]["mAP50"] + match[0]["mAP50_95"]
                if sc > best_score:
                    best_score = sc
                    best_rknn = rp
                    best_rknn_name = lbl
        print(f"[WINNER-RKNN] Selected {best_rknn.name} ({best_rknn_name}) based on 5,000-image val2017 accuracy!")

    # 3. Latency & FPS Ladder: BEFORE (ondemand default) vs AFTER (performance + optimized CPU & NPU Zero-Copy/RGA/MultiCore)
    bench_rows: list[dict] = []
    set_governor("ondemand")
    time.sleep(1.0)
    bench_rows.append(bench_cpu_onnx_preloaded("1_BEFORE_ONNX_FP32_ondemand", REPO_ROOT / "models" / "model_fp32.onnx", bgr_pool, loops=1, fast_mode=False, pin_a76=False, num_threads=4))
    bench_rows.append(bench_cpu_onnx_preloaded("2_BEFORE_ONNX_INT8_Static_ondemand", REPO_ROOT / "models" / "model_int8_static.onnx", bgr_pool, loops=1, fast_mode=False, pin_a76=False, num_threads=4))

    set_governor("performance")
    time.sleep(1.0)
    bench_rows.append(bench_cpu_onnx_preloaded("3_CPU_ONNX_INT8_PinnedA76_Perf", REPO_ROOT / "models" / "model_int8_static.onnx", bgr_pool, loops=1, fast_mode=False, pin_a76=True, num_threads=4))
    bench_rows.append(bench_cpu_onnx_preloaded("4_CPU_ONNX_INT8_FastBlob_PinnedA76", REPO_ROOT / "models" / "model_int8_static.onnx", bgr_pool, loops=1, fast_mode=True, pin_a76=True, num_threads=4))

    if (rknn_dir / "model_fp16_uncut.rknn").exists():
        bench_rows.append(bench_npu_preloaded("5_NPU_FP16_Uncut_Copy_1Core", rknn_dir / "model_fp16_uncut.rknn", bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=False, pass_through_out=False, use_rga=False))
    if (rknn_dir / "model_int8_uncut_normal.rknn").exists():
        bench_rows.append(bench_npu_preloaded("5b_NPU_INT8_Uncut_Copy_1Core", rknn_dir / "model_int8_uncut_normal.rknn", bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=False, pass_through_out=False, use_rga=False))

    bench_rows.append(bench_npu_preloaded("6_NPU_INT8_6Head_Copy_1Core", best_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=False, pass_through_out=False, use_rga=False))
    bench_rows.append(bench_npu_preloaded("7_NPU_INT8_6Head_ZeroCopy_NHWC_1Core", best_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=True, pass_through_out=False, use_rga=False))
    bench_rows.append(bench_npu_preloaded("8_NPU_INT8_6Head_ZeroCopy_NC1_1Core", best_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=False))
    sc_rknn = rknn_dir / "model_int8_6head_normal_sc.rknn"
    if sc_rknn.exists() and sc_rknn != best_rknn:
        bench_rows.append(bench_npu_preloaded("8b_NPU_INT8_6Head_SCMode_ZeroCopy_1Core", sc_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=False))
    bench_rows.append(bench_npu_preloaded("9_NPU_INT8_6Head_ZeroCopy_NC1_SRAM_1Core", best_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=False, enable_sram=True))
    bench_rows.append(bench_npu_preloaded("10_NPU_INT8_6Head_ZeroCopy_NC1_RGA_1Core", best_rknn, bgr_pool, loops=args.bench_loops, n_workers=1, zero_copy=True, pass_through_out=True, use_rga=True))
    bench_rows.append(bench_npu_preloaded("11_NPU_INT8_6Head_ZeroCopy_NC1_3Workers", best_rknn, bgr_pool, loops=args.bench_loops * 2, n_workers=3, zero_copy=True, pass_through_out=True, use_rga=False))
    bench_rows.append(bench_npu_preloaded("12_NPU_INT8_6Head_ZeroCopy_NC1_6Workers", best_rknn, bgr_pool, loops=args.bench_loops * 2, n_workers=6, zero_copy=True, pass_through_out=True, use_rga=False))
    bench_rows.append(bench_npu_preloaded("13_NPU_INT8_6Head_ZeroCopy_RGA_6Workers", best_rknn, bgr_pool, loops=args.bench_loops * 2, n_workers=6, zero_copy=True, pass_through_out=True, use_rga=True))
    if sc_rknn.exists() and sc_rknn != best_rknn:
        bench_rows.append(bench_npu_preloaded("14_NPU_INT8_6Head_SCMode_RGA_6Workers", sc_rknn, bgr_pool, loops=args.bench_loops * 2, n_workers=6, zero_copy=True, pass_through_out=True, use_rga=True))
    hyb_rknn = rknn_dir / "model_int8_6head_hybrid.rknn"
    if hyb_rknn.exists() and hyb_rknn != best_rknn:
        bench_rows.append(bench_npu_preloaded("15_NPU_INT8_6Head_Hybrid_RGA_6Workers", hyb_rknn, bgr_pool, loops=args.bench_loops * 2, n_workers=6, zero_copy=True, pass_through_out=True, use_rga=True))

    df_b = pd.DataFrame(bench_rows)
    df_b.to_csv(out_dir / "latency_fps_before_after.csv", index=False)
    (out_dir / "latency_fps_before_after.json").write_text(json.dumps(bench_rows, indent=2), encoding="utf-8")
    print("\n=== SAVED ALL BENCHMARK REPORTS TO results/orangepi5/ ===")


if __name__ == "__main__":
    main()
