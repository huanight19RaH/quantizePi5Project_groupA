"""Unified 7,500-image COCOeval + Operating-Point + Fused-Op Audit Evaluator for YOLO26n ONNX."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize_pipeline import audit_fused_ort_ops
from src.detector import LetterboxMeta, decode_person_predictions, letterbox_bgr_to_rgb


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


def _run_cocoeval_subset(coco_gt: COCO, coco_results: list[dict], img_ids: list[int]) -> dict[str, float]:
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


def load_ram_test_cache(manifest_path: Path, imgsz: int = 416):
    """Pre-pack all test images from `dataset_manifest.json` into RAM uint8 NHWC cache."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recs = manifest["test_records"]
    buf = np.empty((len(recs), imgsz, imgsz, 3), dtype=np.uint8)
    metas: list[tuple[dict, LetterboxMeta]] = [None] * len(recs)  # type: ignore[list-item]

    def _pack(item: tuple[int, dict]) -> None:
        idx, rec = item
        p = Path(rec["path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing test image {p}. Run `python scripts/prepare_datasets.py --full` first.")
        rgb, meta = letterbox_bgr_to_rgb(img, imgsz=imgsz)
        buf[idx] = rgb
        metas[idx] = (rec, meta)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_pack, enumerate(recs)))
    return buf, metas


def evaluate_model(
    model_path: Path,
    test_buf: np.ndarray,
    test_metas: list[tuple[dict, LetterboxMeta]],
    coco_gt: COCO,
    probe_indices: list[int],
    num_workers: int = 8,
    conf_coco: float = 0.001,
    iou_coco: float = 0.65,
    conf_op: float = 0.25,
    iou_op: float = 0.45,
    ref_op_boxes: list[np.ndarray] | None = None,
    ref_probe_raw: np.ndarray | None = None,
) -> tuple[dict, list[np.ndarray], np.ndarray]:
    """Run parallel ONNX inference on all test images and compute full metrics + fused-op audit."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sessions = [ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"]) for _ in range(num_workers)]
    in_name = sessions[0].get_inputs()[0].name
    is_fp16 = "float16" in str(sessions[0].get_inputs()[0].type)

    n = len(test_metas)
    per_boxes: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    per_scores: list[np.ndarray] = [None] * n  # type: ignore[list-item]
    probe_map = {idx: i for i, idx in enumerate(probe_indices)}
    probe_raw = np.empty((len(probe_indices), 5, 3549), dtype=np.float32)

    t0 = time.perf_counter()

    def _worker(wid: int) -> None:
        sess = sessions[wid]
        dtype_np = np.float16 if is_fp16 else np.float32
        scale_val = np.float32(1.0 / 255.0)
        for idx in range(wid, n, num_workers):
            inp = (test_buf[idx].transpose(2, 0, 1).astype(np.float32) * scale_val).astype(dtype_np)[None, ...]
            raw = sess.run(None, {in_name: inp})[0][0][:5].astype(np.float32)
            if idx in probe_map:
                probe_raw[probe_map[idx]] = raw
            _, lmeta = test_metas[idx]
            bx, sc = decode_person_predictions(raw, lmeta, conf_threshold=conf_coco, iou_threshold=iou_coco, max_det=300)
            per_boxes[idx] = bx
            per_scores[idx] = sc

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(ex.map(_worker, range(num_workers)))
    wall_s = time.perf_counter() - t0

    gt_op = {}
    for rec, _ in test_metas:
        iid = rec["image_id"]
        anns = [a for a in coco_gt.imgToAnns[iid] if a.get("iscrowd", 0) == 0]
        gt_op[iid] = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns], dtype=np.float32) if anns else np.zeros((0, 4), dtype=np.float32)

    coco_res = []
    op_boxes: list[np.ndarray] = []
    tp = fp = fn = neg_fp = neg_imgs = total_op = 0
    matched_ious = []

    for i, (rec, _) in enumerate(test_metas):
        iid = rec["image_id"]
        bx, sc = per_boxes[i], per_scores[i]
        for j in range(len(sc)):
            x1, y1, x2, y2 = float(bx[j, 0]), float(bx[j, 1]), float(bx[j, 2]), float(bx[j, 3])
            coco_res.append({
                "image_id": int(iid),
                "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)],
                "score": round(float(sc[j]), 5),
            })

        mask = sc >= conf_op
        if np.any(mask):
            bx_o, sc_o = bx[mask], sc[mask]
            xywh_o = np.stack([bx_o[:, 0], bx_o[:, 1], np.maximum(0.0, bx_o[:, 2] - bx_o[:, 0]), np.maximum(0.0, bx_o[:, 3] - bx_o[:, 1])], axis=1)
            k2 = cv2.dnn.NMSBoxes(xywh_o.tolist(), sc_o.tolist(), float(conf_op), float(iou_op))
            bx_o = bx_o[np.asarray(k2, dtype=np.int64).reshape(-1)] if len(k2) > 0 else np.zeros((0, 4), dtype=np.float32)
        else:
            bx_o = np.zeros((0, 4), dtype=np.float32)
        op_boxes.append(bx_o)
        total_op += len(bx_o)

        g = gt_op[iid]
        if len(g) == 0:
            if len(bx_o) > 0:
                neg_fp += len(bx_o)
                neg_imgs += 1
            fp += len(bx_o)
        elif len(bx_o) == 0:
            fn += len(g)
        else:
            ious = _box_iou_matrix(bx_o, g)
            used = set()
            for pi in range(len(bx_o)):
                bg = int(np.argmax(ious[pi]))
                if ious[pi, bg] >= 0.50 and bg not in used:
                    used.add(bg)
                    tp += 1
                else:
                    fp += 1
            fn += len(g) - len(used)

        if ref_op_boxes is not None and len(ref_op_boxes[i]) > 0 and len(bx_o) > 0:
            best = _box_iou_matrix(bx_o, ref_op_boxes[i]).max(axis=1)
            matched_ious.extend(best[best > 0.1].tolist())

    all_ids = [r["image_id"] for r, _ in test_metas]
    val_ids = [r["image_id"] for r, _ in test_metas if r.get("slice") == "val2017_official"]
    hard_ids = [r["image_id"] for r, _ in test_metas if r.get("slice") == "matched_train_2500"]

    c_all = _run_cocoeval_subset(coco_gt, coco_res, all_ids)
    c_val = _run_cocoeval_subset(coco_gt, coco_res, val_ids) if val_ids else c_all
    c_hard = _run_cocoeval_subset(coco_gt, coco_res, hard_ids) if hard_ids else c_all

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    ops_audit = audit_fused_ort_ops(model_path)

    raw_cos = 1.0
    box_mae = score_mae = 0.0
    if ref_probe_raw is not None:
        a = probe_raw.reshape(len(probe_indices), -1).astype(np.float64)
        b = ref_probe_raw.reshape(len(probe_indices), -1).astype(np.float64)
        raw_cos = round(float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))), 6)
        box_mae = round(float(np.mean(np.abs(probe_raw[:, :4, :] - ref_probe_raw[:, :4, :]))), 5)
        score_mae = round(float(np.mean(np.abs(probe_raw[:, 4, :] - ref_probe_raw[:, 4, :]))), 6)

    row = {
        "variant": model_path.stem,
        "size_mb": round(model_path.stat().st_size / 1048576, 2),
        "eval_time_s": round(wall_s, 2),
        "throughput_img_s": round(n / max(1e-6, wall_s), 1),
        **c_all,
        "val2017_mAP50": c_val["mAP50"],
        "train2500_mAP50": c_hard["mAP50"],
        "op_precision": round(prec, 4),
        "op_recall": round(rec, 4),
        "op_f1": round(f1, 4),
        "op_preds": total_op,
        "neg_fp_boxes": neg_fp,
        "neg_fp_imgs": neg_imgs,
        "mean_iou_vs_fp32": round(float(np.mean(matched_ious)), 4) if matched_ious else 1.0,
        "probe_raw_cos": raw_cos,
        "probe_box_mae": box_mae,
        "probe_score_mae": score_mae,
        **ops_audit,
    }
    return row, op_boxes, probe_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX models on the 7,500-image distribution-matched test suite.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "datasets" / "dataset_manifest.json")
    parser.add_argument("--gt-json", type=Path, default=REPO_ROOT / "datasets" / "test_7500_person_gt.json")
    parser.add_argument("--models", type=Path, nargs="+", default=[
        REPO_ROOT / "models" / "model_fp32.onnx",
        REPO_ROOT / "models" / "model_fp16.onnx",
        REPO_ROOT / "models" / "model_int8_dynamic.onnx",
        REPO_ROOT / "models" / "model_int8_static.onnx",
    ])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "local_eval_report.csv")
    args = parser.parse_args()

    if not args.gt_json.exists():
        raise FileNotFoundError(f"Missing {args.gt_json}. Run `python scripts/prepare_datasets.py --full` to download the 7,500 test images.")

    coco_gt = COCO(str(args.gt_json))
    test_buf, test_metas = load_ram_test_cache(args.manifest)
    probe_indices = list(range(0, len(test_metas), max(1, len(test_metas) // 256)))[:256]

    rows = []
    ref_op = ref_raw = None
    for idx, mp in enumerate(args.models):
        row, op_bx, p_raw = evaluate_model(mp, test_buf, test_metas, coco_gt, probe_indices, args.workers, ref_op_boxes=ref_op, ref_probe_raw=ref_raw)
        if idx == 0:
            ref_op, ref_raw = op_bx, p_raw
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
