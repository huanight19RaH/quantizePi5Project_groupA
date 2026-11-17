"""Per-block INT8 quantization sensitivity tracker for YOLO26n ONNX.

Quantizes one logical architectural block at a time (with all other blocks kept in FP32)
using a cached calibration table and measures raw tensor drift (`cos_sim`, `box_mae_px`,
`score_mae`) against the FP32 baseline on real calibration/probe images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize_pipeline import ImageFolderCalibrationReader, resolve_calibration_images
from src.detector import letterbox_bgr_to_rgb


def group_nodes_by_block(onnx_path: Path) -> dict[str, list[str]]:
    """Group all ONNX nodes into logical YOLO26n architectural blocks."""
    model = onnx.load(str(onnx_path))
    groups: dict[str, list[str]] = {}
    for node in model.graph.node:
        name = node.name
        if name.startswith("/model.23/"):
            if "/cv2." in name:
                grp = "/model.23/cv2 (Box Tower)"
            elif "/cv3." in name:
                grp = "/model.23/cv3 (Cls Tower)"
            else:
                grp = "/model.23/decode_tail"
        elif "/attn/" in name:
            prefix = "/".join(name.split("/")[:2])
            grp = f"{prefix}/attn (C2PSA)"
        else:
            parts = name.split("/")
            grp = f"/{parts[1]}" if len(parts) > 1 and parts[1].startswith("model.") else "/other"
        groups.setdefault(grp, []).append(name)
    return groups


def audit_blocks(
    fp32_path: Path,
    calib_dir: Path,
    out_csv: Path,
    manifest_path: Path | None = None,
    imgsz: int = 416,
    num_probe: int = 48,
) -> pd.DataFrame:
    """Measure single-block INT8 degradation across all blocks and save sorted sensitivity table."""
    pre_path = fp32_path.parent / "ablation" / "model_fp32_pre.onnx"
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    if not pre_path.exists():
        quant_pre_process(str(fp32_path), str(pre_path))

    calib_imgs = resolve_calibration_images(calib_dir, manifest_path, max_images=200)
    cache_path = fp32_path.parent / "calib_cache_minmax_200.json"
    if not cache_path.exists():
        tmp_init = fp32_path.parent / "_tmp_cache_init.onnx"
        quantize_static(
            model_input=str(pre_path),
            model_output=str(tmp_init),
            calibration_data_reader=ImageFolderCalibrationReader(calib_imgs, imgsz=imgsz),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=[],
            calibration_cache_path=str(cache_path),
        )
        tmp_init.unlink(missing_ok=True)

    probe_paths = sorted(calib_dir.glob("*.jpg"))[:num_probe]
    probe_blobs = [
        (letterbox_bgr_to_rgb(cv2.imread(str(p)), imgsz=imgsz)[0].transpose(2, 0, 1).astype(np.float32) / 255.0)[None, ...]
        for p in probe_paths
    ]

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_options=so, providers=["CPUExecutionProvider"])
    fp32_in = fp32_sess.get_inputs()[0].name
    ref_raw = np.stack([fp32_sess.run(None, {fp32_in: b})[0][0][:5] for b in probe_blobs], axis=0)

    groups = group_nodes_by_block(pre_path)
    all_nodes = [n.name for n in onnx.load(str(pre_path)).graph.node]
    tmp_onnx = fp32_path.parent / "_tmp_sens_blk.onnx"

    t0 = time.perf_counter()
    rows = []
    for grp_name, grp_nodes in sorted(groups.items()):
        excl = [nm for nm in all_nodes if nm not in set(grp_nodes)]
        try:
            quantize_static(
                model_input=str(pre_path),
                model_output=str(tmp_onnx),
                calibration_data_reader=None,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=excl,
                calibration_cache_path=str(cache_path),
            )
            sess = ort.InferenceSession(str(tmp_onnx), sess_options=so, providers=["CPUExecutionProvider"])
            iname = sess.get_inputs()[0].name
            preds = np.stack([sess.run(None, {iname: b})[0][0][:5] for b in probe_blobs], axis=0)
            a = preds.reshape(len(probe_blobs), -1).astype(np.float64)
            b = ref_raw.reshape(len(probe_blobs), -1).astype(np.float64)
            cos = float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
            rows.append({
                "block": grp_name,
                "nodes": len(grp_nodes),
                "cos_sim": round(cos, 6),
                "cos_drop_x1e4": round((1.0 - cos) * 10000.0, 3),
                "box_mae_px": round(float(np.mean(np.abs(preds[:, :4, :] - ref_raw[:, :4, :]))), 4),
                "score_mae": round(float(np.mean(np.abs(preds[:, 4, :] - ref_raw[:, 4, :]))), 6),
            })
        except Exception as exc:
            rows.append({"block": grp_name, "nodes": len(grp_nodes), "cos_sim": 0.0, "cos_drop_x1e4": 9999.0, "box_mae_px": 999.0, "score_mae": 1.0, "err": str(exc)[:60]})

    tmp_onnx.unlink(missing_ok=True)
    df = pd.DataFrame(rows).sort_values("box_mae_px", ascending=False).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SENSITIVITY] Audited {len(df)} blocks in {time.perf_counter() - t0:.1f}s -> {out_csv}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-block INT8 sensitivity audit for YOLO26n ONNX.")
    parser.add_argument("--fp32", type=Path, default=REPO_ROOT / "models" / "model_fp32.onnx")
    parser.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "calibration_data")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "datasets" / "dataset_manifest.json")
    parser.add_argument("--out-csv", type=Path, default=REPO_ROOT / "results" / "block_sensitivity_audit.csv")
    parser.add_argument("--num-probe", type=int, default=48)
    args = parser.parse_args()
    df = audit_blocks(args.fp32, args.calib_dir, args.out_csv, args.manifest, num_probe=args.num_probe)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
