"""Quantization & Optimization Pipeline for YOLO26n Person Detection.

Produces:
  - models/model_fp16.onnx              (ORT float16 transformer pass with keep_io_types=True)
  - models/model_int8_dynamic.onnx      (QUInt8 dynamic weight-only quantization excluding head/Resize/Softmax)
  - models/model_int8_static.onnx       (Pareto Winner: Sensitivity-guided Head+M0+M1+M2 FP32 + C2PSA INT8 + Percentile 99.995%)
  - models/ablation/...                 (Optional full ablation suite when --ablation is passed)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.transformers import float16 as ort_float16

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import letterbox_bgr_to_rgb


class ImageFolderCalibrationReader(CalibrationDataReader):
    """Calibration data reader applying identical YOLO26n 416x416 letterbox preprocessing."""

    def __init__(self, image_paths: list[Path], imgsz: int = 416, input_name: str = "images") -> None:
        self.image_paths = list(image_paths)
        self.imgsz = int(imgsz)
        self.input_name = input_name
        self.pos = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.pos >= len(self.image_paths):
            return None
        path = self.image_paths[self.pos]
        self.pos += 1
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read calibration image: {path}")
        rgb, _ = letterbox_bgr_to_rgb(img_bgr, imgsz=self.imgsz)
        blob = (rgb.transpose(2, 0, 1).astype(np.float32) * np.float32(1.0 / 255.0))[None, ...]
        return {self.input_name: np.ascontiguousarray(blob)}

    def rewind(self) -> None:
        self.pos = 0


def build_exclude_list(
    onnx_path: Path,
    exclude_prefixes: list[str],
    include_attn: bool = False,
    exclude_op_types: tuple[str, ...] = ("Resize", "Upsample", "Softmax"),
) -> list[str]:
    """Select exact node names to keep in FP32 while allowing remaining layers in INT8 QDQ."""
    model = onnx.load(str(onnx_path))
    excl: set[str] = set()
    for node in model.graph.node:
        name = node.name
        if node.op_type in exclude_op_types:
            excl.add(name)
        if include_attn and "/attn/" in name:
            excl.add(name)
        for prefix in exclude_prefixes:
            if name.startswith(prefix):
                excl.add(name)
    return sorted(excl)


def audit_fused_ort_ops(onnx_path: Path) -> dict[str, object]:
    """Inspect ONNX Runtime's post-optimization CPU graph for fused QLinearConv/Conv/QDQ counts."""
    tmp_opt = onnx_path.with_name(f"_tmp_ort_opt_{onnx_path.name}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(tmp_opt)
    _ = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    mf = onnx.load(str(tmp_opt))
    tmp_opt.unlink(missing_ok=True)
    counts = Counter(n.op_type for n in mf.graph.node)
    qconv = counts.get("QLinearConv", 0) + counts.get("ConvInteger", 0)
    fconv = counts.get("Conv", 0)
    return {
        "int8_convs": qconv,
        "fp32_convs": fconv,
        "int8_conv_pct": round(100.0 * qconv / max(1, qconv + fconv), 1),
        "int8_matmuls": counts.get("QLinearMatMul", 0) + counts.get("MatMulInteger", 0),
        "boundary_q_dq": f"{counts.get('QuantizeLinear', 0)}/{counts.get('DequantizeLinear', 0)}",
    }


def resolve_calibration_images(calib_dir: Path, manifest_path: Path | None, max_images: int = 200) -> list[Path]:
    """Load calibration images matching Calib_200 manifest ordering if present, else sorted directory scan."""
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        wanted = [r["file_name"] for r in manifest["calib_records"] if r.get("in_calib200", False)][:max_images]
        paths = [calib_dir / fn for fn in wanted if (calib_dir / fn).exists()]
        if len(paths) == len(wanted) and paths:
            return paths
    paths = sorted(calib_dir.glob("*.jpg"))[:max_images]
    if not paths:
        raise FileNotFoundError(f"No calibration images found in {calib_dir}. Run `python scripts/prepare_datasets.py --calib-only` first.")
    return paths


def run_pipeline(
    fp32_path: Path,
    models_dir: Path,
    calib_dir: Path,
    manifest_path: Path | None = None,
    imgsz: int = 416,
    run_ablation: bool = False,
) -> None:
    """Convert FP32 ONNX into FP16, Dynamic INT8, and Sensitivity-Guided Static INT8 models."""
    ablation_dir = models_dir / "ablation"
    models_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    pre_path = ablation_dir / "model_fp32_pre.onnx"
    if not pre_path.exists():
        print(f"[PREP] Running quant_pre_process -> {pre_path}")
        quant_pre_process(str(fp32_path), str(pre_path))

    # 1. FP16 conversion using ORT transformer pass (safe across Resize Cast types)
    fp16_path = models_dir / "model_fp16.onnx"
    print(f"[FP16] Converting {fp32_path.name} -> {fp16_path.name}")
    m_fp16 = ort_float16.convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
    onnx.save(m_fp16, str(fp16_path))

    # 2. Dynamic INT8 conversion (head & Resize/Upsample/Softmax excluded)
    dyn_path = models_dir / "model_int8_dynamic.onnx"
    dyn_excl = build_exclude_list(pre_path, exclude_prefixes=["/model.23/"], include_attn=False)
    print(f"[INT8-DYN] Converting -> {dyn_path.name} (excluded={len(dyn_excl)} nodes)")
    quantize_dynamic(
        model_input=str(pre_path),
        model_output=str(dyn_path),
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=dyn_excl,
    )

    # Calibration images
    calib_imgs = resolve_calibration_images(calib_dir, manifest_path, max_images=200)
    print(f"[CALIB] Using {len(calib_imgs)} disjoint images from {calib_dir}")

    # 3. Winning Sensitivity-Guided Static INT8 (`INT8_HeadM0M1M2_Pct99995_C200`)
    best_static_path = models_dir / "model_int8_static.onnx"
    cache_pct99995 = models_dir / "calib_cache_pct99995_200.json"
    best_excl = build_exclude_list(
        pre_path,
        exclude_prefixes=["/model.0/", "/model.1/", "/model.2/", "/model.23/"],
        include_attn=False,
    )
    t0 = time.perf_counter()
    print(f"[INT8-STATIC-BEST] Building Head+M0+M1+M2 FP32 + C2PSA INT8 + Percentile 99.995% (excluded={len(best_excl)})")
    quantize_static(
        model_input=str(pre_path),
        model_output=str(best_static_path),
        calibration_data_reader=None if cache_pct99995.exists() else ImageFolderCalibrationReader(calib_imgs, imgsz=imgsz),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        nodes_to_exclude=best_excl,
        extra_options={"CalibPercentile": 99.995, "CalibTensorRangeSymmetric": False},
        calibration_cache_path=str(cache_pct99995),
    )
    print(f"[INT8-STATIC-BEST] Built {best_static_path} in {time.perf_counter() - t0:.2f}s | Audit: {audit_fused_ort_ops(best_static_path)}")

    if run_ablation:
        variants = [
            ("model_int8_static_naive.onnx", False, CalibrationMethod.MinMax, {}, [], False, "calib_cache_minmax_200.json"),
            ("model_int8_stdmix_minmax_c200.onnx", True, CalibrationMethod.MinMax, {}, ["/model.0/", "/model.23/"], True, "calib_cache_minmax_200.json"),
            ("model_int8_stdmix_entropy_c200.onnx", True, CalibrationMethod.Entropy, {}, ["/model.0/", "/model.23/"], True, "calib_cache_entropy_200.json"),
            ("model_int8_stdmix_pct9999_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.23/"], True, "calib_cache_pct9999_200.json"),
            ("model_int8_headonly_pct9999_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}, ["/model.23/"], False, "calib_cache_pct9999_200.json"),
            ("model_int8_sens_m1_pct9999_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.1/", "/model.23/"], True, "calib_cache_pct9999_200.json"),
            ("model_int8_sens_m1m2_pct9999_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.1/", "/model.2/", "/model.23/"], True, "calib_cache_pct9999_200.json"),
            ("model_int8_head_m012_pct9999_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.99, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.1/", "/model.2/", "/model.23/"], False, "calib_cache_pct9999_200.json"),
            ("model_int8_sens_m1m2_pct99995_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.995, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.1/", "/model.2/", "/model.23/"], True, "calib_cache_pct99995_200.json"),
            ("model_int8_sens_m12_16_4_pct99995_c200.onnx", True, CalibrationMethod.Percentile, {"CalibPercentile": 99.995, "CalibTensorRangeSymmetric": False}, ["/model.0/", "/model.1/", "/model.2/", "/model.4/", "/model.16/", "/model.23/"], True, "calib_cache_pct99995_200.json"),
        ]
        for fname, per_ch, meth, opts, pfxs, inc_attn, cfile in variants:
            out_p = ablation_dir / fname
            c_p = models_dir / cfile
            excl = [] if not pfxs else build_exclude_list(pre_path, pfxs, include_attn=inc_attn)
            quantize_static(
                model_input=str(pre_path),
                model_output=str(out_p),
                calibration_data_reader=None if c_p.exists() else ImageFolderCalibrationReader(calib_imgs, imgsz=imgsz),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=per_ch,
                calibrate_method=meth,
                nodes_to_exclude=excl,
                extra_options=opts,
                calibration_cache_path=str(c_p) if pfxs else None,
            )
            print(f"[ABLATION] {fname}: {audit_fused_ort_ops(out_p)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO26n ONNX quantization pipeline.")
    parser.add_argument("--fp32", type=Path, default=REPO_ROOT / "models" / "model_fp32.onnx")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "calibration_data")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "datasets" / "dataset_manifest.json")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--ablation", action="store_true", help="Also build all 10 ablation variants in models/ablation/.")
    args = parser.parse_args()
    run_pipeline(args.fp32, args.models_dir, args.calib_dir, args.manifest, args.imgsz, args.ablation)


if __name__ == "__main__":
    main()
