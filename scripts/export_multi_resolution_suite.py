"""Multi-Resolution Tri-Core Hardware Exploitation Suite for Orange Pi 5 (RK3588).

This script exports and quantizes the TOP 3 models at MULTIPLE RESOLUTIONS:
  - Models: YOLO26n, YOLOv8n, YOLO11n
  - Resolutions: 320x320, 416x416, 512x512, 640x640
  - Quantization Tiers: FP32, FP16, INT8 Static Mixed-Precision (QDQ)

Outputs:
  artifacts/multi_resolution/<model_name>/<imgsz>/
    - model_fp32.onnx
    - model_fp16.onnx
    - model_int8_static.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import letterbox_preprocess

# Top 3 selected candidate models
TOP_MODELS = {
    "yolo26n": {
        "pt": PROJECT_ROOT / "yolo26n.pt",
        "description": "Speed & Overall Efficiency Leader (DFL-free, NMS-free friendly)",
    },
    "yolov8n": {
        "pt": PROJECT_ROOT / "yolov8n.pt",
        "description": "Small Person Detection Specialist (Highest SmallRecall, CSPDarknet53)",
    },
    "yolo11n": {
        "pt": PROJECT_ROOT / "yolo11n.pt",
        "description": "Occlusion & Anti-False-Alarm Specialist (C2PSA Attention)",
    },
}

RESOLUTIONS = [320, 416, 512, 640, 768, 896]


class MultiResCalibrationReader(CalibrationDataReader):
    """Feeds calibration images dynamically resized to target input size."""

    def __init__(
        self,
        calib_dir: Path,
        input_name: str = "images",
        input_size: tuple[int, int] = (416, 416),
        max_images: int = 60,
    ) -> None:
        self.calib_dir = Path(calib_dir)
        self.input_name = input_name
        self.input_size = input_size
        all_imgs = sorted(
            [p for p in self.calib_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        self.image_paths = all_imgs[:max_images]
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        while self._index < len(self.image_paths):
            p = self.image_paths[self._index]
            self._index += 1
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            blob, _ = letterbox_preprocess(img, input_size=self.input_size, input_color="BGR")
            return {self.input_name: blob}
        return None

    def rewind(self) -> None:
        self._index = 0


def discover_sensitive_nodes(onnx_path: Path) -> list[str]:
    """Find sensitive nodes to preserve in FP32 during INT8 quantization."""
    model = onnx.load(str(onnx_path))
    pattern = re.compile(r"^/model\.(\d+)/")
    layer_indices = set()
    for node in model.graph.node:
        m = pattern.match(node.name)
        if m:
            layer_indices.add(int(m.group(1)))

    head_prefix = f"/model.{max(layer_indices)}/" if layer_indices else ""
    static_excluded = set()

    for node in model.graph.node:
        nname = node.name
        is_sp = node.op_type in {"Resize", "Upsample", "Softmax"}
        is_head = head_prefix and nname.startswith(head_prefix)
        if nname.startswith("/model.0/") or "/attn/" in nname or is_sp or is_head:
            static_excluded.add(nname)

    return sorted(static_excluded)


def export_fp32(pt_path: Path, out_onnx: Path, imgsz: int) -> Path:
    """Export PyTorch checkpoint to ONNX with end2end=False."""
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    if out_onnx.exists():
        print(f"  [Skip] FP32 already exists: {out_onnx}")
        return out_onnx

    print(f"  [Export FP32] {pt_path.name} @ {imgsz}x{imgsz}...")
    model = YOLO(str(pt_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        opset=17,
        dynamic=False,
        simplify=True,
        end2end=False,
        device="cpu",
    )
    exported_path = Path(exported)
    if exported_path != out_onnx:
        shutil.move(str(exported_path), str(out_onnx))
    print(f"  [Done FP32] {out_onnx.name} ({out_onnx.stat().st_size / (1024*1024):.2f} MB)")
    return out_onnx


def convert_fp16(fp32_path: Path, fp16_path: Path) -> Path:
    """Convert FP32 ONNX to FP16 with keep_io_types=True."""
    if fp16_path.exists():
        print(f"  [Skip] FP16 already exists: {fp16_path}")
        return fp16_path

    print(f"  [Convert FP16] {fp32_path.name} -> {fp16_path.name}...")
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        model = onnx.load(str(fp32_path))
        model_fp16 = convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(fp16_path))
    except Exception:
        import onnxconverter_common
        model = onnx.load(str(fp32_path))
        model_fp16 = onnxconverter_common.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=True,
        )
        onnx.save(model_fp16, str(fp16_path))
    print(f"  [Done FP16] {fp16_path.name} ({fp16_path.stat().st_size / (1024*1024):.2f} MB)")
    return fp16_path


def quantize_int8_static(
    fp32_path: Path,
    int8_path: Path,
    calib_reader: MultiResCalibrationReader,
    excluded_nodes: list[str],
) -> Path:
    """Quantize to Static Mixed-Precision INT8 with QDQ format."""
    if int8_path.exists():
        print(f"  [Skip] INT8 Static already exists: {int8_path}")
        return int8_path

    print(f"  [Quantize INT8] {fp32_path.name} -> {int8_path.name}...")
    prep_path = int8_path.parent / f"{int8_path.stem}_prep.onnx"
    try:
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(prep_path),
            auto_merge=True,
        )
        src = prep_path
    except Exception as exc:
        print(f"    [Warn] quant_pre_process failed ({exc}), using raw model.")
        src = fp32_path

    quantize_static(
        model_input=str(src),
        model_output=str(int8_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        nodes_to_exclude=excluded_nodes,
    )

    if prep_path.exists():
        prep_path.unlink()

    print(f"  [Done INT8] {int8_path.name} ({int8_path.stat().st_size / (1024*1024):.2f} MB)")
    return int8_path


def main():
    parser = argparse.ArgumentParser(description="Export and quantize top 3 models across multiple resolutions")
    parser.add_argument("--resolutions", nargs="+", type=int, default=RESOLUTIONS, help="Image sizes (e.g. 320 416 512 640)")
    parser.add_argument("--models", nargs="+", default=["yolo26n", "yolov8n", "yolo11n"], help="Subset of models to build")
    parser.add_argument("--calib-dir", type=str, default="data/calibration/images", help="Calibration images folder")
    parser.add_argument("--calib-samples", type=int, default=50, help="Number of calibration samples")
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir)
    if not calib_dir.exists():
        # Fallback to test or train images if calibration/images is empty
        alt_dirs = [
            Path("data/raw/roboflow_cctv_naxyo_v2/train/images"),
            Path("data/raw/roboflow_cctv_naxyo_v2/test/images"),
            Path("orangepi5_benchmark/calibration_data"),
        ]
        for ad in alt_dirs:
            if ad.exists():
                calib_dir = ad
                break

    print(f"\n=======================================================")
    print(f" MULTI-RESOLUTION HARDWARE SUITE BUILDER (ORANGE PI 5)")
    print(f" Selected Models : {args.models}")
    print(f" Resolutions     : {args.resolutions}")
    print(f" Calibration Dir : {calib_dir}")
    print(f"=======================================================\n")

    out_base = PROJECT_ROOT / "artifacts" / "multi_resolution"

    for model_name in args.models:
        if model_name not in TOP_MODELS:
            print(f"[WARN] Unknown model: {model_name}, skipping.")
            continue

        info = TOP_MODELS[model_name]
        pt_path = info["pt"]
        if not pt_path.exists():
            print(f"[ERROR] Checkpoint not found: {pt_path}, skipping.")
            continue

        print(f"\n>>> PROCESSING MODEL: {model_name.upper()} ({info['description']})")

        for res in args.resolutions:
            print(f"\n--- Resolution: {res}x{res} ---")
            target_dir = out_base / model_name / f"res_{res}"
            target_dir.mkdir(parents=True, exist_ok=True)

            fp32_onnx = target_dir / "model_fp32.onnx"
            fp16_onnx = target_dir / "model_fp16.onnx"
            int8_onnx = target_dir / "model_int8_static.onnx"

            # 1. Export FP32
            export_fp32(pt_path, fp32_onnx, res)

            # 2. Convert FP16
            try:
                convert_fp16(fp32_onnx, fp16_onnx)
            except Exception as e:
                print(f"  [ERROR] FP16 conversion failed: {e}")

            # 3. Quantize INT8 Static Mixed-Precision
            try:
                excluded = discover_sensitive_nodes(fp32_onnx)
                calib_reader = MultiResCalibrationReader(
                    calib_dir=calib_dir,
                    input_size=(res, res),
                    max_images=args.calib_samples,
                )
                quantize_int8_static(fp32_onnx, int8_onnx, calib_reader, excluded)
            except Exception as e:
                print(f"  [ERROR] INT8 Quantization failed: {e}")

    print("\n[SUCCESS] Multi-resolution suite build complete!")
    print(f"Outputs saved to: {out_base}\n")


if __name__ == "__main__":
    main()
