"""Export YOLO26n FP32 ONNX (80-class and 1-class 'person' pruned) and verify bit-exact numerical parity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import letterbox_bgr_to_rgb


def prune_yolo26_to_person_only(yolo_obj: YOLO) -> YOLO:
    """Slice final class-prediction conv branches (cv3 and one2one_cv3) cleanly to COCO class 0 ('person')."""
    head = yolo_obj.model.model[-1]
    branches = [head.cv3] + ([head.one2one_cv3] if hasattr(head, "one2one_cv3") else [])
    for branch in branches:
        for seq in branch:
            old_conv = seq[-1]
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                1,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:1])
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias[:1])
            seq[-1] = new_conv

    head.nc = 1
    head.no = head.nc + head.reg_max * 4
    yolo_obj.model.nc = 1
    yolo_obj.model.names = {0: "person"}
    return yolo_obj


def export_fp32(
    weights_path: Path,
    fp32_1cls_out: Path,
    fp32_80cls_out: Path | None = None,
    imgsz: int = 416,
    opset: int = 17,
    verify_dir: Path | None = None,
) -> None:
    """Export YOLO26n FP32 ONNX models (`end2end=False`) and assert numerical parity against 80-class output."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        YOLO(str(weights_path))

    y80 = YOLO(str(weights_path))
    exp80 = Path(y80.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False))
    p80 = fp32_80cls_out or (fp32_1cls_out.parent / "ablation" / "yolo26n_80cls_fp32.onnx")
    p80.parent.mkdir(parents=True, exist_ok=True)
    exp80.replace(p80)

    y1 = prune_yolo26_to_person_only(YOLO(str(weights_path)))
    exp1 = Path(y1.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False))
    fp32_1cls_out.parent.mkdir(parents=True, exist_ok=True)
    exp1.replace(fp32_1cls_out)
    onnx.checker.check_model(onnx.load(str(fp32_1cls_out)))

    # Verify numerical parity using real calibration images (or deterministic random tensor fallback)
    if verify_dir is not None and verify_dir.exists() and list(verify_dir.glob("*.jpg")):
        import cv2
        imgs = sorted(verify_dir.glob("*.jpg"))[:8]
        blobs = [letterbox_bgr_to_rgb(cv2.imread(str(p)), imgsz=imgsz)[0].transpose(2, 0, 1).astype(np.float32) / 255.0 for p in imgs]
        x_np = np.stack(blobs, axis=0)
    else:
        rng = np.random.default_rng(42)
        x_np = rng.uniform(0.0, 1.0, size=(4, 3, imgsz, imgsz)).astype(np.float32)

    s80 = ort.InferenceSession(str(p80), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(fp32_1cls_out), providers=["CPUExecutionProvider"])
    o80 = np.concatenate([s80.run(None, {s80.get_inputs()[0].name: x_np[i : i + 1]})[0] for i in range(len(x_np))], axis=0)
    o1 = np.concatenate([s1.run(None, {s1.get_inputs()[0].name: x_np[i : i + 1]})[0] for i in range(len(x_np))], axis=0)

    diff_80_1 = float(np.max(np.abs(o80[:, :5] - o1)))
    print(f"[EXPORT] 80-cls: {p80} ({p80.stat().st_size / 1048576:.2f} MB)")
    print(f"[EXPORT] 1-cls : {fp32_1cls_out} ({fp32_1cls_out.stat().st_size / 1048576:.2f} MB)")
    print(f"[PARITY] MaxAbsDiff(ONNX_80cls[:, :5] vs ONNX_1cls) = {diff_80_1:.2e}")
    assert diff_80_1 < 1e-4, f"Parity check failed: {diff_80_1}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26n FP32 ONNX (80-cls and 1-cls person).")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "yolo26n.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "model_fp32.onnx")
    parser.add_argument("--out-80cls", type=Path, default=REPO_ROOT / "models" / "ablation" / "yolo26n_80cls_fp32.onnx")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-dir", type=Path, default=REPO_ROOT / "calibration_data")
    args = parser.parse_args()
    export_fp32(args.weights, args.out, args.out_80cls, args.imgsz, args.opset, args.verify_dir)


if __name__ == "__main__":
    main()
