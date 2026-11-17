"""Convert Team_A YOLO26n FP32 ONNX into RK3588 NPU (.rknn) variants using exact letterboxed Calib_200 data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import letterbox_bgr_to_rgb

HEAD_6_OUTPUTS = [
    "/model.23/cv2.0/cv2.0.2/Conv_output_0",
    "/model.23/cv3.0/cv3.0.2/Conv_output_0",
    "/model.23/cv2.1/cv2.1.2/Conv_output_0",
    "/model.23/cv3.1/cv3.1.2/Conv_output_0",
    "/model.23/cv2.2/cv2.2.2/Conv_output_0",
    "/model.23/cv3.2/cv3.2.2/Conv_output_0",
]


def prepare_calib_npy(
    manifest_path: Path,
    calib_dir: Path,
    npy_dir: Path,
    imgsz: int = 416,
    max_images: int = 200,
) -> Path:
    """Create exact RGB NCHW (1, 3, H, W) float32 [0..255] letterbox `.npy` blobs for rknn-toolkit2."""
    npy_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [calib_dir / r["file_name"] for r in manifest["calib_records"] if r.get("in_calib200")][:max_images]
    else:
        files = sorted(calib_dir.glob("*.jpg"))[:max_images]

    list_lines: list[str] = []
    for jpg_p in files:
        out_npy = npy_dir / f"{jpg_p.stem}.npy"
        need_write = True
        if out_npy.exists():
            try:
                arr = np.load(out_npy, mmap_mode="r")
                if arr.shape == (1, 3, imgsz, imgsz) and arr.dtype == np.float32:
                    need_write = False
            except Exception:
                need_write = True
        if need_write:
            img = cv2.imread(str(jpg_p), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read calibration image: {jpg_p}")
            rgb, _ = letterbox_bgr_to_rgb(img, imgsz=imgsz)
            blob = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)
            np.save(out_npy, blob)
        list_lines.append(str(out_npy.resolve()))

    list_file = npy_dir / "calib200_npy.txt"
    list_file.write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    # Stratified 24-image subset preserving pos/neg (~54%/46%) for fast MMSE search
    subset_24 = [list_lines[i] for i in range(0, len(list_lines), max(1, len(list_lines) // 24))][:24]
    list_24_file = npy_dir / "calib24_npy.txt"
    list_24_file.write_text("\n".join(subset_24) + "\n", encoding="utf-8")
    print(f"[CALIB-NPY] Prepared {len(list_lines)} exact RGB NCHW blobs -> {list_file} (+ 24-img subset)")
    return list_file, list_24_file


def build_rknn_variant(
    onnx_path: Path,
    out_rknn: Path,
    outputs: list[str] | None,
    do_quant: bool,
    dataset_txt: Path | None,
    quant_algo: str = "normal",
    auto_hybrid: bool = False,
    auto_hybrid_cos: float = 0.99,
    single_core_mode: bool = False,
) -> None:
    """Compile ONNX into a target RK3588 .rknn artifact."""
    from rknn.api import RKNN

    t0 = time.perf_counter()
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            quantized_dtype="w8a8",
            quantized_algorithm=quant_algo,
            quantized_method="channel",
            target_platform="rk3588",
            optimization_level=3,
            single_core_mode=single_core_mode,
            auto_hybrid_cos_thresh=auto_hybrid_cos,
            disable_rules=["fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa"],
        )
        assert ret == 0, f"rknn.config failed ({ret})"

        ret = rknn.load_onnx(model=str(onnx_path), outputs=outputs)
        assert ret == 0, f"rknn.load_onnx failed ({ret})"

        ret = rknn.build(
            do_quantization=do_quant,
            dataset=str(dataset_txt) if do_quant else None,
            auto_hybrid=auto_hybrid,
        )
        assert ret == 0, f"rknn.build failed ({ret})"

        out_rknn.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(out_rknn))
        assert ret == 0, f"rknn.export_rknn failed ({ret})"
        sz_mb = out_rknn.stat().st_size / 1048576
        print(f"[RKNN] Built {out_rknn.name:34s} | size={sz_mb:5.2f} MiB | time={time.perf_counter() - t0:6.1f}s")
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RK3588 .rknn variants from YOLO26n FP32 ONNX.")
    parser.add_argument("--onnx", type=Path, default=REPO_ROOT / "models" / "model_fp32.onnx")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "models" / "rknn")
    parser.add_argument("--calib-dir", type=Path, default=REPO_ROOT / "calibration_data")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "datasets" / "dataset_manifest.json")
    parser.add_argument("--npy-dir", type=Path, default=REPO_ROOT / "datasets" / "calib200_npy")
    parser.add_argument("--variants", nargs="+", default=["fp16_uncut", "int8_uncut", "int8_6head_normal", "int8_6head_kl", "int8_6head_hybrid", "int8_6head_mmse24"])
    args = parser.parse_args()

    dataset_200, dataset_24 = prepare_calib_npy(args.manifest, args.calib_dir, args.npy_dir)

    specs = {
        "fp16_uncut": ("model_fp16_uncut.rknn", None, False, "normal", False, False, None),
        "int8_uncut": ("model_int8_uncut_normal.rknn", None, True, "normal", False, False, dataset_200),
        "int8_6head_normal": ("model_int8_6head_normal.rknn", HEAD_6_OUTPUTS, True, "normal", False, False, dataset_200),
        "int8_6head_kl": ("model_int8_6head_kl.rknn", HEAD_6_OUTPUTS, True, "kl_divergence", False, False, dataset_200),
        "int8_6head_mmse24": ("model_int8_6head_mmse24.rknn", HEAD_6_OUTPUTS, True, "mmse", False, False, dataset_24),
        "int8_6head_normal_sc": ("model_int8_6head_normal_sc.rknn", HEAD_6_OUTPUTS, True, "normal", False, True, dataset_200),
        "int8_6head_hybrid": ("model_int8_6head_hybrid.rknn", HEAD_6_OUTPUTS, True, "normal", True, False, dataset_200),
    }

    for v in args.variants:
        fname, outs, quant, algo, ahyb, sc, ds = specs[v]
        build_rknn_variant(
            args.onnx,
            args.out_dir / fname,
            outs,
            quant,
            ds,
            quant_algo=algo,
            auto_hybrid=ahyb,
            auto_hybrid_cos=0.992,
            single_core_mode=sc,
        )


if __name__ == "__main__":
    main()
