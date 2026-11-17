"""Unified RK3588 NPU Person Detector backed by C++ Zero-Copy + RGA + Lazy Integer Decode (`libyolo26_rk3588.so`)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _CY26Det(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
    ]


class _CY26Timings(ctypes.Structure):
    _fields_ = [
        ("preproc_ms", ctypes.c_double),
        ("npu_ms", ctypes.c_double),
        ("postproc_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("rga_used", ctypes.c_int),
        ("used_fallback", ctypes.c_int),
    ]


class _CY26EngineInfo(ctypes.Structure):
    _fields_ = [
        ("n_workers", ctypes.c_int),
        ("n_outputs", ctypes.c_int),
        ("imgsz", ctypes.c_int),
        ("is_int8_outputs", ctypes.c_int),
        ("use_zero_copy", ctypes.c_int),
        ("weight_bytes", ctypes.c_uint32),
        ("internal_bytes", ctypes.c_uint32),
        ("total_dma_bytes", ctypes.c_uint64),
    ]


def _load_lib(lib_path: Path | None = None) -> ctypes.CDLL:
    candidate = lib_path or (REPO_ROOT / "cpp" / "libyolo26_rk3588.so")
    if not candidate.exists():
        raise FileNotFoundError(
            f"C++ NPU shared library not found: {candidate}. Build on Orange Pi 5 with `make -C cpp`."
        )
    lib = ctypes.CDLL(str(candidate))
    lib.y26_create_engine.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.y26_create_engine.restype = ctypes.c_void_p

    lib.y26_destroy_engine.argtypes = [ctypes.c_void_p]
    lib.y26_destroy_engine.restype = None

    lib.y26_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CY26EngineInfo)]
    lib.y26_get_info.restype = ctypes.c_int

    lib.y26_detect_bgr.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(_CY26Det),
        ctypes.c_int,
        ctypes.POINTER(_CY26Timings),
    ]
    lib.y26_detect_bgr.restype = ctypes.c_int

    lib.y26_bench_prepacked_multicore.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.y26_bench_prepacked_multicore.restype = ctypes.c_int
    return lib


class NPUDetector:
    """High-speed RK3588 NPU PersonDetector using C++ Zero-Copy DMA, RGA resize, and lazy int8 decode."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        n_workers: int = 1,
        core_mask_mode: int = 1,
        zero_copy: bool = True,
        pass_through_out: bool = True,
        use_rga: bool = False,
        enable_sram: bool = False,
        pin_a76: bool = True,
        lib_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self._lib = _load_lib(Path(lib_path) if lib_path else None)
        self._handle = self._lib.y26_create_engine(
            str(self.model_path).encode("utf-8"),
            int(n_workers),
            int(core_mask_mode),
            int(bool(zero_copy)),
            int(bool(pass_through_out)),
            1 if use_rga else 0,
            int(bool(enable_sram)),
            int(bool(pin_a76)),
        )
        if not self._handle:
            raise RuntimeError(f"Failed to initialize RK3588 NPU engine for {self.model_path}")
        self._det_buf = (_CY26Det * max(32, self.max_det))()

    def info(self) -> dict[str, int]:
        """Return runtime NPU tensor & memory metadata."""
        st = _CY26EngineInfo()
        self._lib.y26_get_info(self._handle, ctypes.byref(st))
        return {
            "n_workers": st.n_workers,
            "n_outputs": st.n_outputs,
            "imgsz": st.imgsz,
            "is_int8_outputs": st.is_int8_outputs,
            "use_zero_copy": st.use_zero_copy,
            "weight_bytes": st.weight_bytes,
            "internal_bytes": st.internal_bytes,
            "total_dma_bytes": st.total_dma_bytes,
        }

    def detect(self, img_bgr: np.ndarray, worker_id: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run C++ NPU inference on an HxWx3 BGR frame and return detections plus microsecond-accurate timings."""
        if img_bgr is None or img_bgr.ndim != 3:
            raise ValueError("Expected non-empty HxWx3 BGR ndarray")
        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)
        oh, ow = img_bgr.shape[:2]
        stride = int(img_bgr.strides[0])
        ptr = img_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        timings = _CY26Timings()
        n = self._lib.y26_detect_bgr(
            self._handle,
            int(worker_id),
            ptr,
            int(ow),
            int(oh),
            stride,
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            self._det_buf,
            len(self._det_buf),
            ctypes.byref(timings),
        )
        if n < 0:
            raise RuntimeError(f"y26_detect_bgr failed with code {n}")
        detections = [
            {
                "bbox": [
                    round(float(self._det_buf[i].x1), 2),
                    round(float(self._det_buf[i].y1), 2),
                    round(float(self._det_buf[i].x2), 2),
                    round(float(self._det_buf[i].y2), 2),
                ],
                "score": round(float(self._det_buf[i].score), 5),
                "class_id": 0,
                "label": "person",
            }
            for i in range(n)
        ]
        return detections, {
            "preprocess_ms": round(timings.preproc_ms, 3),
            "inference_ms": round(timings.npu_ms, 3),
            "postprocess_ms": round(timings.postproc_ms, 3),
            "total_ms": round(timings.total_ms, 3),
            "rga_used": bool(timings.rga_used),
        }

    def bench_prepacked_multicore(self, bgr_images: list[np.ndarray], loops: int = 1, pin_a76: bool = True) -> dict[str, float]:
        """Execute true multi-threaded C++ multi-core NPU throughput benchmark across preloaded BGR frames."""
        n = len(bgr_images)
        kept = [img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img) for img in bgr_images]
        ptrs = (ctypes.c_void_p * n)(*[img.ctypes.data for img in kept])
        ws = (ctypes.c_int * n)(*[int(img.shape[1]) for img in kept])
        hs = (ctypes.c_int * n)(*[int(img.shape[0]) for img in kept])

        wall_ms = ctypes.c_double()
        m_pre = ctypes.c_double()
        m_npu = ctypes.c_double()
        m_post = ctypes.c_double()
        p50 = ctypes.c_double()
        p95 = ctypes.c_double()
        p99 = ctypes.c_double()

        rc = self._lib.y26_bench_prepacked_multicore(
            self._handle,
            ptrs,
            ws,
            hs,
            n,
            int(loops),
            float(self.conf_threshold),
            float(self.iou_threshold),
            int(self.max_det),
            1 if pin_a76 else 0,
            ctypes.byref(wall_ms),
            ctypes.byref(m_pre),
            ctypes.byref(m_npu),
            ctypes.byref(m_post),
            ctypes.byref(p50),
            ctypes.byref(p95),
            ctypes.byref(p99),
        )
        if rc != 0:
            raise RuntimeError(f"C++ multi-core benchmark failed: {rc}")
        total_frames = n * loops
        fps = (1000.0 * total_frames) / max(1e-6, wall_ms.value)
        return {
            "total_frames": total_frames,
            "wall_ms": round(wall_ms.value, 2),
            "fps": round(fps, 2),
            "mean_pre_ms": round(m_pre.value, 3),
            "mean_npu_ms": round(m_npu.value, 3),
            "mean_post_ms": round(m_post.value, 3),
            "p50_ms": round(p50.value, 3),
            "p95_ms": round(p95.value, 3),
            "p99_ms": round(p99.value, 3),
        }

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.y26_destroy_engine(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
