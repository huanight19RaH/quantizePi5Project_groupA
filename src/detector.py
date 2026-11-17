"""Unified ONNX Runtime Person Detector wrapper for FP32, FP16, Dynamic INT8, and Static INT8 models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry metadata recorded during letterbox preprocessing for coordinate unscaling."""

    orig_w: int
    orig_h: int
    scale: float
    pad_x: int
    pad_y: int
    imgsz: int


def letterbox_bgr_to_rgb(
    img_bgr: np.ndarray,
    imgsz: int = 416,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize and pad a BGR image cleanly to square RGB `[imgsz, imgsz, 3]` matching YOLO letterbox."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Input image must be a non-empty HxWx3 BGR array.")
    oh, ow = img_bgr.shape[:2]
    scale = min(imgsz / ow, imgsz / oh)
    nw = int(round(ow * scale))
    nh = int(round(oh * scale))
    px = (imgsz - nw) // 2
    py = (imgsz - nh) // 2

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    canvas[py : py + nh, px : px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    meta = LetterboxMeta(orig_w=ow, orig_h=oh, scale=float(scale), pad_x=int(px), pad_y=int(py), imgsz=imgsz)
    return rgb, meta


def decode_person_predictions(
    raw_output: np.ndarray,
    meta: LetterboxMeta,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    pre_nms_topk: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw `[5, N]` or `[84, N]` YOLO26n output tensor into unscaled `[x1, y1, x2, y2]` boxes and scores."""
    arr = np.asarray(raw_output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] < 5:
        raise ValueError(f"Expected >= 5 channels in raw output, got shape {arr.shape}")

    scores_all = arr[4]
    idx = np.flatnonzero(scores_all >= conf_threshold)
    if idx.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    if idx.size > pre_nms_topk:
        sel = np.argpartition(scores_all[idx], -pre_nms_topk)[-pre_nms_topk:]
        idx = idx[sel]

    sc = scores_all[idx].astype(np.float32)
    cx = arr[0, idx]
    cy = arr[1, idx]
    bw = arr[2, idx]
    bh = arr[3, idx]

    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh

    inv_s = 1.0 / meta.scale
    ox1 = np.clip((x1 - meta.pad_x) * inv_s, 0.0, float(meta.orig_w))
    oy1 = np.clip((y1 - meta.pad_y) * inv_s, 0.0, float(meta.orig_h))
    ox2 = np.clip((x2 - meta.pad_x) * inv_s, 0.0, float(meta.orig_w))
    oy2 = np.clip((y2 - meta.pad_y) * inv_s, 0.0, float(meta.orig_h))

    boxes_xyxy = np.stack([ox1, oy1, ox2, oy2], axis=1).astype(np.float32)
    boxes_xywh = np.stack(
        [ox1, oy1, np.maximum(0.0, ox2 - ox1), np.maximum(0.0, oy2 - oy1)],
        axis=1,
    )

    keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), sc.tolist(), float(conf_threshold), float(iou_threshold))
    if len(keep) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    keep_idx = np.asarray(keep, dtype=np.int64).reshape(-1)
    order = np.argsort(-sc[keep_idx])[:max_det]
    keep_idx = keep_idx[order]
    return boxes_xyxy[keep_idx], sc[keep_idx]


class PersonDetector:
    """Unified ONNX Runtime person detector supporting FP32, FP16, Dynamic INT8, and Static INT8 models."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 416,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        num_threads: int = 4,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(num_threads)
        sess_opts.inter_op_num_threads = 1
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=self.providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in str(inp.type) else np.float32

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        """Convert BGR frame into normalized `[1, 3, H, W]` ONNX input tensor plus letterbox metadata."""
        rgb, meta = letterbox_bgr_to_rgb(img_bgr, imgsz=self.imgsz)
        blob = (rgb.transpose(2, 0, 1).astype(np.float32) * np.float32(1.0 / 255.0)).astype(self.input_dtype)
        return np.ascontiguousarray(blob[None, ...]), meta

    def infer_raw_from_rgb_nhwc(self, rgb_nhwc: np.ndarray) -> np.ndarray:
        """Run raw inference on pre-letterboxed uint8 RGB HxWx3 image and return FP32 `[5, 3549]` tensor."""
        blob = (rgb_nhwc.transpose(2, 0, 1).astype(np.float32) * np.float32(1.0 / 255.0)).astype(self.input_dtype)
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob[None, ...])})[0][0]
        return out[:5].astype(np.float32)

    def detect(self, img_bgr: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run end-to-end detection on a BGR frame and return detections plus stage timings in ms."""
        t0 = time.perf_counter()
        blob, meta = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: blob})[0][0][:5].astype(np.float32)
        t2 = time.perf_counter()
        boxes, scores = decode_person_predictions(
            raw,
            meta,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
        )
        t3 = time.perf_counter()

        detections = [
            {
                "bbox": [round(float(v), 2) for v in boxes[i].tolist()],
                "score": round(float(scores[i]), 5),
                "class_id": 0,
                "label": "person",
            }
            for i in range(len(scores))
        ]
        timings = {
            "preprocess_ms": round((t1 - t0) * 1000.0, 3),
            "inference_ms": round((t2 - t1) * 1000.0, 3),
            "postprocess_ms": round((t3 - t2) * 1000.0, 3),
            "total_ms": round((t3 - t0) * 1000.0, 3),
        }
        return detections, timings

    @staticmethod
    def draw_detections(img_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Overlay bounding boxes and confidence labels onto a copy of `img_bgr`."""
        vis = img_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            sc = det["score"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (16, 185, 129), 2)
            label = f"person {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (16, 185, 129), -1)
            cv2.putText(vis, label, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return vis
