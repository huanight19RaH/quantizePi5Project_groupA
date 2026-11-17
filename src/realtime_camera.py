"""Real-time camera, video, or single-image person detection demo using PersonDetector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import PersonDetector


def build_detector(args):
    """Instantiate ONNX CPU detector or RK3588 Zero-Copy NPU detector based on model suffix/flags."""
    backend = args.backend
    if backend == "auto":
        backend = "npu" if args.model.suffix.lower() == ".rknn" else "onnx"
    if backend == "npu":
        from src.npu_detector import NPUDetector
        return NPUDetector(
            args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            n_workers=args.workers,
            zero_copy=True,
            pass_through_out=True,
            use_rga=args.rga,
            pin_a76=True,
        )
    return PersonDetector(
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        num_threads=args.threads,
        fast_mode=not args.legacy_cpu,
        pin_a76=args.pin_a76,
    )


def run_image(detector, image_path: Path, out_path: Path | None) -> None:
    """Run person detection on a static image and optionally save annotated preview."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    dets, timings = detector.detect(img)
    print(json.dumps({"image": str(image_path), "model": detector.model_path.name, "timings_ms": timings, "detections": dets}, indent=2))
    if out_path is not None:
        vis = PersonDetector.draw_detections(img, dets)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)


def run_stream(
    detector,
    source: str,
    headless: bool = False,
    save_video: Path | None = None,
    max_frames: int = 0,
) -> None:
    """Run real-time stream inference with FPS overlay."""
    cap_src: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    writer = None
    ema_fps = 0.0
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets, timings = detector.detect(frame)
            inst_fps = 1000.0 / max(1e-3, timings["total_ms"])
            ema_fps = inst_fps if frames == 0 else (0.9 * ema_fps + 0.1 * inst_fps)
            frames += 1

            vis = PersonDetector.draw_detections(frame, dets)
            hud = f"{detector.model_path.name} | {ema_fps:.1f} FPS | inf={timings['inference_ms']:.1f}ms | person={len(dets)}"
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if save_video is not None:
                if writer is None:
                    save_video.parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    writer = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)

            if not headless:
                cv2.imshow("Team_A PersonDetector", vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        print(json.dumps({"frames": frames, "final_ema_fps": round(ema_fps, 2)}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Team_A Real-Time Person Detection Demo.")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "model_int8_static.onnx")
    parser.add_argument("--backend", choices=["auto", "onnx", "npu"], default="auto")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0), video path, or image path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rga", action="store_true", help="Enable RK3588 2D RGA hardware letterbox (NPU backend).")
    parser.add_argument("--pin-a76", action="store_true", help="Pin ONNX CPU inference onto Cortex-A76 cores 4-7.")
    parser.add_argument("--legacy-cpu", action="store_true", help="Use unoptimized baseline preprocess on CPU.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-out", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    detector = build_detector(args)
    try:
        src_path = Path(args.source)
        if src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and src_path.exists():
            run_image(detector, src_path, args.save_out)
        else:
            run_stream(detector, args.source, headless=args.headless, save_video=args.save_out, max_frames=args.max_frames)
    finally:
        if hasattr(detector, "close"):
            detector.close()


if __name__ == "__main__":
    main()
