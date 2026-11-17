"""Quick parity check on Orange Pi 5 comparing ONNX FP32 vs NPUDetector (ZeroCopy CPU U8 & RGA HW)."""

from pathlib import Path
import sys
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detector import PersonDetector
from src.npu_detector import NPUDetector


def main() -> None:
    img = cv2.imread(str(REPO_ROOT / "calibration_data" / "000000309695.jpg"), cv2.IMREAD_COLOR)
    d_onnx = PersonDetector(REPO_ROOT / "models" / "model_fp32.onnx", conf_threshold=0.25, iou_threshold=0.45)
    p_o, t_o = d_onnx.detect(img)
    print("ONNX FP32:", p_o, t_o)

    names = sys.argv[1:] if len(sys.argv) > 1 else ["model_int8_6head_normal.rknn", "model_int8_6head_hybrid.rknn"]
    for name in names:
        rknn_p = REPO_ROOT / "models" / "rknn" / name
        print(f"\n=== Checking {name} ===")
        for zc in (False, True):
            for pt in ([False] if not zc else [False, True]):
                for rga in ([False] if not zc else [False, True]):
                    d_npu = NPUDetector(
                        rknn_p,
                        conf_threshold=0.25,
                        iou_threshold=0.45,
                        n_workers=1,
                        zero_copy=zc,
                        pass_through_out=pt,
                        use_rga=rga,
                    )
                    if zc and pt and not rga:
                        print("NPU INFO:", d_npu.info())
                    d_npu.detect(img)
                    p_n, t_n = d_npu.detect(img)
                    print(f"NPU (zc={zc}, pt={pt}, rga={rga}): dets={p_n} | timings={t_n}")
                    d_npu.close()


if __name__ == "__main__":
    main()
