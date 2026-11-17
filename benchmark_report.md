# Benchmark Report — Team_A (`YOLO26n` Person Detection on Orange Pi 5 / RK3588S)

## 1. Executive Summary

- **Task:** Single-class `person` object detection (`YOLO26n`, input `416×416`, `opset=17`, `end2end=False`).
- **Target Edge Device:** Orange Pi 5 (`RK3588S`, `4× Cortex-A76` + `4× Cortex-A55`, `3-core RKNPU v0.9.8` / `librknnrt 2.5.0`, `7.7 GiB RAM`).
- **Evaluation Methodology (Zero Data Leakage / Distribution Preserved):**
  - Evaluated across **7,500 distribution-matched images** (`5,000` full COCO `val2017` + `2,500` proportional stratified COCO `train2017`; `16,190` person boxes; `54.0% pos / 46.0% neg`; `40.5% S / 33.9% M / 25.6% L`).
  - Calibrated using a strictly disjoint **`Calib_200`** subset (`SHA-256 overlap = 0`, `ID overlap = 0`, `54.0% pos / 46.0% neg`, `40.1% S / 35.4% M / 24.5% L`).
- **Best Quantized ONNX Model (`models/model_int8_static.onnx` — `INT8_HeadM0M1M2_Pct99995_C200`):**
  - Protected only sensitivity-verified layers (`/model.0/`, `/model.1/`, `/model.2/`, `/model.23/`, plus `Resize/Upsample/Softmax`), while keeping **`C2PSA` Attention in INT8**.
  - Achieved **`3.20 MiB`** (**-65.6%** vs `9.30 MiB` FP32), **`72/102 (70.6%)` `QLinearConv` + `4/4 QLinearMatMul`** (`1/6` Q/DQ boundaries), **`mAP@50 = 0.69621`** (**-0.00331**, `99.53%` retention), **`mAP@50:95 = 0.46170`** (**-0.00226**, `99.51%` retention), **`AP_small = 0.18094`** (**-0.00068**, `99.63%` retention), **`Mean IoU vs FP32 = 0.9723`**, and fewer negative false positives (`95` vs `102`).

---

## 2. Core ONNX Deliverables Summary (`7,500` Test Images)

| Format / Deliverable | File Path | Size (MiB) | INT8 Conv Coverage | `mAP@50` | `mAP@50:95` | `AP_small` | `F1 (@0.25)` | Neg FP Boxes | Mean IoU vs FP32 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline (1-cls)** | `models/model_fp32.onnx` | 9.30 | 0 / 102 (0.0%) | **0.69952** | **0.46396** | **0.18162** | 0.6749 | 102 | 1.0000 |
| **FP16 (`keep_io_types=True`)** | `models/model_fp16.onnx` | 4.71 | 0 / 102 (0.0%) | **0.69947** | **0.46398** | **0.18181** | 0.6749 | 103 | 0.9960 |
| **INT8 Dynamic (`ConvInteger`)** | `models/model_int8_dynamic.onnx` | 2.96 | 78 / 102 (76.5%) | 0.68987 | 0.45517 | 0.17775 | 0.6670 | 100 | 0.9420 |
| **INT8 Static Naive (`MinMax` All)** | `models/ablation/model_int8_static_naive.onnx` | 2.76 | 102 / 102 (100%) | 0.00000 | 0.00000 | 0.00000 | 0.0000 | 0 | 0.0000 |
| **INT8 Static Winner (Ours)** 🏆 | `models/model_int8_static.onnx` | **3.20** | **72 / 102 (70.6%) + 4 MatMul** | **0.69621** | **0.46170** | **0.18094** | **0.6718** | **95** | **0.9723** |

---

## 3. Verified Orange Pi 5 (`RK3588S`) Hardware Results (`5,000` `val2017` + Before-vs-After Latency/FPS)

1. **Before Optimization Baseline (`ondemand` governor, Python ONNX Runtime CPU):**
   - `ONNX FP32` (`model_fp32.onnx`): **`42.461 ms`** (`23.53 FPS`), `5,000` `val2017` `mAP@50 = 0.69785`, `F1 = 0.6706`.
   - `ONNX INT8 Static` (`model_int8_static.onnx`): **`30.786 ms`** (`32.44 FPS`), `5,000` `val2017` `mAP@50 = 0.69392`, `F1 = 0.6673`.
   - With CPU-only optimization (`FastBlob + A76 Pin + performance`): **`30.066 ms`** (`33.22 FPS`) with bit-exact parity (`mAP@50 = 0.69392`).
2. **After Optimization (`RK3588 NPU` + `6-Head Cut` + C++17 Zero-Copy `NC1HWC2` + Integer-Domain LUT + `RGA` + 3-Core NPU):**
   - **Accuracy (`5,000` `val2017`):** `RKNN_INT8_6Head_Normal` achieves **`mAP@50 = 0.69734`** (`+0.00342` over `ONNX INT8 Static`, `99.93%` of `ONNX FP32`), **`AP_small = 0.17881`**, and **`F1 = 0.6737`** (`NegFP = 71` vs `82` FP32). With hardware RGA active (`RGA_HW`, `70%` of `Calib_200`), **`mAP@50 = 0.69744`** (`99.94%` of FP32), `F1 = 0.6733`. (Uncut `RKNN_INT8_Uncut` collapses to `0.00000`).
   - **Single-Stream Min Latency (`1 Core`, `RGA` + Zero-Copy `NC1`):** **`8.179 ms`** (`122.24 FPS`, `p95 = 8.639 ms`) — **`3.77×`** faster than ONNX INT8 Before (`-73.4%` latency) and **`5.19×`** faster than ONNX FP32 Before (`-80.7%`); **`8.023 ms / 124.64 FPS`** at `640×480` and **`8.401 ms / 119.02 FPS`** at `1080p` via sparse Q11 LUT.
   - **Maximum 3-Core NPU Throughput (`6 Workers`):** **`385.43 FPS`** (`2.59 ms/frame` amortized) — **`11.88×`** FPS vs ONNX INT8 Before and **`16.38×`** vs ONNX FP32 Before, verified with simultaneous 3-core NPU load (`Core0: 76%, Core1: 56%, Core2: 56%`, thermal `54.5–57.3°C`).
3. **Full Reports & Raw Telemetry:**
   - Dataset & Distribution Audit: [reports/01_dataset_distribution_report.md](reports/01_dataset_distribution_report.md)
   - Sensitivity & INT8 Optimization Report: [reports/02_quantization_sensitivity_and_optimization_report.md](reports/02_quantization_sensitivity_and_optimization_report.md)
   - Orange Pi 5 Format & Strategy Analysis: [reports/03_orangepi5_format_fps_analysis.md](reports/03_orangepi5_format_fps_analysis.md)
   - Orange Pi 5 Hardware Utilization Blueprint: [reports/04_orangepi5_max_hardware_utilization_report.md](reports/04_orangepi5_max_hardware_utilization_report.md)
   - **Empirical Orange Pi 5 Before-vs-After Report:** [reports/05_orangepi5_before_after_optimization_report.md](reports/05_orangepi5_before_after_optimization_report.md)
   - Raw Metrics CSV/JSON: [results/full_7500_ablation_report.csv](results/full_7500_ablation_report.csv) | [results/orangepi5/val2017_accuracy_parity.csv](results/orangepi5/val2017_accuracy_parity.csv) | [results/orangepi5/latency_fps_before_after.csv](results/orangepi5/latency_fps_before_after.csv) | [results/orangepi5/resolution_before_after.csv](results/orangepi5/resolution_before_after.csv)
