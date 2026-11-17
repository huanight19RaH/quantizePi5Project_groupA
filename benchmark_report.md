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

## 3. Key Findings & Highest-FPS Format for Orange Pi 5

1. **Highest-FPS Format Overall on Orange Pi 5 (`RK3588S`):**
   - **`.rknn` INT8 (`w8a8` per-channel, 6-Head Cut before DFL/Sigmoid) on the 3-core NPU (`RKNPU v0.9.8` / `librknnrt 2.5.0`)**: Measured on `orangepi5` for YOLO26n `416×416` at **`90.77 FPS` (Python `CORE_0`) / `131.35 FPS` (C++ Zero-Copy `CORE_0`) / `376.10 FPS` (3× NPU) / `384.12–385.10 FPS` (6× NPU sustained)** while maintaining `mAP@50 = 0.7495` vs `0.7529` FP32 on CCTV test frames.
2. **Highest-FPS Format Among Hackathon-Required `.onnx` CPU Models:**
   - **`models/model_int8_static.onnx` (`QLinearConv` + `QLinearMatMul`, `4× Cortex-A76`)**: Measured on `orangepi5` CPU ORT at **`37.22 FPS`** (`26.86 ms` E2E) vs **`24.02 FPS`** (`41.63 ms` E2E) for `ONNX FP32` (+55% speedup), whereas `ONNX INT8 Dynamic` incurs runtime per-frame min/max overhead (even slower than FP32 on CPU ORT in our 7,500-image CPU benchmark: `533.7` vs `739.4 img/s`).
3. **Full Reports & Reproduction:**
   - Dataset & Distribution Audit: [reports/01_dataset_distribution_report.md](reports/01_dataset_distribution_report.md)
   - Sensitivity & INT8 Optimization Report: [reports/02_quantization_sensitivity_and_optimization_report.md](reports/02_quantization_sensitivity_and_optimization_report.md)
   - Orange Pi 5 Highest-FPS Format Analysis: [reports/03_orangepi5_format_fps_analysis.md](reports/03_orangepi5_format_fps_analysis.md)
   - Raw Metrics CSV/JSON: [results/full_7500_ablation_report.csv](results/full_7500_ablation_report.csv)
