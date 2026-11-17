# Team_A — YOLO26n Person Detection Quantization & Orange Pi 5 Edge Optimization

Repository triển khai đầy đủ quy trình thiết kế tập kiểm thử & calibration bảo toàn phân phối tự nhiên, kiểm toán độ nhạy từng khối (Per-Block Sensitivity Audit), tối ưu hóa lượng tử hóa `INT8` (`-65.6%` dung lượng, giữ `99.53%` `mAP@50`), và tối ưu hóa suy luận **C++17 Zero-Copy + RGA + 6-Head Cut + 3-Core NPU** thực chứng trên phần cứng **Orange Pi 5 (`RK3588S`)**.

> **Kết quả thực chiến trên Orange Pi 5 (`5,000` ảnh COCO `val2017`):**
> - **Độ chính xác (`val2017`):** `RKNN_INT8_6Head_Normal` đạt **`mAP@50 = 0.69734`** (`RGA_HW: 0.69744` — giữ **`99.93–99.94%`** so với `ONNX FP32 0.69785` và **cao hơn `+0.00342`** so với chính `ONNX INT8 Static 0.69392`), **`F1 = 0.6737`** (cao hơn FP32 `0.6706`).
> - **Tốc độ & Độ trễ (Before -> After):** Từ `ONNX FP32 CPU` **`42.461 ms / 23.53 FPS`** và `ONNX INT8 Static CPU` **`30.786 ms / 32.44 FPS`** xuống còn **`8.179 ms / 122.24 FPS`** (đơn luồng min-latency, **`-73.4%` latency / `3.77×`** vs ONNX INT8; **`8.023 ms / 124.64 FPS`** ở `640×480`) và **`385.43 FPS`** (**`2.59 ms/frame`**, **`11.88×`** vs ONNX INT8 / **`16.38×`** vs ONNX FP32) khi chạy song song 3 nhân NPU (`6 workers`).

---

## 1. Kết quả Trọng tâm (`INT8_HeadM0M1M2_Pct99995_C200`)

- **Kiến trúc gốc:** `YOLO26n` (`416×416`, `opset=17`, `end2end=False`, cắt tỉa 80 lớp COCO xuống 1 lớp duy nhất `person` với sai số bằng không `MaxAbsDiff = 0.00e+00`).
- **Tập Test chuẩn hóa phân phối:** `7,500` ảnh (`5,000` COCO `val2017` + `2,500` phân tầng tỉ lệ `train2017`; `16,190` boxes; `40.5% S / 33.9% M / 25.6% L`).
- **Tập Calibration sạch (`Calib_200`):** `200` ảnh tách biệt hoàn toàn (`SHA-256 overlap = 0`, `40.1% S / 35.4% M / 24.5% L`, `54.0% Pos / 46.0% Neg`).
- **Chất lượng bản INT8 tối ưu (`models/model_int8_static.onnx`):**
  - Kích thước: **`3.20 MiB`** (`-65.6%` so với `9.30 MiB` FP32).
  - Độ phủ INT8 ORT: **`72/102 (70.6%) QLinearConv` + `4/4 QLinearMatMul`** (chỉ `1/6` biên chuyển đổi Q/DQ do giữ C2PSA Attention chạy INT8).
  - Độ chính xác trên **7,500 ảnh**: **`mAP@50 = 0.69621`** (`-0.00331` vs `0.69952`), **`mAP@50:95 = 0.46170`** (`-0.00226` vs `0.46396`), **`AP_small = 0.18094`** (`-0.00068` vs `0.18162`), **`Mean IoU = 0.9723`**, **`False Positive` trên ảnh không có người giảm từ 102 xuống 95**.

---

## 2. Cấu trúc Repository

```text
Team_A/
├── configs/config.yaml                   # Cấu hình model, đường dẫn, tham số Percentile 99.995% & node bảo vệ
├── calibration_data/                     # 200 ảnh COCO disjoint Calib_200 (SHA-256 overlap = 0)
├── cpp/
│   ├── Makefile                          # Build native C++17 (-O3 -mcpu=cortex-a76+fp16+dotprod)
│   ├── yolo26_rk3588.h / .cpp            # Engine Zero-Copy DMA, NC1HWC2 Pass-Through, Lazy Int8 LUT, Q11 & RGA
│   └── yolo26_edge_cli.cpp               # CLI benchmark C++ thuần không overhead Python
├── datasets/
│   └── dataset_manifest.json             # Danh sách cố định 7,500 test + 400/200 calib theo từng stratum
├── marimo/
│   └── notebook.py                       # Toàn văn marimo notebook đã chạy thực nghiệm 100% (10 cells)
├── models/
│   ├── model_fp32.onnx                   # YOLO26n 1-class ("person") FP32 baseline (9.30 MiB)
│   ├── model_fp16.onnx                   # FP16 qua ORT transformer pass (4.71 MiB)
│   ├── model_int8_dynamic.onnx           # Dynamic INT8 QUInt8 (2.96 MiB)
│   ├── model_int8_static.onnx            # Winner ONNX: Head+M0+M1+M2 FP32, C2PSA INT8, Pct 99.995% (3.20 MiB)
│   ├── rknn/model_int8_6head_normal.rknn # Winner Orange Pi 5 NPU 6-Head INT8 (3.98 MiB)
│   ├── rknn/model_int8_6head_normal_sc.rknn # Bản Single-Core Mode tối giản DMA/RAM (3.56 MiB)
│   ├── ablation/model_int8_static_naive.onnx # Bản Static Naive 100% INT8 dùng đối chứng (mAP=0)
│   └── calib_cache_*.json                # Calibration cache tái sử dụng từ MinMax / Entropy / Percentile
├── reports/
│   ├── 01_dataset_distribution_report.md
│   ├── 02_quantization_sensitivity_and_optimization_report.md
│   ├── 03_orangepi5_format_fps_analysis.md
│   ├── 04_orangepi5_max_hardware_utilization_report.md
│   ├── 05_orangepi5_before_after_optimization_report.md
│   ├── 06_latency_anatomy_and_component_breakdown.md
│   ├── 07_roboflow_cctv_naxyo_benchmark_report.md
│   └── figures/                          # 8 biểu đồ kiểm chứng từ CSV/JSON (offline + Pi 5 + cctv-naxyo)
├── results/
│   ├── orangepi5/                        # Kết quả thực đo Orange Pi 5: val2017 (5K), Before/After, Latency Anatomy
│   ├── cctv_naxyo_orangepi5/             # Benchmark 3,737 ảnh Roboflow cctv-naxyo v2 trên Orange Pi 5
│   ├── block_sensitivity_audit.csv       # Xếp hạng độ nhạy đơn khối cả 28 blocks của YOLO26n
│   ├── full_7500_ablation_report.csv     # Bảng tổng hợp 15 cấu hình đánh giá đầy đủ 7,500 ảnh
│   ├── full_7500_ablation_report.json
│   └── stage*_metrics.json               # Dữ liệu JSON chi tiết từng giai đoạn (2A, 3A, 3B, 3C)
├── scripts/
│   ├── prepare_datasets.py               # Tải Calib_200 hoặc tái tạo toàn bộ Test_7500 + GT JSON
│   ├── prepare_cctv_naxyo.py             # Tải Roboflow cctv-naxyo v2 qua ROBOFLOW_API_KEY
│   ├── export_fp32.py                    # Xuất ONNX FP32 & cắt 80 lớp -> 1 lớp "person" bit-exact
│   ├── quantize_pipeline.py              # Pipeline tạo FP16, Dynamic INT8, Static INT8 Best (+ --ablation)
│   ├── convert_rknn.py                   # Xuất bộ mô hình RKNN 6-Head Cut cho RK3588 NPU
│   ├── benchmark_orangepi5.py            # Đánh giá 5,000 ảnh val2017 + thang đo Before/After trên Pi 5
│   ├── benchmark_cctv_naxyo.py           # Benchmark 3,737 ảnh Roboflow cctv-naxyo + Latency/FPS trên Pi 5
│   ├── check_npu_parity.py               # Kiểm tra parity đầu ra từng biến thể ONNX vs NPU Zero-Copy/RGA
│   ├── tune_orangepi5.sh                 # Khóa governor performance A76 (2.26 GHz) & NPU (1.0 GHz)
│   ├── audit_sensitivity.py              # Kiểm toán độ nhạy đơn khối bằng Calibration Cache
│   ├── evaluate_7500.py                  # Chạy COCOeval + Operating-Point P/R/F1 + Probe + Fused ORT audit
│   └── generate_charts.py                # Vẽ biểu đồ trực tiếp từ CSV/JSON
├── src/
│   ├── detector.py                       # Unified PersonDetector wrapper cho 4 định dạng ONNX (kèm FastBlob/PinA76)
│   ├── npu_detector.py                   # Python ctypes wrapper bọc libyolo26_rk3588.so zero-copy/RGA
│   └── realtime_camera.py                # Demo camera/video/image real-time (auto onnx/npu)
├── benchmark_report.md
├── requirements.txt
└── README.md
```

---

## 3. Hướng dẫn Cài đặt & Tái lập Nhanh

```bash
pip install -r requirements.txt

# 1. Tải sẵn 200 ảnh calibration tách biệt vào calibration_data/
python scripts/prepare_datasets.py --calib-only

# 2. Chạy inference thử qua Unified PersonDetector trên 1 ảnh
python src/realtime_camera.py --model models/model_int8_static.onnx --source calibration_data/000000003442.jpg --headless

# 3. Tái tạo toàn bộ mô hình lượng tử hóa (FP16, Dynamic INT8, Static INT8 Best)
python scripts/quantize_pipeline.py

# 4. Triển khai & benchmark trực tiếp trên Orange Pi 5 (RK3588S NPU + RGA)
make -C cpp -j4
sudo bash scripts/tune_orangepi5.sh performance
python scripts/benchmark_orangepi5.py --bench-loops 5
python src/realtime_camera.py --model models/rknn/model_int8_6head_normal.rknn --backend npu --rga --source calibration_data/000000309695.jpg --headless

# 5. (Tùy chọn) Tải trọn bộ 7,500 ảnh Test + COCO GT JSON và chạy đánh giá offline
python scripts/prepare_datasets.py --full
python scripts/evaluate_7500.py
```

---

## 4. Báo Cáo Chi Tiết

1. [Phân tích Dữ liệu Test 7,500 ảnh & Calibration 200 ảnh Bảo toàn Phân phối](reports/01_dataset_distribution_report.md)
2. [Kiểm toán Độ nhạy 28 Khối & Tối ưu Hóa Độ Chính Xác INT8 (Bảng 15 Mô Hình)](reports/02_quantization_sensitivity_and_optimization_report.md)
3. [Phân Tích Định Dạng Đem Lại FPS Cao Nhất Trên Orange Pi 5 (`RK3588S`)](reports/03_orangepi5_format_fps_analysis.md)
4. [Chiến Lược Khai Thác Tối Đa Phần Cứng Orange Pi 5 (`RK3588S`) Cho Mô Hình Hiện Tại](reports/04_orangepi5_max_hardware_utilization_report.md)
5. [Báo Cáo Thực Nghiệm Before vs After Tối Ưu Inference Trên Orange Pi 5 (`5,000` val2017 + C++ Zero-Copy/RGA/3-Core NPU)](reports/05_orangepi5_before_after_optimization_report.md)
6. [Giải Phẫu & Bóc Tách Toàn Bộ Thành Phần Cấu Tạo Nên Latency Trên Orange Pi 5](reports/06_latency_anatomy_and_component_breakdown.md)
7. [Đánh Giá Thực Nghiệm Trên Bộ Dữ Liệu Roboflow `cctv-naxyo v2` (`3,737` Ảnh CCTV)](reports/07_roboflow_cctv_naxyo_benchmark_report.md)
8. [Báo Cáo Trực Quan Hóa Toàn Bộ Kết Quả (`11` Biểu Đồ Tổng Hợp A→Z)](reports/08_complete_visual_dashboard_report.md)
9. [Benchmark Report Tổng hợp (Chuẩn Hackathon)](benchmark_report.md)
