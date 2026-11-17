# Team_A — YOLO26n Person Detection Quantization & Edge Accuracy Optimization

Repository triển khai đầy đủ quy trình thiết kế tập kiểm thử & calibration bảo toàn phân phối tự nhiên, kiểm toán độ nhạy từng khối (Per-Block Sensitivity Audit), tối ưu hóa lượng tử hóa `INT8` (giảm 65.6% dung lượng, giữ 99.53% `mAP@50`), và phân tích định dạng đạt FPS cao nhất cho **Orange Pi 5 (`RK3588S`)**.

> **Phạm vi hiện tại:** Toàn bộ code, mô hình `.onnx`, tập calibration (`200` ảnh COCO tách biệt), kết quả thô `7,500` ảnh và báo cáo phân tích nằm hoàn toàn trên máy local theo yêu cầu (chưa triển khai xuống Orange Pi 5).

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
├── datasets/
│   └── dataset_manifest.json             # Danh sách cố định 7,500 test + 400/200 calib theo từng stratum
├── marimo/
│   └── notebook.py                       # Toàn văn marimo notebook đã chạy thực nghiệm 100% (10 cells)
├── models/
│   ├── model_fp32.onnx                   # YOLO26n 1-class ("person") FP32 baseline (9.30 MiB)
│   ├── model_fp16.onnx                   # FP16 qua ORT transformer pass (4.71 MiB)
│   ├── model_int8_dynamic.onnx           # Dynamic INT8 QUInt8 (2.96 MiB)
│   ├── model_int8_static.onnx            # Winner: Head+M0+M1+M2 FP32, C2PSA INT8, Pct 99.995% (3.20 MiB)
│   ├── ablation/model_int8_static_naive.onnx # Bản Static Naive 100% INT8 dùng đối chứng (mAP=0)
│   └── calib_cache_*.json                # Calibration cache tái sử dụng từ MinMax / Entropy / Percentile
├── reports/
│   ├── 01_dataset_distribution_report.md
│   ├── 02_quantization_sensitivity_and_optimization_report.md
│   ├── 03_orangepi5_format_fps_analysis.md
│   └── figures/                          # 4 biểu đồ kiểm chứng phân phối, độ nhạy, Pareto & progression
├── results/
│   ├── block_sensitivity_audit.csv       # Xếp hạng độ nhạy đơn khối cả 28 blocks của YOLO26n
│   ├── full_7500_ablation_report.csv     # Bảng tổng hợp 15 cấu hình đánh giá đầy đủ 7,500 ảnh
│   ├── full_7500_ablation_report.json
│   └── stage*_metrics.json               # Dữ liệu JSON chi tiết từng giai đoạn (2A, 3A, 3B, 3C)
├── scripts/
│   ├── prepare_datasets.py               # Tải Calib_200 hoặc tái tạo toàn bộ Test_7500 + GT JSON
│   ├── export_fp32.py                    # Xuất ONNX FP32 & cắt 80 lớp -> 1 lớp "person" bit-exact
│   ├── quantize_pipeline.py              # Pipeline tạo FP16, Dynamic INT8, Static INT8 Best (+ --ablation)
│   ├── audit_sensitivity.py              # Kiểm toán độ nhạy đơn khối bằng Calibration Cache
│   ├── evaluate_7500.py                  # Chạy COCOeval + Operating-Point P/R/F1 + Probe + Fused ORT audit
│   └── generate_charts.py                # Vẽ biểu đồ trực tiếp từ CSV/JSON
├── src/
│   ├── detector.py                       # Unified PersonDetector wrapper cho 4 định dạng ONNX
│   └── realtime_camera.py                # Demo camera/video/image real-time hiển thị FPS telemetry
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

# 4. (Tùy chọn) Tải trọn bộ 7,500 ảnh Test + COCO GT JSON và chạy đánh giá lại từ đầu
python scripts/prepare_datasets.py --full
python scripts/evaluate_7500.py
```

---

## 4. Báo Cáo Chi Tiết

1. [Phân tích Dữ liệu Test 7,500 ảnh & Calibration 200 ảnh Bảo toàn Phân phối](reports/01_dataset_distribution_report.md)
2. [Kiểm toán Độ nhạy 28 Khối & Tối ưu Hóa Độ Chính Xác INT8 (Bảng 15 Mô Hình)](reports/02_quantization_sensitivity_and_optimization_report.md)
3. [Phân Tích Định Dạng Đem Lại FPS Cao Nhất Trên Orange Pi 5 (`RK3588S`)](reports/03_orangepi5_format_fps_analysis.md)
4. [Chiến Lược Khai Thác Tối Đa Phần Cứng Orange Pi 5 (`RK3588S`) Cho Mô Hình Hiện Tại](reports/04_orangepi5_max_hardware_utilization_report.md)
5. [Benchmark Report Tổng hợp (Chuẩn Hackathon)](benchmark_report.md)
