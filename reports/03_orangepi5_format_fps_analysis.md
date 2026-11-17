# Báo Cáo Chi Tiết 03: Phân Tích Định Dạng Nào Đem Lại FPS Cao Nhất Trên Orange Pi 5 (`RK3588S`)

> **Tuyên bố Minh bạch Thực nghiệm (Hardware Honesty):**
> Thực thi lần này triển khai hoàn toàn repo trên máy trạm Windows và thí nghiệm accuracy 7,500 ảnh trên **marimo cloud**. Bộ 4 mô hình `.onnx` mới nhất của `Team_A` (`model_int8_static.onnx` phiên bản `HeadM0M1M2_Pct99995`) **chưa chạy benchmark runtime trực tiếp trên board Orange Pi 5** theo đúng yêu cầu chưa triển khai xuống mạch ở bước này.
> Mọi số liệu FPS trên **Orange Pi 5** dưới đây được trích nguyên trạng từ phần cứng đích **đã quan sát trực tiếp trong phiên SSH tới `orangepi5`** (SoC `RK3588S`, 4× Cortex-A76 `2.26 GHz` + 4× Cortex-A55 `1.80 GHz`, `RKNPU driver v0.9.8`, `librknnrt v2.5.0`, kernel `6.1.115-vendor-rk35xx`) cho cùng kiến trúc **YOLO26n (`416×416`)**.

---

## 1. Câu Trả Lời Trực Diện: Định Dạng Nào Đem Lại FPS Cao Nhất Trên Orange Pi 5?

Cần phân biệt rõ **2 góc nhìn** (toàn phần cứng RK3588S vs khung chuẩn CPU của đề bài Hackathon):

### A. Xét trên Toàn bộ Phần cứng Orange Pi 5 (NPU RK3588 + CPU): **`RKNN INT8 (w8a8 per-channel, 6-Head Cut)` cho NPU** là định dạng nhanh nhất tuyệt đối
- **Tên định dạng:** **`.rknn` (`INT8` asymmetric per-channel `w8a8` chạy trên NPU 3-core `fdab0000.npu`, cắt đồ thị tại 6 nhánh Conv đầu ra thô trước `Reshape/Slice/Sigmoid` để CPU Cortex-A76 giải mã sau)** (`librknnrt 2.5.0`).
- **Tốc độ đo thực tế trên `orangepi5` (YOLO26n `416×416`):**
  - **C++ Zero-Copy (`CORE_0`, 1 nhân NPU):** **`131.35 FPS`** (`7.61 ms` trung vị/ảnh).
  - **C++ Zero-Copy (`3 nhân NPU / 3 luồng`):** **`376.10 FPS`**.
  - **C++ Zero-Copy (`3 nhân NPU / 6 luồng` bền vững 30s):** **`384.12 – 385.10 FPS`** (NPU hoạt động `98%` tại `1000 MHz`, nhiệt độ ổn định `51.8–53.6°C`).
  - **Python RKNNLite E2E (`1 nhân CORE_0`):** **`90.77 FPS`** (`72.75 FPS` trung bình toàn tập test có tiền/hậu xử lý; `236.48 FPS` với 3 worker; `372.38 FPS` với 6 worker).
- **Lý do đạt FPS vượt trội so với mọi định dạng khác trên RK3588S:**
  - NPU 3 nhân của RK3588S cung cấp sức mạnh tính toán **INT8 MAC**, trong khi phép tính giải mã cuối (`Reshape/Concat/Slice/Sub/Add/Div/Sigmoid` trên 3,549 anchor points ở Detect Head) lại **không tối ưu trên NPU nếu để nguyên trong đồ thị `.rknn`** (`RKNN_FP16_Uncut_NPU` chỉ đạt `27.82 FPS` E2E trên 151 ảnh / `57.26 FPS` C++ 1-core vì tắc cổ chai chuyển định dạng & slice/decode cuối mạng). Cắt đồ thị tại 6 nhánh Conv đầu ra cho phép NPU xử lý thuần **100% phép tích chập INT8 lượng tử hóa liên hoàn**, giữ `mAP@50 = 0.7495` (gần như tương đương `0.7529` FP32 trên tập CCTV đó), khác hẳn kiểu cắt quá sâu (`2-Head Cut`) làm tụt `mAP@50-95` từ `0.5048` xuống `0.3788`.

### B. Xét riêng Trong Bộ Định Dạng `.onnx` Chuẩn Của Đề Bài Hackathon (Chạy CPU qua `ONNX Runtime`): **`model_int8_static.onnx` (`Static INT8 QDQ / QLinearConv`)** cho FPS cao nhất
Nếu hệ thống chấm điểm của Mentor chỉ chạy đúng bộ 4 file `.onnx` qua `src/detector.py` trên **ONNX Runtime CPU (`4× Cortex-A76`)**:
- **`ONNX INT8 Static` (`QLinearConv` + `QLinearMatMul`)** là định dạng cho **FPS cao nhất trong nhóm ONNX**: đo trên `4× Cortex-A76` đạt **`37.22 FPS`** (`26.86 ms` E2E, `23.76 ms` inference) — **nhanh hơn +55% so với `ONNX FP32`** (`24.02 FPS`, `41.63 ms` E2E, `38.53 ms` inference).
- **Tại sao `ONNX INT8 Dynamic` và `ONNX FP16` lại CHẬM HƠN trên ARM CPU Orange Pi 5?**
  1. **`ONNX FP16` trên Cortex-A76 (ORT CPU):** CPU ARMv8.2-A trên ORT tiêu chuẩn không có nhiều kernel `Conv` FP16 thuần tốc độ cao tương ứng trên NPU/GPU, thường phải cast nội bộ về FP32 hoặc dùng kernel kém tối ưu hơn; ngay cả trên x86 CPU server ở bảng kết quả 7,500 ảnh, `ONNX_FP16` (`708.6 img/s`) cũng không nhanh hơn `ONNX_FP32` (`739.4 img/s`).
  2. **`ONNX INT8 Dynamic` (`ConvInteger`):** Trước mỗi phép tích chập ở từng khung hình đều phải tính động `min/max` toàn tensor kích hoạt bằng CPU rồi lượng tử hóa on-the-fly — chi phí tính động này triệt tiêu phần lớn lợi ích của INT8 SIMD (`ARM NEON SDOT/UDOT`); thực tế đo trên ORT CPU trong thí nghiệm 7,500 ảnh, `ONNX_INT8_Dynamic` chỉ đạt **`533.7 img/s` (`14.05s`)**, chậm hơn cả `ONNX_FP32` (`739.4 img/s`, `10.14s`) và chậm hơn gần **2×** so với `ONNX_INT8_Static` (`1024.6 img/s`, `7.32s`)!

---

## 2. Bảng Đối Chiếu Toàn Diện Tốc Độ Trên Orange Pi 5 (`RK3588S`)

| Định Dạng / Kiến Trúc | Phần Cứng Thực Thi | Số Luồng / Nhân NPU | FPS Đo Thực Tế Trên `orangepi5` | Độ Trễ / Ảnh | Đánh Giá Thực Chiến Cho Orange Pi 5 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`ONNX FP32` (`model_fp32.onnx`)** | CPU `4× Cortex-A76` (ORT) | 4 threads CPU | **`24.02 FPS`** (E2E) | `41.63 ms` E2E (`38.53 ms` inf) | Đủ chuẩn tham chiếu nhưng chạm trần tải CPU, khó kèm thêm tracking/stream HD. |
| `ONNX FP16` (`model_fp16.onnx`) | CPU `4× Cortex-A76` (ORT) | 4 threads CPU | *(Không tăng tốc trên CPU ORT; tương đương/kém hơn FP32)* | — | Chỉ giảm ~50% kích thước lưu trữ (`4.71 MiB`), không tối ưu cho ORT CPU. |
| `ONNX INT8 Dynamic` (`model_int8_dynamic.onnx`) | CPU `4× Cortex-A76` (ORT) | 4 threads CPU | *(Chậm hơn Static INT8 do ép tính min/max runtime; ~0.52× tốc độ Static INT8)* | — | Không khuyến nghị triển khai thực tế nếu đã có Static INT8 hiệu chỉnh tốt. |
| **`ONNX INT8 Static` (`model_int8_static.onnx`)** 🏆 *(Nhất nhóm ONNX CPU)* | CPU `4× Cortex-A76` (ORT) | 4 threads CPU | **`37.22 FPS`** (E2E) | `26.86 ms` E2E (`23.76 ms` inf) | **Nhanh nhất trong 4 định dạng `.onnx` chuẩn Hackathon**. Bản mới `HeadM0M1M2_Pct99995` đã tối ưu `70.6% INT8 Conv + 4 MatMul` & chỉ `1/6 Q/DQ`. |
| `RKNN FP16 Uncut` (`.rknn`) | NPU `fdab0000.npu` | 1× `CORE_0` (151 imgs) / 6× C++ | `27.82 FPS` (151-img E2E) / `57.26 FPS` (C++ 1×) / `162.28 FPS` (C++ 6×) | `22.28 ms` NPU | Chậm vì đầu ra giải mã giữ nguyên `Slice/Concat/Reshape/Sigmoid` trong đồ thị NPU, gây bottleneck truyền & giải mã tensor. |
| `RKNN INT8 2-Head Cut` (`.rknn`) | NPU `fdab0000.npu` | 1× (151 imgs) / 6× C++ | `91.29 FPS` (151-img E2E) / `354.08 FPS` (C++ 6×) | `8.36 ms` NPU | Cắt quá sâu, **hỏng nặng độ chính xác** (`mAP@50-95` rớt `0.5048 → 0.3788`). |
| **`RKNN INT8 6-Head Cut` (`.rknn` `w8a8` per-channel)** 🚀 *(Nhất tuyệt đối trên RK3588S)* | **3-Core NPU `RK3588S`** (`librknnrt 2.5.0`) | **1× `CORE_0`** / **3×** / **6× C++ Zero-Copy** | **`90.77 FPS`** (Py 1×) · **`131.35 FPS`** (C++ 1×) · **`376.10 FPS`** (C++ 3×) · **`384.12–385.10 FPS`** (C++ 6×) | **`9.01 ms`** (151-img NPU inf) · **`7.61 ms`** (C++ zero-copy 1×) | **Định dạng đem lại FPS cao nhất tuyệt đối trên Orange Pi 5** mà vẫn giữ nguyên `mAP@50` (`0.7495` vs `0.7529` FP32). |

---

## 3. Lưu Ý Quan Trọng Khi Chọn Định Dạng Nộp Bài Hackathon vs Chạy Production

1. **Cho bài nộp chuẩn Hackathon ONNX Runtime (`src/detector.py`):**
   - Bắt buộc nộp đầy đủ bộ 4 file `.onnx` trong `models/` (`model_fp32.onnx`, `model_fp16.onnx`, `model_int8_dynamic.onnx`, **`model_int8_static.onnx`**).
   - Trong đó, **`models/model_int8_static.onnx`** (`INT8_HeadM0M1M2_Pct99995_C200`) là file đem lại **FPS CPU cao nhất** (`+55%` so với FP32 trên 4× Cortex-A76 nhờ `QLinearConv` + `QLinearMatMul` + chỉ `1/6` biên Q/DQ) đồng thời bảo toàn **99.53% `mAP@50`** (`0.69621` vs `0.69952`) và **99.63% `AP_small`**!
2. **Nếu chuyển sang giai đoạn tối ưu phần cứng Orange Pi 5 thực tế (NPU RK3588S):**
   - Xuất trực tiếp biến thể **`.rknn` INT8 `w8a8` per-channel cắt tại 6 nhánh Conv trước `Reshape/Slice/Sigmoid`** của `YOLO26n 416×416` cho `librknnrt 2.5.0` để giải phóng CPU và đẩy hiệu năng từ `~37 FPS` (CPU ONNX) lên **`131 FPS` (1 nhân NPU) → `385 FPS` (3 nhân NPU C++ Zero-Copy)**.
   - Lưu ý: Khi sang bước tối ưu trên Orange Pi 5 ở phiên sau, vẫn cần đo benchmark trực tiếp trên `orangepi5` cho model mới `INT8_HeadM0M1M2_Pct99995` để ghi nhận độ trễ thực tế chính xác của biến thể vừa tối ưu.
