# Báo Cáo Chi Tiết 04: Kiến Trúc Khai Thác Tối Đa Phần Cứng Orange Pi 5 (`RK3588S`) Cho Mô Hình YOLO26n Hiện Tại

> **Tính Xác thực & Minh bạch:**
> - Toàn bộ cấu trúc toán tử ONNX dưới đây được kiểm toán trực tiếp trên file [models/model_fp32.onnx](../models/model_fp32.onnx) và [models/model_int8_static.onnx](../models/model_int8_static.onnx) của repo `Team_A`.
> - Toàn bộ thông số phần cứng `orangepi5` và các mốc FPS đối chứng dưới đây là số liệu thực nghiệm đã quan sát trực tiếp trong phiên làm việc trên máy đích `orangepi5` (chú thích rõ phần nào là cấu hình khuyến nghị cho bước triển khai kế tiếp, chưa chạy trực tiếp trên board với binary mới).

---

## 1. Hồ Sơ Phần Cứng Đích (`orangepi5`) & Giải Phẫu Đồ Thị Mô Hình Hiện Tại

### 1.1 Thông Số Phần Cứng Thực Tế Của `orangepi5`
| Thành Phần | Thông Số Đã Xác Minh Trên `orangepi5` | Vai Trò Lý Tưởng Trong Pipeline |
| :--- | :--- | :--- |
| **SoC / OS / Kernel** | **Rockchip RK3588S** (`aarch64`), Orange Pi 1.2.4 Resolute (`Ubuntu 26.04`), Kernel **`6.1.115-vendor-rk35xx`** | Hỗ trợ kernel driver chuyên dụng của Rockchip (RKNPU, vendor DRM/DMA). |
| **NPU (Bộ Tăng Tốc AI)** | **3 nhân NPU độc lập (`fdab0000.npu`)**, dải xung **`300 – 1000 MHz`**, `RKNPU driver v0.9.8`, **`librknnrt 2.5.0`** | Chạy toàn bộ 102 lớp `Conv`/`SiLU`/`MaxPool` ở INT8 (`w8a8` per-channel). |
| **CPU Cụm Nhỏ (LITTLE)** | **4× Cortex-A55**, **`408 – 1800 MHz`**, Logical **Cores `0–3`** | Đọc/ghi NVMe, nhận stream mạng/camera RTSP, logging, hàng đợi I/O. |
| **CPU Cụm Lớn (big)** | **4× Cortex-A76**, **`408 – 2256 MHz`**, Logical **Cores `4–7`** | Chạy ONNX Runtime (`model_int8_static.onnx`), hoặc hậu xử lý vectorized box/NMS khi chạy NPU. |
| **RAM & Lưu Trữ** | **`7.7 GiB` RAM**, **NVMe SSD `465 GB`** | Giữ toàn bộ multi-context NPU, ring-buffer Zero-Copy DMA, tránh I/O bottleneck của thẻ SD. |

### 1.2 Giải Phẫu Đồ Thị Thực Tế Của Mô Hình `Team_A` (`YOLO26n 416×416 1-class person`)
Kiểm toán trực tiếp đồ thị ONNX ([models/model_fp32.onnx](../models/model_fp32.onnx)):
- **Input / Output gốc:** `images [1, 3, 416, 416]` (`float32` `[0..1]`) → `output0 [1, 5, 3549]` (`3549 = 52×52 + 26×26 + 13×13`).
- **Tổng số node gốc:** **`368` nodes** gồm: `102 Conv`, `88 Sigmoid`, `90 Mul` (tức `SiLU`), `11 Split`, `22 Add`, `23 Concat`, `3 MaxPool`, `12 Reshape`, `4 Transpose`, `4 MatMul`, `2 Softmax`, `2 Resize`, `2 Slice`, `2 Sub`, `1 Div`.
- **3 Điểm Đặc Thù Kiến Trúc Tác Động Trực Tiếp Tới Phần Cứng `RK3588S`:**
  1. **Không có `DFL` (`reg_max = 1`) — chỉ có cụm giải mã hình học cuối `/model.23/`:** 6 lớp `Conv` cuối xuất thẳng **4 kênh hồi quy khoảng cách biên (`ltrb`)** ở `[4, 16, 1, 1]` và **1 kênh logit `person`** ở `[1, 80, 1, 1]`, sau đó đi qua `Reshape -> Concat -> Slice -> Sub -> Add -> Div(/2) -> Mul(*stride)` + `Sigmoid` để ghép thành `[1, 5, 3549]`.
  2. **Có 2 khối Attention `C2PSA` (`/model.10/attn/` và `/model.22/attn/`):** Tổng cộng `28` nodes (`6 Conv + 6 Reshape + 2 Split + 4 Transpose + 4 MatMul + 2 Softmax + 2 Add`). Trong [models/model_int8_static.onnx](../models/model_int8_static.onnx), cả `4/4 MatMul` đều đã lượng tử hóa thành `QLinearMatMul` với độ trôi rất thấp (`0.426 px` ở `/model.10/attn` và `0.086 px` ở `/model.22/attn`).
  3. **3 Tầng Đầu Mạng Cực Nhạy (`/model.0`, `/model.1`, `/model.2`):** Kích thước không gian lớn (`208×208` → `104×104`), `16–32` kênh, chiếm tới `5.23 px` (`/model.1`) + `3.20 px` (`/model.2`) + `1.59 px` (`/model.0`) độ trôi tọa độ nếu lượng tử hóa thô.

---

## 2. Tại Sao Chạy Mặc Định Chỉ Khai Thác Một Phần Rất Nhỏ Phần Cứng?

Nếu copy trực tiếp `models/model_int8_static.onnx` sang chạy bằng `python src/realtime_camera.py` theo mặc định:
1. **3 nhân NPU (`fdab0000.npu`) hoàn toàn nhàn rỗi (`0%`):** `ONNX Runtime` chỉ chạy trên `CPUExecutionProvider`.
2. **Xung đột lập lịch `big.LITTLE`:** OS có thể đẩy luồng OpenCV hoặc ORT sang `4× Cortex-A55` (`1800 MHz`, hiệu năng đơn nhân yếu hơn `A76` đáng kể) hoặc gây cache bouncing qua 2 cụm CPU.
3. **Nút thắt tiền/hậu xử lý trên CPU:** Chuyển `BGR2RGB`, letterbox resize `416×416`, chia `float32 / 255.0`, và `Sigmoid` toàn bộ `3,549` điểm tiêu tốn lượng lớn băng thông RAM và chu kỳ CPU trên mỗi khung hình.

---

## 3. Chiến Lược 5 Tầng Khai Thác Tối Đa Phần Cứng Orange Pi 5 (`RK3588S`)

Dựa trên cấu trúc phần cứng đích và đồ thị mô hình hiện có, có **2 kịch bản vận hành**: (A) Tối ưu tối đa trên CPU nếu đề/hệ thống bắt buộc chạy file `.onnx`, và (B) Khai thác **toàn bộ phần cứng SoC RK3588S (NPU 3 nhân + big.LITTLE + Zero-Copy DMA + RGA/MPP)** cho hiệu năng tối đa.

### Tầng 1: Tối Đa Hóa 3 Nhân NPU (`fdab0000.npu`) Bằng Cắt Đồ Thị & Đa Lõi Độc Lập

1. **Cắt Đồ Thị `.rknn` Chính Xác Tại 6 Đầu Ra Conv Thô (`6-Head Cut`):**
   Khi chuyển từ [models/model_fp32.onnx](../models/model_fp32.onnx) sang `.rknn` (dùng `rknn-toolkit2` phù hợp `librknnrt 2.5.0`), **cắt bỏ phần giải mã cuối `/model.23/Reshape*`–`Concat_3`** bằng cách chỉ định đúng 6 tên tensor đầu ra:
   - P3 (`stride 8`, `52×52`): `"/model.23/cv2.0/cv2.0.2/Conv_output_0"`, `"/model.23/cv3.0/cv3.0.2/Conv_output_0"`
   - P4 (`stride 16`, `26×26`): `"/model.23/cv2.1/cv2.1.2/Conv_output_0"`, `"/model.23/cv3.1/cv3.1.2/Conv_output_0"`
   - P5 (`stride 32`, `13×13`): `"/model.23/cv2.2/cv2.2.2/Conv_output_0"`, `"/model.23/cv3.2/cv3.2.2/Conv_output_0"`
   - **Bằng chứng đo trên `orangepi5` (YOLO26n `416×416`):** Không cắt (`RKNN_FP16_Uncut`) chỉ đạt `27.82 FPS` E2E (151 imgs) / `57.26 FPS` C++ (`1x`), trong khi `6-Head Cut INT8` tăng vọt lên **`90.77 FPS` (Py 1×) / `131.35 FPS` (C++ Zero-Copy 1×)** và giữ nguyên `mAP@50` (`0.7495` vs `0.7529` FP32). Không dùng `2-Head Cut` (đã đo tụt `mAP@50-95` từ `0.5048` → `0.3788`).
2. **Đưa Chuẩn Hóa Input (`/255.0`) Vào Bên Trong Hardware NPU:**
   Cấu hình `mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]]` khi gọi `rknn.config()`. Khi đó tầng đầu vào của `.rknn` nhận thẳng **`uint8 RGB [1, 416, 416, 3]`** (`NHWC`) thay vì `float32`, giảm **4×** lưu lượng ghi DMA đầu vào và xóa bỏ bước chia `/255.0` bằng float trên CPU!
3. **Bật Chế Độ Đa Lõi NPU Theo Pool (`3–6 Context` Trên `CORE_0`, `CORE_1`, `CORE_2`):**
   - **Không dùng `RKNN_NPU_CORE_0_1_2` cho 1 frame đơn lẻ:** YOLO26n `416×416` có kích thước feature map nhỏ (`52×52` → `13×13`), việc chia 1 layer giữa 3 nhân đồng thời gây overhead đồng bộ bộ nhớ.
   - **Chạy song song mức frame/pipeline (`3 đến 6 rknn_context` phân phối đều `CORE_0`, `CORE_1`, `CORE_2`):** Đo thực tế trên `orangepi5` cho thấy hiệu năng tăng gần tuyến tính:
     - `1 worker (CORE_0)`: `131.35 FPS` (C++ Zero-Copy)
     - `3 workers (CORE_0/1/2)`: **`376.10 FPS`** (~`2.86×`)
     - `6 workers (2 queue / core để che trễ DMA↔Compute)`: **`384.12 – 385.10 FPS` bền vững 30s**, đẩy NPU đạt **`98%` tải ở `1000 MHz`**.
4. **Áp Dụng Mixed-Precision NPU Cho Khối Siêu Nhạy (`/model.1`, `/model.2`) Nếu Cần Tối Ưu Độ Chính Xác Trên `.rknn`:**
   Từ bảng kiểm toán độ nhạy [results/block_sensitivity_audit.csv](../results/block_sensitivity_audit.csv) (đã chứng minh trên ONNX giúp giảm `box_mae` 2.4×), khi build `.rknn` có thể dùng `hybrid_quantization_step1/2` để khóa các lớp `/model.1` và `/model.2` ở `float16` nếu cần đẩy sát độ chính xác `AP_small` & `mAP@50:95` của bản `.rknn` về sát FP32 mà chỉ đánh đổi phần nhỏ tốc độ NPU.

---

### Tầng 2: C++ Zero-Copy DMA & Hậu Xử Lý Lọc Ngưỡng Trước `Sigmoid`

1. **Dùng C API `librknnrt.so 2.5.0` Zero-Copy (`rknn_create_mem` + `rknn_set_io_mem`):**
   - Cấp phát sẵn bộ nhớ DMA liên tục (`NC1HWC2` native format hoặc `NHWC uint8` đầu vào) và tái sử dụng suốt vòng đời ứng dụng → loại bỏ hoàn toàn `memcpy` & dequantize toàn tensor của Python wrapper (tăng tốc đo thực tế trên 1 nhân NPU từ `90.77 FPS` → **`131.35 FPS`**, tương đương **+44.7%** chỉ nhờ bỏ overhead Python/copy!).
2. **Thuật Toán Hậu Xử Lý "Lazy Logit-Thresholding" Trên CPU Cortex-A76:**
   - Thay vì tính `1 / (1 + exp(-x))` trên toàn bộ `3,549` điểm lưới ở 3 scale (`52×52 + 26×26 + 13×13`), đổi ngưỡng tin cậy `conf = 0.25` thành ngưỡng **logit thô**:
     $$\tau_{\text{logit}} = \ln\left(\frac{0.25}{1 - 0.25}\right) = -1.098612$$
   - Với bản `.rknn` INT8 `6-Head Cut`, thậm chí đổi tiếp thành **ngưỡng số nguyên `int8` lượng tử hóa** $\tau_{q} = \lfloor \tau_{\text{logit}} / s + zp \rfloor$. Toàn bộ vòng quét trên `3,549` ô lưới chỉ là so sánh **số nguyên 8-bit SIMD (`int8 > tau_q`)**; chỉ những ô (`thường < 20 ô / frame`) vượt ngưỡng mới thực hiện dequant 4 kênh `(l, t, r, b)` và tính box `(cx ± offset) * stride` + NMS! Thời gian hậu xử lý giảm từ `~1.5 ms` xuống **dưới `0.2 ms`**.

---

### Tầng 3: Điều Phối `big.LITTLE` Chính Xác Giữa `Cores 0–3 (A55)` và `Cores 4–7 (A76)`

Kiến trúc RK3588S không đồng nhất (`4× A55 @ 1800 MHz` + `4× A76 @ 2256 MHz`). Để tránh scheduler Linux chạy lung tung gây L2/L3 cache miss:

1. **Kịch bản Chạy Chuẩn ONNX Runtime (`models/model_int8_static.onnx` trên CPU):**
   - Dùng `taskset -c 4-7` (hoặc `os.sched_setaffinity(0, {4, 5, 6, 7})`) để ép toàn bộ luồng inference `ONNX Runtime` nằm trọn trên **4 lõi lớn Cortex-A76**.
   - Trong `src/detector.py`: cấu hình `intra_op_num_threads=4`, `inter_op_num_threads=1` khi chạy 1 luồng camera độ trễ thấp (`~26.86 ms E2E / 37.22 FPS` đo thực tế trên 4× A76); nếu xử lý batch hoặc 2–4 luồng camera song song trên CPU, chạy `2–4` tiến trình độc lập mỗi tiến trình `1–2 threads` ghim cứng vào từng lõi trong nhóm `{4, 5, 6, 7}`.
   - Dời các tác vụ đọc file/decode RTSP/JSON logging sang `Cores 0–3` (`taskset -c 0-3`).
2. **Kịch bản Chạy Pipeline Full-Hardware (NPU + CPU Pipeline Song Song):**
   - **Reader / Network / Disk Threads (`Cores 0–3` Cortex-A55):** Đọc camera/NVMe/RTSP stream vào Ring Buffer.
   - **Preprocess + Postprocess / NMS / Tracker Threads (`Cores 4–7` Cortex-A76):** Ghim mỗi C++ worker thread (`1–4` thread) vào `Core 4..7`, cấp việc sang 3 lõi NPU (`CORE_0/1/2`) theo cơ chế producer-consumer không khóa (`Lock-Free SPSC/MPMC Ring Buffer`). Như vậy khi NPU đang tính khung `t`, A76 đang hậu xử lý khung `t-1` và A55/VPU đang đọc giải mã khung `t+1`.

---

### Tầng 4: Tận Dụng Khối Phần Cứng Giải Mã Video & Tiền Xử Lý 2D Của RK3588S (Khi Có Thiết Bị Kernel Vendor)

SoC RK3588S (chạy Vendor BSP Kernel `6.1.115-vendor-rk35xx`) được thiết kế kèm 2 bộ tăng tốc DMA nhằm giải phóng 100% tải tiền xử lý ảnh khỏi Cortex-A76:
- **VPU / Rockchip MPP (Media Process Platform):** Giải mã phần cứng H.264/H.265 cho nhiều luồng camera RTSP/video song song trực tiếp ra bộ nhớ DMA định dạng `NV12` không tốn CPU decode (`ffmpeg/OpenCV VideoCapture` mặc định chạy CPU software decode rất nặng khi đẩy lên >100 FPS).
- **2D RGA (`librga`):** Thực hiện chuyển màu (`NV12/BGR -> RGB`), co giãn (resize giữ tỉ lệ) và điền viền xám `114` (`letterbox 416×416`) bằng phần cứng 2D trực tiếp từ `dmabuf fd` này sang `dmabuf fd` đầu vào của `librknnrt` (**100% Zero-Copy từ Camera/VPU thẳng vào NPU**).
*(Ghi chú kiểm toán: Do giới hạn truy cập SSH hiện tại chưa đọc node `/dev/rga` và `/dev/mpp_service` trên board `orangepi5`, cần xác minh quyền truy cập `/dev/rga`, `/dev/mpp_service` và gói `librga` trước khi biên dịch nhánh này khi triển khai xuống mạch ở bước sau.)*

---

### Tầng 5: Cố Định Xung Nhịp (`Clock`), Bộ Nhớ (`DMC / NVMe / 7.7 GiB RAM`) & Nhiệt Độ

Để hệ thống đạt hiệu năng ổn định liên tục không jitter (tránh trễ đột ngột khi governor tụt xung xuống `408 MHz` CPU hoặc `300 MHz` NPU):

1. **Khóa Governor Ở Chế Độ `performance` Trên Cả CPU & NPU (cần quyền `root`/`sudo` trên board):**
   - Khóa toàn bộ cụm CPU (`policy0` = A55 `1.80 GHz`, `policy4` & `policy6` = A76 `2.26 GHz`): `scaling_governor = performance`.
   - Khóa NPU `fdab0000.npu` ở xung trần **`1000 MHz`**: đặt `/sys/class/devfreq/fdab0000.npu/governor = performance`.
   - Khóa tần số bộ nhớ `dmc` (nếu có trong `/sys/class/devfreq/dmc/governor`) để băng thông RAM đạt mức cực đại phục vụ DMA NPU 3 nhân chạy đồng thời.
2. **Tận Dụng `465 GB NVMe SSD` & `7.7 GiB RAM`:**
   - Lưu toàn bộ model, video test và cache trên phân vùng NVMe tốc độ cao.
   - Với `7.7 GiB RAM`: Cấp phát tĩnh toàn bộ pool NPU context (`3–6 contexts`, mỗi context chỉ tốn vài chục MB) và ring-buffer frame từ lúc khởi động để chạy **zero-allocation runtime**.
3. **Giám Sát Nhiệt & Chống Thermal Throttling:**
   - Thực nghiệm 30s tải `98%` NPU ở `1000 MHz` (`385.10 FPS`) đo được **`51.8–53.6°C`** trên `orangepi5`. Cần duy trì quạt PWM + heatsink khi chạy đa luồng đồng thời cả NPU (`1000 MHz`) + 4× A76 (`2256 MHz`) + VPU để nhiệt độ không chạm ngưỡng hạ xung của kernel.

---

## 4. Ma Trận Lộ Trình Chọn Kiến Trúc Theo Mục Tiêu Sử Dụng

| Mục Tiêu | Thành Phần Phần Cứng Sử Dụng | FPS Thực Tế Tham Chiếu (`YOLO26n 416×416` trên `orangepi5`) | Ghi Chú Quan Trọng |
| :--- | :--- | :---: | :--- |
| **1. Nộp bài chuẩn Hackathon (`.onnx` bắt buộc)** | `4× Cortex-A76` (`taskset -c 4-7`, `model_int8_static.onnx`, 4 threads) | **`~37.22 FPS`** (đo trên 4× A76) | Đạt chuẩn 100% file `src/detector.py`; NPU chưa được dùng. |
| **2. Chạy Production/Edge 1 Camera (`< 8 ms` Latency)** | `1× NPU CORE_0` + `C++ Zero-Copy` + `1× A76` (Lazy Postproc) + RGA | **`131.35 FPS`** (`7.61 ms` med NPU) | Giảm tải hoàn toàn 7/8 lõi CPU cho logic nghiệp vụ/tracking. |
| **3. Khai Thác Tối Đa 100% Chip RK3588S (Multi-Stream / Stress Max Throughput)** | **3× NPU (`6 workers` @ 1000 MHz)** + **C++ Zero-Copy** + **4× A76 (Postproc)** + **4× A55 (I/O)** + **MPP/RGA** | **`376.10 – 385.10 FPS`** (`98%` NPU load) | Kịch trần thực tế cho YOLO26n `416×416` INT8 `6-Head Cut` trên `orangepi5`. |
