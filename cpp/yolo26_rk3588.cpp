#include "yolo26_rk3588.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sched.h>
#include <thread>
#include <vector>

#include <rga/im2d.h>
#include <rga/rga.h>
#include <rknn_api.h>

namespace {

inline void pin_thread_cores(const std::vector<int>& cores) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int c : cores) CPU_SET(c, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

inline uint16_t f32_to_f16(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03FFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float res;
    std::memcpy(&res, &out, sizeof(res));
    return res;
}

struct HeadPair {
    int box_idx = -1;
    int cls_idx = -1;
    int h = 0;
    int w = 0;
    int stride = 0;
    int box_c2 = 1;
    int cls_c2 = 1;
    int box_is_nc1 = 0;
    int cls_is_nc1 = 0;
    int box_is_int8 = 0;
    int cls_is_int8 = 0;
    int box_raw_dtype = 0; // 0 = INT8, 1 = FP16, 2 = FP32
    int cls_raw_dtype = 0; // 0 = INT8, 1 = FP16, 2 = FP32
    int box_is_nchw = 0;
    int32_t box_zp = 0;
    float box_scale = 1.0f;
    int32_t cls_zp = 0;
    float cls_scale = 1.0f;
    float box_lut[256]{};
    float cls_lut[256]{};
};

struct WorkerCtx {
    rknn_context ctx = 0;
    rknn_tensor_mem* in_mem = nullptr;
    rga_buffer_handle_t rga_dst_handle = 0;
    std::vector<rknn_tensor_mem*> out_mems;
    std::vector<rknn_tensor_attr> out_io_attrs;
    std::vector<uint8_t> tmp_rgb_fallback;
    // Reusable LUT buffers for OpenCV-matched fast Q11 bilinear letterbox
    std::vector<int> x0_lut, x1_lut, wx0_lut, wx1_lut;
    std::vector<int> y0_lut, y1_lut, wy0_lut, wy1_lut;
};

struct Y26Engine {
    int n_workers = 1;
    int n_inputs = 1;
    int n_outputs = 0;
    int imgsz = 416;
    int zero_copy = 1;
    int pass_through_out = 1;
    int preproc_mode = Y26_PREPROC_CPU_U8_BILINEAR;
    int in_is_fp16 = 0;

    rknn_tensor_attr in_attr{};
    std::vector<rknn_tensor_attr> logical_out_attrs;
    std::vector<rknn_tensor_attr> native_out_attrs;
    std::vector<HeadPair> heads; // size 3 when n_outputs == 6
    rknn_mem_size mem_size{};

    std::vector<WorkerCtx> workers;
};

static bool build_q11_bilinear_letterbox_rgb(
    WorkerCtx& wk,
    const uint8_t* bgr,
    int ow,
    int oh,
    int bgr_stride,
    int imgsz,
    uint8_t* dst_rgb,
    int dst_w_stride,
    int& out_nw,
    int& out_nh,
    int& out_px,
    int& out_py,
    float& out_scale
) {
    out_scale = std::min(static_cast<float>(imgsz) / ow, static_cast<float>(imgsz) / oh);
    out_nw = static_cast<int>(std::llround(ow * out_scale));
    out_nh = static_cast<int>(std::llround(oh * out_scale));
    out_px = (imgsz - out_nw) / 2;
    out_py = (imgsz - out_nh) / 2;

    const int dst_row_bytes = dst_w_stride * 3;
    std::memset(dst_rgb, 114, static_cast<size_t>(imgsz) * dst_row_bytes);

    wk.x0_lut.resize(out_nw);
    wk.x1_lut.resize(out_nw);
    wk.wx0_lut.resize(out_nw);
    wk.wx1_lut.resize(out_nw);
    const double scale_x = static_cast<double>(ow) / out_nw;
    for (int dx = 0; dx < out_nw; ++dx) {
        double fx = (dx + 0.5) * scale_x - 0.5;
        int sx = static_cast<int>(std::floor(fx));
        fx -= sx;
        if (sx < 0) { sx = 0; fx = 0.0; }
        if (sx >= ow - 1) { sx = std::max(0, ow - 1); fx = 0.0; }
        int sx1 = std::min(sx + 1, ow - 1);
        int iw1 = static_cast<int>(std::llround(fx * 2048.0));
        wk.x0_lut[dx] = sx * 3;
        wk.x1_lut[dx] = sx1 * 3;
        wk.wx1_lut[dx] = iw1;
        wk.wx0_lut[dx] = 2048 - iw1;
    }

    wk.y0_lut.resize(out_nh);
    wk.y1_lut.resize(out_nh);
    wk.wy0_lut.resize(out_nh);
    wk.wy1_lut.resize(out_nh);
    const double scale_y = static_cast<double>(oh) / out_nh;
    for (int dy = 0; dy < out_nh; ++dy) {
        double fy = (dy + 0.5) * scale_y - 0.5;
        int sy = static_cast<int>(std::floor(fy));
        fy -= sy;
        if (sy < 0) { sy = 0; fy = 0.0; }
        if (sy >= oh - 1) { sy = std::max(0, oh - 1); fy = 0.0; }
        int sy1 = std::min(sy + 1, oh - 1);
        int iy1 = static_cast<int>(std::llround(fy * 2048.0));
        wk.y0_lut[dy] = sy;
        wk.y1_lut[dy] = sy1;
        wk.wy1_lut[dy] = iy1;
        wk.wy0_lut[dy] = 2048 - iy1;
    }

    for (int dy = 0; dy < out_nh; ++dy) {
        const uint8_t* row0 = bgr + static_cast<size_t>(wk.y0_lut[dy]) * bgr_stride;
        const uint8_t* row1 = bgr + static_cast<size_t>(wk.y1_lut[dy]) * bgr_stride;
        uint8_t* dst_row = dst_rgb + static_cast<size_t>(out_py + dy) * dst_row_bytes + out_px * 3;
        const int wy0 = wk.wy0_lut[dy];
        const int wy1 = wk.wy1_lut[dy];

        for (int dx = 0; dx < out_nw; ++dx) {
            const uint8_t* p00 = row0 + wk.x0_lut[dx];
            const uint8_t* p01 = row0 + wk.x1_lut[dx];
            const uint8_t* p10 = row1 + wk.x0_lut[dx];
            const uint8_t* p11 = row1 + wk.x1_lut[dx];
            const int wx0 = wk.wx0_lut[dx];
            const int wx1 = wk.wx1_lut[dx];

            // BGR -> RGB swap directly in output DMA buffer
            for (int c = 0; c < 3; ++c) {
                const int sc = 2 - c;
                const int h0 = p00[sc] * wx0 + p01[sc] * wx1;
                const int h1 = p10[sc] * wx0 + p11[sc] * wx1;
                const int val = (h0 * wy0 + h1 * wy1 + (1 << 21)) >> 22;
                dst_row[dx * 3 + c] = static_cast<uint8_t>(std::min(255, std::max(0, val)));
            }
        }
    }
    return true;
}

static bool try_rga_letterbox_to_dma(
    WorkerCtx& wk,
    const uint8_t* bgr,
    int ow,
    int oh,
    int bgr_stride,
    int imgsz,
    uint8_t* dst_virt,
    int dst_w_stride,
    int& out_nw,
    int& out_nh,
    int& out_px,
    int& out_py,
    float& out_scale
) {
    if (!wk.rga_dst_handle || ow < 16 || oh < 16 || (bgr_stride % 3) != 0) return false;
    out_scale = std::min(static_cast<float>(imgsz) / ow, static_cast<float>(imgsz) / oh);
    out_nw = static_cast<int>(std::llround(ow * out_scale));
    out_nh = static_cast<int>(std::llround(oh * out_scale));
    out_px = (imgsz - out_nw) / 2;
    out_py = (imgsz - out_nh) / 2;

    const int src_w_stride = bgr_stride / 3;
    // RK3588 RGA requires 16-pixel aligned source width stride for BGR888/RGB888; odd drect offsets work natively.
    if (out_nw < 16 || out_nh < 16 || (src_w_stride & 15) != 0) {
        return false;
    }
    // Fill letterbox padding bars (114,114,114) into write-combined DMA memory (RGA colorfill rejects 3-byte RGB888)
    const size_t row_bytes = static_cast<size_t>(dst_w_stride) * 3;
    if (out_py > 0) {
        std::memset(dst_virt, 114, static_cast<size_t>(out_py) * row_bytes);
        const int bot_y = out_py + out_nh;
        if (bot_y < imgsz) {
            std::memset(dst_virt + static_cast<size_t>(bot_y) * row_bytes, 114, static_cast<size_t>(imgsz - bot_y) * row_bytes);
        }
    }
    if (out_px > 0) {
        const int rx = out_px + out_nw;
        for (int y = out_py; y < out_py + out_nh; ++y) {
            uint8_t* rp = dst_virt + static_cast<size_t>(y) * row_bytes;
            std::memset(rp, 114, static_cast<size_t>(out_px) * 3);
            if (rx < imgsz) std::memset(rp + static_cast<size_t>(rx) * 3, 114, static_cast<size_t>(imgsz - rx) * 3);
        }
    }
    rknn_mem_sync(wk.ctx, wk.in_mem, RKNN_MEMORY_SYNC_TO_DEVICE);

    rga_buffer_handle_t src_h = importbuffer_virtualaddr(const_cast<uint8_t*>(bgr), src_w_stride, oh, RK_FORMAT_BGR_888);
    if (!src_h) return false;

    rga_buffer_t src_buf = wrapbuffer_handle(src_h, ow, oh, RK_FORMAT_BGR_888, src_w_stride, oh);
    rga_buffer_t dst_buf = wrapbuffer_handle(wk.rga_dst_handle, imgsz, imgsz, RK_FORMAT_RGB_888, dst_w_stride, imgsz);

    im_rect srect{0, 0, ow, oh};
    im_rect drect{out_px, out_py, out_nw, out_nh};
    im_rect prect{0, 0, 0, 0};

    IM_STATUS st = improcess(src_buf, dst_buf, rga_buffer_t{}, srect, drect, prect, -1, nullptr, nullptr, IM_SYNC);
    releasebuffer_handle(src_h);
    return st == IM_STATUS_SUCCESS;
}

static void fast_nms(std::vector<Y26Det>& cands, float iou_thresh, int max_det, std::vector<Y26Det>& out) {
    out.clear();
    if (cands.empty()) return;
    if (cands.size() > 1200) {
        std::nth_element(cands.begin(), cands.begin() + 1200, cands.end(),
                         [](const Y26Det& a, const Y26Det& b) { return a.score > b.score; });
        cands.resize(1200);
    }
    std::sort(cands.begin(), cands.end(), [](const Y26Det& a, const Y26Det& b) { return a.score > b.score; });

    std::vector<uint8_t> suppressed(cands.size(), 0);
    for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;
        const Y26Det& a = cands[i];
        out.push_back(a);
        if (static_cast<int>(out.size()) >= max_det) break;
        const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (suppressed[j]) continue;
            const Y26Det& b = cands[j];
            const float ix1 = std::max(a.x1, b.x1);
            const float iy1 = std::max(a.y1, b.y1);
            const float ix2 = std::min(a.x2, b.x2);
            const float iy2 = std::min(a.y2, b.y2);
            if (ix2 <= ix1 || iy2 <= iy1) continue;
            const float inter = (ix2 - ix1) * (iy2 - iy1);
            const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
            const float uni = area_a + area_b - inter;
            if (uni > 0.0f && (inter / uni) > iou_thresh) {
                suppressed[j] = 1;
            }
        }
    }
}

static bool init_head_pairs(Y26Engine* eng) {
    struct BranchTag { int idx; int c; int h; int w; };
    std::vector<BranchTag> boxes, clss;
    for (int i = 0; i < eng->n_outputs; ++i) {
        const auto& a = eng->logical_out_attrs[i];
        // Check dims regardless of NCHW/NHWC logical attrs
        int c = 0, h = 0, w = 0;
        if (a.fmt == RKNN_TENSOR_NCHW && a.n_dims == 4) {
            c = static_cast<int>(a.dims[1]);
            h = static_cast<int>(a.dims[2]);
            w = static_cast<int>(a.dims[3]);
        } else if (a.fmt == RKNN_TENSOR_NHWC && a.n_dims == 4) {
            h = static_cast<int>(a.dims[1]);
            w = static_cast<int>(a.dims[2]);
            c = static_cast<int>(a.dims[3]);
        } else if (a.n_dims == 4) {
            // Heuristic on 4D YOLO26 shapes: H==W in {52,26,13}, C in {1,4}
            if (a.dims[2] == a.dims[3]) { c = a.dims[1]; h = a.dims[2]; w = a.dims[3]; }
            else { h = a.dims[1]; w = a.dims[2]; c = a.dims[3]; }
        }
        if (c == 4) boxes.push_back({i, c, h, w});
        else if (c == 1) clss.push_back({i, c, h, w});
    }
    if (boxes.size() != 3 || clss.size() != 3) return false;
    std::sort(boxes.begin(), boxes.end(), [](const BranchTag& a, const BranchTag& b) { return a.h > b.h; });
    std::sort(clss.begin(), clss.end(), [](const BranchTag& a, const BranchTag& b) { return a.h > b.h; });

    eng->heads.resize(3);
    for (int s = 0; s < 3; ++s) {
        if (boxes[s].h != clss[s].h || boxes[s].w != clss[s].w) return false;
        auto& hp = eng->heads[s];
        hp.box_idx = boxes[s].idx;
        hp.cls_idx = clss[s].idx;
        hp.h = boxes[s].h;
        hp.w = boxes[s].w;
        hp.stride = eng->imgsz / hp.h;

        const auto& b_nat = eng->native_out_attrs[hp.box_idx];
        const auto& c_nat = eng->native_out_attrs[hp.cls_idx];
        const auto& b_log = eng->logical_out_attrs[hp.box_idx];
        const auto& c_log = eng->logical_out_attrs[hp.cls_idx];

        const auto& b_src = eng->zero_copy ? b_nat : b_log;
        const auto& c_src = eng->zero_copy ? c_nat : c_log;

        hp.box_zp = b_src.zp;
        hp.box_scale = b_src.scale;
        hp.cls_zp = c_src.zp;
        hp.cls_scale = c_src.scale;
        hp.box_is_int8 = (b_src.type == RKNN_TENSOR_INT8) ? 1 : 0;
        hp.cls_is_int8 = (c_src.type == RKNN_TENSOR_INT8) ? 1 : 0;
        if (eng->zero_copy) {
            hp.box_raw_dtype = hp.box_is_int8 ? 0 : (b_nat.type == RKNN_TENSOR_FLOAT16 ? 1 : 2);
            hp.cls_raw_dtype = hp.cls_is_int8 ? 0 : (c_nat.type == RKNN_TENSOR_FLOAT16 ? 1 : 2);
        } else {
            hp.box_raw_dtype = hp.box_is_int8 ? 0 : 2;
            hp.cls_raw_dtype = hp.cls_is_int8 ? 0 : 2;
        }
        hp.box_is_nc1 = (eng->zero_copy && eng->pass_through_out && b_nat.fmt == RKNN_TENSOR_NC1HWC2 && b_nat.n_dims == 5) ? 1 : 0;
        hp.cls_is_nc1 = (eng->zero_copy && eng->pass_through_out && c_nat.fmt == RKNN_TENSOR_NC1HWC2 && c_nat.n_dims == 5) ? 1 : 0;
        hp.box_is_nchw = (!eng->zero_copy && b_log.fmt == RKNN_TENSOR_NCHW) ? 1 : 0;
        hp.box_c2 = hp.box_is_nc1 ? static_cast<int>(b_nat.dims[4]) : 4;
        hp.cls_c2 = hp.cls_is_nc1 ? static_cast<int>(c_nat.dims[4]) : 1;

        for (int q_u8 = 0; q_u8 < 256; ++q_u8) {
            int q_i8 = static_cast<int>(static_cast<int8_t>(static_cast<uint8_t>(q_u8)));
            hp.box_lut[q_u8] = (q_i8 - hp.box_zp) * hp.box_scale * static_cast<float>(hp.stride);
            float logit = (q_i8 - hp.cls_zp) * hp.cls_scale;
            hp.cls_lut[q_u8] = 1.0f / (1.0f + std::exp(-logit));
        }
    }
    return true;
}

} // namespace

extern "C" {

void* y26_create_engine(
    const char* rknn_path,
    int n_workers,
    int core_mask_mode,
    int zero_copy,
    int pass_through_out,
    int preproc_mode,
    int enable_sram,
    int pin_a76_flag
) {
    if (pin_a76_flag) pin_thread_cores({4, 5, 6, 7});

    std::ifstream ifs(rknn_path, std::ios::binary | std::ios::ate);
    if (!ifs) return nullptr;
    size_t fsz = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::vector<uint8_t> model_data(fsz);
    if (!ifs.read(reinterpret_cast<char*>(model_data.data()), static_cast<std::streamsize>(fsz))) return nullptr;

    auto* eng = new Y26Engine();
    eng->n_workers = std::max(1, n_workers);
    eng->zero_copy = zero_copy ? 1 : 0;
    eng->pass_through_out = pass_through_out ? 1 : 0;
    eng->preproc_mode = preproc_mode;
    eng->workers.resize(eng->n_workers);

    uint32_t flags = 0;
    if (enable_sram) flags |= RKNN_FLAG_ENABLE_SRAM;

    int ret = rknn_init(&eng->workers[0].ctx, model_data.data(), static_cast<uint32_t>(model_data.size()), flags, nullptr);
    if (ret < 0) {
        delete eng;
        return nullptr;
    }
    for (int w = 1; w < eng->n_workers; ++w) {
        ret = rknn_dup_context(&eng->workers[0].ctx, &eng->workers[w].ctx);
        if (ret < 0) {
            y26_destroy_engine(eng);
            return nullptr;
        }
    }

    static const rknn_core_mask kCoreMasks[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    for (int w = 0; w < eng->n_workers; ++w) {
        rknn_core_mask mask = RKNN_NPU_CORE_0;
        if (eng->n_workers > 1) mask = kCoreMasks[w % 3];
        else if (core_mask_mode == 0) mask = RKNN_NPU_CORE_AUTO;
        else if (core_mask_mode == 2) mask = RKNN_NPU_CORE_0_1_2;
        else mask = RKNN_NPU_CORE_0;
        rknn_set_core_mask(eng->workers[w].ctx, mask);
    }

    rknn_query(eng->workers[0].ctx, RKNN_QUERY_MEM_SIZE, &eng->mem_size, sizeof(eng->mem_size));

    rknn_input_output_num io_num{};
    rknn_query(eng->workers[0].ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    eng->n_inputs = static_cast<int>(io_num.n_input);
    eng->n_outputs = static_cast<int>(io_num.n_output);

    eng->in_attr.index = 0;
    rknn_query(eng->workers[0].ctx, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, &eng->in_attr, sizeof(eng->in_attr));
    if (eng->in_attr.dims[1] > 0) eng->imgsz = static_cast<int>(eng->in_attr.dims[1]);
    eng->in_is_fp16 = (eng->in_attr.type == RKNN_TENSOR_FLOAT16) ? 1 : 0;

    eng->logical_out_attrs.resize(eng->n_outputs);
    eng->native_out_attrs.resize(eng->n_outputs);
    for (int i = 0; i < eng->n_outputs; ++i) {
        eng->logical_out_attrs[i].index = i;
        rknn_query(eng->workers[0].ctx, RKNN_QUERY_OUTPUT_ATTR, &eng->logical_out_attrs[i], sizeof(rknn_tensor_attr));

        eng->native_out_attrs[i].index = i;
        if (eng->pass_through_out) {
            rknn_query(eng->workers[0].ctx, RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR, &eng->native_out_attrs[i], sizeof(rknn_tensor_attr));
        } else {
            rknn_query(eng->workers[0].ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &eng->native_out_attrs[i], sizeof(rknn_tensor_attr));
        }
    }

    if (eng->n_outputs == 6) {
        if (!init_head_pairs(eng)) {
            y26_destroy_engine(eng);
            return nullptr;
        }
    }

    // Force zero_copy off for FP16-input uncut models where uint8->fp16 cast runs inside librknnrt
    if (eng->in_is_fp16 && eng->n_outputs == 1) {
        eng->zero_copy = 0;
    }

    if (eng->zero_copy) {
        for (int w = 0; w < eng->n_workers; ++w) {
            auto& wk = eng->workers[w];
            rknn_tensor_attr in_io = eng->in_attr;
            in_io.type = RKNN_TENSOR_UINT8;
            in_io.fmt = RKNN_TENSOR_NHWC;
            in_io.pass_through = 0;
            const uint32_t in_bytes = std::max(in_io.size_with_stride, static_cast<uint32_t>(eng->imgsz * eng->imgsz * 3));
            wk.in_mem = rknn_create_mem(wk.ctx, in_bytes);
            if (!wk.in_mem || rknn_set_io_mem(wk.ctx, wk.in_mem, &in_io) < 0) {
                y26_destroy_engine(eng);
                return nullptr;
            }
            if (wk.in_mem->fd >= 0) {
                const int w_str = in_io.w_stride > 0 ? static_cast<int>(in_io.w_stride) : eng->imgsz;
                wk.rga_dst_handle = importbuffer_fd(wk.in_mem->fd, w_str, eng->imgsz, RK_FORMAT_RGB_888);
            }

            wk.out_mems.resize(eng->n_outputs, nullptr);
            wk.out_io_attrs.resize(eng->n_outputs);
            for (int i = 0; i < eng->n_outputs; ++i) {
                rknn_tensor_attr oa = eng->native_out_attrs[i];
                if (eng->pass_through_out) {
                    oa.pass_through = 1;
                } else {
                    oa.pass_through = 0;
                    if (oa.type != RKNN_TENSOR_INT8 && oa.type != RKNN_TENSOR_FLOAT16) {
                        oa.type = RKNN_TENSOR_FLOAT32;
                    }
                    if (oa.n_dims == 4) oa.fmt = RKNN_TENSOR_NHWC;
                }
                const uint32_t obytes = std::max(oa.size_with_stride, oa.size);
                wk.out_mems[i] = rknn_create_mem(wk.ctx, obytes);
                wk.out_io_attrs[i] = oa;
                if (!wk.out_mems[i] || rknn_set_io_mem(wk.ctx, wk.out_mems[i], &wk.out_io_attrs[i]) < 0) {
                    y26_destroy_engine(eng);
                    return nullptr;
                }
            }
        }
    } else {
        for (int w = 0; w < eng->n_workers; ++w) {
            eng->workers[w].tmp_rgb_fallback.resize(static_cast<size_t>(eng->imgsz) * eng->imgsz * 3);
        }
    }

    return eng;
}

void y26_destroy_engine(void* handle) {
    if (!handle) return;
    auto* eng = static_cast<Y26Engine*>(handle);
    for (auto& wk : eng->workers) {
        if (wk.rga_dst_handle) {
            releasebuffer_handle(wk.rga_dst_handle);
            wk.rga_dst_handle = 0;
        }
        if (wk.in_mem && wk.ctx) {
            rknn_destroy_mem(wk.ctx, wk.in_mem);
            wk.in_mem = nullptr;
        }
        for (auto* om : wk.out_mems) {
            if (om && wk.ctx) rknn_destroy_mem(wk.ctx, om);
        }
        wk.out_mems.clear();
        if (wk.ctx) {
            rknn_destroy(wk.ctx);
            wk.ctx = 0;
        }
    }
    delete eng;
}

int y26_get_info(void* handle, Y26EngineInfo* out_info) {
    if (!handle || !out_info) return -1;
    auto* eng = static_cast<Y26Engine*>(handle);
    out_info->n_workers = eng->n_workers;
    out_info->n_outputs = eng->n_outputs;
    out_info->imgsz = eng->imgsz;
    out_info->is_int8_outputs = (!eng->heads.empty() && eng->heads[0].cls_is_int8) ? 1 : 0;
    out_info->use_zero_copy = eng->zero_copy;
    out_info->weight_bytes = eng->mem_size.total_weight_size;
    out_info->internal_bytes = eng->mem_size.total_internal_size;
    out_info->total_dma_bytes = eng->mem_size.total_dma_allocated_size;
    return 0;
}

int y26_detect_bgr(
    void* handle,
    int worker_id,
    const uint8_t* bgr,
    int orig_w,
    int orig_h,
    int bgr_stride_bytes,
    float conf_thresh,
    float iou_thresh,
    int max_det,
    Y26Det* out_dets,
    int max_out,
    Y26Timings* out_timings
) {
    if (!handle || !bgr || orig_w <= 0 || orig_h <= 0) return -1;
    auto* eng = static_cast<Y26Engine*>(handle);
    const int wid = (worker_id >= 0 && worker_id < eng->n_workers) ? worker_id : 0;
    auto& wk = eng->workers[wid];
    if (bgr_stride_bytes <= 0) bgr_stride_bytes = orig_w * 3;

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    int nw = 0, nh = 0, px = 0, py = 0;
    float scale = 1.0f;
    int rga_used = 0;
    int used_fallback = 0;

    if (eng->zero_copy) {
        uint8_t* dst = static_cast<uint8_t*>(wk.in_mem->virt_addr);
        const int w_str = eng->in_attr.w_stride > 0 ? static_cast<int>(eng->in_attr.w_stride) : eng->imgsz;
        if (eng->preproc_mode == Y26_PREPROC_RGA_HW_BILINEAR) {
            if (try_rga_letterbox_to_dma(wk, bgr, orig_w, orig_h, bgr_stride_bytes, eng->imgsz, dst, w_str, nw, nh, px, py, scale)) {
                rga_used = 1;
            } else {
                used_fallback = 1;
                build_q11_bilinear_letterbox_rgb(wk, bgr, orig_w, orig_h, bgr_stride_bytes, eng->imgsz, dst, w_str, nw, nh, px, py, scale);
            }
        } else {
            build_q11_bilinear_letterbox_rgb(wk, bgr, orig_w, orig_h, bgr_stride_bytes, eng->imgsz, dst, w_str, nw, nh, px, py, scale);
        }
    } else {
        uint8_t* dst = wk.tmp_rgb_fallback.data();
        build_q11_bilinear_letterbox_rgb(wk, bgr, orig_w, orig_h, bgr_stride_bytes, eng->imgsz, dst, eng->imgsz, nw, nh, px, py, scale);
        rknn_input inp{};
        inp.index = 0;
        inp.type = RKNN_TENSOR_UINT8;
        inp.size = static_cast<uint32_t>(wk.tmp_rgb_fallback.size());
        inp.fmt = RKNN_TENSOR_NHWC;
        inp.buf = dst;
        if (rknn_inputs_set(wk.ctx, 1, &inp) < 0) return -2;
    }

    auto t1 = clk::now();

    if (rknn_run(wk.ctx, nullptr) < 0) return -3;

    std::vector<rknn_output> copy_outs;
    if (!eng->zero_copy) {
        copy_outs.resize(eng->n_outputs);
        for (int i = 0; i < eng->n_outputs; ++i) {
            copy_outs[i].index = i;
            copy_outs[i].want_float = (eng->n_outputs == 1 || eng->logical_out_attrs[i].type != RKNN_TENSOR_INT8) ? 1 : 0;
            copy_outs[i].is_prealloc = 0;
        }
        if (rknn_outputs_get(wk.ctx, eng->n_outputs, copy_outs.data(), nullptr) < 0) return -4;
    }

    auto t2 = clk::now();

    std::vector<Y26Det> cands;
    cands.reserve(128);
    const float inv_s = 1.0f / scale;
    const float f_ow = static_cast<float>(orig_w);
    const float f_oh = static_cast<float>(orig_h);

    if (eng->n_outputs == 6) {
        const float p_clamped = std::min(0.999999f, std::max(1e-6f, conf_thresh));
        const float logit_thresh = std::log(p_clamped / (1.0f - p_clamped));

        for (int s = 0; s < 3; ++s) {
            const auto& hp = eng->heads[s];
            const void* box_raw = eng->zero_copy ? wk.out_mems[hp.box_idx]->virt_addr : copy_outs[hp.box_idx].buf;
            const void* cls_raw = eng->zero_copy ? wk.out_mems[hp.cls_idx]->virt_addr : copy_outs[hp.cls_idx].buf;

            const int H = hp.h;
            const int W = hp.w;
            const int HW = H * W;
            const int c_step = hp.cls_is_nc1 ? hp.cls_c2 : 1;
            const int b_step = hp.box_is_nc1 ? hp.box_c2 : 4;
            const float str_f = static_cast<float>(hp.stride);

            int q_min = 0;
            if (hp.cls_raw_dtype == 0) {
                q_min = static_cast<int>(std::ceil((logit_thresh / hp.cls_scale) + hp.cls_zp));
                if (q_min > 127) continue;
            }

            auto read_box_val = [&](int cell, int ch) -> float {
                const int idx = hp.box_is_nchw ? (ch * HW + cell) : (cell * b_step + ch);
                if (hp.box_raw_dtype == 0) {
                    return hp.box_lut[static_cast<const uint8_t*>(box_raw)[idx]];
                }
                const float raw = (hp.box_raw_dtype == 1)
                    ? f16_to_f32(static_cast<const uint16_t*>(box_raw)[idx])
                    : static_cast<const float*>(box_raw)[idx];
                return raw * str_f;
            };

            for (int y = 0; y < H; ++y) {
                const float anchor_y = (static_cast<float>(y) + 0.5f) * str_f;
                const int row_base = y * W;
                for (int x = 0; x < W; ++x) {
                    const int cell = row_base + x;
                    float score = 0.0f;
                    if (hp.cls_raw_dtype == 0) {
                        const int8_t q = static_cast<const int8_t*>(cls_raw)[cell * c_step];
                        if (static_cast<int>(q) < q_min) continue; // Exact integer-domain branch cut: skips ~99% cells
                        score = hp.cls_lut[static_cast<uint8_t>(q)];
                    } else {
                        const float logit = (hp.cls_raw_dtype == 1)
                            ? f16_to_f32(static_cast<const uint16_t*>(cls_raw)[cell * c_step])
                            : static_cast<const float*>(cls_raw)[cell * c_step];
                        if (logit < logit_thresh) continue;
                        score = 1.0f / (1.0f + std::exp(-logit));
                    }

                    const float l = read_box_val(cell, 0);
                    const float t = read_box_val(cell, 1);
                    const float r = read_box_val(cell, 2);
                    const float b = read_box_val(cell, 3);

                    const float anchor_x = (static_cast<float>(x) + 0.5f) * str_f;
                    const float lx1 = anchor_x - l;
                    const float ly1 = anchor_y - t;
                    const float lx2 = anchor_x + r;
                    const float ly2 = anchor_y + b;

                    cands.push_back({
                        std::min(f_ow, std::max(0.0f, (lx1 - px) * inv_s)),
                        std::min(f_oh, std::max(0.0f, (ly1 - py) * inv_s)),
                        std::min(f_ow, std::max(0.0f, (lx2 - px) * inv_s)),
                        std::min(f_oh, std::max(0.0f, (ly2 - py) * inv_s)),
                        score,
                    });
                }
            }
        }
    } else if (eng->n_outputs == 1) {
        // Uncut [1, 5, 3549] float path
        const float* out_f = static_cast<const float*>(copy_outs[0].buf);
        const int N = 3549;
        const float* cx_p = out_f + 0 * N;
        const float* cy_p = out_f + 1 * N;
        const float* bw_p = out_f + 2 * N;
        const float* bh_p = out_f + 3 * N;
        const float* sc_p = out_f + 4 * N;
        for (int i = 0; i < N; ++i) {
            const float sc = sc_p[i];
            if (sc < conf_thresh) continue;
            const float cx = cx_p[i], cy = cy_p[i], hw = 0.5f * bw_p[i], hh = 0.5f * bh_p[i];
            cands.push_back({
                std::min(f_ow, std::max(0.0f, ((cx - hw) - px) * inv_s)),
                std::min(f_oh, std::max(0.0f, ((cy - hh) - py) * inv_s)),
                std::min(f_ow, std::max(0.0f, ((cx + hw) - px) * inv_s)),
                std::min(f_oh, std::max(0.0f, ((cy + hh) - py) * inv_s)),
                sc,
            });
        }
    }

    if (!eng->zero_copy) {
        rknn_outputs_release(wk.ctx, eng->n_outputs, copy_outs.data());
    }

    std::vector<Y26Det> kept;
    fast_nms(cands, iou_thresh, max_det, kept);

    auto t3 = clk::now();

    const int n_write = std::min(max_out, static_cast<int>(kept.size()));
    for (int i = 0; i < n_write; ++i) out_dets[i] = kept[i];

    if (out_timings) {
        out_timings->preproc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        out_timings->npu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        out_timings->postproc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        out_timings->total_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
        out_timings->rga_used = rga_used;
        out_timings->used_fallback = used_fallback;
    }
    return n_write;
}

int y26_bench_prepacked_multicore(
    void* handle,
    const uint8_t* const* bgr_ptrs,
    const int* orig_ws,
    const int* orig_hs,
    int n_images,
    int loops,
    float conf_thresh,
    float iou_thresh,
    int max_det,
    int pin_a76,
    double* out_wall_ms,
    double* out_mean_pre_ms,
    double* out_mean_npu_ms,
    double* out_mean_post_ms,
    double* out_p50_ms,
    double* out_p95_ms,
    double* out_p99_ms
) {
    if (!handle || !bgr_ptrs || n_images <= 0 || loops <= 0) return -1;
    auto* eng = static_cast<Y26Engine*>(handle);
    const int W = eng->n_workers;
    const int total = n_images * loops;

    std::vector<double> pre_lat(total, 0.0), npu_lat(total, 0.0), post_lat(total, 0.0), tot_lat(total, 0.0);

    // Warmup each worker once
    std::vector<Y26Det> wdet(max_det);
    for (int w = 0; w < W; ++w) {
        Y26Timings tm{};
        y26_detect_bgr(handle, w, bgr_ptrs[0], orig_ws[0], orig_hs[0], orig_ws[0] * 3, conf_thresh, iou_thresh, max_det, wdet.data(), max_det, &tm);
    }

    auto t_start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(W);
    for (int w = 0; w < W; ++w) {
        threads.emplace_back([&, w]() {
            if (pin_a76) pin_thread_cores({4 + (w % 4)}); // Spread workers cleanly across 4 Cortex-A76 cores
            std::vector<Y26Det> dets(max_det);
            for (int k = w; k < total; k += W) {
                const int idx = k % n_images;
                Y26Timings tm{};
                y26_detect_bgr(handle, w, bgr_ptrs[idx], orig_ws[idx], orig_hs[idx], orig_ws[idx] * 3, conf_thresh, iou_thresh, max_det, dets.data(), max_det, &tm);
                pre_lat[k] = tm.preproc_ms;
                npu_lat[k] = tm.npu_ms;
                post_lat[k] = tm.postproc_ms;
                tot_lat[k] = tm.total_ms;
            }
        });
    }
    for (auto& th : threads) th.join();
    auto t_end = std::chrono::steady_clock::now();

    if (out_wall_ms) *out_wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    auto mean_vec = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };
    if (out_mean_pre_ms) *out_mean_pre_ms = mean_vec(pre_lat);
    if (out_mean_npu_ms) *out_mean_npu_ms = mean_vec(npu_lat);
    if (out_mean_post_ms) *out_mean_post_ms = mean_vec(post_lat);

    std::sort(tot_lat.begin(), tot_lat.end());
    auto pct = [&](double q) {
        size_t pos = static_cast<size_t>(std::min(static_cast<double>(tot_lat.size() - 1), q * (tot_lat.size() - 1)));
        return tot_lat[pos];
    };
    if (out_p50_ms) *out_p50_ms = pct(0.50);
    if (out_p95_ms) *out_p95_ms = pct(0.95);
    if (out_p99_ms) *out_p99_ms = pct(0.99);
    return 0;
}

} // extern "C"
