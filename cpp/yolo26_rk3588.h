#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    Y26_PREPROC_CPU_U8_BILINEAR = 0,
    Y26_PREPROC_RGA_HW_BILINEAR = 1,
} Y26PreprocMode;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} Y26Det;

typedef struct {
    double preproc_ms;
    double npu_ms;
    double postproc_ms;
    double total_ms;
    int rga_used;
    int used_fallback;
} Y26Timings;

typedef struct {
    int n_workers;
    int n_outputs;
    int imgsz;
    int is_int8_outputs;
    int use_zero_copy;
    uint32_t weight_bytes;
    uint32_t internal_bytes;
    uint64_t total_dma_bytes;
} Y26EngineInfo;

void* y26_create_engine(
    const char* rknn_path,
    int n_workers,
    int core_mask_mode,
    int zero_copy,
    int pass_through_out,
    int preproc_mode,
    int enable_sram,
    int pin_a76_flag
);

void y26_destroy_engine(void* handle);

int y26_get_info(void* handle, Y26EngineInfo* out_info);

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
);

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
);

#ifdef __cplusplus
}
#endif
