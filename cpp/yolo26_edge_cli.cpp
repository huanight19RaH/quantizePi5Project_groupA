#include "yolo26_rk3588.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static bool read_ppm(const std::string& path, std::vector<uint8_t>& bgr, int& w, int& h) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    std::string magic;
    int maxval = 0;
    ifs >> magic >> w >> h >> maxval;
    if (magic != "P6" || w <= 0 || h <= 0 || maxval != 255) return false;
    ifs.get();
    bgr.resize(static_cast<size_t>(w) * h * 3);
    ifs.read(reinterpret_cast<char*>(bgr.data()), static_cast<std::streamsize>(bgr.size()));
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: %s <model.rknn> [--workers W] [--preproc 0|1] [--zerocopy 0|1] [--passthru 0|1] [--sram 0|1] [--ppm-dir DIR] [--loops L]\n", argv[0]);
        return 1;
    }
    const char* rknn_path = argv[1];
    int workers = 1;
    int preproc = 0;
    int zero_copy = 1;
    int pass_thru = 1;
    int sram = 0;
    int loops = 5;
    std::string ppm_dir;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) workers = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--preproc") == 0 && i + 1 < argc) preproc = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--zerocopy") == 0 && i + 1 < argc) zero_copy = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--passthru") == 0 && i + 1 < argc) pass_thru = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sram") == 0 && i + 1 < argc) sram = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc) loops = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ppm-dir") == 0 && i + 1 < argc) ppm_dir = argv[++i];
    }

    void* eng = y26_create_engine(rknn_path, workers, 1, zero_copy, pass_thru, preproc, sram, 1);
    if (!eng) {
        std::fprintf(stderr, "Failed to init engine for %s\n", rknn_path);
        return 2;
    }

    Y26EngineInfo info{};
    y26_get_info(eng, &info);
    std::printf("Engine OK | workers=%d outputs=%d imgsz=%d int8=%d zc=%d weight=%.2fMB dma=%.2fMB\n",
                info.n_workers, info.n_outputs, info.imgsz, info.is_int8_outputs, info.use_zero_copy,
                info.weight_bytes / 1048576.0, info.total_dma_bytes / 1048576.0);

    std::vector<std::vector<uint8_t>> imgs;
    std::vector<int> ws, hs;
    if (!ppm_dir.empty()) {
        for (const auto& entry : fs::directory_iterator(ppm_dir)) {
            if (entry.path().extension() == ".ppm") {
                std::vector<uint8_t> buf;
                int w = 0, h = 0;
                if (read_ppm(entry.path().string(), buf, w, h)) {
                    imgs.push_back(std::move(buf));
                    ws.push_back(w);
                    hs.push_back(h);
                }
            }
        }
    }
    if (imgs.empty()) {
        imgs.emplace_back(640 * 480 * 3, 128);
        ws.push_back(640);
        hs.push_back(480);
    }

    std::vector<const uint8_t*> ptrs(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) ptrs[i] = imgs[i].data();

    double wall_ms = 0, m_pre = 0, m_npu = 0, m_post = 0, p50 = 0, p95 = 0, p99 = 0;
    y26_bench_prepacked_multicore(
        eng, ptrs.data(), ws.data(), hs.data(),
        static_cast<int>(imgs.size()), loops, 0.25f, 0.45f, 300, 1,
        &wall_ms, &m_pre, &m_npu, &m_post, &p50, &p95, &p99
    );
    double fps = (1000.0 * imgs.size() * loops) / (wall_ms > 1e-6 ? wall_ms : 1.0);
    std::printf("BENCH frames=%zu wall_ms=%.2f FPS=%.2f pre=%.3fms npu=%.3fms post=%.3fms p50=%.3fms p95=%.3fms p99=%.3fms\n",
                imgs.size() * static_cast<size_t>(loops), wall_ms, fps, m_pre, m_npu, m_post, p50, p95, p99);

    y26_destroy_engine(eng);
    return 0;
}
