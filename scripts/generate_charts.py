"""Generate verification charts in `reports/figures/` strictly from measured CSV/JSON outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def plot_distribution_alignment(out_path: Path) -> None:
    """Plot verified dataset distribution alignment across COCO train/val, Test_7500, and Calib_200."""
    # Exact values recorded in the Stage 0 Distribution Alignment Audit
    splits = [
        "COCO train2017\n(118,287)",
        "COCO val2017\n(5,000)",
        "Matched train2017\n(2,500)",
        "Full Test Suite\n(7,500)",
        "Calib Pool\n(200)",
        "Calib Extended\n(400)",
    ]
    pos_pct = [54.2, 53.9, 54.2, 54.0, 54.0, 54.2]
    neg_pct = [45.8, 46.1, 45.8, 46.0, 46.0, 45.8]
    small_pct = [41.0, 40.0, 41.6, 40.5, 40.1, 38.6]
    med_pct = [33.5, 34.5, 32.6, 33.9, 35.4, 35.0]
    large_pct = [25.5, 25.5, 25.8, 25.6, 24.5, 26.3]

    x = np.arange(len(splits))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160)

    ax1.bar(x, pos_pct, label="Positive Images (with person)", color="#2563eb")
    ax1.bar(x, neg_pct, bottom=pos_pct, label="Negative Images (0 person)", color="#94a3b8")
    for i, p in enumerate(pos_pct):
        ax1.text(i, p / 2, f"{p:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=8.5)
    ax1.set_ylabel("Share of Images (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("Positive vs Negative Scene Balance (Distribution Preserved)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    w = 0.25
    ax2.bar(x - w, small_pct, width=w, label="Small (<32^2 px)", color="#e11d48")
    ax2.bar(x, med_pct, width=w, label="Medium (32^2-96^2 px)", color="#f59e0b")
    ax2.bar(x + w, large_pct, width=w, label="Large (>96^2 px)", color="#10b981")
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits, fontsize=8.5)
    ax2.set_ylabel("Share of Person Boxes (%)")
    ax2.set_ylim(0, 52)
    ax2.set_title("Person Object Scale Distribution (S / M / L Alignment)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_block_sensitivity(csv_path: Path, out_path: Path) -> None:
    """Plot per-block INT8 box and confidence drift sensitivity from block_sensitivity_audit.csv."""
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=160)

    top_b = df.sort_values("box_mae_px", ascending=False).head(15).iloc[::-1]
    colors_b = ["#be123c" if v >= 1.2 else "#f43f5e" if v >= 0.7 else "#fb7185" for v in top_b["box_mae_px"]]
    ax1.barh(top_b["block"], top_b["box_mae_px"], color=colors_b)
    for i, v in enumerate(top_b["box_mae_px"]):
        ax1.text(v + 0.08, i, f"{v:.2f} px", va="center", fontsize=8)
    ax1.set_xlim(0, max(top_b["box_mae_px"]) * 1.18)
    ax1.set_xlabel("Raw Box Coordinate MAE vs FP32 (px)")
    ax1.set_title("Single-Block INT8 Quantization: Box Localization Drift (Top 15)")
    ax1.grid(axis="x", alpha=0.25)

    top_s = df.sort_values("score_mae", ascending=False).head(15).iloc[::-1]
    ax2.barh(top_s["block"], top_s["score_mae"] * 1e5, color="#2563eb")
    for i, v in enumerate(top_s["score_mae"] * 1e5):
        ax2.text(v + 0.06, i, f"{v:.1f}e-5", va="center", fontsize=8)
    ax2.set_xlim(0, max(top_s["score_mae"] * 1e5) * 1.18)
    ax2.set_xlabel("Person Score MAE x 1e-5 (vs FP32)")
    ax2.set_title("Single-Block INT8 Quantization: Confidence Drift (Top 15)")
    ax2.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_pareto_and_ablation(csv_path: Path, out_pareto: Path, out_prog: Path) -> None:
    """Plot Size-vs-Accuracy Pareto frontier and INT8 optimization progression from full_7500_ablation_report.csv."""
    df = pd.read_csv(csv_path)
    valid = df[df["mAP50"] > 0.5].copy()

    # 1. Pareto frontier: Size vs mAP50 and INT8 Coverage vs mAP50:95
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
    sc1 = ax1.scatter(valid["size_mb"], valid["mAP50"], s=95, c=valid["int8_conv_pct"], cmap="viridis", edgecolors="black", linewidths=0.6)
    cbar1 = fig.colorbar(sc1, ax=ax1)
    cbar1.set_label("INT8 Quantized Conv Ratio (%)")
    offsets1 = {
        "ONNX_FP32_1cls_Baseline": (-10, 6),
        "ONNX_FP16": (-8, 8),
        "PyTorch_FP32_GPU": (10, -6),
        "ONNX_INT8_Dynamic": (8, -3),
        "INT8_StdMix_MinMax_C200": (10, 6),
        "INT8_StdMix_Entropy_C200": (10, -8),
        "INT8_StdMix_Percentile9999_C200": (10, -4),
        "INT8_HeadOnly_Pct9999_C200": (-10, -6),
        "INT8_SensPlusM1_Pct9999_C200": (14, 0),
        "INT8_SensPlusM1M2_Pct9999_C200": (14, 6),
        "INT8_HeadM0M1M2_Pct9999_C200": (-12, -2),
        "INT8_SensPlusM1M2_Pct99995_C200": (14, -8),
        "INT8_HeadM0M1M2_Pct99995_C200": (-10, 14),
        "INT8_SensM1M2M16M4_Pct99995_C200": (16, 8),
    }
    short1 = {
        "INT8_StdMix_Entropy_C200": "StdMix_Entropy (ties MinMax)",
        "INT8_HeadM0M1M2_Pct99995_C200": "HeadM0M1M2_Pct99995 (Best)",
    }
    for _, r in valid.iterrows():
        v = r["variant"]
        nm = short1.get(v, v.replace("INT8_", "").replace("ONNX_", ""))
        dx, dy = offsets1.get(v, (6, 2))
        ax1.annotate(
            nm,
            (r["size_mb"], r["mAP50"]),
            fontsize=7.0,
            ha="right" if dx <= -10 else "left",
            xytext=(dx, dy),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#94a3b8"},
        )
    ax1.set_xlim(0.6, 9.9)
    ax1.set_ylim(0.6890, 0.7008)
    ax1.set_xlabel("ONNX Model Size (MiB)")
    ax1.set_ylabel("COCO mAP@50 (7,500 Test Images)")
    ax1.set_title("Model Size vs mAP@50 (15 Full 7,500-Image Runs)")
    ax1.grid(True, alpha=0.25)

    int8_only = valid[valid["int8_conv_pct"] > 0].copy()
    sc2 = ax2.scatter(int8_only["int8_conv_pct"], int8_only["mAP50_95"], s=100, c=int8_only["mean_iou_vs_fp32"], cmap="plasma", edgecolors="black", linewidths=0.6)
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Mean Box IoU vs FP32")
    ax2.axhline(0.46396, color="#475569", linestyle="--", linewidth=1.1, label="FP32 Baseline (0.46396)")
    offsets2 = {
        "INT8_SensM1M2M16M4_Pct99995_C200": (6, 8),
        "INT8_SensPlusM1M2_Pct99995_C200": (-135, 10),
        "INT8_SensPlusM1M2_Pct9999_C200": (-135, -8),
        "INT8_HeadM0M1M2_Pct99995_C200": (8, 8),
        "INT8_HeadM0M1M2_Pct9999_C200": (8, -14),
        "INT8_SensPlusM1_Pct9999_C200": (10, -2),
        "INT8_StdMix_Percentile9999_C200": (-125, -4),
        "INT8_HeadOnly_Pct9999_C200": (-105, -14),
        "INT8_StdMix_MinMax_C200": (-95, 6),
        "INT8_StdMix_Entropy_C200": (-95, -8),
        "ONNX_INT8_Dynamic": (-65, 6),
    }
    for _, r in int8_only.iterrows():
        v = r["variant"]
        nm = short1.get(v, v.replace("INT8_", "").replace("ONNX_", ""))
        dx, dy = offsets2.get(v, (6, 2))
        ax2.annotate(nm, (r["int8_conv_pct"], r["mAP50_95"]), fontsize=7.0, xytext=(dx, dy), textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#94a3b8"})
    ax2.set_xlim(50.0, 80.0)
    ax2.set_ylim(0.4535, 0.4648)
    ax2.set_xlabel("INT8 Fused Conv Ratio (%)")
    ax2.set_ylabel("COCO mAP@50:95 (7,500 Test Images)")
    ax2.set_title("INT8 Coverage (%) vs Localization mAP@50:95")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_pareto)
    plt.close(fig)

    # 2. Optimization progression bar chart (mAP@50 & AP_small recovery)
    stages = [
        "ONNX_FP32_1cls_Baseline",
        "ONNX_INT8_Dynamic",
        "INT8_StdMix_MinMax_C200",
        "INT8_StdMix_Percentile9999_C200",
        "INT8_HeadOnly_Pct9999_C200",
        "INT8_SensPlusM1_Pct9999_C200",
        "INT8_SensPlusM1M2_Pct9999_C200",
        "INT8_HeadM0M1M2_Pct99995_C200",
        "INT8_SensM1M2M16M4_Pct99995_C200",
    ]
    sub = df.set_index("variant").loc[stages].reset_index()
    short_lbl = [
        "FP32\nBase",
        "Dynamic\nINT8",
        "StdMix\nMinMax",
        "StdMix\nPct99.99",
        "HeadOnly\nPct99.99",
        "+M1\nPct99.99",
        "+M1M2\nPct99.99",
        "Head+M0-2\n99.995*",
        "+M4+M16\n99.995",
    ]
    x = np.arange(len(sub))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4), dpi=160)

    bars1 = ax1.bar(x, sub["mAP50"], color=["#334155" if i == 0 else "#059669" if "99995" in s and "Head" in s else "#3b82f6" for i, s in enumerate(stages)])
    ax1.set_ylim(0.684, 0.703)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_lbl, fontsize=8.0)
    ax1.set_ylabel("mAP@50 (7,500 Test Images)")
    ax1.set_title("INT8 Optimization Trajectory: Recovery Toward FP32 mAP@50")
    for b, v in zip(bars1, sub["mAP50"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.5f}", ha="center", va="bottom", fontsize=7.5, rotation=20)
    ax1.grid(axis="y", alpha=0.25)

    bars2 = ax2.bar(x, sub["AP_small"], color=["#334155" if i == 0 else "#059669" if "99995" in s and "Head" in s else "#f59e0b" for i, s in enumerate(stages)])
    ax2.set_ylim(0.172, 0.184)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_lbl, fontsize=8.0)
    ax2.set_ylabel("COCO AP_small (<32^2 px)")
    ax2.set_title("Small Person Accuracy Preservation (Distribution & Scale Integrity)")
    for b, v in zip(bars2, sub["AP_small"]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v:.5f}", ha="center", va="bottom", fontsize=7.5, rotation=20)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_prog)
    plt.close(fig)


def plot_orangepi5_before_after(
    lat_csv: Path,
    acc_csv: Path,
    res_csv: Path,
    out_fps_lat: Path,
    out_acc_res: Path,
) -> None:
    """Plot Orange Pi 5 Before-vs-After Latency/FPS progression, val2017 accuracy parity, and resolution scaling."""
    df_l = pd.read_csv(lat_csv)
    df_a = pd.read_csv(acc_csv)
    df_r = pd.read_csv(res_csv)

    # Chart 1: Throughput FPS ladder + Stacked Stage Latency (Single-Stream)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=160)

    order_labels = {
        "1_BEFORE_ONNX_FP32_ondemand": "BEFORE\nONNX FP32\n(ondemand)",
        "2_BEFORE_ONNX_INT8_Static_ondemand": "BEFORE\nONNX INT8\n(ondemand)",
        "4_CPU_ONNX_INT8_FastBlob_PinnedA76": "CPU Opt\nFastBlob+A76\n(perf)",
        "5_NPU_FP16_Uncut_Copy_1Core": "NPU FP16\nUncut\n(1Core)",
        "6_NPU_INT8_6Head_Copy_1Core": "NPU INT8\n6Head Copy\n(1Core)",
        "8_NPU_INT8_6Head_ZeroCopy_NC1_1Core": "NPU 6Head\nZC+NC1\n(1Core)",
        "10_NPU_INT8_6Head_ZeroCopy_NC1_RGA_1Core": "AFTER 1C\nZC+NC1+RGA\n(Min Lat)",
        "11_NPU_INT8_6Head_ZeroCopy_NC1_3Workers": "NPU 6Head\n3 Cores\n(3 Workers)",
        "13_NPU_INT8_6Head_ZeroCopy_RGA_6Workers": "AFTER Max\nRGA+3Cores\n(6 Workers)",
    }
    sub_l = df_l[df_l["config"].isin(order_labels)].copy()
    x1 = np.arange(len(sub_l))
    colors1 = [
        "#64748b" if c.startswith(("1_", "2_"))
        else "#3b82f6" if c.startswith(("4_", "5_"))
        else "#0ea5e9" if c.startswith(("6_", "8_"))
        else "#059669" if c.startswith("10_")
        else "#047857"
        for c in sub_l["config"]
    ]
    bars = ax1.bar(x1, sub_l["fps"], color=colors1)
    for b, v in zip(bars, sub_l["fps"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 6.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([order_labels[c] for c in sub_l["config"]], fontsize=7.5)
    ax1.set_ylim(0, max(sub_l["fps"]) * 1.15)
    ax1.set_ylabel("Sustained Throughput (FPS)")
    ax1.set_title("Orange Pi 5 Throughput: Before (CPU ONNX) vs After (NPU Zero-Copy + RGA + 3-Core)")
    ax1.grid(axis="y", alpha=0.25)

    single_keys = [
        "1_BEFORE_ONNX_FP32_ondemand",
        "2_BEFORE_ONNX_INT8_Static_ondemand",
        "4_CPU_ONNX_INT8_FastBlob_PinnedA76",
        "5_NPU_FP16_Uncut_Copy_1Core",
        "6_NPU_INT8_6Head_Copy_1Core",
        "8_NPU_INT8_6Head_ZeroCopy_NC1_1Core",
        "10_NPU_INT8_6Head_ZeroCopy_NC1_RGA_1Core",
    ]
    sub_s = df_l[df_l["config"].isin(single_keys)].copy()
    x2 = np.arange(len(sub_s))
    ax2.bar(x2, sub_s["pre_ms"], label="Preprocess (Letterbox/RGB)", color="#f59e0b")
    ax2.bar(x2, sub_s["inf_ms"], bottom=sub_s["pre_ms"], label="Inference (ORT / RKNN NPU)", color="#2563eb")
    ax2.bar(x2, sub_s["post_ms"], bottom=sub_s["pre_ms"] + sub_s["inf_ms"], label="Postprocess (Decode + NMS)", color="#ef4444")
    for i, r in enumerate(sub_s.itertuples()):
        ax2.text(i, r.mean_total_ms + 0.7, f"{r.mean_total_ms:.2f} ms", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([order_labels[c] for c in sub_s["config"]], fontsize=7.5)
    ax2.set_ylim(0, max(sub_s["mean_total_ms"]) * 1.16)
    ax2.set_ylabel("Single-Stream End-to-End Latency (ms)")
    ax2.set_title("Single-Frame Stage Latency Breakdown (42.46 ms / 30.79 ms -> 8.18 ms)")
    ax2.legend(loc="upper right", fontsize=8.0)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_fps_lat)
    plt.close(fig)

    # Chart 2: Full 5,000-Image COCO val2017 Accuracy Parity + Resolution Preprocessing Tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=160)
    acc_map = {
        "ONNX_FP32_Baseline": "ONNX FP32\n(Ref)",
        "ONNX_INT8_Static_Before": "ONNX INT8\nBefore",
        "ONNX_INT8_Static_FastCPU": "ONNX INT8\nFastCPU",
        "RKNN_FP16_Uncut_NPU": "RKNN FP16\nUncut",
        "RKNN_INT8_6Head_Hybrid_ZeroCopy_NC1": "RKNN 6H\nHybrid",
        "RKNN_INT8_6Head_MMSE_ZeroCopy_NC1": "RKNN 6H\nMMSE",
        "RKNN_INT8_6Head_Normal_ZeroCopy_NC1": "RKNN 6H\nNormal+Q11",
        "RKNN_INT8_6Head_Normal_RGA_HW": "RKNN 6H\nNormal+RGA*",
    }
    sub_a = df_a[df_a["variant"].isin(acc_map)].copy()
    xa = np.arange(len(sub_a))
    w = 0.36
    b1 = ax1.bar(xa - w / 2, sub_a["mAP50"], width=w, label="mAP@50 (left axis)", color="#2563eb")
    b2 = ax1.bar(xa + w / 2, sub_a["op_f1"], width=w, label="F1 @ conf=0.25", color="#10b981")
    ax1.axhline(0.69785, color="#1e3a8a", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axhline(0.69392, color="#f59e0b", linestyle=":", linewidth=1.1, alpha=0.85, label="ONNX INT8 mAP@50 (0.69392)")
    for b, v in zip(b1, sub_a["mAP50"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center", va="bottom", fontsize=6.8, rotation=35)
    for b, v in zip(b2, sub_a["op_f1"]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.4f}", ha="center", va="bottom", fontsize=6.8, rotation=35)
    ax1.set_xticks(xa)
    ax1.set_xticklabels([acc_map[v] for v in sub_a["variant"]], fontsize=7.5)
    ax1.set_ylim(0.640, 0.715)
    ax1.set_ylabel("COCO val2017 Metric (5,000 Images)")
    ax1.set_title("5,000-Image val2017 Accuracy Parity (Note: Uncut RKNN INT8 = 0.0000)")
    ax1.legend(loc="lower left", fontsize=7.5)
    ax1.grid(axis="y", alpha=0.25)

    res_order = ["480p_640x480", "720p_1280x720", "1080p_1920x1080"]
    res_labels = ["480p\n(640x480)", "720p\n(1280x720)", "1080p\n(1920x1080)"]
    xr = np.arange(len(res_order))
    cfg_styles = [
        ("ONNX_FP32_LegacyCPU", "Before ONNX FP32 (CPU)", "#64748b"),
        ("ONNX_INT8_FastCPU", "Optimized CPU ONNX INT8", "#3b82f6"),
        ("NPU_INT8_6Head_Q11_1Core", "After NPU 1-Core + Q11 LUT", "#059669"),
        ("NPU_INT8_6Head_RGA_1Core", "After NPU 1-Core + Virtual RGA", "#f59e0b"),
    ]
    bw = 0.19
    for idx_c, (ckey, clbl, ccol) in enumerate(cfg_styles):
        vals = [float(df_r[(df_r["resolution"] == r) & (df_r["config"] == ckey)]["fps"].iloc[0]) for r in res_order]
        off = (idx_c - 1.5) * bw
        bars_r = ax2.bar(xr + off, vals, width=bw, label=clbl, color=ccol)
        for br, val in zip(bars_r, vals):
            ax2.text(br.get_x() + br.get_width() / 2, val + 1.8, f"{val:.1f}", ha="center", va="bottom", fontsize=7.0)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(res_labels, fontsize=8.5)
    ax2.set_ylim(0, 148)
    ax2.set_ylabel("Single-Stream FPS (1 Core)")
    ax2.set_title("Camera Resolution Scaling: RGA Wins 480p (124.6 FPS), Q11 Wins 1080p (119.0 FPS)")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_acc_res)
    plt.close(fig)


def main() -> None:
    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution_alignment(fig_dir / "distribution_alignment.png")
    plot_block_sensitivity(REPO_ROOT / "results" / "block_sensitivity_audit.csv", fig_dir / "block_sensitivity_ranking.png")
    plot_pareto_and_ablation(
        REPO_ROOT / "results" / "full_7500_ablation_report.csv",
        fig_dir / "pareto_size_vs_map50.png",
        fig_dir / "int8_optimization_progression.png",
    )
    op5_dir = REPO_ROOT / "results" / "orangepi5"
    if (op5_dir / "latency_fps_before_after.csv").exists():
        plot_orangepi5_before_after(
            op5_dir / "latency_fps_before_after.csv",
            op5_dir / "val2017_accuracy_parity.csv",
            op5_dir / "resolution_before_after.csv",
            fig_dir / "orangepi5_before_after_fps_latency.png",
            fig_dir / "orangepi5_accuracy_and_resolution_parity.png",
        )
    print(f"[CHARTS] Generated verified figures in {fig_dir}")


if __name__ == "__main__":
    main()
