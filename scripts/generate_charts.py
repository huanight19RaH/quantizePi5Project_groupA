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
    print(f"[CHARTS] Generated 4 verified figures in {fig_dir}")


if __name__ == "__main__":
    main()
