#!/usr/bin/env python3
"""
Generate key plots from benchmark results (RESULTS.md).
Uses classic matplotlib. Run: python plot_results.py
Saves figures in the plots/ folder.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "plots"

# --- Data from RESULTS.md ---

# dummy_mouse_enhancers_ensembl
models_mouse = ["CNN", "BiLSTM", "DNABERT-1", "DNABERT-2"]
valid_roc_mouse = [0.794, 0.811, 0.797, 0.882]
test_roc_mouse = [0.751, 0.774, 0.801, 0.816]
test_acc_mouse = [0.686, 0.711, 0.694, 0.719]
time_sec_mouse = [4, 21, 157, 102]

# human_nontata_promoters
models_prom = ["Random Forest", "DNABERT-2"]
test_roc_prom = [0.980, 0.9663]


def plot_test_roc_mouse():
    """Test ROC-AUC by model (dummy_mouse_enhancers_ensembl)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models_mouse))
    bars = ax.bar(x, test_roc_mouse, color=["#6b8e9a", "#6b8e9a", "#7a9e8a", "#4a7c59"], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("dummy_mouse_enhancers_ensembl — Test ROC-AUC by model")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(test_roc_mouse):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_test_roc_mouse.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/plot_test_roc_mouse.png")


def plot_valid_vs_test_mouse():
    """Valid vs Test ROC-AUC (dummy_mouse)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models_mouse))
    w = 0.35
    ax.bar(x - w / 2, valid_roc_mouse, w, label="Valid", color="steelblue", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, test_roc_mouse, w, label="Test", color="coral", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("dummy_mouse_enhancers_ensembl — Valid vs Test ROC-AUC")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_valid_vs_test_mouse.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/plot_valid_vs_test_mouse.png")


def plot_time_mouse():
    """Training time by model (dummy_mouse)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(models_mouse))
    bars = ax.bar(x, time_sec_mouse, color=["#8ba3a8", "#8ba3a8", "#9aad8e", "#6a9a5e"], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("dummy_mouse_enhancers_ensembl — Training time by model")
    for i, v in enumerate(time_sec_mouse):
        ax.text(i, v + 3, f"{v}s", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_time_mouse.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/plot_time_mouse.png")


def plot_promoters():
    """Test ROC-AUC for human_nontata_promoters (RF vs DNABERT-2)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(models_prom))
    ax.bar(x, test_roc_prom, color=["#7a9e8a", "#4a7c59"], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_prom, rotation=15, ha="right")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("human_nontata_promoters — Test ROC-AUC")
    ax.set_ylim(0.9, 1.0)
    for i, v in enumerate(test_roc_prom):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_promoters_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/plot_promoters_roc.png")


def plot_summary_2x2():
    """Summary: 2x2 subplots in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    # (0,0) Test ROC mouse
    ax = axes[0, 0]
    x = np.arange(len(models_mouse))
    ax.bar(x, test_roc_mouse, color=["#6b8e9a", "#6b8e9a", "#7a9e8a", "#4a7c59"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Mouse enhancers — Test ROC-AUC")
    ax.set_ylim(0, 1)

    # (0,1) Valid vs Test mouse
    ax = axes[0, 1]
    w = 0.35
    ax.bar(x - w / 2, valid_roc_mouse, w, label="Valid", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, test_roc_mouse, w, label="Test", color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Mouse enhancers — Valid vs Test")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # (1,0) Time mouse
    ax = axes[1, 0]
    ax.bar(x, time_sec_mouse, color=["#8ba3a8", "#8ba3a8", "#9aad8e", "#6a9a5e"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models_mouse)
    ax.set_ylabel("Time (s)")
    ax.set_title("Mouse enhancers — Training time")

    # (1,1) Promoters
    ax = axes[1, 1]
    xp = np.arange(len(models_prom))
    ax.bar(xp, test_roc_prom, color=["#7a9e8a", "#4a7c59"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(xp)
    ax.set_xticklabels(models_prom, rotation=15, ha="right")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Human promoters — Test ROC-AUC")
    ax.set_ylim(0.9, 1.0)

    fig.suptitle("Benchmark results summary (dc-genomics-review)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/plot_summary.png")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_test_roc_mouse()
    plot_valid_vs_test_mouse()
    plot_time_mouse()
    plot_promoters()
    plot_summary_2x2()
    print("Done. Figures saved in plots/.")
