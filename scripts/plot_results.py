#!/usr/bin/env python3
"""Generate visualization plots for NQS-SQD experiment results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_epoch_sweep():
    """Plot VMC/SQD energy error vs training epochs."""
    sweep_files = list((RESULTS_DIR / "epoch_sweep").glob("LiH_epoch_sweep_*.json"))
    if not sweep_files:
        print("No epoch sweep data found")
        return

    data = load_json(sweep_files[0])
    results = data["results"]
    fci = data["E_FCI"]

    epochs = [r["epochs"] for r in results]
    vmc_errors = [r["dE_VMC_mHa"] for r in results]
    sqd_errors = [r["dE_SQD_mHa"] for r in results]
    conservation = [r["conservation_ratio"] * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Energy error vs epochs
    ax1 = axes[0]
    ax1.semilogy(epochs, vmc_errors, "o-", label="VMC", color="tab:blue", markersize=8)
    ax1.semilogy(epochs, sqd_errors, "s-", label="SQD", color="tab:orange", markersize=8)
    ax1.axhline(1.6, color="gray", linestyle="--", alpha=0.5, label="Chemical accuracy")
    ax1.set_xlabel("Training Epochs", fontsize=12)
    ax1.set_ylabel("Energy Error (mHa)", fontsize=12)
    ax1.set_title("LiH (STO-3G): Error vs Training", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)

    # Right: Conservation ratio vs epochs
    ax2 = axes[1]
    ax2.plot(epochs, conservation, "^-", color="tab:green", markersize=8)
    ax2.set_xlabel("Training Epochs", fontsize=12)
    ax2.set_ylabel("Conservation Ratio (%)", fontsize=12)
    ax2.set_title("Sample Conservation vs Training", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    out_path = FIGURES_DIR / "epoch_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "epoch_sweep.pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_ablation_comparison():
    """Plot NQS vs Baseline SQD energy across sample budgets."""
    ablation_files = list((RESULTS_DIR / "ablation").glob("LiH_ablation_*.json"))
    if not ablation_files:
        print("No ablation data found")
        return

    data = load_json(ablation_files[0])
    fci = data["fci_energy"]
    budgets = data["sample_budgets"]

    nqs_stats = data["nqs_stats"]
    baseline_stats = data["baseline_stats"]

    nqs_means = [nqs_stats[str(b)]["energy_mean"] for b in budgets]
    nqs_stds = [nqs_stats[str(b)]["energy_std"] for b in budgets]
    baseline_means = [baseline_stats[str(b)]["energy_mean"] for b in budgets]
    baseline_stds = [baseline_stats[str(b)]["energy_std"] for b in budgets]

    # Convert to error in mHa
    nqs_errors = [(fci - e) * 1000 for e in nqs_means]
    baseline_errors = [(fci - e) * 1000 for e in baseline_means]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Absolute energy
    ax1 = axes[0]
    ax1.errorbar(budgets, nqs_means, yerr=nqs_stds, fmt="o-",
                 label="NQS + SQD", capsize=4, markersize=8)
    ax1.errorbar(budgets, baseline_means, yerr=baseline_stds, fmt="s-",
                 label="Baseline SQD", capsize=4, markersize=8)
    ax1.axhline(fci, color="red", linestyle="--", alpha=0.7, label=f"FCI = {fci:.4f} Ha")
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample Budget", fontsize=12)
    ax1.set_ylabel("SQD Energy (Ha)", fontsize=12)
    ax1.set_title("LiH: SQD Energy vs Sample Budget", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Energy error
    ax2 = axes[1]
    ax2.semilogy(budgets, nqs_errors, "o-", label="NQS + SQD", markersize=8)
    ax2.semilogy(budgets, [max(e, 1e-6) for e in baseline_errors], "s-",
                 label="Baseline SQD", markersize=8)
    ax2.axhline(1.6, color="gray", linestyle="--", alpha=0.5, label="Chemical accuracy")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sample Budget", fontsize=12)
    ax2.set_ylabel("Energy Error (mHa)", fontsize=12)
    ax2.set_title("LiH: Error vs Sample Budget", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "ablation_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "ablation_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_vmc_convergence():
    """Plot VMC training convergence curve."""
    nqs_files = list((RESULTS_DIR / "raw").glob("LiH_nqs_*.json"))
    if not nqs_files:
        print("No NQS raw data found")
        return

    # Use the most recent file
    data = load_json(sorted(nqs_files)[-1])
    energy_history = data["vmc_energy_history"]
    std_history = data["vmc_std_history"]
    fci = data["fci_energy"]
    hf = data["hf_energy"]

    epochs = list(range(1, len(energy_history) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, energy_history, "-", alpha=0.8, linewidth=1.5, label="VMC Energy")
    ax.fill_between(
        epochs,
        [e - s for e, s in zip(energy_history, std_history)],
        [e + s for e, s in zip(energy_history, std_history)],
        alpha=0.2
    )
    ax.axhline(fci, color="red", linestyle="--", alpha=0.7, label=f"FCI = {fci:.4f} Ha")
    ax.axhline(hf, color="green", linestyle=":", alpha=0.7, label=f"HF = {hf:.4f} Ha")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Energy (Ha)", fontsize=12)
    ax.set_title("LiH (STO-3G): VMC Training Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "vmc_convergence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "vmc_convergence.pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_scatter_vmc_vs_sqd():
    """Scatter plot: VMC error vs SQD error colored by conservation ratio."""
    sweep_files = list((RESULTS_DIR / "epoch_sweep").glob("LiH_epoch_sweep_*.json"))
    if not sweep_files:
        print("No epoch sweep data found")
        return

    data = load_json(sweep_files[0])
    results = data["results"]

    vmc_errors = [r["dE_VMC_mHa"] for r in results]
    sqd_errors = [r["dE_SQD_mHa"] for r in results]
    conservation = [r["conservation_ratio"] * 100 for r in results]
    epochs = [r["epochs"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(vmc_errors, sqd_errors, c=conservation, s=100,
                         cmap="viridis", edgecolor="black", linewidth=1)

    for i, ep in enumerate(epochs):
        ax.annotate(f"ep={ep}", (vmc_errors[i], sqd_errors[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Conservation Ratio (%)", fontsize=11)

    ax.set_xlabel("VMC Energy Error (mHa)", fontsize=12)
    ax.set_ylabel("SQD Energy Error (mHa)", fontsize=12)
    ax.set_title("LiH: VMC vs SQD Error (by Epochs)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "scatter_vmc_sqd.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "scatter_vmc_sqd.pdf", bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    print("Generating experiment plots...")
    print("-" * 40)
    plot_epoch_sweep()
    plot_ablation_comparison()
    plot_vmc_convergence()
    plot_scatter_vmc_vs_sqd()
    print("-" * 40)
    print(f"All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
