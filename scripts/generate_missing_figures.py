#!/usr/bin/env python3
"""Generate missing figures for experiment report.

Creates:
1. conservation_distributions.png - Histogram of Conservation Ratio by method
2. training_analysis.png - Training epochs effect on SQD error
3. error_distributions.png - SQD error distribution histograms
4. heatmap_analysis.png - Multi-dimensional heatmap analysis
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_all_results(results_dir: pathlib.Path) -> pd.DataFrame:
    """Load all experiment results from JSON files."""
    all_data = []

    for json_file in results_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    valid_data = [d for d in all_data if "error" not in d]
    return pd.DataFrame(valid_data)


def plot_conservation_distributions(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot Conservation Ratio distributions by method and molecule."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Overall distribution by method
    ax = axes[0, 0]
    for method, color in [("nqs", "steelblue"), ("baseline", "coral")]:
        method_df = df[df["method"] == method]
        if not method_df.empty:
            ax.hist(
                method_df["conservation_ratio"] * 100,
                bins=30,
                alpha=0.6,
                label=method.upper(),
                color=color,
                edgecolor="black",
            )
    ax.set_xlabel("Conservation Ratio (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Overall Conservation Ratio Distribution", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By molecule - NQS
    ax = axes[0, 1]
    nqs_df = df[df["method"] == "nqs"]
    for mol in sorted(nqs_df["molecule"].unique()):
        mol_data = nqs_df[nqs_df["molecule"] == mol]["conservation_ratio"] * 100
        ax.hist(mol_data, bins=20, alpha=0.5, label=mol, edgecolor="black")
    ax.set_xlabel("Conservation Ratio (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("NQS Conservation Ratio by Molecule", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # By molecule - Baseline
    ax = axes[1, 0]
    baseline_df = df[df["method"] == "baseline"]
    for mol in sorted(baseline_df["molecule"].unique()):
        mol_data = baseline_df[baseline_df["molecule"] == mol]["conservation_ratio"] * 100
        ax.hist(mol_data, bins=20, alpha=0.5, label=mol, edgecolor="black")
    ax.set_xlabel("Conservation Ratio (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Baseline Conservation Ratio by Molecule", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[1, 1]
    plot_data = []
    for mol in sorted(df["molecule"].unique()):
        for method in ["nqs", "baseline"]:
            subset = df[(df["molecule"] == mol) & (df["method"] == method)]
            if not subset.empty:
                for val in subset["conservation_ratio"] * 100:
                    plot_data.append({"Molecule": mol, "Method": method.upper(), "Conservation (%)": val})

    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x="Molecule", y="Conservation (%)", hue="Method", ax=ax)
        ax.set_title("Conservation Ratio Comparison", fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "conservation_distributions.png", dpi=150)
    plt.close()
    print("[INFO] Saved conservation_distributions.png")


def plot_training_analysis(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot training epochs effect on SQD error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    nqs_df = df[df["method"] == "nqs"].copy()
    if nqs_df.empty:
        print("[WARN] No NQS data for training analysis")
        return

    # 1. SQD error vs epochs (scatter with trend)
    ax = axes[0, 0]
    epochs = sorted(nqs_df["vmc_epochs"].unique())

    means = []
    stds = []
    for ep in epochs:
        ep_data = nqs_df[nqs_df["vmc_epochs"] == ep]["sqd_error_mha"]
        means.append(ep_data.mean())
        stds.append(ep_data.std())

    ax.errorbar(epochs, means, yerr=stds, marker="o", capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel("SQD Error (mHa)", fontsize=12)
    ax.set_title("SQD Error vs Training Epochs", fontsize=14)
    ax.set_yscale("log")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa")
    ax.axhline(1.6, color="orange", linestyle="--", alpha=0.5, label="1.6 mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. VMC error vs epochs
    ax = axes[0, 1]
    vmc_means = []
    vmc_stds = []
    for ep in epochs:
        ep_data = nqs_df[nqs_df["vmc_epochs"] == ep]["vmc_error_mha"].dropna()
        if len(ep_data) > 0:
            vmc_means.append(ep_data.mean())
            vmc_stds.append(ep_data.std())
        else:
            vmc_means.append(np.nan)
            vmc_stds.append(np.nan)

    ax.errorbar(epochs, vmc_means, yerr=vmc_stds, marker="s", capsize=5, linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel("VMC Error (mHa)", fontsize=12)
    ax.set_title("VMC Error vs Training Epochs", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Correlation: VMC error vs SQD error
    ax = axes[1, 0]
    valid_data = nqs_df[nqs_df["vmc_error_mha"].notna() & nqs_df["sqd_error_mha"].notna()]

    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    for ep, color in zip(epochs, colors):
        ep_data = valid_data[valid_data["vmc_epochs"] == ep]
        ax.scatter(
            ep_data["vmc_error_mha"],
            ep_data["sqd_error_mha"],
            c=[color],
            label=f"ep={ep}",
            s=60,
            alpha=0.7,
            edgecolors="black",
        )

    ax.set_xlabel("VMC Error (mHa)", fontsize=12)
    ax.set_ylabel("SQD Error (mHa)", fontsize=12)
    ax.set_title("VMC vs SQD Error (Negative Correlation)", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Conservation ratio vs epochs
    ax = axes[1, 1]
    cons_means = []
    cons_stds = []
    for ep in epochs:
        ep_data = nqs_df[nqs_df["vmc_epochs"] == ep]["conservation_ratio"] * 100
        cons_means.append(ep_data.mean())
        cons_stds.append(ep_data.std())

    ax.errorbar(epochs, cons_means, yerr=cons_stds, marker="^", capsize=5, linewidth=2, markersize=8, color="purple")
    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel("Conservation Ratio (%)", fontsize=12)
    ax.set_title("Conservation Ratio vs Training Epochs", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_analysis.png", dpi=150)
    plt.close()
    print("[INFO] Saved training_analysis.png")


def plot_error_distributions(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot SQD error distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Overall SQD error distribution (log scale)
    ax = axes[0, 0]
    for method, color in [("nqs", "steelblue"), ("baseline", "coral")]:
        method_df = df[df["method"] == method]
        errors = method_df["sqd_error_mha"]
        errors = errors[errors > 1e-10]  # Filter near-zero
        if len(errors) > 0:
            ax.hist(
                np.log10(errors + 1e-12),
                bins=30,
                alpha=0.6,
                label=method.upper(),
                color=color,
                edgecolor="black",
            )
    ax.set_xlabel("log10(SQD Error / mHa)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SQD Error Distribution (Log Scale)", fontsize=14)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="1 mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. NQS error by molecule
    ax = axes[0, 1]
    nqs_df = df[df["method"] == "nqs"]
    for mol in sorted(nqs_df["molecule"].unique()):
        errors = nqs_df[nqs_df["molecule"] == mol]["sqd_error_mha"]
        errors = errors[errors > 1e-10]
        if len(errors) > 0:
            ax.hist(np.log10(errors + 1e-12), bins=20, alpha=0.5, label=mol, edgecolor="black")
    ax.set_xlabel("log10(SQD Error / mHa)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("NQS SQD Error by Molecule", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Baseline error by molecule
    ax = axes[1, 0]
    baseline_df = df[df["method"] == "baseline"]
    for mol in sorted(baseline_df["molecule"].unique()):
        errors = baseline_df[baseline_df["molecule"] == mol]["sqd_error_mha"]
        errors = errors[errors > 1e-10]
        if len(errors) > 0:
            ax.hist(np.log10(errors + 1e-12), bins=20, alpha=0.5, label=mol, edgecolor="black")
        else:
            # All near-zero
            ax.axvline(-9, linestyle="--", alpha=0.5, label=f"{mol} (< 1e-9)")
    ax.set_xlabel("log10(SQD Error / mHa)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Baseline SQD Error by Molecule", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cumulative distribution
    ax = axes[1, 1]
    for method, color in [("nqs", "steelblue"), ("baseline", "coral")]:
        method_df = df[df["method"] == method]
        errors = np.sort(method_df["sqd_error_mha"].values)
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=method.upper(), color=color, linewidth=2)

    ax.set_xlabel("SQD Error (mHa)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("Cumulative Distribution of SQD Error", fontsize=14)
    ax.set_xscale("log")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa")
    ax.axvline(1.6, color="orange", linestyle="--", alpha=0.5, label="1.6 mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", dpi=150)
    plt.close()
    print("[INFO] Saved error_distributions.png")


def plot_heatmap_analysis(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot multi-dimensional heatmap analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    nqs_df = df[df["method"] == "nqs"].copy()

    # 1. Heatmap: epochs vs samples -> SQD error (LiH)
    ax = axes[0, 0]
    lih_nqs = nqs_df[nqs_df["molecule"] == "LiH"]
    if not lih_nqs.empty:
        pivot = lih_nqs.pivot_table(
            values="sqd_error_mha",
            index="vmc_epochs",
            columns="n_samples",
            aggfunc="mean",
        )
        if not pivot.empty:
            sns.heatmap(
                np.log10(pivot + 1e-12),
                annot=True,
                fmt=".1f",
                cmap="RdYlGn_r",
                ax=ax,
                cbar_kws={"label": "log10(SQD Error)"},
            )
            ax.set_title("LiH: log10(SQD Error) - Epochs vs Samples", fontsize=14)
            ax.set_xlabel("Number of Samples", fontsize=12)
            ax.set_ylabel("Training Epochs", fontsize=12)

    # 2. Heatmap: epochs vs samples -> Conservation ratio (LiH)
    ax = axes[0, 1]
    if not lih_nqs.empty:
        pivot = lih_nqs.pivot_table(
            values="conservation_ratio",
            index="vmc_epochs",
            columns="n_samples",
            aggfunc="mean",
        )
        if not pivot.empty:
            sns.heatmap(
                pivot * 100,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=ax,
                cbar_kws={"label": "Conservation (%)"},
            )
            ax.set_title("LiH: Conservation Ratio (%) - Epochs vs Samples", fontsize=14)
            ax.set_xlabel("Number of Samples", fontsize=12)
            ax.set_ylabel("Training Epochs", fontsize=12)

    # 3. Heatmap: molecule vs method -> mean SQD error
    ax = axes[1, 0]
    pivot = df.pivot_table(
        values="sqd_error_mha",
        index="molecule",
        columns="method",
        aggfunc="mean",
    )
    if not pivot.empty:
        sns.heatmap(
            np.log10(pivot + 1e-12),
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": "log10(SQD Error)"},
        )
        ax.set_title("Mean log10(SQD Error) by Molecule and Method", fontsize=14)
        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Molecule", fontsize=12)

    # 4. Heatmap: molecule vs method -> mean Conservation ratio
    ax = axes[1, 1]
    pivot = df.pivot_table(
        values="conservation_ratio",
        index="molecule",
        columns="method",
        aggfunc="mean",
    )
    if not pivot.empty:
        sns.heatmap(
            pivot * 100,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Conservation (%)"},
        )
        ax.set_title("Mean Conservation Ratio (%) by Molecule and Method", fontsize=14)
        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Molecule", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_analysis.png", dpi=150)
    plt.close()
    print("[INFO] Saved heatmap_analysis.png")


def main():
    results_dir = pathlib.Path("results/phase_diagram")
    output_dir = pathlib.Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading results...")
    df = load_all_results(results_dir)
    print(f"[INFO] Loaded {len(df)} experiment records")

    if df.empty:
        print("[ERROR] No data found")
        return

    print(f"[INFO] Molecules: {df['molecule'].unique()}")
    print(f"[INFO] Methods: {df['method'].unique()}")

    # Generate missing figures
    plot_conservation_distributions(df, output_dir)
    plot_training_analysis(df, output_dir)
    plot_error_distributions(df, output_dir)
    plot_heatmap_analysis(df, output_dir)

    print(f"\n[INFO] All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
