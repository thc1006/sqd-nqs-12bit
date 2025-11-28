#!/usr/bin/env python3
"""Generate Phase Diagram visualizations from experiment results.

Creates:
1. VMC Error vs SQD Error scatter plots (by epochs)
2. Sample Efficiency curves (NQS vs Baseline)
3. Phase Diagram contours (x=VMC_error, y=samples, color=SQD_error)
4. H-chain scaling comparison

Usage:
    python scripts/generate_phase_diagram.py --input results/phase_diagram/
    python scripts/generate_phase_diagram.py --input results/phase_diagram/all_results_*.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata


def load_results(input_path: pathlib.Path) -> pd.DataFrame:
    """Load experiment results from JSON files."""
    if input_path.is_dir():
        # Load all JSON files in directory
        files = list(input_path.glob("*.json"))
    else:
        files = [input_path]

    all_data = []
    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    # Filter out error entries
    valid_data = [d for d in all_data if "error" not in d]
    return pd.DataFrame(valid_data)


def plot_vmc_vs_sqd_scatter(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot VMC error vs SQD error scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter NQS results only
    nqs_df = df[df["method"] == "nqs"].copy()
    if nqs_df.empty:
        print("[WARN] No NQS results for VMC vs SQD scatter")
        return

    # Group by epochs
    epochs = sorted(nqs_df["vmc_epochs"].unique())
    colors = cm.viridis(np.linspace(0, 1, len(epochs)))

    for epoch, color in zip(epochs, colors):
        subset = nqs_df[nqs_df["vmc_epochs"] == epoch]
        ax.scatter(
            subset["vmc_error_mha"],
            subset["sqd_error_mha"],
            c=[color],
            label=f"epochs={epoch}",
            s=80,
            alpha=0.7,
            edgecolors="black",
        )

    ax.set_xlabel("VMC Energy Error (mHa)", fontsize=12)
    ax.set_ylabel("SQD Energy Error (mHa)", fontsize=12)
    ax.set_title("VMC vs SQD Error by Training Epochs", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add reference lines
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa (12-bit)")
    ax.axhline(1.6, color="orange", linestyle="--", alpha=0.5, label="1.6 mHa (chem. acc.)")

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "vmc_vs_sqd_scatter.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved vmc_vs_sqd_scatter.png")


def plot_sample_efficiency(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot sample efficiency curves: NQS vs Baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by molecule
    molecules = df["molecule"].unique()

    for mol in molecules:
        mol_df = df[df["molecule"] == mol]

        # NQS results
        nqs_df = mol_df[mol_df["method"] == "nqs"]
        baseline_df = mol_df[mol_df["method"] == "baseline"]

        if nqs_df.empty and baseline_df.empty:
            continue

        # Left plot: SQD error vs samples
        ax = axes[0]
        if not nqs_df.empty:
            nqs_grouped = nqs_df.groupby("n_samples")["sqd_error_mha"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                nqs_grouped["n_samples"],
                nqs_grouped["mean"],
                yerr=nqs_grouped["std"],
                marker="o",
                label=f"{mol} NQS",
                capsize=3,
            )

        if not baseline_df.empty:
            base_grouped = baseline_df.groupby("n_samples")["sqd_error_mha"].agg(["mean", "std"]).reset_index()
            ax.errorbar(
                base_grouped["n_samples"],
                base_grouped["mean"],
                yerr=base_grouped["std"],
                marker="s",
                linestyle="--",
                label=f"{mol} Baseline",
                capsize=3,
            )

        # Right plot: Conservation ratio vs samples
        ax2 = axes[1]
        if not nqs_df.empty:
            nqs_cons = nqs_df.groupby("n_samples")["conservation_ratio"].agg(["mean", "std"]).reset_index()
            ax2.errorbar(
                nqs_cons["n_samples"],
                nqs_cons["mean"] * 100,
                yerr=nqs_cons["std"] * 100,
                marker="o",
                label=f"{mol} NQS",
                capsize=3,
            )

        if not baseline_df.empty:
            base_cons = baseline_df.groupby("n_samples")["conservation_ratio"].agg(["mean", "std"]).reset_index()
            ax2.errorbar(
                base_cons["n_samples"],
                base_cons["mean"] * 100,
                yerr=base_cons["std"] * 100,
                marker="s",
                linestyle="--",
                label=f"{mol} Baseline",
                capsize=3,
            )

    # Format left plot
    axes[0].set_xlabel("Number of Samples", fontsize=12)
    axes[0].set_ylabel("SQD Energy Error (mHa)", fontsize=12)
    axes[0].set_title("Sample Efficiency: SQD Error", fontsize=14)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Format right plot
    axes[1].set_xlabel("Number of Samples", fontsize=12)
    axes[1].set_ylabel("Conservation Ratio (%)", fontsize=12)
    axes[1].set_title("Conservation Ratio vs Samples", fontsize=14)
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sample_efficiency.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved sample_efficiency.png")


def plot_phase_diagram_contour(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot phase diagram: x=VMC_error, y=samples, color=SQD_error."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter NQS results with valid VMC errors
    nqs_df = df[(df["method"] == "nqs") & (df["vmc_error_mha"].notna())].copy()
    if len(nqs_df) < 10:
        print("[WARN] Not enough data for phase diagram contour")
        return

    # Get data points
    x = nqs_df["vmc_error_mha"].values
    y = nqs_df["n_samples"].values
    z = nqs_df["sqd_error_mha"].values

    # Create grid for interpolation
    xi = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
    yi = np.logspace(np.log10(y.min()), np.log10(y.max()), 50)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate (using log scale)
    try:
        Zi = griddata(
            (np.log10(x), np.log10(y)),
            np.log10(z + 1e-10),
            (np.log10(Xi), np.log10(Yi)),
            method="linear",
        )
    except Exception as e:
        print(f"[WARN] Interpolation failed: {e}")
        return

    # Plot contour
    levels = np.logspace(-4, 2, 20)
    cs = ax.contourf(Xi, Yi, 10**Zi, levels=levels, cmap="RdYlGn_r", norm=plt.matplotlib.colors.LogNorm())
    cbar = plt.colorbar(cs, ax=ax, label="SQD Error (mHa)")

    # Add contour lines for key thresholds
    ax.contour(Xi, Yi, 10**Zi, levels=[0.1, 1.0, 1.6], colors=["green", "red", "orange"], linewidths=2)

    # Scatter original points
    scatter = ax.scatter(x, y, c=z, cmap="RdYlGn_r", norm=plt.matplotlib.colors.LogNorm(vmin=1e-4, vmax=100),
                         edgecolors="black", s=60, zorder=5)

    ax.set_xlabel("VMC Energy Error (mHa)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("SQD Phase Diagram: VMC Quality vs Sample Budget", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add annotations
    ax.text(0.95, 0.95, "1 mHa (12-bit)", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", color="red")
    ax.text(0.95, 0.90, "1.6 mHa (chem. acc.)", transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right", color="orange")

    plt.tight_layout()
    plt.savefig(output_dir / "phase_diagram_contour.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved phase_diagram_contour.png")


def plot_hchain_scaling(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot H-chain scaling comparison."""
    # Filter H-chain molecules
    hchain_df = df[df["molecule"].str.match(r"H\d+")].copy()
    if hchain_df.empty:
        print("[WARN] No H-chain data for scaling plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SQD error vs system size
    ax = axes[0]
    for method in ["nqs", "baseline"]:
        method_df = hchain_df[hchain_df["method"] == method]
        if method_df.empty:
            continue

        # Extract chain length from molecule name
        method_df = method_df.copy()
        method_df["chain_length"] = method_df["molecule"].str.extract(r"H(\d+)").astype(int)

        grouped = method_df.groupby("chain_length")["sqd_error_mha"].agg(["mean", "std"]).reset_index()

        marker = "o" if method == "nqs" else "s"
        linestyle = "-" if method == "nqs" else "--"
        ax.errorbar(
            grouped["chain_length"] * 2,  # spin orbitals
            grouped["mean"],
            yerr=grouped["std"],
            marker=marker,
            linestyle=linestyle,
            label=method.upper(),
            capsize=3,
        )

    ax.set_xlabel("Number of Spin Orbitals", fontsize=12)
    ax.set_ylabel("SQD Energy Error (mHa)", fontsize=12)
    ax.set_title("H-chain Scaling: SQD Error vs System Size", fontsize=14)
    ax.set_yscale("log")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Conservation ratio vs system size
    ax2 = axes[1]
    for method in ["nqs", "baseline"]:
        method_df = hchain_df[hchain_df["method"] == method]
        if method_df.empty:
            continue

        method_df = method_df.copy()
        method_df["chain_length"] = method_df["molecule"].str.extract(r"H(\d+)").astype(int)

        grouped = method_df.groupby("chain_length")["conservation_ratio"].agg(["mean", "std"]).reset_index()

        marker = "o" if method == "nqs" else "s"
        linestyle = "-" if method == "nqs" else "--"
        ax2.errorbar(
            grouped["chain_length"] * 2,
            grouped["mean"] * 100,
            yerr=grouped["std"] * 100,
            marker=marker,
            linestyle=linestyle,
            label=method.upper(),
            capsize=3,
        )

    ax2.set_xlabel("Number of Spin Orbitals", fontsize=12)
    ax2.set_ylabel("Conservation Ratio (%)", fontsize=12)
    ax2.set_title("H-chain Scaling: Conservation Ratio", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hchain_scaling.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved hchain_scaling.png")


def plot_bond_length_scan(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Plot LiH bond length scan results."""
    lih_df = df[df["molecule"] == "LiH"].copy()
    if lih_df.empty:
        print("[WARN] No LiH data for bond length scan")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SQD error vs bond length
    ax = axes[0]
    for method in ["nqs", "baseline"]:
        method_df = lih_df[lih_df["method"] == method]
        if method_df.empty:
            continue

        grouped = method_df.groupby("bond_length")["sqd_error_mha"].agg(["mean", "std"]).reset_index()

        marker = "o" if method == "nqs" else "s"
        linestyle = "-" if method == "nqs" else "--"
        ax.errorbar(
            grouped["bond_length"],
            grouped["mean"],
            yerr=grouped["std"],
            marker=marker,
            linestyle=linestyle,
            label=method.upper(),
            capsize=3,
        )

    ax.set_xlabel("Bond Length (A)", fontsize=12)
    ax.set_ylabel("SQD Energy Error (mHa)", fontsize=12)
    ax.set_title("LiH: SQD Error vs Bond Length", fontsize=14)
    ax.set_yscale("log")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: VMC error vs bond length (NQS only)
    ax2 = axes[1]
    nqs_df = lih_df[lih_df["method"] == "nqs"]
    if not nqs_df.empty:
        grouped = nqs_df.groupby("bond_length")["vmc_error_mha"].agg(["mean", "std"]).reset_index()
        ax2.errorbar(
            grouped["bond_length"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=3,
            color="blue",
        )

    ax2.set_xlabel("Bond Length (A)", fontsize=12)
    ax2.set_ylabel("VMC Energy Error (mHa)", fontsize=12)
    ax2.set_title("LiH: VMC Error vs Bond Length (NQS)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "lih_bond_scan.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved lih_bond_scan.png")


def generate_summary_table(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Generate summary statistics table."""
    summary_rows = []

    for molecule in df["molecule"].unique():
        mol_df = df[df["molecule"] == molecule]

        for method in ["nqs", "baseline"]:
            method_df = mol_df[mol_df["method"] == method]
            if method_df.empty:
                continue

            row = {
                "molecule": molecule,
                "method": method,
                "n_experiments": len(method_df),
                "sqd_error_mean_mha": method_df["sqd_error_mha"].mean(),
                "sqd_error_std_mha": method_df["sqd_error_mha"].std(),
                "sqd_error_min_mha": method_df["sqd_error_mha"].min(),
                "conservation_ratio_mean": method_df["conservation_ratio"].mean(),
                "conservation_ratio_std": method_df["conservation_ratio"].std(),
            }

            if method == "nqs" and "vmc_error_mha" in method_df.columns:
                vmc_errors = method_df["vmc_error_mha"].dropna()
                if len(vmc_errors) > 0:
                    row["vmc_error_mean_mha"] = vmc_errors.mean()
                    row["vmc_error_std_mha"] = vmc_errors.std()

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)
    print(f"[INFO] Saved experiment_summary.csv")

    # Print to console
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase Diagram visualizations")
    parser.add_argument("--input", type=str, default="results/phase_diagram", help="Input directory or file")
    parser.add_argument("--output", type=str, default="results/figures", help="Output directory")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading results from {input_path}")
    df = load_results(input_path)
    print(f"[INFO] Loaded {len(df)} experiment records")

    if df.empty:
        print("[ERROR] No valid experiment data found")
        return

    # Generate all plots
    plot_vmc_vs_sqd_scatter(df, output_dir)
    plot_sample_efficiency(df, output_dir)
    plot_phase_diagram_contour(df, output_dir)
    plot_hchain_scaling(df, output_dir)
    plot_bond_length_scan(df, output_dir)
    generate_summary_table(df, output_dir)

    print(f"\n[INFO] All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
