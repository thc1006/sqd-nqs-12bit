"""Ablation experiment: NQS vs Baseline sampler comparison.

This script systematically compares:
1. NQS (Neural Quantum State) sampling with VMC-trained FFNN
2. Baseline (random Bernoulli) sampling

Across different sample budgets to demonstrate sample efficiency gains.

Key metrics:
- SQD energy vs reference FCI energy
- Conservation ratio (fraction passing N_elec, S_z filter)
- Error vs sample budget (sample complexity curve)

Usage:
    # Run full ablation study
    python -m src.experiments.ablation_nqs_vs_baseline --config configs/ablation_config.yaml

    # Quick test with fewer samples
    python -m src.experiments.ablation_nqs_vs_baseline --quick

    # Specific molecule
    python -m src.experiments.ablation_nqs_vs_baseline --molecule LiH --bond-length 0.8
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from src.sqd_interface.hamiltonian import (
    MoleculeConfig,
    build_molecular_hamiltonian,
    print_molecular_info,
)
from src.sqd_interface.sqd_runner import run_sqd_on_samples, SQDConfig
from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.nqs_models.vmc_training import VMCConfig, train_nqs_vmc


@dataclass
class AblationConfig:
    """Configuration for ablation experiment."""

    # Sample budgets to test
    sample_budgets: List[int] = None

    # Number of random seeds for statistical averaging
    n_seeds: int = 5

    # VMC training epochs
    vmc_epochs: int = 100

    # NQS architecture
    nqs_alpha: int = 4  # hidden_dim = n_visible * alpha
    nqs_layers: int = 2
    nqs_activation: str = "tanh"

    # SQD configuration
    sqd_max_iterations: int = 5
    sqd_num_batches: int = 3
    sqd_samples_per_batch: int = 100

    def __post_init__(self):
        if self.sample_budgets is None:
            self.sample_budgets = [100, 500, 1000, 2500, 5000, 10000]


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_dir: pathlib.Path, prefix: str) -> pathlib.Path:
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{prefix}_{timestamp}.json"

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2)

    print(f"[INFO] Results saved to {output_path}")
    return output_path


def run_single_experiment(
    mol_data,
    n_samples: int,
    use_nqs: bool,
    seed: int,
    vmc_config: VMCConfig,
    nqs_config: FFNNNQSConfig,
    sqd_config: SQDConfig,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """Run a single NQS or baseline experiment.

    Returns
    -------
    result : dict
        Dictionary containing energy, conservation ratio, and diagnostics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_nqs:
        # Train NQS model
        model = FFNNNQS(nqs_config, device=device)

        vmc_result = train_nqs_vmc(
            model=model,
            mol_data=mol_data,
            config=vmc_config,
            device=device,
            verbose=verbose,
        )

        # Sample from trained model
        n_chains = min(256, n_samples)
        n_samples_per_chain = (n_samples + n_chains - 1) // n_chains

        samples = model.sample_mcmc(
            n_samples_per_chain=n_samples_per_chain,
            n_chains=n_chains,
            burn_in_steps=vmc_config.burn_in_steps,
            step_interval=vmc_config.step_interval,
            device=device,
        )

        vmc_energy = vmc_result.best_energy
    else:
        # Generate random Bernoulli samples
        n_spin_orb = mol_data.n_spin_orb
        samples = 2 * torch.randint(0, 2, (n_samples, n_spin_orb), device=device).float() - 1
        vmc_energy = None

    # Run SQD
    sqd_result = run_sqd_on_samples(
        mol_data=mol_data,
        samples=samples,
        config=sqd_config,
        verbose=verbose,
    )

    return {
        "sqd_energy": sqd_result.energy,
        "sqd_electronic_energy": sqd_result.electronic_energy,
        "conservation_ratio": sqd_result.conservation_ratio,
        "n_samples_used": sqd_result.n_samples_used,
        "n_samples_total": sqd_result.n_samples_total,
        "subspace_dimension": sqd_result.subspace_dimension,
        "vmc_energy": vmc_energy,
        "seed": seed,
    }


def run_ablation_study(
    mol_data,
    ablation_config: AblationConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run complete ablation study comparing NQS vs baseline.

    Returns
    -------
    results : dict
        Complete ablation results with all sample budgets and seeds.
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: NQS vs Baseline Sampling")
    print("=" * 70)
    print(f"Sample budgets: {ablation_config.sample_budgets}")
    print(f"Random seeds: {ablation_config.n_seeds}")
    print(f"FCI reference: {mol_data.fci_energy:.6f} Ha")
    print("=" * 70)

    # Set up configs
    n_visible = mol_data.n_spin_orb

    nqs_config = FFNNNQSConfig(
        n_visible=n_visible,
        n_hidden=int(n_visible * ablation_config.nqs_alpha),
        n_layers=ablation_config.nqs_layers,
        activation=ablation_config.nqs_activation,
    )

    vmc_config = VMCConfig(
        n_samples=500,
        n_chains=256,
        burn_in_steps=100,
        step_interval=10,
        learning_rate=0.01,
        sr_regularization=1e-4,
        n_epochs=ablation_config.vmc_epochs,
        patience=100,
        lr_schedule="cosine",
    )

    sqd_config = SQDConfig(
        max_iterations=ablation_config.sqd_max_iterations,
        num_batches=ablation_config.sqd_num_batches,
        samples_per_batch=ablation_config.sqd_samples_per_batch,
        symmetrize_spin=True,
    )

    # Results storage
    nqs_results = {n: [] for n in ablation_config.sample_budgets}
    baseline_results = {n: [] for n in ablation_config.sample_budgets}

    total_runs = len(ablation_config.sample_budgets) * ablation_config.n_seeds * 2
    current_run = 0

    for n_samples in ablation_config.sample_budgets:
        print(f"\n{'='*60}")
        print(f"Sample Budget: {n_samples}")
        print(f"{'='*60}")

        for seed_idx in range(ablation_config.n_seeds):
            seed = 42 + seed_idx * 1000

            # Run NQS experiment
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] NQS @ {n_samples} samples, seed={seed}")

            try:
                nqs_result = run_single_experiment(
                    mol_data=mol_data,
                    n_samples=n_samples,
                    use_nqs=True,
                    seed=seed,
                    vmc_config=vmc_config,
                    nqs_config=nqs_config,
                    sqd_config=sqd_config,
                    device=device,
                    verbose=verbose,
                )
                nqs_results[n_samples].append(nqs_result)
                error_mha = abs(nqs_result["sqd_energy"] - mol_data.fci_energy) * 1000
                print(f"    NQS Energy: {nqs_result['sqd_energy']:.6f} Ha (error: {error_mha:.3f} mHa)")
                print(f"    Conservation: {nqs_result['conservation_ratio']*100:.2f}%")
            except Exception as e:
                print(f"    NQS FAILED: {e}")
                nqs_results[n_samples].append({"error": str(e), "seed": seed})

            # Run baseline experiment
            current_run += 1
            print(f"[{current_run}/{total_runs}] Baseline @ {n_samples} samples, seed={seed}")

            try:
                baseline_result = run_single_experiment(
                    mol_data=mol_data,
                    n_samples=n_samples,
                    use_nqs=False,
                    seed=seed,
                    vmc_config=vmc_config,
                    nqs_config=nqs_config,
                    sqd_config=sqd_config,
                    device=device,
                    verbose=verbose,
                )
                baseline_results[n_samples].append(baseline_result)
                error_mha = abs(baseline_result["sqd_energy"] - mol_data.fci_energy) * 1000
                print(f"    Baseline Energy: {baseline_result['sqd_energy']:.6f} Ha (error: {error_mha:.3f} mHa)")
                print(f"    Conservation: {baseline_result['conservation_ratio']*100:.2f}%")
            except Exception as e:
                print(f"    Baseline FAILED: {e}")
                baseline_results[n_samples].append({"error": str(e), "seed": seed})

    # Compute statistics
    def compute_stats(results_list):
        valid = [r for r in results_list if "error" not in r]
        if not valid:
            return {"mean": float("nan"), "std": float("nan"), "n_valid": 0}

        energies = [r["sqd_energy"] for r in valid]
        conservation = [r["conservation_ratio"] for r in valid]

        return {
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "conservation_mean": float(np.mean(conservation)),
            "conservation_std": float(np.std(conservation)),
            "n_valid": len(valid),
        }

    nqs_stats = {n: compute_stats(nqs_results[n]) for n in ablation_config.sample_budgets}
    baseline_stats = {n: compute_stats(baseline_results[n]) for n in ablation_config.sample_budgets}

    return {
        "nqs_results": nqs_results,
        "baseline_results": baseline_results,
        "nqs_stats": nqs_stats,
        "baseline_stats": baseline_stats,
        "sample_budgets": ablation_config.sample_budgets,
        "n_seeds": ablation_config.n_seeds,
        "fci_energy": mol_data.fci_energy,
        "hf_energy": mol_data.hf_energy,
    }


def print_summary_table(results: dict) -> None:
    """Print a summary table of ablation results."""
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Samples':>10} | {'NQS Energy (Ha)':>20} | {'Baseline Energy (Ha)':>20} | {'NQS Cons.':>10} | {'Baseline Cons.':>12}")
    print("-" * 80)

    fci = results["fci_energy"]

    for n in results["sample_budgets"]:
        nqs = results["nqs_stats"][n]
        base = results["baseline_stats"][n]

        nqs_str = f"{nqs['energy_mean']:.6f} +/- {nqs['energy_std']:.4f}" if nqs["n_valid"] > 0 else "FAILED"
        base_str = f"{base['energy_mean']:.6f} +/- {base['energy_std']:.4f}" if base["n_valid"] > 0 else "FAILED"
        nqs_cons = f"{nqs['conservation_mean']*100:.1f}%" if nqs["n_valid"] > 0 else "N/A"
        base_cons = f"{base['conservation_mean']*100:.1f}%" if base["n_valid"] > 0 else "N/A"

        print(f"{n:>10} | {nqs_str:>20} | {base_str:>20} | {nqs_cons:>10} | {base_cons:>12}")

    print("-" * 80)
    print(f"FCI Reference: {fci:.6f} Ha")
    print("=" * 80)

    # Error comparison
    print("\n" + "=" * 80)
    print("ERROR COMPARISON (mHa from FCI)")
    print("=" * 80)
    print(f"{'Samples':>10} | {'NQS Error':>15} | {'Baseline Error':>15} | {'Improvement':>15}")
    print("-" * 80)

    for n in results["sample_budgets"]:
        nqs = results["nqs_stats"][n]
        base = results["baseline_stats"][n]

        if nqs["n_valid"] > 0 and base["n_valid"] > 0:
            nqs_err = abs(nqs["energy_mean"] - fci) * 1000
            base_err = abs(base["energy_mean"] - fci) * 1000
            improvement = base_err - nqs_err
            print(f"{n:>10} | {nqs_err:>12.3f} mHa | {base_err:>12.3f} mHa | {improvement:>12.3f} mHa")
        else:
            print(f"{n:>10} | {'N/A':>15} | {'N/A':>15} | {'N/A':>15}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study: NQS vs Baseline sampling for SQD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="H2",
        choices=["H2", "LiH", "H4", "H6"],
        help="Molecule to study (default: H2).",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=None,
        help="Bond length in Angstrom.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set (default: sto-3g).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with fewer samples and seeds.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of random seeds for averaging.",
    )
    parser.add_argument(
        "--vmc-epochs",
        type=int,
        default=100,
        help="Number of VMC training epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during experiments.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load config if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        print(f"[INFO] Loaded config from {args.config}")

    # Build molecule
    molecule_name = cfg.get("molecule", {}).get("name", args.molecule)
    bond_length = cfg.get("molecule", {}).get("bond_length", args.bond_length)
    basis = cfg.get("molecule", {}).get("basis", args.basis)

    # Set default bond lengths
    if bond_length is None:
        bond_length = {"H2": 0.74, "LiH": 0.8, "H4": 1.0, "H6": 1.0}.get(molecule_name, 1.0)

    mol_config = MoleculeConfig(
        name=molecule_name,
        bond_length=bond_length,
        basis=basis,
    )
    mol_data = build_molecular_hamiltonian(mol_config)
    print_molecular_info(mol_data)

    # Ablation configuration
    if args.quick:
        # Quick test mode
        ablation_config = AblationConfig(
            sample_budgets=[100, 500, 1000],
            n_seeds=2,
            vmc_epochs=50,
        )
        print("[INFO] Quick test mode enabled")
    else:
        ablation_cfg = cfg.get("ablation", {})
        ablation_config = AblationConfig(
            sample_budgets=ablation_cfg.get("sample_budgets", [100, 500, 1000, 2500, 5000, 10000]),
            n_seeds=ablation_cfg.get("n_seeds", args.n_seeds),
            vmc_epochs=ablation_cfg.get("vmc_epochs", args.vmc_epochs),
            nqs_alpha=ablation_cfg.get("nqs_alpha", 4),
            nqs_layers=ablation_cfg.get("nqs_layers", 2),
            sqd_max_iterations=ablation_cfg.get("sqd_max_iterations", 5),
            sqd_num_batches=ablation_cfg.get("sqd_num_batches", 3),
            sqd_samples_per_batch=ablation_cfg.get("sqd_samples_per_batch", 100),
        )

    # Run ablation study
    results = run_ablation_study(
        mol_data=mol_data,
        ablation_config=ablation_config,
        device=device,
        verbose=args.verbose,
    )

    # Add metadata
    results["molecule"] = molecule_name
    results["bond_length"] = bond_length
    results["basis"] = basis
    results["n_spin_orb"] = mol_data.n_spin_orb
    results["n_elec"] = mol_data.n_elec
    results["ablation_config"] = {
        "sample_budgets": ablation_config.sample_budgets,
        "n_seeds": ablation_config.n_seeds,
        "vmc_epochs": ablation_config.vmc_epochs,
        "nqs_alpha": ablation_config.nqs_alpha,
        "nqs_layers": ablation_config.nqs_layers,
    }

    # Print summary
    print_summary_table(results)

    # Save results
    output_dir = pathlib.Path(args.output_dir)
    save_results(results, output_dir, f"{molecule_name}_ablation")


if __name__ == "__main__":
    main()
