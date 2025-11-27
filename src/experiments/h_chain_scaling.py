"""H-chain scaling experiments (H4, H6, H8, ...).

This script studies how NQS-SQD performance scales with system size
using linear hydrogen chains as a model system.

Key questions:
- How does sample efficiency scale with number of qubits?
- How does conservation ratio change with system size?
- What is the computational cost scaling?

Usage:
    # Run H4 (8-bit) experiment
    python -m src.experiments.h_chain_scaling --chain-length 4

    # Run scaling study from H2 to H8
    python -m src.experiments.h_chain_scaling --scaling

    # Quick test
    python -m src.experiments.h_chain_scaling --quick
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
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
class HChainConfig:
    """Configuration for H-chain experiment."""

    chain_lengths: List[int] = None  # e.g., [2, 4, 6, 8]
    bond_length: float = 1.0  # H-H distance in Angstrom
    basis: str = "sto-3g"
    n_samples: int = 5000
    n_seeds: int = 3
    vmc_epochs: int = 100
    nqs_alpha: int = 4

    def __post_init__(self):
        if self.chain_lengths is None:
            self.chain_lengths = [2, 4, 6]


def save_results(results: dict, output_dir: pathlib.Path, prefix: str) -> pathlib.Path:
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{prefix}_{timestamp}.json"

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


def run_single_chain_experiment(
    chain_length: int,
    bond_length: float,
    basis: str,
    n_samples: int,
    use_nqs: bool,
    seed: int,
    vmc_epochs: int,
    nqs_alpha: int,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """Run experiment for a single H-chain configuration.

    Parameters
    ----------
    chain_length : int
        Number of hydrogen atoms (2, 4, 6, 8, ...).
    bond_length : float
        H-H bond distance in Angstrom.
    basis : str
        Basis set name.
    n_samples : int
        Number of samples for SQD.
    use_nqs : bool
        Whether to use NQS or baseline sampler.
    seed : int
        Random seed.
    vmc_epochs : int
        VMC training epochs.
    nqs_alpha : int
        NQS hidden dimension multiplier.
    device : torch.device
        Compute device.
    verbose : bool
        Verbose output.

    Returns
    -------
    result : dict
        Experiment results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build H-chain molecule name
    mol_name = f"H{chain_length}"

    mol_config = MoleculeConfig(
        name=mol_name,
        bond_length=bond_length,
        basis=basis,
    )

    try:
        mol_data = build_molecular_hamiltonian(mol_config)
    except Exception as e:
        print(f"[ERROR] Failed to build {mol_name}: {e}")
        return {"error": str(e), "chain_length": chain_length, "seed": seed}

    n_spin_orb = mol_data.n_spin_orb

    # Configure NQS
    nqs_config = FFNNNQSConfig(
        n_visible=n_spin_orb,
        n_hidden=int(n_spin_orb * nqs_alpha),
        n_layers=2 if n_spin_orb <= 8 else 3,
        activation="tanh",
    )

    # Configure VMC (scale with system size)
    vmc_config = VMCConfig(
        n_samples=min(1000, 100 * n_spin_orb),
        n_chains=256,
        burn_in_steps=100 + 20 * n_spin_orb,
        step_interval=10 + n_spin_orb,
        learning_rate=0.01 / np.sqrt(n_spin_orb / 4),
        sr_regularization=1e-4,
        n_epochs=vmc_epochs,
        patience=100,
        lr_schedule="cosine",
    )

    # Configure SQD
    sqd_config = SQDConfig(
        max_iterations=5,
        num_batches=3,
        samples_per_batch=min(200, n_samples // 10),
        symmetrize_spin=True,
    )

    if use_nqs:
        # Train NQS
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
        # Baseline sampling
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
        "chain_length": chain_length,
        "n_spin_orb": n_spin_orb,
        "n_elec": mol_data.n_elec,
        "sqd_energy": sqd_result.energy,
        "sqd_electronic_energy": sqd_result.electronic_energy,
        "conservation_ratio": sqd_result.conservation_ratio,
        "n_samples_used": sqd_result.n_samples_used,
        "n_samples_total": sqd_result.n_samples_total,
        "subspace_dimension": sqd_result.subspace_dimension,
        "fci_energy": mol_data.fci_energy,
        "hf_energy": mol_data.hf_energy,
        "vmc_energy": vmc_energy,
        "seed": seed,
        "use_nqs": use_nqs,
    }


def run_scaling_study(
    config: HChainConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run complete H-chain scaling study.

    Returns
    -------
    results : dict
        Scaling study results across all chain lengths.
    """
    print("\n" + "=" * 70)
    print("H-CHAIN SCALING STUDY")
    print("=" * 70)
    print(f"Chain lengths: {config.chain_lengths}")
    print(f"Bond length: {config.bond_length} A")
    print(f"Samples: {config.n_samples}, Seeds: {config.n_seeds}")
    print("=" * 70)

    all_results = {
        "nqs": {n: [] for n in config.chain_lengths},
        "baseline": {n: [] for n in config.chain_lengths},
    }

    total_runs = len(config.chain_lengths) * config.n_seeds * 2
    current_run = 0

    for chain_len in config.chain_lengths:
        print(f"\n{'='*60}")
        print(f"H{chain_len} Chain ({chain_len * 2} spin orbitals)")
        print(f"{'='*60}")

        for seed_idx in range(config.n_seeds):
            seed = 42 + seed_idx * 1000

            # NQS experiment
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] H{chain_len} NQS, seed={seed}")

            try:
                nqs_result = run_single_chain_experiment(
                    chain_length=chain_len,
                    bond_length=config.bond_length,
                    basis=config.basis,
                    n_samples=config.n_samples,
                    use_nqs=True,
                    seed=seed,
                    vmc_epochs=config.vmc_epochs,
                    nqs_alpha=config.nqs_alpha,
                    device=device,
                    verbose=verbose,
                )
                all_results["nqs"][chain_len].append(nqs_result)

                if "error" not in nqs_result:
                    error_mha = abs(nqs_result["sqd_energy"] - nqs_result["fci_energy"]) * 1000
                    print(f"    Energy: {nqs_result['sqd_energy']:.6f} Ha (error: {error_mha:.3f} mHa)")
                    print(f"    Conservation: {nqs_result['conservation_ratio']*100:.2f}%")
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results["nqs"][chain_len].append({"error": str(e), "seed": seed})

            # Baseline experiment
            current_run += 1
            print(f"[{current_run}/{total_runs}] H{chain_len} Baseline, seed={seed}")

            try:
                baseline_result = run_single_chain_experiment(
                    chain_length=chain_len,
                    bond_length=config.bond_length,
                    basis=config.basis,
                    n_samples=config.n_samples,
                    use_nqs=False,
                    seed=seed,
                    vmc_epochs=config.vmc_epochs,
                    nqs_alpha=config.nqs_alpha,
                    device=device,
                    verbose=verbose,
                )
                all_results["baseline"][chain_len].append(baseline_result)

                if "error" not in baseline_result:
                    error_mha = abs(baseline_result["sqd_energy"] - baseline_result["fci_energy"]) * 1000
                    print(f"    Energy: {baseline_result['sqd_energy']:.6f} Ha (error: {error_mha:.3f} mHa)")
                    print(f"    Conservation: {baseline_result['conservation_ratio']*100:.2f}%")
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results["baseline"][chain_len].append({"error": str(e), "seed": seed})

    # Compute statistics
    def compute_stats(results_list):
        valid = [r for r in results_list if "error" not in r]
        if not valid:
            return {"n_valid": 0}

        energies = [r["sqd_energy"] for r in valid]
        fci = valid[0]["fci_energy"]
        conservation = [r["conservation_ratio"] for r in valid]
        n_spin_orb = valid[0]["n_spin_orb"]

        return {
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "error_mean_mha": float(np.mean([abs(e - fci) * 1000 for e in energies])),
            "conservation_mean": float(np.mean(conservation)),
            "conservation_std": float(np.std(conservation)),
            "fci_energy": fci,
            "n_spin_orb": n_spin_orb,
            "n_valid": len(valid),
        }

    stats = {
        "nqs": {n: compute_stats(all_results["nqs"][n]) for n in config.chain_lengths},
        "baseline": {n: compute_stats(all_results["baseline"][n]) for n in config.chain_lengths},
    }

    return {
        "raw_results": all_results,
        "stats": stats,
        "config": {
            "chain_lengths": config.chain_lengths,
            "bond_length": config.bond_length,
            "basis": config.basis,
            "n_samples": config.n_samples,
            "n_seeds": config.n_seeds,
        },
    }


def print_scaling_table(results: dict) -> None:
    """Print scaling study summary table."""
    print("\n" + "=" * 90)
    print("H-CHAIN SCALING SUMMARY")
    print("=" * 90)
    print(f"{'Chain':>8} | {'Qubits':>8} | {'NQS Error (mHa)':>18} | {'Baseline Error (mHa)':>20} | {'NQS Cons.':>10}")
    print("-" * 90)

    stats = results["stats"]
    for chain_len in results["config"]["chain_lengths"]:
        nqs = stats["nqs"].get(chain_len, {})
        base = stats["baseline"].get(chain_len, {})

        n_qubits = nqs.get("n_spin_orb", chain_len * 2)

        nqs_err = f"{nqs.get('error_mean_mha', float('nan')):.3f}" if nqs.get("n_valid", 0) > 0 else "N/A"
        base_err = f"{base.get('error_mean_mha', float('nan')):.3f}" if base.get("n_valid", 0) > 0 else "N/A"
        nqs_cons = f"{nqs.get('conservation_mean', 0)*100:.1f}%" if nqs.get("n_valid", 0) > 0 else "N/A"

        print(f"H{chain_len:>7} | {n_qubits:>8} | {nqs_err:>18} | {base_err:>20} | {nqs_cons:>10}")

    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H-chain scaling experiments for NQS-SQD"
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=None,
        help="Single chain length to run (e.g., 4 for H4).",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run full scaling study from H2 to H8.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode.",
    )
    parser.add_argument(
        "--bond-length",
        type=float,
        default=1.0,
        help="H-H bond length in Angstrom.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples for SQD.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling",
        help="Output directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if args.quick:
        config = HChainConfig(
            chain_lengths=[2, 4],
            bond_length=args.bond_length,
            n_samples=500,
            n_seeds=1,
            vmc_epochs=20,
        )
        print("[INFO] Quick test mode")
    elif args.scaling:
        config = HChainConfig(
            chain_lengths=[2, 4, 6],
            bond_length=args.bond_length,
            n_samples=args.n_samples,
            n_seeds=args.n_seeds,
            vmc_epochs=100,
        )
    elif args.chain_length:
        config = HChainConfig(
            chain_lengths=[args.chain_length],
            bond_length=args.bond_length,
            n_samples=args.n_samples,
            n_seeds=args.n_seeds,
            vmc_epochs=100,
        )
    else:
        config = HChainConfig(
            chain_lengths=[2, 4, 6],
            bond_length=args.bond_length,
            n_samples=args.n_samples,
            n_seeds=args.n_seeds,
        )

    results = run_scaling_study(
        config=config,
        device=device,
        verbose=args.verbose,
    )

    print_scaling_table(results)

    output_dir = pathlib.Path(args.output_dir)
    save_results(results, output_dir, "h_chain_scaling")


if __name__ == "__main__":
    main()
