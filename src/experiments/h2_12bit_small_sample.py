"""Main experiment: NQS-SQD for small molecules in few-sample regime.

This script implements the complete NQS + SQD pipeline:
1. Build molecular Hamiltonian (PySCF)
2. Train NQS using VMC with Stochastic Reconfiguration
3. Sample from trained NQS using Metropolis MCMC
4. Run SQD diagonalization on samples
5. Compare NQS vs baseline (random) sampling

Usage:
    # Run with NQS sampler (default)
    python -m src.experiments.h2_12bit_small_sample --config configs/h2_12bit_nqs.yaml

    # Run with baseline (Bernoulli) sampler
    python -m src.experiments.h2_12bit_small_sample --config configs/h2_12bit_baseline.yaml --no-nqs

    # Run with LiH (12-bit)
    python -m src.experiments.h2_12bit_small_sample --molecule LiH --bond-length 0.8

Key research questions:
- How does NQS sampling improve sample efficiency vs random baseline?
- What is the minimum sample budget needed to reach FCI accuracy?
- How does the conservation filter ratio affect SQD convergence?
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml

from src.sqd_interface.hamiltonian import (
    MoleculeConfig,
    H2Config,
    LiHConfig,
    build_molecular_hamiltonian,
    build_h2_hamiltonian,
    build_lih_hamiltonian,
    print_molecular_info,
)
from src.sqd_interface.sqd_runner import run_sqd_on_samples, SQDConfig
from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.nqs_models.vmc_training import VMCConfig, train_nqs_vmc


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(results: dict, output_dir: pathlib.Path, prefix: str) -> None:
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


def run_nqs_sqd_experiment(
    mol_data,
    nqs_config: FFNNNQSConfig,
    vmc_config: VMCConfig,
    sqd_config: SQDConfig,
    n_final_samples: int,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run complete NQS + SQD experiment.

    Returns
    -------
    results : dict
        Dictionary containing:
        - vmc_best_energy: Best VMC energy during training
        - vmc_energy_history: Training energy history
        - sqd_energy: Final SQD energy
        - sqd_conservation_ratio: Fraction of samples passing filter
        - fci_energy: Reference FCI energy
    """
    if verbose:
        print("\n" + "="*60)
        print("NQS + SQD Experiment")
        print("="*60)

    # Create NQS model
    model = FFNNNQS(nqs_config, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n[NQS] Model created with {n_params} parameters")
        print(f"[NQS] Architecture: {nqs_config.n_visible} -> {nqs_config.n_hidden} x {nqs_config.n_layers} -> 1")

    # Train with VMC
    if verbose:
        print(f"\n[VMC] Training for up to {vmc_config.n_epochs} epochs...")

    vmc_result = train_nqs_vmc(
        model=model,
        mol_data=mol_data,
        config=vmc_config,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[VMC] Training complete. Best energy: {vmc_result.best_energy:.6f} Ha")

    # Sample from trained model
    if verbose:
        print(f"\n[Sampling] Generating {n_final_samples} samples from trained NQS...")

    n_chains = min(256, n_final_samples)
    n_samples_per_chain = (n_final_samples + n_chains - 1) // n_chains

    samples = model.sample_mcmc(
        n_samples_per_chain=n_samples_per_chain,
        n_chains=n_chains,
        burn_in_steps=vmc_config.burn_in_steps,
        step_interval=vmc_config.step_interval,
        device=device,
    )

    if verbose:
        print(f"[Sampling] Generated {samples.shape[0]} samples")

    # Run SQD
    if verbose:
        print(f"\n[SQD] Running Sample-based Quantum Diagonalization...")

    sqd_result = run_sqd_on_samples(
        mol_data=mol_data,
        samples=samples,
        config=sqd_config,
        verbose=verbose,
    )

    # Compile results
    results = {
        "vmc_best_energy": vmc_result.best_energy,
        "vmc_energy_history": vmc_result.energy_history,
        "vmc_std_history": vmc_result.std_history,
        "sqd_energy": sqd_result.energy,
        "sqd_electronic_energy": sqd_result.electronic_energy,
        "sqd_conservation_ratio": sqd_result.conservation_ratio,
        "sqd_n_samples_used": sqd_result.n_samples_used,
        "sqd_n_samples_total": sqd_result.n_samples_total,
        "sqd_subspace_dimension": sqd_result.subspace_dimension,
        "fci_energy": mol_data.fci_energy,
        "hf_energy": mol_data.hf_energy,
        "nuclear_repulsion": mol_data.nuclear_repulsion_energy,
    }

    return results


def run_baseline_sqd_experiment(
    mol_data,
    sqd_config: SQDConfig,
    n_samples: int,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Run baseline (random Bernoulli) + SQD experiment.

    Returns
    -------
    results : dict
        Dictionary containing SQD results from random sampling.
    """
    if verbose:
        print("\n" + "="*60)
        print("Baseline (Random) + SQD Experiment")
        print("="*60)

    # Generate random ±1 samples
    n_spin_orb = mol_data.n_spin_orb
    if verbose:
        print(f"\n[Baseline] Generating {n_samples} random Bernoulli samples...")

    samples = 2 * torch.randint(0, 2, (n_samples, n_spin_orb), device=device).float() - 1

    if verbose:
        print(f"[Baseline] Generated {samples.shape[0]} samples")

    # Run SQD
    if verbose:
        print(f"\n[SQD] Running Sample-based Quantum Diagonalization...")

    sqd_result = run_sqd_on_samples(
        mol_data=mol_data,
        samples=samples,
        config=sqd_config,
        verbose=verbose,
    )

    # Compile results
    results = {
        "sqd_energy": sqd_result.energy,
        "sqd_electronic_energy": sqd_result.electronic_energy,
        "sqd_conservation_ratio": sqd_result.conservation_ratio,
        "sqd_n_samples_used": sqd_result.n_samples_used,
        "sqd_n_samples_total": sqd_result.n_samples_total,
        "sqd_subspace_dimension": sqd_result.subspace_dimension,
        "fci_energy": mol_data.fci_energy,
        "hf_energy": mol_data.hf_energy,
        "nuclear_repulsion": mol_data.nuclear_repulsion_energy,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NQS-SQD experiment for small molecules in few-sample regime"
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
        default="LiH",
        choices=["H2", "LiH", "H4", "H6"],
        help="Molecule to study (default: LiH for 12-bit).",
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
        "--no-nqs",
        action="store_true",
        help="Use baseline (random) sampler instead of NQS.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2560,
        help="Number of final samples for SQD.",
    )
    parser.add_argument(
        "--vmc-epochs",
        type=int,
        default=100,
        help="Number of VMC training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # SQD config
    sqd_cfg = cfg.get("sqd", {})
    sqd_config = SQDConfig(
        energy_tol=sqd_cfg.get("energy_tol", 1e-6),
        occupancies_tol=sqd_cfg.get("occupancies_tol", 1e-6),
        max_iterations=sqd_cfg.get("max_iterations", 5),
        num_batches=sqd_cfg.get("num_batches", 3),
        samples_per_batch=sqd_cfg.get("samples_per_batch", 100),
        symmetrize_spin=sqd_cfg.get("symmetrize_spin", True),
        carryover_threshold=sqd_cfg.get("carryover_threshold", 1e-4),
        max_cycle=sqd_cfg.get("max_cycle", 200),
        spin_sq=sqd_cfg.get("spin_sq", 0.0),
        seed=args.seed,
    )

    n_samples = cfg.get("sampling", {}).get("n_samples", args.n_samples)

    if args.no_nqs:
        # Baseline experiment
        results = run_baseline_sqd_experiment(
            mol_data=mol_data,
            sqd_config=sqd_config,
            n_samples=n_samples,
            device=device,
            verbose=True,
        )
        prefix = f"{molecule_name}_baseline"
    else:
        # NQS experiment
        nqs_cfg = cfg.get("nqs", {})
        n_visible = mol_data.n_spin_orb
        alpha = nqs_cfg.get("alpha", 4)

        nqs_config = FFNNNQSConfig(
            n_visible=n_visible,
            n_hidden=int(n_visible * alpha),
            n_layers=nqs_cfg.get("n_layers", 2),
            activation=nqs_cfg.get("activation", "tanh"),
        )

        vmc_cfg = cfg.get("vmc", {})
        vmc_config = VMCConfig(
            n_samples=vmc_cfg.get("n_samples", 500),
            n_chains=vmc_cfg.get("n_chains", 256),
            burn_in_steps=vmc_cfg.get("burn_in_steps", 100),
            step_interval=vmc_cfg.get("step_interval", 10),
            learning_rate=vmc_cfg.get("learning_rate", 0.01),
            sr_regularization=vmc_cfg.get("sr_regularization", 1e-4),
            n_epochs=vmc_cfg.get("n_epochs", args.vmc_epochs),
            patience=vmc_cfg.get("patience", 100),
            lr_schedule=vmc_cfg.get("lr_schedule", "cosine"),
        )

        results = run_nqs_sqd_experiment(
            mol_data=mol_data,
            nqs_config=nqs_config,
            vmc_config=vmc_config,
            sqd_config=sqd_config,
            n_final_samples=n_samples,
            device=device,
            verbose=True,
        )
        prefix = f"{molecule_name}_nqs"

    # Add metadata
    results["molecule"] = molecule_name
    results["bond_length"] = bond_length
    results["basis"] = basis
    results["n_spin_orb"] = mol_data.n_spin_orb
    results["n_elec"] = mol_data.n_elec
    results["seed"] = args.seed
    results["use_nqs"] = not args.no_nqs

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Molecule: {molecule_name} @ {bond_length:.3f} A ({basis})")
    print(f"Spin orbitals: {mol_data.n_spin_orb} ({mol_data.n_spin_orb}-bit)")
    print(f"Electrons: {mol_data.n_elec}")
    print("-"*60)
    print(f"SQD Energy:  {results['sqd_energy']:.6f} Ha")
    print(f"FCI Energy:  {results['fci_energy']:.6f} Ha")
    print(f"Error:       {abs(results['sqd_energy'] - results['fci_energy'])*1000:.3f} mHa")
    print(f"Conservation ratio: {results['sqd_conservation_ratio']*100:.2f}%")
    print("="*60)

    # Save results
    output_dir = pathlib.Path(args.output_dir)
    save_results(results, output_dir, prefix)


if __name__ == "__main__":
    main()
