#!/usr/bin/env python3
"""Full research plan execution script.

Executes all phases of the NQS-SQD research plan:
- Phase 0-1: LiH bond length scan (0.8, 1.0, 1.5, 2.0 A)
- Phase 1-1: NQS quality control (epoch sweep)
- Phase 1-2: Sample count and subspace dimension sweep
- Phase 1-3: Phase Diagram generation
- Phase 2: H4 chain experiments (16 spin orbitals)
- Phase 3: H6 chain experiments (24 spin orbitals)

Usage:
    python scripts/run_full_research_plan.py --all
    python scripts/run_full_research_plan.py --phase 0
    python scripts/run_full_research_plan.py --phase 1
    python scripts/run_full_research_plan.py --phase 2

Target: Maximize RTX 4090 utilization for thorough experiments.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.sqd_interface.hamiltonian import (
    MoleculeConfig,
    build_molecular_hamiltonian,
    print_molecular_info,
)
from src.sqd_interface.sqd_runner import run_sqd_on_samples, SQDConfig
from src.nqs_models.ffn_nqs import FFNNNQS, FFNNNQSConfig
from src.nqs_models.vmc_training import VMCConfig, train_nqs_vmc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Single experiment result."""
    molecule: str
    bond_length: float
    method: str  # "nqs" or "baseline"
    n_samples: int
    vmc_epochs: int
    nqs_alpha: int
    nqs_layers: int
    seed: int

    # Results
    vmc_energy: Optional[float]
    sqd_energy: float
    fci_energy: float
    hf_energy: float

    # Errors (in mHa)
    vmc_error_mha: Optional[float]
    sqd_error_mha: float

    # Diagnostics
    conservation_ratio: float
    subspace_dimension: int
    n_samples_used: int

    # Timing
    vmc_time_sec: float
    sqd_time_sec: float
    total_time_sec: float


def save_results(results: List[Dict], output_path: pathlib.Path) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"[INFO] Saved results to {output_path}")


def run_single_experiment(
    mol_data,
    n_samples: int,
    use_nqs: bool,
    seed: int,
    vmc_epochs: int,
    nqs_alpha: int,
    nqs_layers: int,
    device: torch.device,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a single NQS or baseline experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    vmc_time = 0.0
    sqd_time = 0.0

    n_visible = mol_data.n_spin_orb

    # NQS configuration
    nqs_config = FFNNNQSConfig(
        n_visible=n_visible,
        n_hidden=int(n_visible * nqs_alpha),
        n_layers=nqs_layers,
        activation="tanh",
    )

    # VMC configuration
    vmc_config = VMCConfig(
        n_samples=min(1000, 100 * n_visible),
        n_chains=256,
        burn_in_steps=100 + 10 * n_visible,
        step_interval=10,
        learning_rate=0.01,
        sr_regularization=1e-4,
        n_epochs=vmc_epochs,
        patience=max(50, vmc_epochs // 2),
        lr_schedule="cosine",
    )

    # SQD configuration
    sqd_config = SQDConfig(
        max_iterations=5,
        num_batches=3,
        samples_per_batch=min(200, n_samples // 10),
        symmetrize_spin=True,
    )

    vmc_energy = None

    if use_nqs:
        # Train NQS
        vmc_start = time.time()
        model = FFNNNQS(nqs_config, device=device)

        vmc_result = train_nqs_vmc(
            model=model,
            mol_data=mol_data,
            config=vmc_config,
            device=device,
            verbose=verbose,
        )

        vmc_energy = vmc_result.best_energy
        vmc_time = time.time() - vmc_start

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
    else:
        # Baseline: random Bernoulli sampling
        samples = 2 * torch.randint(0, 2, (n_samples, n_visible), device=device).float() - 1

    # Run SQD
    sqd_start = time.time()
    sqd_result = run_sqd_on_samples(
        mol_data=mol_data,
        samples=samples,
        config=sqd_config,
        verbose=verbose,
    )
    sqd_time = time.time() - sqd_start

    total_time = time.time() - start_time

    # Compute errors
    vmc_error_mha = None
    if vmc_energy is not None and mol_data.fci_energy is not None:
        vmc_error_mha = (vmc_energy - mol_data.fci_energy) * 1000

    sqd_error_mha = 0.0
    if mol_data.fci_energy is not None:
        sqd_error_mha = abs(sqd_result.energy - mol_data.fci_energy) * 1000

    return ExperimentResult(
        molecule=mol_data.molecule_name,
        bond_length=mol_data.bond_length,
        method="nqs" if use_nqs else "baseline",
        n_samples=n_samples,
        vmc_epochs=vmc_epochs,
        nqs_alpha=nqs_alpha,
        nqs_layers=nqs_layers,
        seed=seed,
        vmc_energy=vmc_energy,
        sqd_energy=sqd_result.energy,
        fci_energy=mol_data.fci_energy,
        hf_energy=mol_data.hf_energy,
        vmc_error_mha=vmc_error_mha,
        sqd_error_mha=sqd_error_mha,
        conservation_ratio=sqd_result.conservation_ratio,
        subspace_dimension=sqd_result.subspace_dimension,
        n_samples_used=sqd_result.n_samples_used,
        vmc_time_sec=vmc_time,
        sqd_time_sec=sqd_time,
        total_time_sec=total_time,
    )


# -----------------------------------------------------------------------------
# Phase 0-1: LiH Bond Length Scan
# -----------------------------------------------------------------------------

def run_phase_0_1(device: torch.device, output_dir: pathlib.Path) -> List[Dict]:
    """Phase 0-1: LiH bond length scan at 0.8, 1.0, 1.5, 2.0 A."""
    print("\n" + "=" * 80)
    print("PHASE 0-1: LiH BOND LENGTH SCAN")
    print("=" * 80)

    bond_lengths = [0.8, 1.0, 1.5, 2.0]
    epochs_list = [2, 5, 10, 20, 40, 100, 200]
    n_samples = 5000
    n_seeds = 3

    results = []
    total_runs = len(bond_lengths) * len(epochs_list) * n_seeds * 2
    current_run = 0

    for bond_length in bond_lengths:
        print(f"\n{'='*60}")
        print(f"LiH @ {bond_length} A")
        print(f"{'='*60}")

        mol_config = MoleculeConfig(name="LiH", bond_length=bond_length, basis="sto-3g")
        mol_data = build_molecular_hamiltonian(mol_config)
        print_molecular_info(mol_data)

        for vmc_epochs in epochs_list:
            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx * 1000

                # NQS experiment
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] LiH@{bond_length}A, epochs={vmc_epochs}, seed={seed} (NQS)")

                try:
                    result = run_single_experiment(
                        mol_data=mol_data,
                        n_samples=n_samples,
                        use_nqs=True,
                        seed=seed,
                        vmc_epochs=vmc_epochs,
                        nqs_alpha=4,
                        nqs_layers=2,
                        device=device,
                        verbose=False,
                    )
                    results.append(asdict(result))
                    print(f"    VMC: {result.vmc_energy:.6f} Ha (error: {result.vmc_error_mha:.1f} mHa)")
                    print(f"    SQD: {result.sqd_energy:.6f} Ha (error: {result.sqd_error_mha:.4f} mHa)")
                    print(f"    Conservation: {result.conservation_ratio*100:.2f}%")
                except Exception as e:
                    print(f"    FAILED: {e}")
                    results.append({"error": str(e), "bond_length": bond_length, "epochs": vmc_epochs, "seed": seed, "method": "nqs"})

                # Baseline experiment
                current_run += 1
                print(f"[{current_run}/{total_runs}] LiH@{bond_length}A, epochs={vmc_epochs}, seed={seed} (Baseline)")

                try:
                    result = run_single_experiment(
                        mol_data=mol_data,
                        n_samples=n_samples,
                        use_nqs=False,
                        seed=seed,
                        vmc_epochs=vmc_epochs,
                        nqs_alpha=4,
                        nqs_layers=2,
                        device=device,
                        verbose=False,
                    )
                    results.append(asdict(result))
                    print(f"    SQD: {result.sqd_energy:.6f} Ha (error: {result.sqd_error_mha:.4f} mHa)")
                    print(f"    Conservation: {result.conservation_ratio*100:.2f}%")
                except Exception as e:
                    print(f"    FAILED: {e}")
                    results.append({"error": str(e), "bond_length": bond_length, "epochs": vmc_epochs, "seed": seed, "method": "baseline"})

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, output_dir / f"phase_0_1_lih_bond_scan_{timestamp}.json")

    return results


# -----------------------------------------------------------------------------
# Phase 1-1: NQS Quality Control Scan
# -----------------------------------------------------------------------------

def run_phase_1_1(device: torch.device, output_dir: pathlib.Path) -> List[Dict]:
    """Phase 1-1: NQS quality control scan (epoch, alpha, layers)."""
    print("\n" + "=" * 80)
    print("PHASE 1-1: NQS QUALITY CONTROL SCAN")
    print("=" * 80)

    # LiH @ 0.8 A as reference
    mol_config = MoleculeConfig(name="LiH", bond_length=0.8, basis="sto-3g")
    mol_data = build_molecular_hamiltonian(mol_config)
    print_molecular_info(mol_data)

    # Scan parameters
    epochs_list = [2, 5, 10, 20, 40, 100, 200, 500]
    alpha_list = [2, 4, 8]
    layers_list = [1, 2, 3]
    n_samples = 5000
    n_seeds = 3

    results = []

    # First: epoch scan with fixed alpha=4, layers=2
    print("\n--- Epoch Scan (alpha=4, layers=2) ---")
    for epochs in epochs_list:
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000
            print(f"\nEpochs={epochs}, seed={seed}")

            try:
                result = run_single_experiment(
                    mol_data=mol_data,
                    n_samples=n_samples,
                    use_nqs=True,
                    seed=seed,
                    vmc_epochs=epochs,
                    nqs_alpha=4,
                    nqs_layers=2,
                    device=device,
                    verbose=False,
                )
                results.append({**asdict(result), "scan_type": "epoch"})
                print(f"    VMC: {result.vmc_error_mha:.1f} mHa, SQD: {result.sqd_error_mha:.4f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")

    # Second: alpha scan with fixed epochs=100, layers=2
    print("\n--- Alpha Scan (epochs=100, layers=2) ---")
    for alpha in alpha_list:
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000
            print(f"\nAlpha={alpha}, seed={seed}")

            try:
                result = run_single_experiment(
                    mol_data=mol_data,
                    n_samples=n_samples,
                    use_nqs=True,
                    seed=seed,
                    vmc_epochs=100,
                    nqs_alpha=alpha,
                    nqs_layers=2,
                    device=device,
                    verbose=False,
                )
                results.append({**asdict(result), "scan_type": "alpha"})
                print(f"    VMC: {result.vmc_error_mha:.1f} mHa, SQD: {result.sqd_error_mha:.4f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")

    # Third: layers scan with fixed epochs=100, alpha=4
    print("\n--- Layers Scan (epochs=100, alpha=4) ---")
    for layers in layers_list:
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 1000
            print(f"\nLayers={layers}, seed={seed}")

            try:
                result = run_single_experiment(
                    mol_data=mol_data,
                    n_samples=n_samples,
                    use_nqs=True,
                    seed=seed,
                    vmc_epochs=100,
                    nqs_alpha=4,
                    nqs_layers=layers,
                    device=device,
                    verbose=False,
                )
                results.append({**asdict(result), "scan_type": "layers"})
                print(f"    VMC: {result.vmc_error_mha:.1f} mHa, SQD: {result.sqd_error_mha:.4f} mHa")
            except Exception as e:
                print(f"    FAILED: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, output_dir / f"phase_1_1_nqs_quality_{timestamp}.json")

    return results


# -----------------------------------------------------------------------------
# Phase 1-2: Sample Count and Subspace Dimension Scan
# -----------------------------------------------------------------------------

def run_phase_1_2(device: torch.device, output_dir: pathlib.Path) -> List[Dict]:
    """Phase 1-2: Sample count and subspace dimension scan."""
    print("\n" + "=" * 80)
    print("PHASE 1-2: SAMPLE COUNT AND DIMENSION SCAN")
    print("=" * 80)

    mol_config = MoleculeConfig(name="LiH", bond_length=0.8, basis="sto-3g")
    mol_data = build_molecular_hamiltonian(mol_config)
    print_molecular_info(mol_data)

    # Sample budgets
    sample_budgets = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
    epochs_list = [10, 50, 100, 200]
    n_seeds = 3

    results = []

    for epochs in epochs_list:
        print(f"\n--- Epochs={epochs} ---")

        for n_samples in sample_budgets:
            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx * 1000
                print(f"\nSamples={n_samples}, epochs={epochs}, seed={seed}")

                # NQS
                try:
                    result = run_single_experiment(
                        mol_data=mol_data,
                        n_samples=n_samples,
                        use_nqs=True,
                        seed=seed,
                        vmc_epochs=epochs,
                        nqs_alpha=4,
                        nqs_layers=2,
                        device=device,
                        verbose=False,
                    )
                    results.append({**asdict(result), "scan_type": "samples"})
                    print(f"    NQS: VMC={result.vmc_error_mha:.1f}mHa, SQD={result.sqd_error_mha:.4f}mHa, cons={result.conservation_ratio*100:.1f}%")
                except Exception as e:
                    print(f"    NQS FAILED: {e}")

                # Baseline
                try:
                    result = run_single_experiment(
                        mol_data=mol_data,
                        n_samples=n_samples,
                        use_nqs=False,
                        seed=seed,
                        vmc_epochs=epochs,
                        nqs_alpha=4,
                        nqs_layers=2,
                        device=device,
                        verbose=False,
                    )
                    results.append({**asdict(result), "scan_type": "samples"})
                    print(f"    Baseline: SQD={result.sqd_error_mha:.4f}mHa, cons={result.conservation_ratio*100:.1f}%")
                except Exception as e:
                    print(f"    Baseline FAILED: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, output_dir / f"phase_1_2_sample_scan_{timestamp}.json")

    return results


# -----------------------------------------------------------------------------
# Phase 2: H4 Chain (16 spin orbitals)
# -----------------------------------------------------------------------------

def run_phase_2(device: torch.device, output_dir: pathlib.Path) -> List[Dict]:
    """Phase 2: H4 chain experiments (16 spin orbitals)."""
    print("\n" + "=" * 80)
    print("PHASE 2: H4 CHAIN (16 SPIN ORBITALS)")
    print("=" * 80)

    # H4 with different bond lengths
    bond_lengths = [0.8, 1.0, 1.2, 1.5]
    epochs_list = [50, 100, 200, 500]
    sample_budgets = [1000, 5000, 10000, 25000]
    n_seeds = 3

    results = []

    for bond_length in bond_lengths:
        print(f"\n{'='*60}")
        print(f"H4 @ {bond_length} A")
        print(f"{'='*60}")

        mol_config = MoleculeConfig(name="H4", bond_length=bond_length, basis="sto-3g")
        mol_data = build_molecular_hamiltonian(mol_config)
        print_molecular_info(mol_data)

        for epochs in epochs_list:
            for n_samples in sample_budgets:
                for seed_idx in range(n_seeds):
                    seed = 42 + seed_idx * 1000
                    print(f"\nH4@{bond_length}A, epochs={epochs}, samples={n_samples}, seed={seed}")

                    # NQS
                    try:
                        result = run_single_experiment(
                            mol_data=mol_data,
                            n_samples=n_samples,
                            use_nqs=True,
                            seed=seed,
                            vmc_epochs=epochs,
                            nqs_alpha=4,
                            nqs_layers=3,
                            device=device,
                            verbose=False,
                        )
                        results.append({**asdict(result), "phase": "2"})
                        print(f"    NQS: VMC={result.vmc_error_mha:.1f}mHa, SQD={result.sqd_error_mha:.4f}mHa")
                    except Exception as e:
                        print(f"    NQS FAILED: {e}")

                    # Baseline
                    try:
                        result = run_single_experiment(
                            mol_data=mol_data,
                            n_samples=n_samples,
                            use_nqs=False,
                            seed=seed,
                            vmc_epochs=epochs,
                            nqs_alpha=4,
                            nqs_layers=3,
                            device=device,
                            verbose=False,
                        )
                        results.append({**asdict(result), "phase": "2"})
                        print(f"    Baseline: SQD={result.sqd_error_mha:.4f}mHa")
                    except Exception as e:
                        print(f"    Baseline FAILED: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, output_dir / f"phase_2_h4_chain_{timestamp}.json")

    return results


# -----------------------------------------------------------------------------
# Phase 3: H6 Chain (24 spin orbitals)
# -----------------------------------------------------------------------------

def run_phase_3(device: torch.device, output_dir: pathlib.Path) -> List[Dict]:
    """Phase 3: H6 chain experiments (24 spin orbitals)."""
    print("\n" + "=" * 80)
    print("PHASE 3: H6 CHAIN (24 SPIN ORBITALS)")
    print("=" * 80)

    bond_lengths = [1.0, 1.5]
    epochs_list = [100, 200, 500]
    sample_budgets = [5000, 10000, 50000]
    n_seeds = 3

    results = []

    for bond_length in bond_lengths:
        print(f"\n{'='*60}")
        print(f"H6 @ {bond_length} A")
        print(f"{'='*60}")

        mol_config = MoleculeConfig(name="H6", bond_length=bond_length, basis="sto-3g")
        mol_data = build_molecular_hamiltonian(mol_config)
        print_molecular_info(mol_data)

        for epochs in epochs_list:
            for n_samples in sample_budgets:
                for seed_idx in range(n_seeds):
                    seed = 42 + seed_idx * 1000
                    print(f"\nH6@{bond_length}A, epochs={epochs}, samples={n_samples}, seed={seed}")

                    # NQS
                    try:
                        result = run_single_experiment(
                            mol_data=mol_data,
                            n_samples=n_samples,
                            use_nqs=True,
                            seed=seed,
                            vmc_epochs=epochs,
                            nqs_alpha=4,
                            nqs_layers=3,
                            device=device,
                            verbose=False,
                        )
                        results.append({**asdict(result), "phase": "3"})
                        print(f"    NQS: VMC={result.vmc_error_mha:.1f}mHa, SQD={result.sqd_error_mha:.4f}mHa")
                    except Exception as e:
                        print(f"    NQS FAILED: {e}")

                    # Baseline
                    try:
                        result = run_single_experiment(
                            mol_data=mol_data,
                            n_samples=n_samples,
                            use_nqs=False,
                            seed=seed,
                            vmc_epochs=epochs,
                            nqs_alpha=4,
                            nqs_layers=3,
                            device=device,
                            verbose=False,
                        )
                        results.append({**asdict(result), "phase": "3"})
                        print(f"    Baseline: SQD={result.sqd_error_mha:.4f}mHa")
                    except Exception as e:
                        print(f"    Baseline FAILED: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, output_dir / f"phase_3_h6_chain_{timestamp}.json")

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run full NQS-SQD research plan")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=str, choices=["0", "1", "2", "3"], help="Run specific phase")
    parser.add_argument("--output-dir", type=str, default="results/phase_diagram", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    all_results = []

    if args.all or args.phase == "0":
        results = run_phase_0_1(device, output_dir)
        all_results.extend(results)

    if args.all or args.phase == "1":
        results = run_phase_1_1(device, output_dir)
        all_results.extend(results)

        results = run_phase_1_2(device, output_dir)
        all_results.extend(results)

    if args.all or args.phase == "2":
        results = run_phase_2(device, output_dir)
        all_results.extend(results)

    if args.all or args.phase == "3":
        results = run_phase_3(device, output_dir)
        all_results.extend(results)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED: Total time = {total_time/3600:.2f} hours")
    print(f"{'='*80}")

    # Save combined results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(all_results, output_dir / f"all_results_{timestamp}.json")


if __name__ == "__main__":
    main()
