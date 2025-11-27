"""Sample-based Quantum Diagonalization (SQD) runner.

This module implements:
1. Conservation law filtering (N_elec, S_z)
2. Spin orbital reordering (interleaved -> blocked)
3. BitArray conversion for qiskit-addon-sqd
4. Fermionic Hamiltonian diagonalization

Physical conventions:
- NQS samples use interleaved spin ordering: 1up, 1down, 2up, 2down, ...
- qiskit-addon-sqd uses blocked spin ordering: 1up, 2up, ..., 1down, 2down, ...
- ±1 encoding from NQS: +1 = occupied, -1 = unoccupied
- 0/1 encoding for SQD: 1 = occupied, 0 = unoccupied

References:
- Kanno et al., arXiv:2405.05068 (2024) - SQD method
- qiskit-addon-sqd documentation
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
import torch

from qiskit_addon_sqd.counts import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

from .hamiltonian import MolecularData


@dataclass
class SQDConfig:
    """Configuration for SQD diagonalization.

    Attributes
    ----------
    energy_tol : float
        Energy convergence tolerance.
    occupancies_tol : float
        Occupancy convergence tolerance.
    max_iterations : int
        Maximum SQD iterations.
    num_batches : int
        Number of sample batches for averaging.
    samples_per_batch : int
        Samples per batch.
    symmetrize_spin : bool
        Whether to symmetrize spin.
    carryover_threshold : float
        Threshold for carrying configurations between iterations.
    max_cycle : int
        Maximum SCI solver cycles.
    spin_sq : float
        Target S^2 value (0.0 for singlet).
    seed : int | None
        Random seed for reproducibility.
    """

    energy_tol: float = 1e-6
    occupancies_tol: float = 1e-6
    max_iterations: int = 5
    num_batches: int = 3
    samples_per_batch: int = 100
    symmetrize_spin: bool = True
    carryover_threshold: float = 1e-4
    max_cycle: int = 200
    spin_sq: float = 0.0
    seed: int | None = 12345


@dataclass
class SQDResult:
    """Result of SQD diagonalization.

    Attributes
    ----------
    energy : float
        Ground state energy estimate (including nuclear repulsion).
    electronic_energy : float
        Electronic energy only.
    subspace_dimension : int
        Final subspace dimension.
    n_samples_used : int
        Number of samples that passed conservation filtering.
    n_samples_total : int
        Total number of input samples.
    conservation_ratio : float
        Fraction of samples passing conservation filter.
    iteration_history : list
        Energy at each SQD iteration.
    """

    energy: float
    electronic_energy: float
    subspace_dimension: int
    n_samples_used: int
    n_samples_total: int
    conservation_ratio: float
    iteration_history: list


def samples_to_bitstrings(
    samples: torch.Tensor,
) -> dict[str, int]:
    """Convert ±1 samples to bitstring counts.

    Parameters
    ----------
    samples : torch.Tensor
        Samples with ±1 values, shape (n_samples, n_spin_orb).

    Returns
    -------
    config_counts : dict[str, int]
        Dictionary mapping bitstring to count.
        Bitstring uses "0"/"1" encoding.
    """
    # Convert to list of tuples for counting
    samples_list = [tuple(s.cpu().numpy().astype(int)) for s in samples]
    config_counts_raw = Counter(samples_list)

    # Convert ±1 to "0"/"1" strings
    config_str_counts = Counter()
    for config_tuple, count in config_counts_raw.items():
        # Map: -1 -> "0", +1 -> "1"
        mapped_bits = ["0" if s == -1 else "1" for s in config_tuple]
        config_str = "".join(mapped_bits)
        config_str_counts[config_str] += count

    return dict(config_str_counts)


def filter_conserved_configurations(
    config_counts: dict[str, int],
    n_elec: tuple[int, int],
    target_sz: int = 0,
) -> dict[str, int]:
    """Filter configurations by conservation laws (N_elec and S_z).

    Parameters
    ----------
    config_counts : dict[str, int]
        Dictionary mapping bitstring to count.
    n_elec : tuple[int, int]
        Number of (alpha, beta) electrons.
    target_sz : int
        Target S_z × 2 value (N_up - N_down). Default 0 for singlet.

    Returns
    -------
    conserved_counts : dict[str, int]
        Filtered configurations satisfying conservation laws.

    Notes
    -----
    Assumes interleaved spin ordering: 1up, 1down, 2up, 2down, ...
    - Even indices (0, 2, 4, ...) are alpha (up) spin
    - Odd indices (1, 3, 5, ...) are beta (down) spin
    """
    expected_n_elec = n_elec[0] + n_elec[1]
    conserved_counts = {}

    for config_str, count in config_counts.items():
        # Check total electron number
        if config_str.count("1") != expected_n_elec:
            continue

        # Check S_z conservation
        # Count up spins (even indices) and down spins (odd indices)
        n_up = sum(1 for i in range(0, len(config_str), 2) if config_str[i] == "1")
        n_down = sum(1 for i in range(1, len(config_str), 2) if config_str[i] == "1")

        s_z_times_2 = n_up - n_down
        if s_z_times_2 == target_sz:
            conserved_counts[config_str] = count

    return conserved_counts


def reorder_interleaved_to_blocked(config_str: str) -> str:
    """Reorder spin orbitals from interleaved to blocked format.

    Parameters
    ----------
    config_str : str
        Bitstring in interleaved format: 1up, 1down, 2up, 2down, ...

    Returns
    -------
    reordered : str
        Bitstring in blocked format: 1up, 2up, ..., 1down, 2down, ...

    Example
    -------
    >>> reorder_interleaved_to_blocked("10110100")
    # Input:  1up=1, 1down=0, 2up=1, 2down=1, 3up=0, 3down=1, 4up=0, 4down=0
    # Output: 1up, 2up, 3up, 4up, 1down, 2down, 3down, 4down
    #         1,   1,   0,   0,   0,     1,     1,     0
    "11000110"
    """
    # Extract up (even indices) and down (odd indices)
    up_part = "".join(config_str[i] for i in range(0, len(config_str), 2))
    down_part = "".join(config_str[i] for i in range(1, len(config_str), 2))
    return up_part + down_part


def configs_to_bitarray(
    config_counts: dict[str, int],
    reorder_spins: bool = True,
) -> BitArray:
    """Convert configuration counts to qiskit-addon-sqd BitArray.

    Parameters
    ----------
    config_counts : dict[str, int]
        Dictionary mapping bitstring to count.
    reorder_spins : bool
        Whether to reorder from interleaved to blocked spin format.

    Returns
    -------
    bit_array : BitArray
        BitArray suitable for diagonalize_fermionic_hamiltonian.
    """
    if not config_counts:
        raise ValueError("No configurations to convert")

    # Optionally reorder spins
    if reorder_spins:
        reordered_counts = {}
        for config_str, count in config_counts.items():
            reordered = reorder_interleaved_to_blocked(config_str)
            reordered_counts[reordered] = reordered_counts.get(reordered, 0) + count
        config_counts = reordered_counts

    # Get bitstring list and parameters
    bitstrings = list(config_counts.keys())
    num_bits = len(bitstrings[0])
    num_bytes = (num_bits + 7) // 8

    # Expand by counts to create sample array
    samples = []
    for bitstring, count in config_counts.items():
        samples.extend([int(bitstring, 2)] * count)

    # Convert to bytes
    data = b"".join(val.to_bytes(num_bytes, "big") for val in samples)
    array = np.frombuffer(data, dtype=np.uint8)

    # Create BitArray
    bit_array = BitArray(array.reshape(-1, num_bytes), num_bits=num_bits)

    return bit_array


def run_sqd_on_samples(
    mol_data: MolecularData,
    samples: torch.Tensor | dict[str, int],
    config: SQDConfig | None = None,
    verbose: bool = True,
) -> SQDResult:
    """Run SQD diagonalization on NQS samples.

    This function performs the complete SQD pipeline:
    1. Convert ±1 samples to bitstring counts
    2. Filter by conservation laws (N_elec, S_z)
    3. Reorder spin orbitals (interleaved -> blocked)
    4. Convert to BitArray
    5. Run fermionic Hamiltonian diagonalization

    Parameters
    ----------
    mol_data : MolecularData
        Molecular integrals and metadata.
    samples : torch.Tensor | dict[str, int]
        Either raw ±1 samples (batch, n_spin_orb) or pre-computed config counts.
    config : SQDConfig, optional
        SQD configuration. Uses defaults if not provided.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    result : SQDResult
        SQD results including energy and diagnostics.
    """
    if config is None:
        config = SQDConfig()

    # Convert samples to config counts if needed
    if isinstance(samples, torch.Tensor):
        config_counts = samples_to_bitstrings(samples)
        n_samples_total = samples.shape[0]
    else:
        config_counts = samples
        n_samples_total = sum(config_counts.values())

    if verbose:
        print(f"\n[SQD] Total samples: {n_samples_total}")
        print(f"[SQD] Unique configurations: {len(config_counts)}")

    # Filter by conservation laws
    conserved_counts = filter_conserved_configurations(
        config_counts,
        n_elec=mol_data.n_elec,
        target_sz=0,
    )

    n_samples_used = sum(conserved_counts.values())
    conservation_ratio = n_samples_used / n_samples_total if n_samples_total > 0 else 0

    if verbose:
        print(f"[SQD] Conserved configurations: {len(conserved_counts)}")
        print(f"[SQD] Conserved samples: {n_samples_used} ({conservation_ratio*100:.2f}%)")

    if not conserved_counts:
        print("[SQD] WARNING: No configurations passed conservation filter!")
        return SQDResult(
            energy=float("nan"),
            electronic_energy=float("nan"),
            subspace_dimension=0,
            n_samples_used=0,
            n_samples_total=n_samples_total,
            conservation_ratio=0.0,
            iteration_history=[],
        )

    # Convert to BitArray
    bit_array = configs_to_bitarray(conserved_counts, reorder_spins=True)

    if verbose:
        print(f"[SQD] BitArray: {bit_array}")

    # Set up SCI solver
    sci_solver = partial(
        solve_sci_batch,
        spin_sq=config.spin_sq,
        max_cycle=config.max_cycle,
    )

    # Track iteration history
    iteration_history = []

    def callback(results: list[SCIResult]) -> None:
        iteration = len(iteration_history) + 1
        energies = [r.energy + mol_data.nuclear_repulsion_energy for r in results]
        iteration_history.append(energies)
        if verbose:
            print(f"[SQD] Iteration {iteration}: Energies = {[f'{e:.6f}' for e in energies]}")

    # Run SQD
    try:
        result = diagonalize_fermionic_hamiltonian(
            mol_data.hcore,
            mol_data.eri,
            bit_array,
            samples_per_batch=config.samples_per_batch,
            norb=mol_data.n_orb,
            nelec=mol_data.n_elec,
            num_batches=config.num_batches,
            energy_tol=config.energy_tol,
            occupancies_tol=config.occupancies_tol,
            max_iterations=config.max_iterations,
            sci_solver=sci_solver,
            symmetrize_spin=config.symmetrize_spin,
            carryover_threshold=config.carryover_threshold,
            callback=callback,
            seed=config.seed,
        )

        electronic_energy = result.energy
        total_energy = electronic_energy + mol_data.nuclear_repulsion_energy

        # Get subspace dimension from the SCI result
        subspace_dim = np.prod(result.sci_state.amplitudes.shape) if hasattr(result, 'sci_state') else 0

        if verbose:
            print(f"\n[SQD] Final electronic energy: {electronic_energy:.6f} Ha")
            print(f"[SQD] Final total energy: {total_energy:.6f} Ha")
            print(f"[SQD] Reference FCI: {mol_data.fci_energy:.6f} Ha" if mol_data.fci_energy else "")

        return SQDResult(
            energy=total_energy,
            electronic_energy=electronic_energy,
            subspace_dimension=int(subspace_dim),
            n_samples_used=n_samples_used,
            n_samples_total=n_samples_total,
            conservation_ratio=conservation_ratio,
            iteration_history=iteration_history,
        )

    except Exception as e:
        print(f"[SQD] ERROR: {e}")
        raise
