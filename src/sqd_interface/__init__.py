"""Interfaces to qiskit-addon-sqd and Hamiltonian construction utilities."""

from .hamiltonian import (
    MolecularData,
    H2Config,
    LiHConfig,
    MoleculeConfig,
    build_molecular_hamiltonian,
    build_h2_hamiltonian,
    build_lih_hamiltonian,
    build_h2_hamiltonian_12bit,
    print_molecular_info,
)

from .sqd_runner import (
    SQDConfig,
    SQDResult,
    samples_to_bitstrings,
    filter_conserved_configurations,
    reorder_interleaved_to_blocked,
    configs_to_bitarray,
    run_sqd_on_samples,
)

__all__ = [
    # Hamiltonian
    "MolecularData",
    "H2Config",
    "LiHConfig",
    "MoleculeConfig",
    "build_molecular_hamiltonian",
    "build_h2_hamiltonian",
    "build_lih_hamiltonian",
    "build_h2_hamiltonian_12bit",
    "print_molecular_info",
    # SQD
    "SQDConfig",
    "SQDResult",
    "samples_to_bitstrings",
    "filter_conserved_configurations",
    "reorder_interleaved_to_blocked",
    "configs_to_bitarray",
    "run_sqd_on_samples",
]
