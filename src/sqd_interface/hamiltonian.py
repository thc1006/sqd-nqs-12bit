"""Molecular Hamiltonian builders and bitstring mappings.

Uses PySCF to generate molecular integrals (hcore, eri) for small molecules
like H2 and LiH. These integrals are then used by qiskit-addon-sqd for
Sample-based Quantum Diagonalization.

Physical conventions:
- n_orb: number of spatial orbitals (e.g., 6 for LiH in STO-3G)
- n_spin_orb: number of spin orbitals = 2 * n_orb (for NQS sampling)
- n_elec: tuple (n_alpha, n_beta) for alpha/beta electrons
- Spin orbital ordering for NQS: interleaved (1up, 1down, 2up, 2down, ...)
- Spin orbital ordering for SQD: blocked (1up, 2up, ..., 1down, 2down, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MolecularData:
    """Container for molecular integrals and metadata.

    All integrals are in the spatial orbital basis (not spin orbitals).
    The spin orbital indices are handled during sampling and SQD.

    Attributes
    ----------
    hcore : NDArray
        One-electron integrals, shape (n_orb, n_orb).
    eri : NDArray
        Two-electron integrals in chemist notation, shape (n_orb, n_orb, n_orb, n_orb).
    nuclear_repulsion_energy : float
        Nuclear repulsion energy in Hartree.
    n_orb : int
        Number of spatial orbitals.
    n_elec : tuple[int, int]
        Number of (alpha, beta) electrons.
    n_spin_orb : int
        Number of spin orbitals (= 2 * n_orb).
    hf_energy : float
        Hartree-Fock energy (reference).
    fci_energy : float | None
        Full CI energy if available (exact ground state for comparison).
    ccsd_energy : float | None
        CCSD energy if available.
    molecule_name : str
        Name of the molecule (e.g., "H2", "LiH").
    basis : str
        Basis set used (e.g., "sto-3g", "6-31g").
    bond_length : float
        Bond length in Angstrom (for diatomics).
    """

    hcore: NDArray[np.float64]
    eri: NDArray[np.float64]
    nuclear_repulsion_energy: float
    n_orb: int
    n_elec: tuple[int, int]
    n_spin_orb: int
    hf_energy: float
    fci_energy: float | None = None
    ccsd_energy: float | None = None
    molecule_name: str = ""
    basis: str = "sto-3g"
    bond_length: float = 0.0


@dataclass
class H2Config:
    """Configuration for H2 molecule.

    Attributes
    ----------
    bond_length : float
        H-H bond length in Angstrom. Equilibrium is ~0.74 A.
    basis : str
        Basis set. "sto-3g" gives 2 spatial orbitals (4 spin orbitals).
        "6-31g" gives 4 spatial orbitals (8 spin orbitals).
    """

    bond_length: float = 0.74
    basis: str = "sto-3g"


@dataclass
class LiHConfig:
    """Configuration for LiH molecule.

    Attributes
    ----------
    bond_length : float
        Li-H bond length in Angstrom. Equilibrium is ~1.6 A.
    basis : str
        Basis set. "sto-3g" gives 6 spatial orbitals (12 spin orbitals).
    """

    bond_length: float = 1.6
    basis: str = "sto-3g"


@dataclass
class MoleculeConfig:
    """Generic molecule configuration.

    Attributes
    ----------
    name : str
        Molecule name ("H2", "LiH", "H4", "H6").
    bond_length : float
        Bond length in Angstrom.
    basis : str
        Basis set name.
    geometry : list | None
        Custom geometry as list of (atom, (x, y, z)) tuples.
        If None, default geometry is used based on name and bond_length.
    """

    name: Literal["H2", "LiH", "H4", "H6"] = "H2"
    bond_length: float = 0.74
    basis: str = "sto-3g"
    geometry: list | None = None


def _get_default_geometry(
    name: str, bond_length: float
) -> list[tuple[str, tuple[float, float, float]]]:
    """Generate default geometry for common molecules."""
    if name == "H2":
        return [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, bond_length)),
        ]
    elif name == "LiH":
        return [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, bond_length)),
        ]
    elif name == "H4":
        # Linear H4 chain
        return [
            ("H", (0.0, 0.0, i * bond_length))
            for i in range(4)
        ]
    elif name == "H6":
        # Linear H6 chain
        return [
            ("H", (0.0, 0.0, i * bond_length))
            for i in range(6)
        ]
    else:
        raise ValueError(f"Unknown molecule: {name}")


def build_molecular_hamiltonian(config: MoleculeConfig) -> MolecularData:
    """Build molecular Hamiltonian using PySCF.

    This function generates the one-electron (hcore) and two-electron (eri)
    integrals needed for qiskit-addon-sqd, along with reference energies
    from HF, FCI, and CCSD calculations.

    Parameters
    ----------
    config : MoleculeConfig
        Molecule configuration specifying geometry, basis, etc.

    Returns
    -------
    MolecularData
        Container with all molecular integrals and metadata.

    Notes
    -----
    - hcore and eri are in spatial orbital basis
    - For NQS sampling, use n_spin_orb = 2 * n_orb
    - The eri tensor uses chemist notation: (pq|rs)
    """
    from pyscf import gto, scf, fci, cc

    # Build geometry
    if config.geometry is not None:
        geometry = config.geometry
    else:
        geometry = _get_default_geometry(config.name, config.bond_length)

    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = config.basis
    mol.spin = 0  # Singlet state
    mol.charge = 0
    mol.build()

    # Run Hartree-Fock
    mf = scf.RHF(mol)
    mf.verbose = 0
    hf_energy = mf.kernel()

    # Get integrals in MO basis
    # hcore: one-electron integrals
    # eri: two-electron integrals in chemist notation (pq|rs)
    hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = mol.ao2mo(mf.mo_coeff, aosym="s1").reshape(
        mol.nao, mol.nao, mol.nao, mol.nao
    )

    nuclear_repulsion_energy = mol.energy_nuc()
    n_orb = mol.nao
    n_elec = (mol.nelec[0], mol.nelec[1])  # (alpha, beta)
    n_spin_orb = 2 * n_orb

    # Run FCI for reference (exact ground state)
    fci_energy = None
    try:
        cisolver = fci.FCI(mf)
        cisolver.verbose = 0
        fci_energy, _ = cisolver.kernel()
    except Exception:
        pass  # FCI may fail for larger systems

    # Run CCSD for reference
    ccsd_energy = None
    try:
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        ccsd_energy = mycc.e_tot
    except Exception:
        pass  # CCSD may fail

    return MolecularData(
        hcore=hcore,
        eri=eri,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        n_orb=n_orb,
        n_elec=n_elec,
        n_spin_orb=n_spin_orb,
        hf_energy=hf_energy,
        fci_energy=fci_energy,
        ccsd_energy=ccsd_energy,
        molecule_name=config.name,
        basis=config.basis,
        bond_length=config.bond_length,
    )


def build_h2_hamiltonian(config: H2Config) -> MolecularData:
    """Build H2 Hamiltonian.

    Convenience wrapper for H2 molecule.

    Parameters
    ----------
    config : H2Config
        H2 configuration.

    Returns
    -------
    MolecularData
        Molecular integrals and metadata.
    """
    mol_config = MoleculeConfig(
        name="H2",
        bond_length=config.bond_length,
        basis=config.basis,
    )
    return build_molecular_hamiltonian(mol_config)


def build_lih_hamiltonian(config: LiHConfig) -> MolecularData:
    """Build LiH Hamiltonian.

    Convenience wrapper for LiH molecule. LiH in STO-3G basis gives
    6 spatial orbitals = 12 spin orbitals, which is the "12-bit" encoding
    mentioned in the project goals.

    Parameters
    ----------
    config : LiHConfig
        LiH configuration.

    Returns
    -------
    MolecularData
        Molecular integrals and metadata.

    Notes
    -----
    Reference energies for LiH @ 0.8 A (STO-3G):
    - HF:      ~-7.616 Ha
    - FCI:     ~-7.634 Ha (target for SQD)
    - CCSD:    ~-7.634 Ha
    """
    mol_config = MoleculeConfig(
        name="LiH",
        bond_length=config.bond_length,
        basis=config.basis,
    )
    return build_molecular_hamiltonian(mol_config)


# Backward compatibility alias
def build_h2_hamiltonian_12bit(cfg: H2Config) -> MolecularData:
    """Backward compatibility wrapper.

    Note: H2 in STO-3G is only 4 spin orbitals (4-bit), not 12-bit.
    For true 12-bit, use LiH in STO-3G (12 spin orbitals).
    """
    return build_h2_hamiltonian(cfg)


def print_molecular_info(data: MolecularData) -> None:
    """Print molecular information for debugging."""
    print(f"\n{'='*50}")
    print(f"Molecule: {data.molecule_name}")
    print(f"Basis: {data.basis}")
    print(f"Bond length: {data.bond_length:.3f} A")
    print(f"{'='*50}")
    print(f"Spatial orbitals (n_orb): {data.n_orb}")
    print(f"Spin orbitals (n_spin_orb): {data.n_spin_orb}")
    print(f"Electrons (alpha, beta): {data.n_elec}")
    print(f"Total electrons: {sum(data.n_elec)}")
    print(f"{'='*50}")
    print(f"Nuclear repulsion: {data.nuclear_repulsion_energy:.6f} Ha")
    print(f"HF energy:         {data.hf_energy:.6f} Ha")
    if data.fci_energy is not None:
        print(f"FCI energy:        {data.fci_energy:.6f} Ha")
    if data.ccsd_energy is not None:
        print(f"CCSD energy:       {data.ccsd_energy:.6f} Ha")
    print(f"{'='*50}\n")
