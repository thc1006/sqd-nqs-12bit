"""Variational Monte Carlo (VMC) training for Neural Quantum States.

This module implements:
1. Local energy calculation for fermionic Hamiltonians
2. Stochastic Reconfiguration (SR) optimization
3. VMC training loop with early stopping

Physical background:
- VMC minimizes E[ψ] = <ψ|H|ψ>/<ψ|ψ> variationally
- Local energy: E_L(x) = <x|H|ψ>/<x|ψ> satisfies E[E_L] = E[ψ]
- SR is a second-order optimizer that uses the quantum Fisher matrix
- SR update: δθ = S^(-1) * grad, where S_ij = <O_i* O_j> - <O_i*><O_j>

References:
- Sorella, PRB 64, 024512 (2001) - Stochastic Reconfiguration
- Carleo & Troyer, Science 355, 602 (2017) - NQS + VMC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from .ffn_nqs import FFNNNQS, efficient_parallel_sampler
from ..sqd_interface.hamiltonian import MolecularData


@dataclass
class VMCConfig:
    """Configuration for VMC training.

    Attributes
    ----------
    n_samples : int
        Total number of samples per epoch.
    n_chains : int
        Number of parallel MCMC chains.
    burn_in_steps : int
        MCMC burn-in steps.
    step_interval : int
        Steps between collected samples.
    learning_rate : float
        Initial learning rate for SR.
    sr_regularization : float
        Regularization for SR matrix inversion (diagonal shift).
    n_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience.
    lr_schedule : str
        Learning rate schedule: "constant", "cosine", "exp".
    """

    n_samples: int = 1000
    n_chains: int = 256
    burn_in_steps: int = 100
    step_interval: int = 10
    learning_rate: float = 0.01
    sr_regularization: float = 1e-4
    n_epochs: int = 500
    patience: int = 100
    lr_schedule: str = "cosine"


@dataclass
class TrainingResult:
    """Result of VMC training.

    Attributes
    ----------
    best_energy : float
        Best energy achieved during training.
    best_state_dict : dict
        Model state dict at best energy.
    energy_history : list[float]
        Energy at each epoch.
    std_history : list[float]
        Energy standard error at each epoch.
    """

    best_energy: float
    best_state_dict: dict
    energy_history: list[float]
    std_history: list[float]


def local_energy_batch(
    model: FFNNNQS,
    samples: torch.Tensor,
    hcore: np.ndarray,
    eri: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Compute local energies for a batch of samples.

    The local energy is defined as:
        E_L(x) = sum_{x'} H_{x,x'} * ψ(x') / ψ(x)

    For a fermionic Hamiltonian in second quantization:
        H = sum_{pq} h_pq a^†_p a_q + 0.5 * sum_{pqrs} g_pqrs a^†_p a^†_q a_s a_r

    where h = hcore and g = eri (electron repulsion integrals).

    Parameters
    ----------
    model : FFNNNQS
        The NQS model.
    samples : torch.Tensor
        Batch of configurations, shape (batch, n_spin_orb) with ±1 values.
    hcore : np.ndarray
        One-electron integrals in MO basis, shape (n_orb, n_orb).
    eri : np.ndarray
        Two-electron integrals in chemist notation, shape (n_orb, n_orb, n_orb, n_orb).
    device : torch.device
        Computation device.

    Returns
    -------
    local_energies : torch.Tensor
        Local energies for each sample, shape (batch,).

    Notes
    -----
    This implementation assumes:
    - Spin-orbital ordering: interleaved (1up, 1down, 2up, 2down, ...)
    - ±1 encoding: +1 = occupied, -1 = unoccupied
    - Real-valued NQS (no phase)
    """
    batch_size = samples.shape[0]
    n_spin_orb = samples.shape[1]
    n_orb = n_spin_orb // 2

    # Convert ±1 to occupation (0/1): occ = (sample + 1) / 2
    occupations = (samples + 1) / 2  # (batch, n_spin_orb), values in {0, 1}

    # Compute log|ψ(x)| for current samples
    log_psi = model(samples)  # (batch,)

    # Initialize local energies
    local_energies = torch.zeros(batch_size, device=device, dtype=torch.float64)

    # Convert integrals to torch
    hcore_t = torch.tensor(hcore, device=device, dtype=torch.float64)
    eri_t = torch.tensor(eri, device=device, dtype=torch.float64)

    # --- One-electron terms: sum_{pq} h_pq <x|a^†_p a_q|ψ>/ψ(x) ---
    # For diagonal: <x|a^†_p a_p|x> = n_p (occupation of orbital p)
    # For off-diagonal: <x|a^†_p a_q|x'> where x' = a^†_p a_q |x>

    for p_spatial in range(n_orb):
        for q_spatial in range(n_orb):
            h_pq = hcore_t[p_spatial, q_spatial]
            if abs(h_pq) < 1e-12:
                continue

            # Both alpha and beta spin contributions
            for spin in [0, 1]:  # 0=alpha (even), 1=beta (odd)
                p = 2 * p_spatial + spin  # spin orbital index
                q = 2 * q_spatial + spin

                if p == q:
                    # Diagonal: h_pp * n_p
                    local_energies += h_pq * occupations[:, p]
                else:
                    # Off-diagonal: h_pq * <x|a^†_p a_q|ψ>/ψ(x)
                    # This is non-zero only if q is occupied and p is empty in x
                    # Then x' = x with p occupied and q empty
                    mask = (occupations[:, q] > 0.5) & (occupations[:, p] < 0.5)

                    if mask.any():
                        # Create x' by flipping p and q
                        samples_new = samples.clone()
                        samples_new[:, p] = samples[:, q]  # copy occupation
                        samples_new[:, q] = -samples[:, q]  # flip q to unoccupied

                        # For fermionic sign: count occupied orbitals between p and q
                        min_idx, max_idx = min(p, q), max(p, q)
                        sign = torch.ones(batch_size, device=device, dtype=torch.float64)
                        for k in range(min_idx + 1, max_idx):
                            sign *= (1 - 2 * occupations[:, k])  # -1 if occupied

                        log_psi_new = model(samples_new).double()
                        ratio = torch.exp(log_psi_new - log_psi.double())

                        local_energies += torch.where(
                            mask,
                            h_pq * sign * ratio,
                            torch.zeros_like(local_energies)
                        )

    # --- Two-electron terms: 0.5 * sum_{pq} (J_pq - K_pq * delta_spin) * n_p * n_q ---
    # In chemist notation: J_pq = (pp|qq) = eri[p,p,q,q], K_pq = (pq|qp) = eri[p,q,q,p]
    # Exchange only for same-spin electrons

    # Iterate over spin orbitals
    for p in range(n_spin_orb):
        for q in range(n_spin_orb):
            if p == q:
                continue  # Skip self-interaction

            p_sp = p // 2  # spatial orbital index
            q_sp = q // 2
            p_spin = p % 2  # 0=alpha, 1=beta
            q_spin = q % 2

            # Coulomb integral (pp|qq) - always present
            J = eri_t[p_sp, p_sp, q_sp, q_sp]

            # Exchange integral (pq|qp) - only for same spin
            if p_spin == q_spin:
                K = eri_t[p_sp, q_sp, q_sp, p_sp]
            else:
                K = 0.0

            local_energies += 0.5 * (J - K) * occupations[:, p] * occupations[:, q]

    return local_energies.float()


def stochastic_reconfiguration_update(
    model: FFNNNQS,
    samples: torch.Tensor,
    hcore: np.ndarray,
    eri: np.ndarray,
    lr: float,
    reg: float,
    device: torch.device,
) -> float:
    """Perform one Stochastic Reconfiguration (SR) update step.

    SR minimizes the energy using a natural gradient approach:
        δθ = -lr * S^(-1) * grad_E

    where S is the quantum Fisher information matrix:
        S_ij = <O_i* O_j> - <O_i*><O_j>
        O_i = ∂log(ψ)/∂θ_i

    Parameters
    ----------
    model : FFNNNQS
        The NQS model to update.
    samples : torch.Tensor
        Batch of configurations, shape (batch, n_spin_orb) with ±1 values.
    hcore : np.ndarray
        One-electron integrals.
    eri : np.ndarray
        Two-electron integrals.
    lr : float
        Learning rate.
    reg : float
        Regularization for matrix inversion (diagonal shift).
    device : torch.device
        Computation device.

    Returns
    -------
    energy : float
        Mean energy after update.
    """
    model.train()
    batch_size = samples.shape[0]

    # Compute local energies (no grad needed for this)
    with torch.no_grad():
        local_energies = local_energy_batch(model, samples, hcore, eri, device)
        E_mean = local_energies.mean()
        E_centered = local_energies - E_mean

    # Compute log derivatives O_i = ∂log(ψ)/∂θ_i
    # We need gradients w.r.t. parameters for each sample
    log_psi = model(samples)  # (batch,)

    # Get parameter gradients for each sample
    n_params = sum(p.numel() for p in model.parameters())
    O_matrix = torch.zeros(batch_size, n_params, device=device)

    for i in range(batch_size):
        model.zero_grad()
        log_psi[i].backward(retain_graph=(i < batch_size - 1))

        grad_flat = []
        for p in model.parameters():
            if p.grad is not None:
                grad_flat.append(p.grad.view(-1).clone())
            else:
                grad_flat.append(torch.zeros(p.numel(), device=device))
        O_matrix[i] = torch.cat(grad_flat)

    # Compute S matrix: S_ij = <O_i O_j> - <O_i><O_j>
    O_mean = O_matrix.mean(dim=0)  # (n_params,)
    O_centered = O_matrix - O_mean  # (batch, n_params)

    S = (O_centered.T @ O_centered) / batch_size  # (n_params, n_params)

    # Regularization: S -> S + reg * I
    S += reg * torch.eye(n_params, device=device)

    # Compute gradient of energy: grad_E = 2 * Re[<O* (E_L - <E_L>)>]
    # For real NQS: grad_E = 2 * <O (E_L - <E_L>)>
    grad_E = 2.0 * (O_centered.T @ E_centered) / batch_size  # (n_params,)

    # Solve S @ delta = -grad_E for delta
    try:
        delta = torch.linalg.solve(S, -lr * grad_E)
    except Exception:
        # Fallback to pseudoinverse if singular
        delta = -lr * torch.linalg.lstsq(S, grad_E).solution

    # Apply update to parameters
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.add_(delta[idx:idx + numel].view_as(p))
            idx += numel

    return E_mean.item()


def adjust_lr(
    initial_lr: float,
    epoch: int,
    schedule_type: str,
    T_max: int,
    decay_rate: float = 0.98,
) -> float:
    """Adjust learning rate according to schedule.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate.
    epoch : int
        Current epoch.
    schedule_type : str
        Schedule type: "constant", "cosine", "exp".
    T_max : int
        Total number of epochs (for cosine schedule).
    decay_rate : float
        Decay rate per epoch (for exp schedule).

    Returns
    -------
    lr : float
        Adjusted learning rate.
    """
    if schedule_type == "cosine":
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / T_max))
    elif schedule_type == "exp":
        return initial_lr * (decay_rate ** epoch)
    else:
        return initial_lr


def train_nqs_vmc(
    model: FFNNNQS,
    mol_data: MolecularData,
    config: VMCConfig,
    device: torch.device,
    verbose: bool = True,
) -> TrainingResult:
    """Train NQS model using VMC with Stochastic Reconfiguration.

    Parameters
    ----------
    model : FFNNNQS
        The NQS model to train.
    mol_data : MolecularData
        Molecular data with integrals.
    config : VMCConfig
        Training configuration.
    device : torch.device
        Computation device.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    result : TrainingResult
        Training results including best energy and model state.
    """
    hcore = mol_data.hcore
    eri = mol_data.eri
    n_spin_orb = mol_data.n_spin_orb

    best_energy = float("inf")
    best_state_dict = None
    no_improve_count = 0

    energy_history = []
    std_history = []

    n_samples_per_chain = config.n_samples // config.n_chains

    for epoch in range(config.n_epochs):
        # Adjust learning rate
        lr = adjust_lr(
            config.learning_rate,
            epoch,
            config.lr_schedule,
            config.n_epochs,
        )

        # Sample from current model
        samples = efficient_parallel_sampler(
            model=model,
            n_samples_per_chain=n_samples_per_chain,
            n_chains=config.n_chains,
            n_visible=n_spin_orb,
            burn_in_steps=config.burn_in_steps,
            step_interval=config.step_interval,
            device=device,
        )

        # SR update
        energy = stochastic_reconfiguration_update(
            model=model,
            samples=samples,
            hcore=hcore,
            eri=eri,
            lr=lr,
            reg=config.sr_regularization,
            device=device,
        )

        # Evaluate
        with torch.no_grad():
            eval_local_energies = local_energy_batch(model, samples, hcore, eri, device)
            eval_mean = eval_local_energies.mean().item()
            eval_std = eval_local_energies.std().item() / np.sqrt(len(eval_local_energies))

        # Add nuclear repulsion
        eval_mean_total = eval_mean + mol_data.nuclear_repulsion_energy

        energy_history.append(eval_mean_total)
        std_history.append(eval_std)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1:4d}] Energy: {eval_mean_total:.6f} ± {eval_std:.6f} Ha (lr={lr:.6f})")

        # Early stopping
        if eval_mean_total < best_energy:
            best_energy = eval_mean_total
            best_state_dict = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= config.patience:
            if verbose:
                print(f"[Early Stopping] No improvement for {config.patience} epochs.")
            break

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            print(f"[Training Complete] Best energy: {best_energy:.6f} Ha")

    return TrainingResult(
        best_energy=best_energy,
        best_state_dict=best_state_dict,
        energy_history=energy_history,
        std_history=std_history,
    )
