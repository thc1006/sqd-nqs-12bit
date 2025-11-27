"""FFNN-based Neural Quantum State (NQS) models with Metropolis MCMC sampling.

This module implements:
1. FFNNNQS: Feed-forward neural network for log-amplitude parameterization
2. Metropolis MCMC sampling with parallel chains
3. GPU-accelerated sampling for RTX 4090

Physical conventions:
- Input encoding: ±1 (not 0/1), where -1 = unoccupied, +1 = occupied
- Output: log|ψ(x)| (real-valued log-amplitude)
- Probability: p(x) ∝ |ψ(x)|² = exp(2 * log|ψ(x)|)

References:
- Carleo & Troyer, Science 355, 602 (2017) - original NQS paper
- Hibat-Allah et al., PRR 2, 023358 (2020) - autoregressive NQS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FFNNNQSConfig:
    """Configuration for FFNN-based NQS.

    Attributes
    ----------
    n_visible : int
        Number of visible units (spin orbitals). For LiH in STO-3G, this is 12.
    n_hidden : int
        Hidden layer width. Typically alpha * n_visible where alpha ~ 2-4.
    n_layers : int
        Number of hidden layers. 2-3 layers usually sufficient.
    activation : str
        Activation function: "relu", "tanh", "gelu", "silu".
    use_bias : bool
        Whether to use bias in linear layers.
    """

    n_visible: int
    n_hidden: int = 64
    n_layers: int = 2
    activation: str = "tanh"
    use_bias: bool = True


class FFNNNQS(nn.Module):
    """Feed-forward Neural Quantum State with Metropolis MCMC sampling.

    This model parameterizes log|ψ(x)| where x ∈ {-1, +1}^n is a spin configuration.
    The probability distribution is p(x) ∝ |ψ(x)|² = exp(2 * log|ψ(x)|).

    Architecture:
        Input (n_visible) -> [Linear -> Activation] × n_layers -> Linear -> Output (1)

    Notes
    -----
    - Uses ±1 encoding (not 0/1) to match standard physics convention
    - For VMC training, we need gradients of log|ψ| w.r.t. parameters
    - The network is real-valued; complex phase could be added for sign problem
    """

    def __init__(self, config: FFNNNQSConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self._device = device or torch.device("cpu")

        # Build network layers
        layers = []
        in_dim = config.n_visible

        # Select activation function
        activation_fn = self._get_activation(config.activation)

        for _ in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.n_hidden, bias=config.use_bias))
            layers.append(activation_fn())
            in_dim = config.n_hidden

        # Output layer: scalar log-amplitude
        layers.append(nn.Linear(in_dim, 1, bias=config.use_bias))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(self._device)

    def _get_activation(self, name: str) -> type:
        """Get activation function class by name."""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform for tanh-like activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log-amplitude log|ψ(x)| for a batch of configurations.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, n_visible) with ±1 values.

        Returns
        -------
        log_psi : torch.Tensor
            Tensor of shape (batch,) with real-valued log-amplitudes.
        """
        # Ensure float type for network
        x = x.float()
        out = self.net(x)
        return out.squeeze(-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probability log p(x) ∝ 2 * log|ψ(x)| (unnormalized).

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, n_visible) with ±1 values.

        Returns
        -------
        log_prob : torch.Tensor
            Tensor of shape (batch,) with unnormalized log probabilities.
        """
        return 2.0 * self.forward(x)

    @torch.no_grad()
    def sample_mcmc(
        self,
        n_samples_per_chain: int,
        n_chains: int,
        burn_in_steps: int = 100,
        step_interval: int = 10,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from p(x) ∝ |ψ(x)|² using parallel Metropolis MCMC.

        This implements the standard Metropolis-Hastings algorithm with
        single-site flip proposals, running multiple chains in parallel
        for efficient GPU utilization.

        Parameters
        ----------
        n_samples_per_chain : int
            Number of samples to collect from each chain (after burn-in).
        n_chains : int
            Number of parallel MCMC chains.
        burn_in_steps : int
            Number of burn-in steps before collecting samples.
        step_interval : int
            Steps between collected samples (for decorrelation).
        device : torch.device, optional
            Device for computation. Defaults to model's device.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples_per_chain * n_chains, n_visible) with ±1 values.

        Notes
        -----
        - Proposal: flip a randomly chosen spin (single-site flip)
        - Acceptance: Metropolis criterion min(1, p(x')/p(x))
        - All chains are run in parallel using vectorized operations
        """
        device = device or self._device
        n_visible = self.config.n_visible

        # Initialize chains randomly with ±1 values
        # Shape: (n_chains, n_visible)
        states = 2 * torch.randint(0, 2, (n_chains, n_visible), device=device).float() - 1

        # Compute initial log probabilities
        log_probs = self.log_prob(states)  # (n_chains,)

        # Burn-in phase
        for _ in range(burn_in_steps):
            states, log_probs = self._mcmc_step(states, log_probs, device)

        # Collection phase
        samples = []
        for i in range(n_samples_per_chain):
            # Take step_interval steps between samples
            for _ in range(step_interval):
                states, log_probs = self._mcmc_step(states, log_probs, device)
            samples.append(states.clone())

        # Stack all samples: (n_samples_per_chain, n_chains, n_visible)
        # Then reshape to (n_samples_per_chain * n_chains, n_visible)
        samples = torch.stack(samples, dim=0)  # (n_samples_per_chain, n_chains, n_visible)
        samples = samples.view(-1, n_visible)  # (total_samples, n_visible)

        return samples

    def _mcmc_step(
        self,
        states: torch.Tensor,
        log_probs: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one Metropolis MCMC step for all chains in parallel.

        Parameters
        ----------
        states : torch.Tensor
            Current states, shape (n_chains, n_visible).
        log_probs : torch.Tensor
            Current log probabilities, shape (n_chains,).
        device : torch.device
            Computation device.

        Returns
        -------
        new_states : torch.Tensor
            Updated states after Metropolis step.
        new_log_probs : torch.Tensor
            Updated log probabilities.
        """
        n_chains = states.shape[0]
        n_visible = states.shape[1]

        # Propose: randomly select one spin to flip for each chain
        flip_idx = torch.randint(0, n_visible, (n_chains,), device=device)

        # Create proposed states by flipping selected spins
        proposed = states.clone()
        proposed[torch.arange(n_chains, device=device), flip_idx] *= -1

        # Compute log probabilities of proposed states
        proposed_log_probs = self.log_prob(proposed)

        # Metropolis acceptance criterion
        # Accept if log(p_new) - log(p_old) > log(uniform)
        log_accept_ratio = proposed_log_probs - log_probs
        log_uniform = torch.log(torch.rand(n_chains, device=device))
        accept = log_accept_ratio > log_uniform

        # Update states and log probs where accepted
        # Use where for vectorized conditional update
        new_states = torch.where(accept.unsqueeze(-1), proposed, states)
        new_log_probs = torch.where(accept, proposed_log_probs, log_probs)

        return new_states, new_log_probs

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Convenience wrapper for MCMC sampling with default parameters.

        Parameters
        ----------
        n_samples : int
            Total number of samples desired.
        device : torch.device, optional
            Device for computation.

        Returns
        -------
        samples : torch.Tensor
            Tensor of shape (n_samples, n_visible) with ±1 values.
        """
        # Use reasonable defaults
        n_chains = min(256, n_samples)
        n_samples_per_chain = (n_samples + n_chains - 1) // n_chains

        samples = self.sample_mcmc(
            n_samples_per_chain=n_samples_per_chain,
            n_chains=n_chains,
            burn_in_steps=100,
            step_interval=10,
            device=device,
        )

        # Return exactly n_samples
        return samples[:n_samples]


def efficient_parallel_sampler(
    model: FFNNNQS,
    n_samples_per_chain: int,
    n_chains: int,
    n_visible: int,
    burn_in_steps: int,
    step_interval: int,
    device: torch.device,
) -> torch.Tensor:
    """Efficient parallel MCMC sampler matching notebook interface.

    This function provides the same interface as the `efficient_parallel_sampler`
    function used in the reference notebook (NQS-SQD-Qiskit.ipynb).

    Parameters
    ----------
    model : FFNNNQS
        The NQS model to sample from.
    n_samples_per_chain : int
        Number of samples per chain.
    n_chains : int
        Number of parallel chains.
    n_visible : int
        Number of visible units (not used, taken from model).
    burn_in_steps : int
        Burn-in steps before collecting.
    step_interval : int
        Steps between collected samples.
    device : torch.device
        Computation device.

    Returns
    -------
    samples : torch.Tensor
        Tensor of shape (n_samples_per_chain * n_chains, n_visible) with ±1 values.
    """
    return model.sample_mcmc(
        n_samples_per_chain=n_samples_per_chain,
        n_chains=n_chains,
        burn_in_steps=burn_in_steps,
        step_interval=step_interval,
        device=device,
    )


# Alias for backward compatibility with notebook code
FFNN = FFNNNQS
