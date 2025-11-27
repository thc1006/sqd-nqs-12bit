"""Neural Quantum State model definitions and training utilities."""

from .ffn_nqs import (
    FFNNNQS,
    FFNNNQSConfig,
    FFNN,
    efficient_parallel_sampler,
)

from .vmc_training import (
    VMCConfig,
    TrainingResult,
    local_energy_batch,
    stochastic_reconfiguration_update,
    adjust_lr,
    train_nqs_vmc,
)

from .utils import (
    init_weights_xavier,
    count_parameters,
)

from .gpu_optimized import (
    batched_mcmc_sampler,
    vectorized_local_energy,
    optimized_sr_update,
    AMPTrainer,
    enable_tf32,
    benchmark_mcmc,
)

__all__ = [
    # Models
    "FFNNNQS",
    "FFNNNQSConfig",
    "FFNN",
    # Sampling
    "efficient_parallel_sampler",
    # Training
    "VMCConfig",
    "TrainingResult",
    "local_energy_batch",
    "stochastic_reconfiguration_update",
    "adjust_lr",
    "train_nqs_vmc",
    # Utilities
    "init_weights_xavier",
    "count_parameters",
    # GPU Optimized
    "batched_mcmc_sampler",
    "vectorized_local_energy",
    "optimized_sr_update",
    "AMPTrainer",
    "enable_tf32",
    "benchmark_mcmc",
]
