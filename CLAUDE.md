# Project Context

You are collaborating with Ting-Yi (蔡秀吉) on a quantum chemistry research project.
The goal is to study **sample-efficient Sample-based Quantum Diagonalization (SQD)**
using **Neural Quantum States (NQS) based on feed-forward neural networks** as classical
samplers, primarily for small molecules like H₂ under **12–14-bit** representations.

## About This Project

- Programming language: Python 3.10+
- Core libraries:
  - Qiskit and `qiskit-addon-sqd` for Sample-based Quantum Diagonalization (SQD)
  - PyTorch for feed-forward NQS (FFNN-based amplitude / log-ψ models)
  - NumPy / SciPy / Matplotlib / Pandas for analysis and plotting
- Hardware target: single NVIDIA RTX 4090 GPU on Linux / WSL2

**Important research constraints:**

- Do **NOT** use adaptive basis methods.
- Avoid trivial "system comparison" papers; focus on **sample-efficiency**, **bias/variance**,
  and **post-processing that pushes estimates from ~ -5.6 Ha toward ~ -7.63 Ha** with as few
  samples as possible.
- Emphasize clear ablations:
  - NQS vs. baseline samplers
  - different sample budgets (e.g. 1e2, 1e3, 1e4, …)
  - different bit-depth / qubit counts when relevant

Whenever you propose changes, keep alignment with these goals.

## Key Directories

- `src/nqs_models/`
  - `ffn_nqs.py` : FFNN-based NQS models (real-valued log-ψ or amplitude network)
  - `utils.py`   : sampling utilities, parameter initializers, reweighting helpers
- `src/sqd_interface/`
  - `hamiltonian.py`      : builds molecular Hamiltonians (H₂, H₄, …) and 12-bit mappings
  - `sqd_runner.py`       : thin wrapper around `qiskit-addon-sqd` pipelines
  - `sampling_adapters.py`: adapters exposing a simple sampler API to SQD
- `src/experiments/`
  - `h2_12bit_small_sample.py`      : main experiment for H₂ @ 12-bit, few-sample regime
  - `h_chain_scaling.py`            : optional scaling experiments (H₄ / H₆)
  - `ablation_nqs_vs_baseline.py`   : head-to-head sampler comparison
- `configs/`
  - YAML configs describing molecule, basis mapping, sampler hyperparameters, SQD options
- `data/`
  - `molecule_integrals/` : optional precomputed molecular integrals or reference data
  - `cached_samples/`     : re-usable sample sets to avoid recomputation
- `results/`
  - `raw/`       : raw logs, JSON/NPZ dumps of runs
  - `processed/` : aggregated CSV/Parquet summaries
  - `figures/`   : final plots for papers / slides
- `notebooks/`
  - Jupyter notebooks used for sanity checks and exploratory analysis

## Standards

- Use **type hints** on all public functions.
- Prefer simple, readable code over clever one-liners.
- All core logic should eventually be covered by smoke tests where feasible.
- Document **all** non-trivial math assumptions and approximations in comments or docstrings.
- Use docstrings to explain:
  - physical meaning of parameters (e.g., what "12-bit" means in your encoding),
  - how FFNN output is converted into probability distributions for sampling.

## Common Commands

```bash
# Create virtual environment (example using uv)
uv venv
source .venv/bin/activate

# Install project (editable)
uv pip install -e .

# Run the main H2 12-bit NQS vs baseline experiment (currently a stub)
uv run python -m src.experiments.h2_12bit_small_sample   --config configs/h2_12bit_nqs.yaml

# Run quick baseline SQD without NQS (currently a stub)
uv run python -m src.experiments.h2_12bit_small_sample   --config configs/h2_12bit_baseline.yaml --no-nqs
```

If you need to discover available scripts, first run:

```bash
ls -R src/experiments
```

and then inspect the file(s) before modifying them.

## How to Work with This Repo (Claude Workflow)

When working on this repository, follow this workflow:

1. **Read context first**
   - Skim `README.md`, `src/experiments/` files, and relevant configs.
   - For theory questions, prefer `/nqs-theory-review` custom command.
2. **Plan before coding**
   - For any non-trivial change, first propose a brief plan in bullet points.
   - Highlight how the change affects:
     - sample complexity,
     - numerical stability,
     - GPU memory footprint on a single 4090.
3. **Small, iterative diffs**
   - Make changes in small, reviewable patches.
   - When touching numerical code, include a quick numerical sanity check
     (e.g. confirming known limits or very small test systems).
4. **Be explicit about stochasticity**
   - Always specify random seeds when relevant.
   - Clearly distinguish between:
     - Monte Carlo noise,
     - approximation error from the NQS model,
     - SQD algorithmic error.

## MCP Configuration

This project may use MCP servers configured via `.mcp.json` in the repository root.
When you see available tools, prefer:

- a **python-exec** / **code-exec** style server for running experiments and scripts in
  a controlled way (instead of inventing ad-hoc shell commands),
- a **shell** server only for simple file system operations (listing files, moving results).

Before relying on these tools, briefly inspect `.mcp.json` to understand what servers
exist and which commands they expose.

## Notes & Warnings

- Do **NOT** commit secrets, API keys, or proprietary credentials.
- When summarizing or rewriting research notes, keep both:
  - physics / quantum chemistry correctness, and
  - computational constraints (12–14 bits, single 4090) in mind.
- When proposing new experiments, explicitly state:
  - approximate compute cost,
  - expected difficulty of implementation,
  - what new scientific insight it would add beyond existing runs.
