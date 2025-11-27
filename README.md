# sqd-nqs-12bit

Sample-efficient Sample-based Quantum Diagonalization (SQD) with feed-forward
Neural Quantum State (NQS) samplers for small molecules (starting from H₂)
under 12–14-bit encodings.

This repository is designed to be used together with **Claude Code** (Opus 4.5)
as an agentic research assistant. The key idea is to:

- use an FFNN-based NQS as a **classical sampler** that generates bitstring
  configurations for a molecular Hamiltonian,
- feed these samples into **`qiskit-addon-sqd`** to perform Sample-based Quantum
  Diagonalization,
- study how **few samples** (and imperfect samplers) can still recover accurate
  ground-state energies (e.g. pushing from ~ -5.6 Ha toward ~ -7.63 Ha).

## Quick start

1. Create a virtual environment (example using `uv`):

   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies in editable mode:

   ```bash
   uv pip install -e .
   ```

   If you are not using `uv`, you can instead do:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. Open the project with **Claude Code** from this folder. Claude will read
   `CLAUDE.md`, discover `skills/` and `.claude/commands/`, and can help you
   implement the missing pieces step by step.

4. Run the (currently stub) main experiment script:

   ```bash
   uv run python -m src.experiments.h2_12bit_small_sample      --config configs/h2_12bit_nqs.yaml
   ```

   This script currently only prints placeholders and checks that imports work.
   You can ask Claude Code to flesh it out into a real experiment.

## Project layout

See `CLAUDE.md` for a detailed breakdown of directories and workflows.
