# Project Context

You are collaborating with  (蔡秀吉) on a quantum chemistry research project.
The goal is to study **sample-efficient Sample-based Quantum Diagonalization (SQD)**
using **Neural Quantum States (NQS) based on feed-forward neural networks** as classical
samplers, primarily for small molecules like H₂ under **12–14-bit** representations.

## Development Plan & Progress Tracking

**開發計畫文件**: [`DEVELOPMENT_PLAN.md`](./DEVELOPMENT_PLAN.md)

斷線恢復時，請先讀取 `DEVELOPMENT_PLAN.md` 查看當前進度。

實作時請務必保持程式碼自然，不要讓人看出是 AI 生成。此外，實作也請務必遵循 Code Readability，記住代碼是寫給人看的，所以要有自然的代碼流程

### 關鍵參考檔案 (已驗證有效)

> 暫時權限密碼：Kinoshita15@

所有紀錄用的文檔都必須要精簡但保留所有必要的關鍵的資訊，遵循 TLDR，同時不可以有任何的 emoji


以下兩個檔案包含**已驗證成功**的完整流程 (LiH → FCI 精度 -7.634167 Ha):

| 檔案 | 說明 | 優先級 |
|------|------|--------|
| `nqs_sqd_qiskit.py` | Notebook 導出的 .py 版本，包含完整訓練+採樣+SQD流程 | **最高** |
| `NQS-SQD-Qiskit.ipynb` | Jupyter notebook，有輸出結果可驗證 | **最高** |

**移植代碼時必須參考這兩個檔案**，它們包含:
- `efficient_parallel_sampler()`: Metropolis MCMC 採樣
- `stochastic_reconfiguration_update()`: SR 訓練
- `local_energy_batch()`: 局域能量計算
- 守恆條件篩選邏輯 (N_elec, S_z)
- BitArray 格式轉換
- `diagonalize_fermionic_hamiltonian()` 調用

## About This Project

- Programming language: Python 3.12+
- **Package manager: uv** (preferred over pip for speed and reproducibility)
- Core libraries:
  - Qiskit 2.2+ and `qiskit-addon-sqd` for Sample-based Quantum Diagonalization (SQD)
  - PyTorch 2.5+ (with CUDA 12.1) for feed-forward NQS (FFNN-based amplitude / log-ψ models)
  - NumPy / SciPy / Matplotlib / Pandas for analysis and plotting
- Hardware target: single NVIDIA RTX 4090 GPU on Linux
- Search tool: ripgrep (`rg`) for fast code search

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

## Repository Structure

```
sqd-nqs-12bit/
├── CLAUDE.md                     # Project guide (this file)
├── DEVELOPMENT_PLAN.md           # Research phases and progress
├── plan.md                       # PI meeting notes
├── pyproject.toml                # Project dependencies
│
├── src/
│   ├── nqs_models/
│   │   ├── ffn_nqs.py            # FFNN NQS model (log-psi network)
│   │   ├── vmc_training.py       # VMC trainer with SR (FIXED: ERI indexing)
│   │   └── gpu_optimized.py      # GPU-optimized local energy (FIXED: chemist notation)
│   │
│   ├── sqd_interface/
│   │   ├── hamiltonian.py        # PySCF molecular integrals (H2, LiH, H4, H6)
│   │   └── sqd_runner.py         # qiskit-addon-sqd wrapper
│   │
│   └── experiments/
│       ├── h2_12bit_small_sample.py
│       ├── ablation_nqs_vs_baseline.py
│       └── h_chain_scaling.py
│
├── scripts/
│   ├── run_full_research_plan.py # Full experiment runner (Phase 0-3)
│   ├── generate_phase_diagram.py # Visualization generation
│   └── plot_results.py           # Legacy plotting
│
├── configs/
│   ├── h2_12bit_nqs.yaml
│   ├── h2_12bit_baseline.yaml
│   ├── ablation_config.yaml
│   └── lih_12bit_ablation.yaml
│
├── results/
│   ├── raw/                      # Raw JSON experiment dumps
│   ├── ablation/                 # Ablation study results
│   ├── epoch_sweep/              # Epoch sweep data
│   ├── phase_diagram/            # Phase diagram data (Phase 0-3)
│   ├── scaling/                  # H-chain scaling results
│   ├── figures/                  # Generated plots (PNG)
│   └── EXPERIMENT_SUMMARY.md     # Results summary with bug fixes
│
└── notebooks/
    └── NQS-SQD-Qiskit.ipynb      # Reference notebook (external vmc_cal.py)
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/nqs_models/vmc_training.py` | VMC training with SR | Fixed (chemist notation) |
| `src/nqs_models/gpu_optimized.py` | GPU local energy | Fixed (chemist notation) |
| `src/sqd_interface/hamiltonian.py` | Molecular integrals | Working |
| `src/sqd_interface/sqd_runner.py` | SQD pipeline | Working |
| `scripts/run_full_research_plan.py` | Phase 0-3 runner | New |
| `scripts/generate_phase_diagram.py` | Visualization | New |

## Chemist vs Physicist Notation

PySCF uses chemist notation for electron repulsion integrals (ERI):

| Integral | Chemist Notation | PySCF Index |
|----------|------------------|-------------|
| Coulomb J_pq | (pp\|qq) | `eri[p,p,q,q]` |
| Exchange K_pq | (pq\|qp) | `eri[p,q,q,p]` |

Previously fixed bugs in `vmc_training.py:195-221` and `gpu_optimized.py:190-197`.

## Standards

- Use **type hints** on all public functions.
- Prefer simple, readable code over clever one-liners.
- All core logic should eventually be covered by smoke tests where feasible.
- Document **all** non-trivial math assumptions and approximations in comments or docstrings.
- Use docstrings to explain:
  - physical meaning of parameters (e.g., what "12-bit" means in your encoding),
  - how FFNN output is converted into probability distributions for sampling.

## Environment Setup

```bash
# uv and ripgrep are installed in ~/.local/bin/
# Add to PATH if needed:
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version      # uv 0.9.13
rg --version      # ripgrep 14.1.1
```

## Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies (use uv, NOT pip)
uv pip install -e ".[dev]"

# Add a new dependency
uv pip install <package>

# Run the main H2 12-bit NQS vs baseline experiment (currently a stub)
uv run python -m src.experiments.h2_12bit_small_sample --config configs/h2_12bit_nqs.yaml

# Run quick baseline SQD without NQS (currently a stub)
uv run python -m src.experiments.h2_12bit_small_sample --config configs/h2_12bit_baseline.yaml --no-nqs

# Fast code search with ripgrep
rg "pattern" src/            # search in src/
rg -t py "def train"         # search only Python files
rg -l "NQS"                   # list files containing "NQS"
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
