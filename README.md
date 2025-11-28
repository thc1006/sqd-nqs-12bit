# sqd-nqs-12bit

Sample-efficient Sample-based Quantum Diagonalization (SQD) with feed-forward
Neural Quantum State (NQS) samplers for small molecules under 12-bit encodings.

## Key Finding

**NQS training quality and SQD accuracy are negatively correlated.**

In 12-bit systems (LiH, H4, H6), uniform random sampling (Baseline) consistently
outperforms well-trained NQS samplers. This counterintuitive result suggests that
SQD success depends on sample diversity rather than sample quality.

| Molecule | Method | SQD Error (mHa) | Conservation Ratio |
|----------|--------|-----------------|-------------------|
| LiH | NQS | 312.18 | 6.5% |
| LiH | Baseline | 65.38 | 5.1% |
| H4 | NQS | 156.07 | 35.7% |
| H4 | Baseline | ~0 | 14.5% |
| H6 | NQS | 385.77 | 25.1% |
| H6 | Baseline | ~0 | 9.8% |

## Experiment Summary

- **Total experiments**: 1788 runs across LiH, H4, H6
- **Systems tested**: 8-12 spin orbitals, 4-6 electrons
- **Parameters scanned**: epochs (50-500), samples (100-5000), bond lengths (1.0-3.0 A)

Full results: [`results/figures/experiment_report.md`](results/figures/experiment_report.md)

## Quick Start

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run full experiment pipeline
python scripts/run_full_research_plan.py

# Generate visualizations
python scripts/generate_phase_diagram.py
python scripts/generate_missing_figures.py
```

## Project Structure

```
sqd-nqs-12bit/
├── src/
│   ├── nqs_models/          # FFNN NQS, VMC training, GPU optimization
│   ├── sqd_interface/       # PySCF integrals, qiskit-addon-sqd wrapper
│   └── experiments/         # Experiment runners
├── scripts/
│   ├── run_full_research_plan.py   # Main experiment runner
│   ├── generate_phase_diagram.py   # Visualization
│   └── hackmd_sync.py              # HackMD integration
├── results/
│   ├── phase_diagram/       # Raw JSON data
│   └── figures/             # Generated plots and reports
├── DEVELOPMENT_PLAN.md      # Research progress tracking
└── CLAUDE.md                # Project guide for Claude Code
```

## Generated Figures

| Figure | Description |
|--------|-------------|
| `vmc_vs_sqd_scatter.png` | VMC vs SQD error (negative correlation) |
| `training_analysis.png` | Training epochs effect on accuracy |
| `sample_efficiency.png` | Sample count vs SQD error |
| `hchain_scaling.png` | H4/H6 system scaling |
| `conservation_distributions.png` | Conservation ratio histograms |
| `heatmap_analysis.png` | Multi-parameter heatmaps |

## Requirements

- Python 3.12+
- PyTorch 2.5+ (CUDA 12.1)
- qiskit-addon-sqd
- PySCF
- NVIDIA RTX 4090 (recommended)

## References

- [qiskit-addon-sqd](https://qiskit.github.io/qiskit-addon-sqd/)
- [SKQD](https://arxiv.org/html/2501.09702v1)
- [SQD Limitations](https://arxiv.org/html/2501.07231v1)
- [Fermionic NQS](https://www.nature.com/articles/s41467-020-15724-9)
