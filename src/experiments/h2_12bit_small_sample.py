"""Main experiment: H2 @ 12-bit, few-sample regime (stub).

This script is intentionally minimal so that you can extend it with Claude Code.
For now, it:

- parses basic CLI flags,
- loads a config YAML file,
- constructs a dummy sampler and runs a dummy SQD call,
- prints the result.

You should replace the placeholders with real Hamiltonian construction,
training loops, and analysis.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import yaml

from src.sqd_interface.hamiltonian import H2Config, build_h2_hamiltonian_12bit
from src.sqd_interface.sampling_adapters import BaselineBernoulliSampler
from src.sqd_interface.sqd_runner import run_sqd_on_samples


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="H2 12-bit NQS vs baseline experiment (stub)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h2_12bit_baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--no-nqs",
        action="store_true",
        help="Ignore any NQS-related settings and use a baseline sampler.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Loaded config from {args.config}:") 
    print(cfg)

    # Placeholder: construct a dummy H2 Hamiltonian config.
    h2_cfg = H2Config(
        bond_length=cfg.get("molecule", {}).get("bond_length", 0.74),
        bit_depth=cfg.get("encoding", {}).get("bits", 12),
    )
    try:
        hamiltonian = build_h2_hamiltonian_12bit(h2_cfg)
    except NotImplementedError:
        hamiltonian = None
        print("[WARN] build_h2_hamiltonian_12bit is not implemented yet; using None as placeholder.")

    n_visible = cfg.get("encoding", {}).get("bits", 12)
    n_samples = cfg.get("sampling", {}).get("n_samples", 256)

    sampler = BaselineBernoulliSampler(n_visible=n_visible)
    print(f"[INFO] Drawing {n_samples} baseline samples with n_visible={n_visible}.")
    samples = sampler.sample(n_samples=n_samples)

    result = run_sqd_on_samples(hamiltonian=hamiltonian, samples=samples)
    print("[RESULT] Dummy SQD result (stub):", result)
    print("[NOTE] You can now ask Claude Code to replace placeholders with real logic.")


if __name__ == "__main__":
    main()
