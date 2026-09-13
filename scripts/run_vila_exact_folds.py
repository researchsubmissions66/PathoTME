#!/usr/bin/env python3
"""Run exact-coverage ViLa/TME folds in one process and summarize them."""
from __future__ import annotations

import argparse
from pathlib import Path

import run_vila_fusion
import summarize_vila_exact_folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 2, 3])
    args = parser.parse_args()

    for fold in args.folds:
        status = run_vila_fusion.main([
            "--experiment-config", str(args.experiment_config),
            "--fold", str(fold),
            "--device", args.device,
        ])
        if status:
            return status
    return summarize_vila_exact_folds.main([
        "--results-root", str(args.results_root),
        "--folds", *map(str, args.folds),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
