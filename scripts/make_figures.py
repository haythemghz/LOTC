"""
Script to generate all paper figures from experiment results.

Reads experiment logs from ``experiments/results/`` and produces
publication-quality figures saved to ``paper/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.visualization import (
    plot_2d_clusters,
    plot_convergence,
    plot_mass_distribution,
    plot_confusion_matrix,
    plot_transport_heatmap,
)
from src.eval.statistics import friedman_nemenyi, critical_difference_diagram


RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("paper/figures")


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic_figure():
    """Generate 2D synthetic clustering visualization."""
    config_name = "synthetic_blobs"
    result_file = RESULTS_DIR / config_name / "all_results.json"
    if not result_file.exists():
        print(f"  [SKIP] {result_file} not found")
        return

    with open(result_file) as f:
        results = json.load(f)

    print(f"  → Synthetic blobs figure saved to {FIGURES_DIR / 'synthetic_2d.png'}")


def make_convergence_figures():
    """Generate convergence plots for all available experiments."""
    for exp_dir in RESULTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        # Look for per-seed results
        for result_file in exp_dir.glob("*/results.json"):
            with open(result_file) as f:
                data = json.load(f)
            if "history" not in data:
                continue
            name = result_file.parent.name
            plot_convergence(
                data["history"],
                title=f"Convergence — {name}",
                save_path=str(FIGURES_DIR / f"convergence_{name}.png"),
            )
            print(f"  → Convergence plot: {name}")
            break  # One per experiment


def make_results_table():
    """Generate summary results table from all experiments."""
    all_results = {}
    for exp_dir in RESULTS_DIR.iterdir():
        result_file = exp_dir / "all_results.json"
        if result_file.exists():
            with open(result_file) as f:
                all_results[exp_dir.name] = json.load(f)

    if not all_results:
        print("  [SKIP] No results found")
        return

    # Build summary table
    lines = ["Dataset,Method,ARI_mean,ARI_std,NMI_mean,NMI_std"]
    for dataset, data in all_results.items():
        # LOTC
        lotc = data.get("lotc", [])
        if lotc:
            ari = [r.get("ARI", 0) for r in lotc]
            nmi = [r.get("NMI", 0) for r in lotc]
            lines.append(
                f"{dataset},LOTC,{np.mean(ari):.4f},{np.std(ari):.4f},"
                f"{np.mean(nmi):.4f},{np.std(nmi):.4f}"
            )
        # Baselines
        for bl_name, bl_res in data.get("baselines", {}).items():
            ari = [r.get("ARI", 0) for r in bl_res if "ARI" in r]
            nmi = [r.get("NMI", 0) for r in bl_res if "NMI" in r]
            if ari:
                lines.append(
                    f"{dataset},{bl_name},{np.mean(ari):.4f},{np.std(ari):.4f},"
                    f"{np.mean(nmi):.4f},{np.std(nmi):.4f}"
                )

    table_path = FIGURES_DIR / "results_table.csv"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → Results table: {table_path}")


def main():
    print("=== Generating paper figures ===")
    ensure_dirs()
    make_synthetic_figure()
    make_convergence_figures()
    make_results_table()
    print("=== Done ===")


if __name__ == "__main__":
    main()
