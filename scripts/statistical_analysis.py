#!/usr/bin/env python3
"""
Statistical Analysis for LOTC Benchmark Results.

Loads YAML results, computes mean±std, runs Wilcoxon signed-rank tests,
and generates LaTeX-ready tables.

Usage:
    python scripts/statistical_analysis.py --results_dir experiments/results/full_benchmark
"""

import argparse
import os
import sys
import yaml
import numpy as np
from scipy import stats
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> dict:
    """Load all YAML result files from a directory."""
    all_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.yaml') and fname != 'all_results.yaml':
            config_key = fname.replace('.yaml', '')
            with open(os.path.join(results_dir, fname)) as f:
                all_results[config_key] = yaml.load(f, Loader=yaml.UnsafeLoader)
    return all_results


def extract_metric(runs: list[dict], metric: str = 'ARI') -> np.ndarray:
    """Extract a metric array from a list of run dicts, skipping errors."""
    vals = [r[metric] for r in runs if metric in r and 'error' not in r]
    return np.array(vals, dtype=np.float64)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test. Returns p-value."""
    if len(a) < 5 or len(b) < 5:
        return float('nan')
    # Pair-wise on matching seeds
    n = min(len(a), len(b))
    diffs = a[:n] - b[:n]
    if np.allclose(diffs, 0):
        return 1.0
    try:
        _, p = stats.wilcoxon(diffs, alternative='two-sided')
        return p
    except ValueError:
        return float('nan')


def generate_summary(all_results: dict, metric: str = 'ARI'):
    """Generate summary table with mean±std and significance tests."""
    print(f"\n{'='*80}")
    print(f" Summary Table: {metric}")
    print(f"{'='*80}")

    for config_key, methods_data in sorted(all_results.items()):
        print(f"\n--- {config_key} ---")
        print(f"{'Method':<20} {'Mean':<10} {'Std':<10} {'N':<5} {'vs LOTC (p)':<12}")
        print("-" * 60)

        lotc_vals = None
        if 'lotc' in methods_data:
            lotc_vals = extract_metric(methods_data['lotc'], metric)

        for method, runs in sorted(methods_data.items()):
            vals = extract_metric(runs, metric)
            if len(vals) == 0:
                print(f"{method:<20} {'N/A':<10} {'N/A':<10} {0:<5}")
                continue

            mean_val = np.mean(vals)
            std_val = np.std(vals)
            n = len(vals)

            p_str = ""
            if lotc_vals is not None and method != 'lotc' and len(lotc_vals) >= 5:
                p_val = wilcoxon_test(lotc_vals, vals)
                if not np.isnan(p_val):
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    p_str = f"{p_val:.4f} {sig}"

            print(f"{method:<20} {mean_val:<10.4f} {std_val:<10.4f} {n:<5} {p_str}")


def generate_latex_table(all_results: dict, metric: str = 'ARI'):
    """Generate LaTeX-ready table."""
    print(f"\n{'='*80}")
    print(f" LaTeX Table: {metric}")
    print(f"{'='*80}")

    for config_key, methods_data in sorted(all_results.items()):
        print(f"\n% --- {config_key} ---")
        methods = sorted(methods_data.keys())

        # Find best method
        best_method = None
        best_mean = -1
        for method in methods:
            vals = extract_metric(methods_data[method], metric)
            if len(vals) > 0 and np.mean(vals) > best_mean:
                best_mean = np.mean(vals)
                best_method = method

        for method in methods:
            vals = extract_metric(methods_data[method], metric)
            if len(vals) == 0:
                continue
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if method == best_method:
                print(f"{method} & $\\mathbf{{{mean_val:.3f} \\pm {std_val:.3f}}}$ \\\\")
            else:
                print(f"{method} & ${mean_val:.3f} \\pm {std_val:.3f}$ \\\\")


def generate_runtime_table(all_results: dict):
    """Generate runtime/memory comparison table."""
    print(f"\n{'='*80}")
    print(f" Runtime & Memory Comparison")
    print(f"{'='*80}")
    print(f"{'Config':<30} {'Method':<20} {'Runtime(s)':<12} {'Memory(MB)':<12} {'Params':<12}")
    print("-" * 90)

    for config_key, methods_data in sorted(all_results.items()):
        for method, runs in sorted(methods_data.items()):
            runtimes = [r.get('runtime_s', float('nan')) for r in runs if 'error' not in r]
            memories = [r.get('peak_mem_mb', 0) for r in runs if 'error' not in r]
            params = [r.get('n_params', 0) for r in runs if 'error' not in r]

            if runtimes:
                rt = np.mean(runtimes)
                mem = np.mean(memories) if any(m > 0 for m in memories) else 0
                n_p = params[0] if params and params[0] > 0 else 0
                print(f"{config_key:<30} {method:<20} {rt:<12.2f} {mem:<12.1f} {n_p:<12}")


def generate_mass_analysis(all_results: dict):
    """Analyze mass collapse frequency."""
    print(f"\n{'='*80}")
    print(f" Mass Collapse Analysis (LOTC only)")
    print(f"{'='*80}")

    for config_key, methods_data in sorted(all_results.items()):
        if 'lotc' not in methods_data:
            continue

        runs = [r for r in methods_data['lotc'] if 'error' not in r]
        if not runs:
            continue

        mass_mins = [r.get('mass_min', float('nan')) for r in runs]
        mass_entropies = [r.get('mass_entropy', float('nan')) for r in runs]

        K = 10  # default
        collapse_threshold = 1.0 / (10 * K)
        n_collapsed = sum(1 for m in mass_mins if m < collapse_threshold)

        print(f"{config_key}: {n_collapsed}/{len(runs)} runs with collapsed masses "
              f"(min(alpha) < {collapse_threshold:.4f})")
        print(f"  Mass entropy: {np.mean(mass_entropies):.4f} ± {np.std(mass_entropies):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis")
    parser.add_argument('--results_dir', type=str,
                        default='experiments/results/full_benchmark')
    parser.add_argument('--metric', type=str, default='ARI')
    args = parser.parse_args()

    all_results = load_results(args.results_dir)

    if not all_results:
        print(f"No results found in {args.results_dir}")
        return

    for metric in ['ARI', 'NMI', 'ACC']:
        generate_summary(all_results, metric)

    generate_latex_table(all_results, args.metric)
    generate_runtime_table(all_results)
    generate_mass_analysis(all_results)


if __name__ == '__main__':
    main()
