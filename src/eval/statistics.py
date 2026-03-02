"""
Statistical tests for comparing clustering methods.

Implements Wilcoxon signed-rank, Friedman + Nemenyi posthoc, and
critical difference diagram generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def wilcoxon_pairwise(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: Scores for method A across seeds/datasets.
        scores_b: Scores for method B across seeds/datasets.

    Returns:
        Dict with ``"statistic"`` and ``"p_value"``.
    """
    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float]:
    """Paired t-test for evaluating direct superiority.
    
    Returns:
        Dict with ``"statistic"`` and ``"p_value"``.
    """
    stat, p = stats.ttest_rel(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p)}


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Computes Cohen's d effect size for paired samples."""
    diff = scores_a - scores_b
    return float(np.mean(diff) / np.std(diff, ddof=1))


def friedman_nemenyi(
    score_matrix: np.ndarray,
    method_names: list[str] | None = None,
) -> dict[str, Any]:
    """Friedman test + Nemenyi posthoc for multiple methods.

    Args:
        score_matrix: Shape ``(n_datasets, n_methods)``. Each row is one
            dataset, each column is one method's score.
        method_names: Optional method names.

    Returns:
        Dict with ``"friedman_stat"``, ``"friedman_p"``, ``"avg_ranks"``,
        ``"nemenyi_cd"`` (critical difference at alpha=0.05).
    """
    n_datasets, n_methods = score_matrix.shape

    # Rank methods per dataset (higher score = lower rank = better)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        order = np.argsort(-score_matrix[i])  # descending
        for rank_val, idx in enumerate(order):
            ranks[i, idx] = rank_val + 1

    avg_ranks = ranks.mean(axis=0)

    # Friedman test
    stat, p = stats.friedmanchisquare(*[score_matrix[:, j] for j in range(n_methods)])

    # Nemenyi critical difference
    # CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    # q_alpha values for Nemenyi at alpha=0.05 (approximation)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(n_methods, 2.569)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    result = {
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "avg_ranks": avg_ranks.tolist(),
        "nemenyi_cd": float(cd),
    }
    if method_names:
        result["method_names"] = method_names

    return result


def critical_difference_diagram(
    avg_ranks: list[float] | np.ndarray,
    method_names: list[str],
    cd: float,
    title: str = "Critical Difference Diagram",
    save_path: str | None = None,
) -> plt.Figure:
    """Generate a critical difference diagram.

    Methods whose average rank difference is less than ``cd`` are connected
    by a horizontal bar, indicating no significant difference.

    Args:
        avg_ranks: Average rank per method.
        method_names: Names of methods.
        cd: Nemenyi critical difference.
        title: Plot title.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure.
    """
    avg_ranks = np.array(avg_ranks)
    n = len(method_names)
    sorted_idx = np.argsort(avg_ranks)

    fig, ax = plt.subplots(1, 1, figsize=(10, 2 + n * 0.3))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-0.5, n * 0.6 + 1)
    ax.set_xlabel("Average Rank", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_xaxis()
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Draw axis
    ax.axhline(y=0, color="black", linewidth=0.8)

    # Place methods
    for i, idx in enumerate(sorted_idx):
        rank = avg_ranks[idx]
        y_pos = -0.3
        ax.plot(rank, 0, "ko", markersize=6)
        side = "left" if i < n // 2 else "right"
        offset = 0.3 + (i % (n // 2 + 1)) * 0.5
        ax.annotate(
            f"{method_names[idx]} ({rank:.2f})",
            xy=(rank, 0),
            xytext=(rank, offset),
            fontsize=9,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    # Draw CD bar
    ax.plot([1, 1 + cd], [-0.2, -0.2], "r-", linewidth=2)
    ax.text(1 + cd / 2, -0.35, f"CD={cd:.2f}", ha="center", fontsize=9, color="red")

    # Draw cliques (groups that are not significantly different)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(avg_ranks[sorted_idx[i]] - avg_ranks[sorted_idx[j]]) < cd:
                r1 = avg_ranks[sorted_idx[i]]
                r2 = avg_ranks[sorted_idx[j]]
                y_line = -0.15 - 0.05 * i
                ax.plot([r1, r2], [y_line, y_line], "b-", linewidth=2, alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
