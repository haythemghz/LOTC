"""
Visualization utilities for LOTC experiments.

Generates publication-quality plots for clusters, transport plans,
convergence, mass distributions, and confusion matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns


def plot_2d_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray | None = None,
    P: np.ndarray | None = None,
    title: str = "LOTC Clustering",
    save_path: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    show_transport: bool = True,
    max_arrows: int = 200,
) -> plt.Figure:
    """Plot 2D clusters with prototype locations and transport arrows.

    Args:
        X: Data points, shape ``(n, 2)``.
        labels: Cluster labels, shape ``(n,)``.
        prototypes: Prototype locations, shape ``(K, 2)``.
        P: Transport plan, shape ``(n, K)``. Used for transport arrows.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        show_transport: Whether to draw transport arrows.
        max_arrows: Max number of arrows to draw (random subsample).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Unique labels
    unique_labels = np.unique(labels)
    K = len(unique_labels)
    colors = cm.Set2(np.linspace(0, 1, max(K, 8)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[colors[i % len(colors)]],
            s=15, alpha=0.6, label=f"Cluster {label}",
            edgecolors="none",
        )

    # Draw transport arrows
    if show_transport and P is not None and prototypes is not None:
        n = X.shape[0]
        K_proto = prototypes.shape[0]
        # Subsample for readability
        if n > max_arrows:
            idx = np.random.choice(n, max_arrows, replace=False)
        else:
            idx = np.arange(n)
        for i in idx:
            j = np.argmax(P[i])
            weight = P[i, j]
            if weight > 1e-6:
                ax.annotate(
                    "",
                    xy=prototypes[j],
                    xytext=X[i],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        alpha=min(weight * n * 0.3, 0.4),
                        lw=0.5,
                    ),
                )

    # Plot prototypes
    if prototypes is not None:
        ax.scatter(
            prototypes[:, 0], prototypes[:, 1],
            c="red", marker="*", s=300, edgecolors="black",
            linewidths=1.5, zorder=10, label="Prototypes",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_convergence(
    history: list[dict[str, Any]],
    metrics: list[str] | None = None,
    title: str = "Training Convergence",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training loss and metrics over epochs.

    Args:
        history: List of per-epoch log dicts (from trainer).
        metrics: List of metric keys to plot (e.g., ["ARI", "NMI"]).
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    if metrics is None:
        metrics = ["ARI", "NMI"]

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]

    n_plots = 1 + len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(epochs, losses, "b-", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].grid(True, alpha=0.3)

    # Metric curves
    for i, metric in enumerate(metrics):
        vals = [(h["epoch"], h[metric]) for h in history if metric in h]
        if vals:
            ep, val = zip(*vals)
            axes[i + 1].plot(ep, val, "o-", linewidth=1.5, markersize=3)
            axes[i + 1].set_xlabel("Epoch")
            axes[i + 1].set_ylabel(metric)
            axes[i + 1].set_title(metric)
            axes[i + 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_mass_distribution(
    learned_masses: np.ndarray,
    true_sizes: np.ndarray | None = None,
    title: str = "Learned Mass Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing learned masses to true cluster sizes.

    Args:
        learned_masses: Learned prototype masses, shape ``(K,)``.
        true_sizes: True cluster proportions, shape ``(K,)``.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    K = len(learned_masses)
    x = np.arange(K)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, learned_masses, width, label="Learned", color="steelblue")
    if true_sizes is not None:
        ax.bar(x + width / 2, true_sizes, width, label="True", color="coral", alpha=0.7)
    ax.set_xlabel("Prototype Index")
    ax.set_ylabel("Mass / Proportion")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Uses the Hungarian algorithm to align predicted labels with true labels
    for best visual correspondence.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        save_path: Path to save figure.

    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    # Build confusion matrix
    cm_raw = confusion_matrix(y_true, y_pred)

    # Hungarian matching for best alignment
    row_ind, col_ind = linear_sum_assignment(-cm_raw)
    # Reorder columns
    cm_aligned = cm_raw[:, col_ind]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_aligned, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_transport_heatmap(
    P: np.ndarray,
    title: str = "Transport Plan Heatmap",
    save_path: str | None = None,
    max_rows: int = 100,
) -> plt.Figure:
    """Heatmap of the transport plan matrix.

    Args:
        P: Transport plan, shape ``(n, K)``.
        title: Plot title.
        save_path: Path to save.
        max_rows: Maximum number of data points to display.

    Returns:
        Matplotlib figure.
    """
    if P.shape[0] > max_rows:
        idx = np.random.choice(P.shape[0], max_rows, replace=False)
        idx.sort()
        P = P[idx]

    fig, ax = plt.subplots(figsize=(6, max(4, P.shape[0] * 0.05)))
    sns.heatmap(P, cmap="YlOrRd", ax=ax, xticklabels=True)
    ax.set_xlabel("Prototype Index")
    ax.set_ylabel("Data Point")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
