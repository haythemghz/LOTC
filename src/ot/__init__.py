"""Optimal Transport building blocks: Sinkhorn solver and cost functions."""

from .sinkhorn import sinkhorn_log_domain
from .costs import squared_euclidean_cost, cosine_cost, MahalanobisCost

__all__ = [
    "sinkhorn_log_domain",
    "squared_euclidean_cost",
    "cosine_cost",
    "MahalanobisCost",
]
