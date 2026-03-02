import torch
import torch.nn as nn
import torch.nn.functional as F


def squared_euclidean_cost(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise squared Euclidean distances.
    Args:
        z: (B, d) tensor of data embeddings
        c: (K, d) tensor of prototype locations
    Returns:
        (B, K) cost matrix
    """
    # Using torch.cdist for efficiency (computes ||z||^2 + ||c||^2 - 2z^Tc)
    return torch.cdist(z, c, p=2) ** 2


def cosine_cost(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise cosine cost (1 - cosine_similarity).
    Args:
        z: (B, d) tensor of data embeddings
        c: (K, d) tensor of prototype locations
    Returns:
        (B, K) cost matrix
    """
    z_norm = F.normalize(z, p=2, dim=-1)
    c_norm = F.normalize(c, p=2, dim=-1)
    sim = torch.mm(z_norm, c_norm.t())
    return 1.0 - sim


class MahalanobisCost(nn.Module):
    """
    Computes pairwise Mahalanobis distance with a learnable PSD matrix.
    M = L @ L^T where L is lower-triangular.
    Cost = (z_i - c_j)^T M (z_i - c_j)
         = ||L^T z_i - L^T c_j||_2^2
    """

    def __init__(self, d: int):
        super().__init__()
        # Initialise L as the identity matrix
        self.L = nn.Parameter(torch.eye(d))

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Enforce lower-triangular structure for strict Cholesky,
        # though simply doing z @ L is also valid for PSD M = L L^T.
        L_tril = torch.tril(self.L)
        
        # Project inputs
        z_proj = z @ L_tril
        c_proj = c @ L_tril
        
        # Compute squared Euclidean distance in projected space
        return squared_euclidean_cost(z_proj, c_proj)
