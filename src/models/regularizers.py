import torch
import torch.nn.functional as F


def mass_entropy_reg(masses: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative entropy of the mass distribution.
    Minimising this encourages balanced mass distribution (maximum entropy).
    Args:
        masses: (K,) tensor of probabilities (softmax normalised)
    Returns:
        Scalar penalty: \sum_j \alpha_j \log \alpha_j
    """
    return torch.sum(masses * torch.log(masses + 1e-12))


def mass_kl_reg(masses: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
    """
    KL divergence from masses to a prior distribution.
    When prior=None, defaults to uniform (equivalent to negative entropy).
    For imbalanced data, prior can be set to the empirical class frequencies.
    
    Args:
        masses: (K,) tensor of predicted mass probabilities
        prior: (K,) tensor of prior probabilities (default: uniform)
    Returns:
        KL(masses || prior) scalar
    """
    if prior is None:
        prior = torch.ones_like(masses) / len(masses)
    prior = prior.to(masses.device)
    # KL(q || p) = sum q * log(q/p)
    return F.kl_div(
        prior.log(),   # log p  (input)
        masses,         # q      (target)
        reduction='sum',
        log_target=False
    )


def dispersion_reg_l2(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Computes L2 shrinkage on prototypes.
    """
    return torch.sum(prototypes ** 2)


def dispersion_reg_collision(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Computes a repulsive collision penalty between all pairs of prototypes.
    Returns a POSITIVE value that DECREASES as prototypes move apart.
    Minimising this pushes prototypes apart.
    
    We use 1/(d^2 + eps) so the penalty is large when prototypes are close
    and small when they are far apart.
    """
    K = prototypes.size(0)
    if K <= 1:
        return torch.tensor(0.0, device=prototypes.device)
        
    distances_sq = torch.cdist(prototypes, prototypes, p=2) ** 2
    # Sum over strictly upper triangular part
    idx = torch.triu_indices(K, K, offset=1)
    pairwise_sq = distances_sq[idx[0], idx[1]]
    # Inverse-distance repulsion: large when close, small when far
    return torch.sum(1.0 / (pairwise_sq + 1e-4))


def graph_laplacian_reg(prototypes: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Computes the graph Laplacian regularisation c^T L_G c.
    Args:
        prototypes: (K, d) tensor
        W: (K, K) symmetric adjacency matrix for the prototype graph
    Returns:
        Scalar penalty \sum_{j,k} W_{jk} ||c_j - c_k||^2
    """
    # L_G = D - W
    D = torch.diag(torch.sum(W, dim=1))
    L_G = D - W
    
    # Compute Tr(C^T L_G C)
    res = torch.trace(prototypes.t() @ L_G @ prototypes)
    return res
