import torch

def sinkhorn_log_domain(
    C: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 50,
    return_plan: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the entropic Optimal Transport plan and cost in the log-domain.
    
    Args:
        C: Cost matrix of shape (B, K)
        u: Source marginals of shape (B,)
        v: Target marginals of shape (K,)
        epsilon: Entropic regularization parameter
        max_iter: Number of Sinkhorn iterations
        return_plan: If True, returns (plan, cost). If False, returns (f, g) dual potentials.
        
    Returns:
        P: Transport plan (if return_plan=True)
        cost: Scalar unregularized cost <P, C> (if return_plan=True)
        (f, g): Potentials (if return_plan=False)
    """
    B, K = C.shape
    u = u.view(-1)
    v = v.view(-1)
    
    # Initialize potentials in log domain
    device = C.device
    dtype = C.dtype
    
    f = torch.zeros(B, dtype=dtype, device=device)
    g = torch.zeros(K, dtype=dtype, device=device)
    
    log_u = torch.log(u + 1e-12)
    log_v = torch.log(v + 1e-12)
    
    C_eps = C / epsilon
    
    for _ in range(max_iter):
        term_f = -C_eps + g.view(1, K)
        f = log_u - torch.logsumexp(term_f, dim=1)
        
        term_g = -C_eps + f.view(B, 1)
        g = log_v - torch.logsumexp(term_g, dim=0)
        
        # Center potentials for numerical stability
        center = f.mean()
        f = f - center
        g = g + center
        
    if not return_plan:
        return f, g

    # Final transport plan with clamping
    log_P = f.view(B, 1) + g.view(1, K) - C_eps
    P = torch.exp(log_P).clamp(min=1e-12, max=1.0)
    cost = torch.sum(P * C)
    
    return P, cost

def sinkhorn_divergence(
    C_uv: torch.Tensor,
    C_uu: torch.Tensor,
    C_vv: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 50
) -> torch.Tensor:
    """
    Computes the Sinkhorn Divergence with stability centering.
    """
    f_uv, g_uv = sinkhorn_log_domain(C_uv, u, v, epsilon, max_iter, return_plan=False)
    ot_uv = epsilon * (torch.sum(u * f_uv) + torch.sum(v * g_uv))
    
    f_uu, g_uu = sinkhorn_log_domain(C_uu, u, u, epsilon, max_iter, return_plan=False)
    ot_uu = epsilon * (torch.sum(u * f_uu) + torch.sum(u * g_uu))
    
    f_vv, g_vv = sinkhorn_log_domain(C_vv, v, v, epsilon, max_iter, return_plan=False)
    ot_vv = epsilon * (torch.sum(v * f_vv) + torch.sum(v * g_vv))
    
    return (ot_uv - 0.5 * ot_uu - 0.5 * ot_vv).clamp(min=0.0)
