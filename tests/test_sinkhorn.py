import torch
import pytest
import ot  # Python Optimal Transport (POT)
from src.ot.sinkhorn import sinkhorn_log_domain
from src.ot.costs import squared_euclidean_cost

def test_sinkhorn_marginals():
    # Setup toy data
    B, K = 5, 3
    z = torch.randn(B, 2)
    c = torch.randn(K, 2)
    C = squared_euclidean_cost(z, c)
    
    u = torch.ones(B) / B
    v = torch.softmax(torch.randn(K), dim=0)
    
    P, cost = sinkhorn_log_domain(C, u, v, epsilon=0.01, max_iter=100)
    
    # Check shape
    assert P.shape == (B, K)
    
    # Check marginals
    assert torch.allclose(P.sum(dim=1), u, atol=1e-3)
    assert torch.allclose(P.sum(dim=0), v, atol=1e-3)
    
def test_sinkhorn_cost_matches_pot():
    # Compare againstPOT library
    B, K = 4, 3
    z = torch.randn(B, 2)
    c = torch.randn(K, 2)
    C = squared_euclidean_cost(z, c)
    
    u = torch.ones(B) / B
    v = torch.ones(K) / K
    
    epsilon = 0.05
    P, cost = sinkhorn_log_domain(C, u, v, epsilon=epsilon, max_iter=200)
    
    # POT comparison
    C_np = C.numpy()
    u_np = u.numpy()
    v_np = v.numpy()
    P_pot = ot.sinkhorn(u_np, v_np, C_np, reg=epsilon, method='sinkhorn_log')
    
    # Assert plan is close
    assert torch.allclose(P, torch.tensor(P_pot, dtype=torch.float32), atol=1e-3)
    
def test_sinkhorn_differentiability():
    # Verify gradients flow through unrolled iterations
    B, K = 3, 2
    z = torch.randn(B, 2, requires_grad=True)
    c = torch.randn(K, 2, requires_grad=True)
    
    # Learnable logits for v
    logits = torch.randn(K, requires_grad=True)
    
    # Forward pass
    u = torch.ones(B) / B
    v = torch.softmax(logits, dim=0)
    C = squared_euclidean_cost(z, c)
    
    P, cost = sinkhorn_log_domain(C, u, v, epsilon=0.1, max_iter=50)
    
    # Backward pass
    cost.backward()
    
    # Check gradients are populated
    assert z.grad is not None
    assert c.grad is not None
    assert logits.grad is not None
    
    # Gradients should not be strictly zero
    assert torch.norm(z.grad) > 0
    assert torch.norm(c.grad) > 0
    assert torch.norm(logits.grad) > 0
