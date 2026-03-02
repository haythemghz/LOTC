import torch
import torch.nn as nn
from src.models.encoders import IdentityEncoder, MLPEncoder
from src.models.lotc_model import LOTCModel

def test_identity_encoder():
    x = torch.randn(10, 5)
    enc = IdentityEncoder()
    assert torch.allclose(enc(x), x)
    
def test_mlp_encoder():
    enc = MLPEncoder(input_dim=10, hidden_dims=[32, 16], output_dim=5)
    x = torch.randn(8, 10)
    out = enc(x)
    assert out.shape == (8, 5)

def test_lotc_model_forward():
    B = 16
    d0 = 20
    d = 8
    K = 5
    
    encoder = MLPEncoder(input_dim=d0, hidden_dims=[16], output_dim=d)
    model = LOTCModel(encoder, num_prototypes=K, embed_dim=d, cost_type='squared_euclidean')
    
    x = torch.randn(B, d0)
    
    # Forward pass without Laplacian graph
    out = model(x, epsilon=0.1, sinkhorn_iter=10, lambda_mass=0.1, lambda_disp=0.1)
    
    # Check outputs exist and have correct shape
    assert out['z'].shape == (B, d)
    assert out['P'].shape == (B, K)
    assert out['hard_assignments'].shape == (B,)
    assert out['soft_assignments'].shape == (B, K)
    
    # Check losses
    assert out['ot_cost'] > 0
    assert out['reg_mass'] < 0
    assert out['reg_disp'] >= 0  # L2 dispersion is positive
    assert out['total_loss'].requires_grad
    
    # Check probabilities
    assert torch.allclose(out['soft_assignments'].sum(dim=1), torch.ones(B))
    
def test_lotc_model_mahalanobis():
    encoder = IdentityEncoder()
    model = LOTCModel(encoder, num_prototypes=4, embed_dim=3, cost_type='mahalanobis')
    
    x = torch.randn(10, 3)
    out = model(x, epsilon=0.05, sinkhorn_iter=5)
    
    # The initial Cholesky is Identity, so cost should match squared Euclidean initially
    z = out['z']
    c = model.prototypes.prototypes
    
    from src.ot.costs import squared_euclidean_cost
    C_euc = squared_euclidean_cost(z, c)
    C_mah = model.compute_cost_matrix(z, c)
    
    assert torch.allclose(C_euc, C_mah)
