import torch
from src.models.prototypes import PrototypeModule
from src.models.regularizers import mass_entropy_reg, dispersion_reg_l2, dispersion_reg_collision, graph_laplacian_reg

def test_prototype_module():
    K, d = 3, 2
    model = PrototypeModule(K, d)
    
    # Check default initialisation
    assert model.prototypes.shape == (K, d)
    assert model.mass_logits.shape == (K,)
    
    # Check softmax masses
    masses = model.masses
    assert torch.allclose(masses.sum(), torch.tensor(1.0))
    assert torch.all(masses > 0)
    
    # Check K-Means init
    data = torch.randn(10, d)
    model.init_from_kmeans(data)
    
    # After init, logits should be zero again
    assert torch.allclose(model.mass_logits, torch.zeros(K))
    assert torch.allclose(model.masses, torch.ones(K) / K)

def test_regularizers():
    K, d = 3, 2
    masses = torch.ones(K) / K
    
    # Entropy should be negative and scalar
    ent = mass_entropy_reg(masses)
    assert ent.dim() == 0
    assert ent < 0
    
    prototypes = torch.randn(K, d)
    disp1 = dispersion_reg_l2(prototypes)
    disp2 = dispersion_reg_collision(prototypes)
    
    assert disp1.dim() == 0 and disp1 >= 0
    assert disp2.dim() == 0 and disp2 <= 0
    
    # Laplacian for disconnected graph should be 0
    W = torch.zeros(K, K)
    lap = graph_laplacian_reg(prototypes, W)
    assert torch.allclose(lap, torch.tensor(0.0))
    
    # Laplacian for fully connected graph
    W = torch.ones(K, K) - torch.eye(K)
    lap = graph_laplacian_reg(prototypes, W)
    assert lap > 0
