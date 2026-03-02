import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ot.sinkhorn import sinkhorn_log_domain
from src.ot.costs import squared_euclidean_cost, cosine_cost, MahalanobisCost
from src.models.prototypes import PrototypeModule
from src.models.encoders import IdentityEncoder
from src.models.regularizers import mass_entropy_reg, mass_kl_reg, dispersion_reg_l2, dispersion_reg_collision, graph_laplacian_reg

class LOTCModel(nn.Module):
    """
    Learned Optimal Transport Clustering (LOTC) Model.
    Unifies representation learning, prototype locations, and cluster masses.
    """
    def __init__(self, encoder: nn.Module, num_prototypes: int, embed_dim: int, 
                 cost_type: str = 'cosine', normalize: bool = True):
        super().__init__()
        self.encoder = encoder
        self.prototypes = PrototypeModule(num_prototypes, embed_dim)
        self.cost_type = cost_type
        self.normalize = normalize
        
        if cost_type == 'mahalanobis':
            self.mahalanobis_cost = MahalanobisCost(embed_dim)

    def compute_cost_matrix(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.cost_type == 'squared_euclidean':
            return squared_euclidean_cost(z, c)
        elif self.cost_type == 'cosine':
            return cosine_cost(z, c)
        elif self.cost_type == 'mahalanobis':
            return self.mahalanobis_cost(z, c)
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and optionally normalize to unit sphere."""
        z = self.encoder(x)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z

    def get_prototypes(self) -> torch.Tensor:
        """Get prototypes, optionally normalized to unit sphere."""
        c = self.prototypes.prototypes
        if self.normalize:
            c = F.normalize(c, dim=1)
        return c

    def forward(self, x: torch.Tensor, epsilon: float = 0.05, sinkhorn_iter: int = 50,
                lambda_mass: float = 0.0, lambda_disp: float = 0.0, lambda_lap: float = 0.0,
                disp_type: str = 'collision', W_graph: torch.Tensor | None = None,
                use_divergence: bool = False, mass_prior: torch.Tensor | None = None) -> dict:
        """
        Forward pass returning optimal transport plan, costs, and assignments.
        """
        # 1. Embed data
        z = self.encode(x)
        
        # 2. Get prototypes  
        c = self.get_prototypes()
        masses = self.prototypes.masses
        
        # 3. Compute cost matrix
        C = self.compute_cost_matrix(z, c)
        
        # 4. Sinkhorn OT
        B = z.size(0)
        u = torch.ones(B, dtype=z.dtype, device=z.device) / B
        
        if use_divergence:
            from src.ot.sinkhorn import sinkhorn_divergence
            C_uu = self.compute_cost_matrix(z, z)
            C_vv = self.compute_cost_matrix(c, c)
            ot_cost = sinkhorn_divergence(C, C_uu, C_vv, u, masses, epsilon=epsilon, max_iter=sinkhorn_iter)
            # Still need P for assignments
            P, _ = sinkhorn_log_domain(C, u, masses, epsilon=epsilon, max_iter=sinkhorn_iter)
        else:
            P, ot_cost = sinkhorn_log_domain(C, u, masses, epsilon=epsilon, max_iter=sinkhorn_iter)
        
        # 5. Regularisers
        if mass_prior is not None:
            reg_mass = mass_kl_reg(masses, mass_prior)
        else:
            reg_mass = mass_entropy_reg(masses)
        
        if disp_type == 'l2':
            reg_disp = dispersion_reg_l2(c)
        else:
            reg_disp = dispersion_reg_collision(c)
            
        reg_lap = torch.tensor(0.0, device=z.device)
        if lambda_lap > 0 and W_graph is not None:
            reg_lap = graph_laplacian_reg(c, W_graph)
            
        # 6. Total loss (OT + regularizers)
        total_loss = ot_cost + lambda_mass * reg_mass + lambda_disp * reg_disp + lambda_lap * reg_lap
        
        # 7. Assignments
        hard_assignments = torch.argmax(P, dim=1)
        soft_assignments = P / (P.sum(dim=1, keepdim=True) + 1e-12)
        
        return {
            'z': z,
            'P': P,
            'ot_cost': ot_cost,
            'reg_mass': reg_mass,
            'reg_disp': reg_disp,
            'reg_lap': reg_lap,
            'total_loss': total_loss,
            'hard_assignments': hard_assignments,
            'soft_assignments': soft_assignments
        }
