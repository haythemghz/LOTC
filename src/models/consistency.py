import torch
import torch.nn as nn
import torch.nn.functional as F

class AssignmentConsistencyLoss(nn.Module):
    """
    Consistency Regularization for clustering.
    Minimizes the KL divergence between assignments of two views of the same data.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        """
        P1, P2: Assignment plans of shape (B, K).
        """
        # Ensure non-zero before power
        P1 = P1.clamp(min=1e-12)
        P2 = P2.clamp(min=1e-12)
        
        P1_sharpened = P1 ** (1 / self.temperature)
        P1_sharpened = P1_sharpened / P1_sharpened.sum(dim=1, keepdim=True)
        
        # log(P2) is safe due to clamp
        loss = F.kl_div(torch.log(P2), P1_sharpened, reduction='batchmean')
        return loss

class ContrastiveLoss(nn.Module):
    """
    InfoNCE loss for self-supervised pre-training.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: Normalized embeddings of shape (B, d).
        """
        B = z1.size(0)
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate for all-pairs similarity
        z = torch.cat([z1, z2], dim=0) # (2B, d)
        sim = torch.mm(z, z.t()) / self.temperature # (2B, 2B)
        
        # Labels for InfoNCE
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
        
        # Mask out self-similarities
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim = sim[~mask].view(2*B, 2*B - 1)
        
        # Positive indices in the masked matrix
        pos_indices = torch.cat([torch.arange(B, 2*B) - 1, torch.arange(0, B)], dim=0).to(z.device)
        
        # Cross Entropy
        loss = F.cross_entropy(sim, pos_indices)
        return loss

def entropy_loss(P: torch.Tensor) -> torch.Tensor:
    """
    Optional entropy minimization on assignments to prevent over-uniformity.
    """
    # Sum_j P_ij log P_ij
    ent = -torch.sum(P * torch.log(P + 1e-12), dim=1)
    return torch.mean(ent)
