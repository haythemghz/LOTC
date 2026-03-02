import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np

class PrototypeModule(nn.Module):
    """
    Manages the prototype locations (c) and mass logits (l).
    Enforces the simplex constraint on masses via softmax.
    """
    def __init__(self, num_prototypes: int, dim: int):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.dim = dim
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        # Initialise to zero for uniform masses
        self.mass_logits = nn.Parameter(torch.zeros(num_prototypes))

    @property
    def masses(self) -> torch.Tensor:
        """Returns the softmax-normalised masses."""
        return torch.softmax(self.mass_logits, dim=0)

    @torch.no_grad()
    def init_from_kmeans(self, data: torch.Tensor):
        """
        Initialise prototypes using K-Means on a data sample.
        Args:
            data: (N, d) tensor of data embeddings
        """
        # Ensure we don't try to fit more clusters than data points
        if len(data) < self.num_prototypes:
            raise ValueError("KMeans init requires more data points than prototypes.")

        data_np = data.cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_prototypes, n_init=10, random_state=42)
        kmeans.fit(data_np)
        
        # Set prototypes to cluster centers
        centers = torch.tensor(kmeans.cluster_centers_, dtype=data.dtype, device=data.device)
        self.prototypes.copy_(centers)
        
        # Reset masses to uniform
        self.mass_logits.zero_()

    @torch.no_grad()
    def init_random(self, data: torch.Tensor):
        """
        Initialise prototypes using random sampling from data.
        """
        indices = torch.randperm(len(data))[:self.num_prototypes]
        self.prototypes.copy_(data[indices])
        self.mass_logits.zero_()
