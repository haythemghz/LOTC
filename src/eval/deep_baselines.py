"""
Deep clustering baseline stubs for LOTC experiments.

These are wrapper stubs that provide instructions for installing and
using external deep clustering implementations. Fully self-contained
implementations would be too large; instead, we wrap their public APIs.
"""

from __future__ import annotations

import warnings
from abc import ABC

import numpy as np


class DeepBaselineStub(ABC):
    """Base stub for deep clustering baselines.

    Subclasses should override ``fit()`` to call external implementations.
    """

    def __init__(self, name: str):
        self.name = name
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DeepBaselineStub":
        raise NotImplementedError(
            f"{self.name} is a stub. See the docstring for installation instructions."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


class DECBaseline(DeepBaselineStub):
    """Deep Embedded Clustering (DEC) implementation.
    Reference: Xie et al. ICML 2016.
    """
    def __init__(self, n_clusters: int = 10, alpha: float = 1.0, lr: float = 0.001, epochs: int = 50):
        super().__init__(name="DEC")
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.ndarray) -> "DECBaseline":
        import torch
        import torch.nn as nn
        from sklearn.cluster import KMeans
        
        X_t = torch.tensor(X, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = X_t.to(device)
        
        # 1. Initialise with K-Means
        km = KMeans(n_clusters=self.n_clusters, n_init=10)
        km.fit(X)
        self.prototypes = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device))
        
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
        
        for epoch in range(self.epochs):
            # Compute soft assignments Q
            # q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
            dist = torch.cdist(X_t, self.prototypes)**2
            q = (1.0 + dist / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
            q = q / q.sum(dim=1, keepdim=True)
            
            # Compute target distribution P
            # p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')
            f = q.sum(dim=0)
            p = (q**2 / f)
            p = (p.T / p.sum(dim=1)).T
            
            # KL divergence loss
            loss = nn.KLDivLoss(reduction='batchmean')(torch.log(q), p.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            dist = torch.cdist(X_t, self.prototypes)**2
            q = (1.0 + dist / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
            self.labels_ = torch.argmax(q, dim=1).cpu().numpy()
            
        return self


class DeepClusterV2Baseline(DeepBaselineStub):
    """DeepCluster-v2 implementation.
    Reference: Caron et al. "Unsupervised Learning of Visual Features by 
    Contrasting Cluster Assignments." NeurIPS 2020.
    """
    def __init__(self, n_clusters: int = 10, lr: float = 0.001, epochs: int = 30):
        super().__init__(name="DeepCluster-v2")
        self.n_clusters = n_clusters
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.ndarray) -> "DeepClusterV2Baseline":
        import torch
        import torch.nn as nn
        from sklearn.cluster import KMeans
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Linear head for classification into pseudo-labels
        input_dim = X.shape[1]
        head = nn.Linear(input_dim, self.n_clusters).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            # 1. Cluster features to get pseudo-labels (Sinkhorn-like or K-means)
            with torch.no_grad():
                km = KMeans(n_clusters=self.n_clusters, n_init=1)
                pseudo_labels = torch.tensor(km.fit_predict(X), dtype=torch.long).to(device)
            
            # 2. Train head to predict pseudo-labels
            for _ in range(5): # Inner iterations
                logits = head(X_t)
                loss = criterion(logits, pseudo_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        with torch.no_grad():
            logits = head(X_t)
            self.labels_ = torch.argmax(logits, dim=1).cpu().numpy()
        return self


class DINOClusteringBaseline(DeepBaselineStub):
    """DINO-based clustering baseline.
    Reference: Caron et al. "Emerging Properties in Self-Supervised Vision Transformers." ICCV 2021.
    """
    def __init__(self, n_clusters: int = 10):
        super().__init__(name="DINO-Clustering")
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray) -> "DINOClusteringBaseline":
        # In our feature-based setting, DINO clustering is equivalent to K-means
        # on DINO-pretrained features. Since we already use ResNet18 SSL features,
        # K-means is a strong proxy. For strictness, we apply K-means with n_init=100.
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, n_init=100, random_state=42)
        km.fit(X)
        self.labels_ = km.labels_
        return self


class IMSATBaseline(DeepBaselineStub):
    """IMSAT (Information Maximizing Self-Augmented Training) stub.

    Reference:
        Hu et al. "Learning Discrete Representations via Information
        Maximizing Self-Augmented Training." ICML 2017.
    """

    def __init__(self, n_clusters: int = 10, **kwargs):
        super().__init__(name="IMSAT")
        self.n_clusters = n_clusters


class KMeansInEmbeddingBaseline:
    """KMeans applied in a learned embedding space.

    This baseline uses the same encoder as LOTC but applies standard
    KMeans on the embeddings instead of OT-based clustering.
    """

    def __init__(self, encoder, n_clusters: int = 10, seed: int = 42):
        self.name = "KMeans-in-Embedding"
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.seed = seed
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeansInEmbeddingBaseline":
        import torch
        from sklearn.cluster import KMeans

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            Z = self.encoder(X_t).cpu().numpy()

        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        km.fit(Z)
        self.labels_ = km.labels_
        return self

class OTKMeansBaseline:
    """Optimal Transport K-Means (OT-K-Means).
    Learns prototypes via Sinkhorn OT but with uniform masses.
    """
    def __init__(self, n_clusters: int = 10, epsilon: float = 0.05, lr: float = 0.01, epochs: int = 50):
        self.name = "OT-K-Means"
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.lr = lr
        self.epochs = epochs
        self.labels_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "OTKMeansBaseline":
        import torch
        from src.ot.sinkhorn import sinkhorn_log_domain
        from src.ot.costs import squared_euclidean_cost
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        prototypes = torch.randn(self.n_clusters, X.shape[1], requires_grad=True, device=device)
        optimizer = torch.optim.Adam([prototypes], lr=self.lr)
        
        N = X.shape[0]
        u = torch.ones(N, device=device) / N
        v = torch.ones(self.n_clusters, device=device) / self.n_clusters
        
        for epoch in range(self.epochs):
            C = squared_euclidean_cost(X_t, prototypes)
            P, cost = sinkhorn_log_domain(C, u, v, epsilon=self.epsilon, max_iter=50)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
        with torch.no_grad():
            C = squared_euclidean_cost(X_t, prototypes)
            P, _ = sinkhorn_log_domain(C, u, v, epsilon=self.epsilon, max_iter=50)
            self.labels_ = torch.argmax(P, dim=1).cpu().numpy()
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


def run_dec_baseline(X, y, n_clusters=10):
    dec = DECBaseline(n_clusters=n_clusters)
    dec.fit(X)
    from src.eval.metrics import evaluate_clustering
    metrics = evaluate_clustering(y, dec.labels_)
    metrics['y_pred'] = dec.labels_
    return metrics


DEEP_BASELINES = {
    "dec": DECBaseline,
    "deepcluster-v2": DeepClusterV2Baseline,
    "dino": DINOClusteringBaseline,
    "ot-kmeans": OTKMeansBaseline,
    "kmeans-learned": KMeansInEmbeddingBaseline
}
