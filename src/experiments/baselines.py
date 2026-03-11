"""
SOTA Deep Clustering Baselines for head-to-head comparison with LOTC.

All baselines operate on identical pretrained features for fair comparison.
Provides: K-Means, DEC, DCEC, IMSAT-proxy, and P²OT-proxy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from src.eval.metrics import evaluate_clustering as compute_all_metrics


# ---------------------------------------------------------------------------
# 1. K-Means (sklearn baseline)
# ---------------------------------------------------------------------------
def run_kmeans(X: np.ndarray, y: np.ndarray, K: int, n_init: int = 20,
               seed: int = 42) -> dict:
    """Run K-Means and return clustering metrics."""
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed, max_iter=300)
    preds = km.fit_predict(X)
    metrics = compute_all_metrics(y, preds)
    metrics['method'] = 'KMeans'
    return metrics


# ---------------------------------------------------------------------------
# 2. DEC (Deep Embedded Clustering)
# ---------------------------------------------------------------------------
class DECHead(nn.Module):
    """DEC clustering head: soft assignment via Student's t-distribution."""
    def __init__(self, n_clusters: int, embed_dim: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # q_{ij} = (1 + ||z_i - mu_j||^2 / alpha)^{-(alpha+1)/2}
        dist = torch.cdist(z, self.cluster_centers).pow(2)
        q = (1.0 + dist / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary target distribution p from q."""
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        return (weight / weight.sum(dim=1, keepdim=True)).detach()


def run_dec(X: np.ndarray, y: np.ndarray, K: int, embed_dim: int = 128,
            epochs: int = 100, lr: float = 1e-3, batch_size: int = 256,
            seed: int = 42, device: str = 'cpu') -> dict:
    """Run DEC on precomputed features.

    Uses: MLP autoencoder pretrain → KMeans init → KL-sharpening.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    input_dim = X_t.shape[1]

    # Simple MLP encoder
    encoder = nn.Sequential(
        nn.Linear(input_dim, 500), nn.ReLU(),
        nn.Linear(500, 500), nn.ReLU(),
        nn.Linear(500, 2000), nn.ReLU(),
        nn.Linear(2000, embed_dim),
    ).to(device)

    # Autoencoder pretrain
    decoder = nn.Sequential(
        nn.Linear(embed_dim, 2000), nn.ReLU(),
        nn.Linear(2000, 500), nn.ReLU(),
        nn.Linear(500, 500), nn.ReLU(),
        nn.Linear(500, input_dim),
    ).to(device)

    ae_params = list(encoder.parameters()) + list(decoder.parameters())
    ae_opt = torch.optim.Adam(ae_params, lr=lr)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Pretrain autoencoder
    for _ in range(50):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = encoder(batch_x)
            recon = decoder(z)
            loss = F.mse_loss(recon, batch_x)
            ae_opt.zero_grad()
            loss.backward()
            ae_opt.step()

    # KMeans init for cluster centers
    with torch.no_grad():
        all_z = encoder(X_t.to(device)).cpu().numpy()
    km = KMeans(n_clusters=K, n_init=20, random_state=seed)
    km.fit(all_z)

    dec_head = DECHead(K, embed_dim).to(device)
    dec_head.cluster_centers.data = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device)

    # DEC fine-tuning
    dec_opt = torch.optim.Adam(list(encoder.parameters()) + list(dec_head.parameters()), lr=lr * 0.1)
    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            z = encoder(batch_x)
            q = dec_head(z)
            p = DECHead.target_distribution(q)
            loss = F.kl_div(q.log(), p, reduction='batchmean')
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()

    # Final eval
    with torch.no_grad():
        all_z = encoder(X_t.to(device))
        q = dec_head(all_z)
        preds = q.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y, preds)
    metrics['method'] = 'DEC'
    return metrics


# ---------------------------------------------------------------------------
# 3. SCAN-proxy (Nearest-Neighbor + K-Means)
# ---------------------------------------------------------------------------
def run_scan_proxy(X: np.ndarray, y: np.ndarray, K: int,
                   n_neighbors: int = 20, seed: int = 42) -> dict:
    """SCAN-like proxy: KNN-consistent K-Means.

    Since full SCAN requires data augmentations and a multi-stage pipeline,
    this is a simplified version that uses KNN-graph consistency on
    pretrained features.
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    X_norm = torch.nn.functional.normalize(X_t, p=2, dim=1)
    
    n_samples = X_t.shape[0]
    X_smooth = torch.zeros_like(X_t)
    
    # Process in batches to avoid OOM
    batch_size = 2000
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch = X_norm[i:end]
        
        # Compute cosine similarity
        sim = torch.mm(batch, X_norm.T)
        
        # Get top-k closest (highest similarity)
        topk_sim, topk_indices = torch.topk(sim, k=n_neighbors, dim=1)
        
        # Convert similarity to distance-like weights
        distances = 1.0 - topk_sim
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted sum of neighbor features
        neighbor_features = X_t[topk_indices] # (batch, n_neighbors, D)
        smoothed_batch = (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
        X_smooth[i:end] = smoothed_batch
        
    X_smooth_np = X_smooth.cpu().numpy()
    
    # K-Means on smoothed features
    km = KMeans(n_clusters=K, n_init=20, random_state=seed)
    preds = km.fit_predict(X_smooth_np)
    metrics = compute_all_metrics(y, preds)
    metrics['method'] = 'SCAN-proxy'
    return metrics


# ---------------------------------------------------------------------------
# 4. P²OT-proxy (Partial OT for imbalanced clustering)
# ---------------------------------------------------------------------------
def run_p2ot_proxy(X: np.ndarray, y: np.ndarray, K: int,
                   transport_fraction: float = 0.8,
                   epochs: int = 50, seed: int = 42,
                   device: str = 'cpu') -> dict:
    """P²OT-like proxy: partial OT assignment with progressive transport.

    Approximates the P²OT approach (Zhang et al., ICLR 2024) by using
    partial Sinkhorn with a transport fraction that increases progressively.
    """
    torch.manual_seed(seed)
    from src.ot.sinkhorn import sinkhorn_log_domain

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    n = X_t.shape[0]

    # KMeans initialization for prototypes
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(X)
    prototypes = torch.tensor(km.cluster_centers_, dtype=torch.float32,
                              device=device, requires_grad=True)

    masses = torch.ones(K, device=device) / K
    opt = torch.optim.Adam([prototypes], lr=0.01)

    for epoch in range(epochs):
        # Progressive transport fraction
        frac = min(1.0, transport_fraction + (1.0 - transport_fraction) * epoch / epochs)

        # Cost matrix
        C = torch.cdist(X_t, prototypes).pow(2)
        u = torch.ones(n, device=device) / n

        # Partial Sinkhorn: scale source mass by fraction
        u_partial = u * frac
        P, cost = sinkhorn_log_domain(C, u_partial, masses, epsilon=0.05, max_iter=50)

        opt.zero_grad()
        cost.backward()
        opt.step()

    # Final assignment
    with torch.no_grad():
        C = torch.cdist(X_t, prototypes).pow(2)
        u = torch.ones(n, device=device) / n
        P, _ = sinkhorn_log_domain(C, u, masses, epsilon=0.05, max_iter=50)
        preds = P.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y, preds)
    metrics['method'] = 'P2OT-proxy'
    return metrics


# ---------------------------------------------------------------------------
# 5. IMSAT-proxy (Information Maximization + Self-Augmented Training)
# ---------------------------------------------------------------------------
def run_imsat_proxy(X: np.ndarray, y: np.ndarray, K: int,
                    epochs: int = 100, lr: float = 1e-3,
                    batch_size: int = 256, seed: int = 42,
                    device: str = 'cpu') -> dict:
    """IMSAT-like proxy: MI maximization with perturbation consistency.

    Uses additive Gaussian noise as the perturbation instead of data
    augmentations.
    """
    torch.manual_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32)
    input_dim = X_t.shape[1]

    net = nn.Sequential(
        nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, K),
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            # Clean and noisy forward
            logits_clean = net(batch_x)
            noise = torch.randn_like(batch_x) * 0.1
            logits_noisy = net(batch_x + noise)

            p_clean = F.softmax(logits_clean, dim=1)
            p_noisy = F.softmax(logits_noisy, dim=1)

            # Marginal entropy (maximize) - Conditional entropy (minimize)
            p_avg = p_clean.mean(dim=0)
            H_marginal = -(p_avg * (p_avg + 1e-10).log()).sum()
            H_conditional = -(p_clean * (p_clean + 1e-10).log()).sum(dim=1).mean()

            # Consistency (minimize KL between clean and noisy)
            consistency = F.kl_div(p_noisy.log(), p_clean.detach(), reduction='batchmean')

            loss = -H_marginal + H_conditional + 1.0 * consistency
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = net(X_t.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    metrics = compute_all_metrics(y, preds)
    metrics['method'] = 'IMSAT-proxy'
    return metrics


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BASELINE_METHODS = {
    'kmeans': run_kmeans,
    'dec': run_dec,
    'scan_proxy': run_scan_proxy,
    'p2ot_proxy': run_p2ot_proxy,
    'imsat_proxy': run_imsat_proxy,
}
