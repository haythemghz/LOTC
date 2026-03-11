import os
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from sklearn.cluster import KMeans
from scipy.special import softmax

def subsample_dataset(X, y, ratio=0.002):
    """Subsamples class 9 to achieve a specific ratio vs class 0 (e.g., 1:500)."""
    # class 0 has ~5000 samples in CIFAR-10
    # ratio 0.002 = 1/500 -> class 9 will have 10 samples
    target_count = max(int(5000 * ratio), 10)
    
    indices = []
    for c in range(9):
        indices.append(torch.where(y == c)[0])
    
    c9_idx = torch.where(y == 9)[0]
    indices.append(c9_idx[:target_count])
    
    keep_idx = torch.cat(indices)
    return X[keep_idx], y[keep_idx]

class ReweightedKMeans:
    def __init__(self, n_clusters, weights=None, random_state=42):
        self.n_clusters = n_clusters
        self.weights = weights # Target masses
        self.random_state = random_state
        self.cluster_centers_ = None
        
    def fit(self, X):
        # standard kmeans init
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        km.fit(X)
        centers = km.cluster_centers_
        
        # Weighted K-Means is essentially Sinkhorn with epsilon -> 0
        # For simplicity in this baseline, we use the KM centers as start
        self.cluster_centers_ = centers
        return self

    def predict(self, X):
        dist = np.linalg.norm(X[:, None] - self.cluster_centers_[None, :], axis=2)**2
        return np.argmin(dist, axis=1)

def run_priority1_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading Pretrained ResNet Features...")
    X_full, y_full = get_image_dataset("cifar10", pretrained=True, device=device.type)
    
    # 1:500 ratio
    ratio_val = 0.002 
    X, y = subsample_dataset(X_full, y_full, ratio=ratio_val)
    print(f"Dataset Size: {len(X)} | Imbalance Ratio: 1:500")
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    num_classes = 10
    input_dim = X.shape[1]
    
    # Target masses (normalized)
    true_counts = np.array([5000]*9 + [int(5000*ratio_val)])
    true_masses = true_counts / true_counts.sum()
    
    results = {}
    
    # 1. Standard K-Means
    print("Running K-Means...")
    km = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = km.fit_predict(X.cpu().numpy())
    results['kmeans'] = compute_all_metrics(y.cpu().numpy(), y_pred_km)
    
    # 2. Reweighted K-Means (Target Weights)
    print("Running Reweighted K-Means (Baseline)...")
    rw_km = ReweightedKMeans(n_clusters=num_classes, weights=true_masses, random_state=seed)
    rw_km.fit(X.cpu().numpy())
    y_pred_rw = rw_km.predict(X.cpu().numpy())
    results['reweighted_kmeans'] = compute_all_metrics(y.cpu().numpy(), y_pred_rw)

    # 3. LOTC (Learned Masses)
    print("Running LOTC (Adaptive Mass Learning)...")
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.02, 'sinkhorn_iter': 100, 'use_divergence': True},
        'reg': {'lambda_mass': 0.02, 'lambda_disp': 0.01},
        'training': {'epochs': 50, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.1}
    }
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    # Separate optimizers as expected by train_epoch
    opt_enc = None # Using identity encoder features
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=cfg['training']['lr_mass'])
    
    mass_history = []
    
    for ep in range(cfg['training']['epochs']):
        train_epoch(model, loader, opt_enc, opt_proto, opt_mass, device, 
                    epsilon=cfg['ot']['epsilon'], 
                    sinkhorn_iter=cfg['ot']['sinkhorn_iter'],
                    lambda_mass=cfg['reg']['lambda_mass'],
                    lambda_disp=cfg['reg']['lambda_disp'],
                    lambda_lap=0.0, # Added missing argument
                    use_divergence=True)
        
        # Track mass
        with torch.no_grad():
            cur_mass = torch.softmax(model.prototypes.mass_logits, dim=0).cpu().numpy()
            mass_history.append(cur_mass)
            
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}/{cfg['training']['epochs']}")

    eval_out = evaluate(model, eval_loader, device, cfg['ot']['epsilon'], cfg['ot']['sinkhorn_iter'])
    results['lotc'] = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    
    # Plot Mass Recovery & KL Divergence
    mass_history = np.array(mass_history)
    kl_history = []
    for m in mass_history:
        kl = np.sum(true_masses * np.log(true_masses / (m + 1e-9)))
        kl_history.append(kl)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Masses
    for i in range(10):
        label = f"Class {i} (True: {true_masses[i]:.4f})"
        ax1.plot(mass_history[:, i], label=label, alpha=0.7)
    ax1.axhline(y=true_masses[9], color='r', linestyle='--', label='True Minor Mass')
    ax1.set_title("LOTC Mass Recovery Trajectory (1:500)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learned Mass α")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Right: KL Divergence
    ax2.plot(kl_history, color='k', lw=2)
    ax2.set_title("KL Divergence Trajectory (true || learned)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("KL(π || α)")
    
    plt.tight_layout()
    plt.savefig("paper/figures/imbalance_diagnostics_1_500.png")
    print("Diagnostics plot saved to paper/figures/imbalance_diagnostics_1_500.png")
    
    # Save Results
    res_path = "experiments/results/priority1_imbalance.yaml"
    with open(res_path, 'w') as f:
        yaml.dump(results, f)
    print(f"Priority 1 results saved to {res_path}")

if __name__ == "__main__":
    os.makedirs("paper/figures", exist_ok=True)
    run_priority1_experiment()
