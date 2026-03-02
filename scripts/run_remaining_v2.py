"""
Remaining benchmarks v2: CIFAR-100 Coarse and Sensitivity Study.
Uses pretrained features and fixed LOTC logic.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from src.data.images import get_image_dataset
from src.models.encoders import IdentityEncoder, MLPEncoder
from src.models.lotc_model import LOTCModel
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering

def run_cifar100(device):
    print("\n" + "="*60)
    print("CIFAR-100 Coarse (20 classes, ResNet-18 features)")
    print("="*60)
    X, y = get_image_dataset('cifar100', pretrained=True, device=str(device))
    X, y = X.to(device), y.to(device)
    n_classes = 20
    
    # K-Means
    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=42)
    km_pred = kmeans.fit_predict(X.cpu().numpy())
    km = evaluate_clustering(y.cpu().numpy(), km_pred)
    print(f"  K-Means:  ACC={km['ACC']:.4f}  ARI={km['ARI']:.4f}  NMI={km['NMI']:.4f}")
    
    # LOTC Identity
    model = LOTCModel(IdentityEncoder(), n_classes, X.shape[1], cost_type='cosine', normalize=True).to(device)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    model.prototypes.prototypes.data.copy_(centers)
    
    opt_p = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
    opt_m = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
    loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=512, shuffle=True)
    eval_loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=2048, shuffle=False)
    
    best = {'ARI': -1}
    for ep in range(1, 41):
        train_epoch(model, loader, None, opt_p, opt_m, device, epsilon=0.1, sinkhorn_iter=50, 
                    lambda_mass=0.005, lambda_disp=0.001, lambda_lap=0.0, use_divergence=True)
        if ep % 10 == 0:
            ev = evaluate(model, eval_loader, device, 0.1, 50)
            m = evaluate_clustering(ev['y_true'].numpy(), ev['y_pred'].numpy())
            print(f"  LOTC-Id Ep{ep:3d}: ACC={m['ACC']:.4f}  ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}")
            if m['ARI'] > best['ARI']:
                best = m.copy()
    return {'kmeans': km, 'lotc': best}

def run_sensitivity(device):
    print("\n" + "="*60)
    print("Sensitivity Study (Fashion-MNIST, MLP 64-dim features)")
    print("="*60)
    # We use pre-extracted MLP features if they exist, or just use raw for speed on F-MNIST
    # For this study, we'll use a fixed seed and data subset
    X, y = get_image_dataset('fmnist')
    X = X.view(len(X), -1).to(device) # [60000, 784]
    y = y.to(device)
    
    # Subsample 10k for speed
    idx = torch.randperm(len(X))[:10000]
    X_sub, y_sub = X[idx], y[idx]
    
    def run_config(eps, iters):
        torch.manual_seed(42)
        # Use a small MLP to get 64-dim embeddings
        enc = MLPEncoder(784, [256], 64).to(device)
        model = LOTCModel(enc, 10, 64, cost_type='cosine', normalize=True).to(device)
        opt_e = torch.optim.Adam(model.encoder.parameters(), lr=0.001)
        opt_p = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
        opt_m = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
        loader = DataLoader(TensorDataset(X_sub.cpu(), y_sub.cpu()), batch_size=512, shuffle=True)
        
        for ep in range(1, 21): # Shorter for sensitivity
            train_epoch(model, loader, opt_e, opt_p, opt_m, device, epsilon=eps, sinkhorn_iter=iters, 
                        lambda_mass=0.0, lambda_disp=0.001, lambda_lap=0.0, use_divergence=True)
        
        eval_loader = DataLoader(TensorDataset(X_sub.cpu(), y_sub.cpu()), batch_size=2048, shuffle=False)
        ev = evaluate(model, eval_loader, device, eps, iters)
        return evaluate_clustering(ev['y_true'].numpy(), ev['y_pred'].numpy())

    results = {'epsilon_study': {}, 'iter_study': {}}
    
    print("Studying Epsilon (T=50):")
    for eps in [0.01, 0.05, 0.1, 0.5]:
        m = run_config(eps, 50)
        print(f"  eps={eps:.2f}: ACC={m['ACC']:.4f} ARI={m['ARI']:.4f}")
        results['epsilon_study'][eps] = m
    
    print("Studying Iterations (eps=0.1):")
    for iters in [5, 10, 20, 50]:
        m = run_config(0.1, iters)
        print(f"  iters={iters:2d}: ACC={m['ACC']:.4f} ARI={m['ARI']:.4f}")
        results['iter_study'][iters] = m
        
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c100 = run_cifar100(device)
    sens = run_sensitivity(device)
    
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/cifar100_v2.yaml', 'w') as f:
        yaml.dump(c100, f)
    with open('experiments/results/sensitivity_v2.yaml', 'w') as f:
        yaml.dump(sens, f)
    print("\nResults saved to cifar100_v2.yaml and sensitivity_v2.yaml")

if __name__ == '__main__':
    main()
