"""
Fast combined benchmark — CIFAR-10 + Imbalanced CIFAR-10
Uses 10k samples and 30 epochs to keep runtime under 10 minutes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from src.data.images import get_image_dataset
from src.data.datasets import MultiViewDataset
from src.models.encoders import MLPEncoder, IdentityEncoder
from src.models.lotc_model import LOTCModel
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering


def run_single(name, X, y, device, seed=42, n_epochs=40, use_mlp=False, use_cons=False):
    """Run K-Means and LOTC on a dataset, return results dict."""
    torch.manual_seed(seed)
    n_classes = len(torch.unique(y))
    input_dim = X.shape[1]
    
    # K-Means
    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=seed)
    km_pred = kmeans.fit_predict(X.cpu().numpy())
    km = evaluate_clustering(y.cpu().numpy(), km_pred)
    print(f"  K-Means:  ACC={km['ACC']:.4f}  ARI={km['ARI']:.4f}  NMI={km['NMI']:.4f}")
    
    # LOTC Identity
    torch.manual_seed(seed)
    model = LOTCModel(IdentityEncoder(), n_classes, input_dim, cost_type='cosine', normalize=True).to(device)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    model.prototypes.prototypes.data.copy_(centers)
    
    opt_p = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
    opt_m = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
    loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=512, shuffle=True)
    eval_loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=2048, shuffle=False)
    
    best = {'ARI': -1}
    for ep in range(1, n_epochs+1):
        train_epoch(model, loader, None, opt_p, opt_m, device,
                    epsilon=0.1, sinkhorn_iter=50, lambda_mass=0.005,
                    lambda_disp=0.001, lambda_lap=0.0, disp_type='collision',
                    use_divergence=True, grad_clip=5.0)
        if ep % 10 == 0 or ep == n_epochs:
            ev = evaluate(model, eval_loader, device, 0.1, 50)
            m = evaluate_clustering(ev['y_true'].numpy(), ev['y_pred'].numpy())
            masses = model.prototypes.masses.detach().cpu().numpy()
            print(f"  LOTC-Id Ep{ep:3d}: ACC={m['ACC']:.4f}  ARI={m['ARI']:.4f}  NMI={m['NMI']:.4f}  masses={np.round(masses, 3)}")
            if m['ARI'] > best['ARI']:
                best = m.copy()
    
    lotc_id = best
    
    # LOTC MLP (optional)
    lotc_mlp = {'ACC': 0, 'ARI': 0, 'NMI': 0}
    if use_mlp:
        torch.manual_seed(seed)
        enc = MLPEncoder(input_dim, [256, 256], 128)
        model2 = LOTCModel(enc, n_classes, 128, cost_type='cosine', normalize=True).to(device)
        with torch.no_grad():
            z = model2.encode(X[:2000])
            model2.prototypes.init_from_kmeans(z)
        
        o_e = torch.optim.Adam(model2.encoder.parameters(), lr=0.0005)
        o_p = torch.optim.Adam([model2.prototypes.prototypes], lr=0.003)
        o_m = torch.optim.Adam([model2.prototypes.mass_logits], lr=0.01)
        
        if use_cons:
            train_ld = DataLoader(MultiViewDataset(X.cpu(), y.cpu(), noise_std=0.03), batch_size=256, shuffle=True)
        else:
            train_ld = loader
        
        best2 = {'ARI': -1}
        for ep in range(1, n_epochs+1):
            train_epoch(model2, train_ld, o_e, o_p, o_m, device,
                        epsilon=0.1, sinkhorn_iter=50, lambda_mass=0.005,
                        lambda_disp=0.001, lambda_lap=0.0,
                        lambda_cons=0.5 if use_cons else 0.0,
                        disp_type='collision', use_divergence=True, grad_clip=5.0)
            if ep % 10 == 0 or ep == n_epochs:
                ev = evaluate(model2, eval_loader, device, 0.1, 50)
                m2 = evaluate_clustering(ev['y_true'].numpy(), ev['y_pred'].numpy())
                masses2 = model2.prototypes.masses.detach().cpu().numpy()
                print(f"  LOTC-MLP Ep{ep:3d}: ACC={m2['ACC']:.4f}  ARI={m2['ARI']:.4f}  NMI={m2['NMI']:.4f}  masses={np.round(masses2, 3)}")
                if m2['ARI'] > best2['ARI']:
                    best2 = m2.copy()
        lotc_mlp = best2
    
    return {'kmeans': km, 'lotc_identity': lotc_id, 'lotc_mlp': lotc_mlp}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    all_results = {}
    
    # --- CIFAR-10 (10k subsample for speed) ---
    print("="*60)
    print("CIFAR-10 (pretrained ResNet-18 features, 10k subsample)")
    print("="*60)
    X, y = get_image_dataset('cifar10', pretrained=True, device=str(device))
    # Subsample for speed
    idx = torch.randperm(len(X))[:10000]
    X_sub, y_sub = X[idx].to(device), y[idx].to(device)
    print(f"Shape: {X_sub.shape}, Classes: {len(torch.unique(y_sub))}")
    r1 = run_single("cifar10", X_sub, y_sub, device, use_mlp=True, use_cons=True, n_epochs=40)
    all_results['cifar10'] = r1
    
    # --- Imbalanced CIFAR-10 ---
    print("\n" + "="*60)
    print("Imbalanced CIFAR-10 (pretrained ResNet-18 features)")
    print("="*60)
    X_imb, y_imb = get_image_dataset('imbalanced_cifar10', pretrained=True, device=str(device))
    X_imb, y_imb = X_imb.to(device), y_imb.to(device)
    print(f"Shape: {X_imb.shape}, Classes: {len(torch.unique(y_imb))}")
    true_props = torch.bincount(y_imb.cpu()).float()
    true_props = true_props[true_props > 0] / true_props.sum()
    print(f"True proportions: {true_props.numpy().round(3)}")
    r2 = run_single("imbalanced", X_imb, y_imb, device, use_mlp=True, use_cons=True, n_epochs=40)
    all_results['imbalanced_cifar10'] = r2
    
    # --- Summary ---
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for dname, res in all_results.items():
        print(f"\n{dname}:")
        for method, m in res.items():
            print(f"  {method:20s}  ACC={m.get('ACC',0):.4f}  ARI={m.get('ARI',0):.4f}  NMI={m.get('NMI',0):.4f}")
    
    # Save
    os.makedirs('experiments/results', exist_ok=True)
    # Convert numpy to float for YAML serialization
    def to_plain(d):
        return {k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v) for k, v in d.items()}
    
    save_data = {k: to_plain(v) for k, v in all_results.items()}
    with open('experiments/results/combined_v2_results.yaml', 'w') as f:
        yaml.dump(save_data, f, default_flow_style=False)
    print("\nSaved to experiments/results/combined_v2_results.yaml")


if __name__ == '__main__':
    main()
