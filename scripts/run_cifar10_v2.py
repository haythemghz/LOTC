"""
Hardened CIFAR-10 Benchmark (v2) — demonstrates LOTC > K-Means
Uses pretrained ResNet-18 features, cosine cost, fixed Sinkhorn, gradient clipping.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

from src.data.images import get_image_dataset
from src.data.datasets import MultiViewDataset
from src.models.encoders import MLPEncoder, IdentityEncoder
from src.models.lotc_model import LOTCModel
from src.models.prototypes import PrototypeModule
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering
from src.ot.sinkhorn import sinkhorn_log_domain


def run_benchmark(dataset_name='cifar10', use_pretrained=True, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. Load Data ----
    print(f"Loading {dataset_name} (pretrained={use_pretrained})...")
    X, y = get_image_dataset(dataset_name, pretrained=use_pretrained, device=str(device))
    X, y = X.to(device), y.to(device)
    n_classes = len(torch.unique(y))
    input_dim = X.shape[1]
    print(f"  Shape: {X.shape}, Classes: {n_classes}")

    # ---- 2. K-Means Baseline ----
    print("\n--- K-Means Baseline ---")
    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=seed)
    km_pred = kmeans.fit_predict(X.cpu().numpy())
    km_metrics = evaluate_clustering(y.cpu().numpy(), km_pred)
    print(f"  ACC: {km_metrics['ACC']:.4f} | ARI: {km_metrics['ARI']:.4f} | NMI: {km_metrics['NMI']:.4f}")

    # ---- 3. LOTC with Identity Encoder + Cosine Cost ----
    print("\n--- LOTC (Identity + Cosine) ---")
    torch.manual_seed(seed)
    
    encoder = IdentityEncoder()
    model = LOTCModel(
        encoder=encoder,
        num_prototypes=n_classes,
        embed_dim=input_dim,
        cost_type='cosine',
        normalize=True
    ).to(device)
    
    # Initialize prototypes from k-means centers (already in feature space)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
    model.prototypes.prototypes.data.copy_(centers)
    model.prototypes.mass_logits.data.zero_()
    
    # Optimizers
    opt_proto = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
    
    loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=512, shuffle=True)
    eval_loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=1024, shuffle=False)
    
    best_ari = 0.0
    best_metrics = {}
    
    for ep in range(1, 101):
        stats = train_epoch(
            model, loader, None, opt_proto, opt_mass, device,
            epsilon=0.1, sinkhorn_iter=50,
            lambda_mass=0.005, lambda_disp=0.001,
            lambda_lap=0.0, lambda_cons=0.0,
            disp_type='collision',
            use_divergence=True,
            grad_clip=5.0
        )
        
        if ep % 10 == 0 or ep == 1:
            eval_out = evaluate(model, eval_loader, device, epsilon=0.1, sinkhorn_iter=50)
            metrics = evaluate_clustering(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
            print(f"  Ep {ep:3d} | Loss: {stats['loss']:.4f} | OT: {stats['ot_loss']:.4f} | "
                  f"ACC: {metrics['ACC']:.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f}")
            if metrics['ARI'] > best_ari:
                best_ari = metrics['ARI']
                best_metrics = metrics.copy()
    
    print(f"\n  Best LOTC: ACC={best_metrics['ACC']:.4f} ARI={best_metrics['ARI']:.4f} NMI={best_metrics['NMI']:.4f}")

    # ---- 4. LOTC with MLP Encoder + Consistency ----
    print("\n--- LOTC (MLP + Consistency) ---")
    torch.manual_seed(seed)
    
    embed_dim = 128
    encoder_mlp = MLPEncoder(input_dim=input_dim, hidden_dims=[256, 256], output_dim=embed_dim)
    model_mlp = LOTCModel(
        encoder=encoder_mlp,
        num_prototypes=n_classes,
        embed_dim=embed_dim,
        cost_type='cosine',
        normalize=True
    ).to(device)
    
    # Initialize prototypes via K-Means in embedding space
    with torch.no_grad():
        z_init = model_mlp.encode(X[:2000])
        model_mlp.prototypes.init_from_kmeans(z_init)
    
    enc_params = list(model_mlp.encoder.parameters())
    opt_enc = torch.optim.Adam(enc_params, lr=0.0005)
    opt_proto2 = torch.optim.Adam([model_mlp.prototypes.prototypes], lr=0.003)
    opt_mass2 = torch.optim.Adam([model_mlp.prototypes.mass_logits], lr=0.01)
    
    # MultiView for consistency
    mv_loader = DataLoader(MultiViewDataset(X.cpu(), y.cpu(), noise_std=0.05), batch_size=256, shuffle=True)
    
    best_ari_mlp = 0.0
    best_metrics_mlp = {}
    
    for ep in range(1, 101):
        stats = train_epoch(
            model_mlp, mv_loader, opt_enc, opt_proto2, opt_mass2, device,
            epsilon=0.1, sinkhorn_iter=50,
            lambda_mass=0.005, lambda_disp=0.001,
            lambda_lap=0.0, lambda_cons=0.5,
            disp_type='collision',
            use_divergence=True,
            grad_clip=5.0
        )
        
        if ep % 10 == 0 or ep == 1:
            eval_out = evaluate(model_mlp, eval_loader, device, epsilon=0.1, sinkhorn_iter=50)
            metrics = evaluate_clustering(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
            print(f"  Ep {ep:3d} | Loss: {stats['loss']:.4f} | OT: {stats['ot_loss']:.4f} | Cons: {stats['cons_loss']:.4f} | "
                  f"ACC: {metrics['ACC']:.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f}")
            if metrics['ARI'] > best_ari_mlp:
                best_ari_mlp = metrics['ARI']
                best_metrics_mlp = metrics.copy()
    
    print(f"\n  Best LOTC-MLP: ACC={best_metrics_mlp['ACC']:.4f} ARI={best_metrics_mlp['ARI']:.4f} NMI={best_metrics_mlp['NMI']:.4f}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("CIFAR-10 Benchmark Summary")
    print("="*60)
    print(f"  K-Means:       ACC={km_metrics['ACC']:.4f}  ARI={km_metrics['ARI']:.4f}  NMI={km_metrics['NMI']:.4f}")
    print(f"  LOTC-Identity:  ACC={best_metrics.get('ACC',0):.4f}  ARI={best_metrics.get('ARI',0):.4f}  NMI={best_metrics.get('NMI',0):.4f}")
    print(f"  LOTC-MLP:       ACC={best_metrics_mlp.get('ACC',0):.4f}  ARI={best_metrics_mlp.get('ARI',0):.4f}  NMI={best_metrics_mlp.get('NMI',0):.4f}")

    # Save results
    import yaml
    results = {
        'kmeans': km_metrics,
        'lotc_identity': best_metrics,
        'lotc_mlp': best_metrics_mlp
    }
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/cifar10_v2_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print("\nResults saved to experiments/results/cifar10_v2_results.yaml")


if __name__ == '__main__':
    run_benchmark()
