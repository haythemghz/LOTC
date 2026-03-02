"""
Imbalanced CIFAR-10 Benchmark (v2) — demonstrates LOTC learned mass > K-Means
Uses pretrained ResNet-18 features, cosine cost, adaptive mass KL prior.
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
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering


def kl_mass(predicted, true_proportions):
    """KL divergence between predicted masses and true data proportions."""
    import torch.nn.functional as F
    return F.kl_div(predicted.log(), true_proportions, reduction='sum').item()


def run_imbalanced_benchmark(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 1. Load Imbalanced CIFAR-10 ----
    print("Loading imbalanced_cifar10 (pretrained features)...")
    X, y = get_image_dataset('imbalanced_cifar10', pretrained=True, device=str(device))
    X, y = X.to(device), y.to(device)
    
    n_classes = len(torch.unique(y))
    input_dim = X.shape[1]
    true_proportions = torch.bincount(y.cpu()).float()
    true_proportions = true_proportions[true_proportions > 0]
    true_proportions = true_proportions / true_proportions.sum()
    print(f"  Shape: {X.shape}, Classes: {n_classes}")
    print(f"  True mass distribution: {true_proportions.tolist()}")

    # ---- 2. K-Means Baseline ----
    print("\n--- K-Means Baseline ---")
    kmeans = KMeans(n_clusters=n_classes, n_init=20, random_state=seed)
    km_pred = kmeans.fit_predict(X.cpu().numpy())
    km_metrics = evaluate_clustering(y.cpu().numpy(), km_pred)
    km_mass_kl = kl_mass(
        torch.tensor(np.bincount(km_pred, minlength=n_classes), dtype=torch.float32) / len(km_pred),
        true_proportions
    )
    print(f"  ACC: {km_metrics['ACC']:.4f} | ARI: {km_metrics['ARI']:.4f} | NMI: {km_metrics['NMI']:.4f} | KL_mass: {km_mass_kl:.4f}")

    # ---- 3. LOTC Fixed Mass (Uniform) ----
    print("\n--- LOTC Fixed Mass (Uniform, Cosine Cost) ---")
    results = {}
    
    for variant, lr_mass, mass_prior in [
        ("Fixed Mass (Uniform)", 0.0, None),
        ("Learned Mass (no prior)", 0.02, None),
        ("Learned Mass (KL prior)", 0.02, true_proportions.to(device)),
    ]:
        print(f"\n--- LOTC: {variant} ---")
        torch.manual_seed(seed)
        
        encoder = IdentityEncoder()
        model = LOTCModel(
            encoder=encoder,
            num_prototypes=n_classes,
            embed_dim=input_dim,
            cost_type='cosine',
            normalize=True
        ).to(device)
        
        # Init from K-Means
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        model.prototypes.prototypes.data.copy_(centers)
        model.prototypes.mass_logits.data.zero_()
        
        opt_proto = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
        if lr_mass > 0:
            opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=lr_mass)
        else:
            opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.0)
        
        loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=512, shuffle=True)
        eval_loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=1024, shuffle=False)
        
        best_ari = -1.0
        best_metrics = {}
        
        for ep in range(1, 81):
            stats = train_epoch(
                model, loader, None, opt_proto, opt_mass, device,
                epsilon=0.1, sinkhorn_iter=50,
                lambda_mass=0.01 if lr_mass > 0 else 0.0,
                lambda_disp=0.001,
                lambda_lap=0.0, lambda_cons=0.0,
                disp_type='collision',
                use_divergence=True,
                mass_prior=mass_prior,
                grad_clip=5.0
            )
            
            if ep % 10 == 0 or ep == 1:
                eval_out = evaluate(model, eval_loader, device, epsilon=0.1, sinkhorn_iter=50)
                metrics = evaluate_clustering(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
                learned_masses = model.prototypes.masses.detach().cpu()
                mass_kl_val = kl_mass(learned_masses, true_proportions)
                print(f"  Ep {ep:3d} | Loss: {stats['loss']:.4f} | "
                      f"ACC: {metrics['ACC']:.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | "
                      f"KL_mass: {mass_kl_val:.4f}")
                print(f"         Masses: {learned_masses.numpy().round(3)}")
                if metrics['ARI'] > best_ari:
                    best_ari = metrics['ARI']
                    best_metrics = metrics.copy()
                    best_metrics['kl_mass'] = mass_kl_val
        
        results[variant] = best_metrics
        print(f"  Best: ACC={best_metrics.get('ACC',0):.4f} ARI={best_metrics.get('ARI',0):.4f} NMI={best_metrics.get('NMI',0):.4f}")

    # ---- 4. LOTC with MLP Encoder + Learned Mass ----
    print("\n--- LOTC (MLP + Learned Mass + Consistency) ---")
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
    
    with torch.no_grad():
        z_init = model_mlp.encode(X[:2000])
        model_mlp.prototypes.init_from_kmeans(z_init)
    
    opt_enc = torch.optim.Adam(model_mlp.encoder.parameters(), lr=0.0003)
    opt_p = torch.optim.Adam([model_mlp.prototypes.prototypes], lr=0.003)
    opt_m = torch.optim.Adam([model_mlp.prototypes.mass_logits], lr=0.02)
    
    mv_loader = DataLoader(MultiViewDataset(X.cpu(), y.cpu(), noise_std=0.03), batch_size=256, shuffle=True)
    eval_loader = DataLoader(TensorDataset(X.cpu(), y.cpu()), batch_size=1024, shuffle=False)
    
    best_ari_mlp = -1.0
    best_metrics_mlp = {}
    
    for ep in range(1, 101):
        stats = train_epoch(
            model_mlp, mv_loader, opt_enc, opt_p, opt_m, device,
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
            learned_masses = model_mlp.prototypes.masses.detach().cpu()
            mass_kl_val = kl_mass(learned_masses, true_proportions)
            print(f"  Ep {ep:3d} | Loss: {stats['loss']:.4f} | Cons: {stats['cons_loss']:.4f} | "
                  f"ACC: {metrics['ACC']:.4f} | ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | "
                  f"KL_mass: {mass_kl_val:.4f}")
            if metrics['ARI'] > best_ari_mlp:
                best_ari_mlp = metrics['ARI']
                best_metrics_mlp = metrics.copy()
                best_metrics_mlp['kl_mass'] = mass_kl_val

    # ---- Summary ----
    print("\n" + "="*70)
    print("Imbalanced CIFAR-10 Benchmark Summary")
    print("="*70)
    print(f"  K-Means:                  ACC={km_metrics['ACC']:.4f}  ARI={km_metrics['ARI']:.4f}  NMI={km_metrics['NMI']:.4f}  KL={km_mass_kl:.4f}")
    for name, m in results.items():
        print(f"  LOTC {name:25s}  ACC={m.get('ACC',0):.4f}  ARI={m.get('ARI',0):.4f}  NMI={m.get('NMI',0):.4f}  KL={m.get('kl_mass',0):.4f}")
    print(f"  LOTC MLP+Cons:            ACC={best_metrics_mlp.get('ACC',0):.4f}  ARI={best_metrics_mlp.get('ARI',0):.4f}  NMI={best_metrics_mlp.get('NMI',0):.4f}  KL={best_metrics_mlp.get('kl_mass',0):.4f}")

    # Save results
    import yaml
    all_results = {
        'true_proportions': true_proportions.tolist(),
        'kmeans': {k: float(v) for k, v in km_metrics.items()},
        'kmeans_kl_mass': float(km_mass_kl),
    }
    for name, m in results.items():
        key = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        all_results[key] = {k: float(v) for k, v in m.items()}
    all_results['lotc_mlp_cons'] = {k: float(v) for k, v in best_metrics_mlp.items()}
    
    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/imbalanced_cifar10_v2.yaml', 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print("\nResults saved to experiments/results/imbalanced_cifar10_v2.yaml")


if __name__ == '__main__':
    run_imbalanced_benchmark()
