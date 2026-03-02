"""
Comprehensive stress testing for LOTC: extreme imbalance, feature corruption, and weak backbone.
Generates results for main.tex robustness table.
"""
import os, torch, numpy as np, yaml
from torch.utils.data import DataLoader, TensorDataset
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from sklearn.cluster import KMeans

def create_imbalanced(X, y, K, ratio):
    """Create imbalanced dataset with given max:min ratio."""
    classes = sorted(torch.unique(y).tolist())[:K]
    n_max = len(y) // K
    n_min = max(1, n_max // ratio)
    
    # Exponential decay from n_max to n_min
    counts = np.geomspace(n_max, n_min, K).astype(int)
    
    X_imb, y_imb = [], []
    for i, c in enumerate(classes):
        mask = y == c
        idx = torch.where(mask)[0][:counts[i]]
        X_imb.append(X[idx])
        y_imb.append(y[idx])
    return torch.cat(X_imb), torch.cat(y_imb)

def add_noise(X, sigma):
    """Add Gaussian noise to features."""
    return X + sigma * torch.randn_like(X)

def run_lotc_single(X, y, device, cfg, num_classes):
    """Run LOTC once and return metrics."""
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    input_dim = X.shape[1]
    
    cfg_copy = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg_copy['model'] = cfg['model'].copy()
    cfg_copy['model']['num_prototypes'] = num_classes
    
    model = build_model(cfg_copy, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    enc_params = list(model.encoder.parameters())
    opt_enc = torch.optim.Adam(enc_params, lr=cfg_copy['training']['lr_enc']) if enc_params else None
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg_copy['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=cfg_copy['training']['lr_mass'])
    
    for ep in range(1, cfg_copy['training']['epochs'] + 1):
        train_epoch(model, loader, opt_enc, opt_proto, opt_mass, device,
                    epsilon=cfg_copy['ot']['epsilon'],
                    sinkhorn_iter=cfg_copy['ot']['sinkhorn_iter'],
                    lambda_mass=cfg_copy['reg']['lambda_mass'],
                    lambda_disp=cfg_copy['reg']['lambda_disp'],
                    lambda_lap=cfg_copy['reg']['lambda_lap'],
                    lambda_cons=0.0,
                    disp_type=cfg_copy['reg']['disp_type'],
                    use_divergence=cfg_copy['ot']['use_divergence'])
    
    eval_out = evaluate(model, eval_loader, device, cfg_copy['ot']['epsilon'], cfg_copy['ot']['sinkhorn_iter'])
    return compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(42); torch.manual_seed(42)
    
    print("Loading CIFAR-10 features...")
    X, y = get_image_dataset("cifar10", pretrained=True, device=device.type)
    K = 10
    
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': X.shape[1], 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
        'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
    }
    
    results = {}
    
    # === 1. Extreme Imbalance ===
    print("\n=== Extreme Imbalance Stress Tests ===")
    for ratio in [200, 500]:
        print(f"  Ratio 1:{ratio}...")
        X_imb, y_imb = create_imbalanced(X, y, K, ratio)
        
        # K-Means
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        km_pred = km.fit_predict(X_imb.numpy())
        km_m = compute_all_metrics(y_imb.numpy(), km_pred)
        
        # LOTC
        lotc_m = run_lotc_single(X_imb, y_imb, device, cfg, K)
        
        results[f'imbalance_1_{ratio}'] = {
            'kmeans': km_m, 'lotc': lotc_m,
            'n_samples': len(y_imb),
            'delta_ari': lotc_m['ARI'] - km_m['ARI']
        }
        print(f"    KM ARI={km_m['ARI']:.4f} | LOTC ARI={lotc_m['ARI']:.4f} | Δ={lotc_m['ARI']-km_m['ARI']:+.4f}")
    
    # === 2. Feature Corruption Noise ===
    print("\n=== Feature Corruption Tests ===")
    for sigma in [0.1, 0.3, 0.5]:
        print(f"  σ={sigma}...")
        X_noisy = add_noise(X, sigma)
        
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        km_pred = km.fit_predict(X_noisy.numpy())
        km_m = compute_all_metrics(y.numpy(), km_pred)
        
        lotc_m = run_lotc_single(X_noisy, y, device, cfg, K)
        
        results[f'noise_sigma_{sigma}'] = {
            'kmeans': km_m, 'lotc': lotc_m,
            'delta_ari': lotc_m['ARI'] - km_m['ARI']
        }
        print(f"    KM ARI={km_m['ARI']:.4f} | LOTC ARI={lotc_m['ARI']:.4f} | Δ={lotc_m['ARI']-km_m['ARI']:+.4f}")
    
    # === 3. Weak Backbone (Random Features) ===
    print("\n=== Weak Backbone (Random Init) ===")
    X_rand = torch.randn_like(X)  # Completely random features
    
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km_pred = km.fit_predict(X_rand.numpy())
    km_m = compute_all_metrics(y.numpy(), km_pred)
    
    lotc_m = run_lotc_single(X_rand, y, device, cfg, K)
    
    results['random_backbone'] = {
        'kmeans': km_m, 'lotc': lotc_m,
        'delta_ari': lotc_m['ARI'] - km_m['ARI']
    }
    print(f"  KM ARI={km_m['ARI']:.4f} | LOTC ARI={lotc_m['ARI']:.4f} | Δ={lotc_m['ARI']-km_m['ARI']:+.4f}")
    
    # Save
    out_path = "experiments/results/stress_tests.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nAll stress tests complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
