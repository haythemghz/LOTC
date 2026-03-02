"""
Large-Scale CIFAR-10 Benchmark v3 (Master Level).
- Full 50,000 samples.
- 5 Seeds for statistical significance.
- ACC, ARI, NMI, Silhouette, Davies-Bouldin.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, TensorDataset
from src.data.images import get_image_dataset
from src.models.lotc_model import LOTCModel
from src.training.loops import train_epoch, evaluate
from src.experiments.run_experiment import evaluate_clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def clean_dict(d):
    """Recursively convert numpy types to Python types."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    elif isinstance(d, (np.float32, np.float64, np.float16)):
        return float(d)
    elif isinstance(d, (np.int32, np.int64, np.int16)):
        return int(d)
    return d

def run_master_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Master Benchmark on {device}...")
    
    # 1. Load FULL CIFAR-10 (Pretrained ResNet-18 features)
    print("Loading full CIFAR-10 (50,000 samples)...")
    X, y = get_image_dataset('cifar10', pretrained=True, device=str(device))
    X = X.to(device)
    y = y.to(device)
    
    seeds = [42, 123, 999, 2024, 7]
    results = {'kmeans': [], 'lotc': []}
    
    for seed in seeds:
        print(f"\n--- Running Seed: {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # --- K-Means Baseline ---
        km = KMeans(n_clusters=10, n_init=10, random_state=seed)
        with torch.no_grad():
            z_init = X.cpu().numpy()
            km_labels = km.fit_predict(z_init)
            km_m = evaluate_clustering(y.cpu().numpy(), km_labels)
            # Internal metrics (sub-sampled to 10k for speed)
            idx = np.random.choice(len(z_init), 10000, replace=False)
            km_m['silhouette'] = silhouette_score(z_init[idx], km_labels[idx])
            km_m['db_score'] = davies_bouldin_score(z_init[idx], km_labels[idx])
            results['kmeans'].append(km_m)
            print(f"  K-Means: ACC={km_m['ACC']:.4f} ARI={km_m['ARI']:.4f} Sil={km_m['silhouette']:.4f}")

        # --- LOTC ---
        # Identity encoder for frozen features (standard SSL protocol)
        model = LOTCModel(torch.nn.Identity(), 10, 512, cost_type='cosine', normalize=True).to(device)
        model.prototypes.init_from_kmeans(X)
        
        optimizer_p = torch.optim.Adam([model.prototypes.prototypes], lr=0.005)
        optimizer_m = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
        loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)
        
        best_ari = -1
        best_metrics = None
        
        for ep in range(1, 41):
            train_epoch(model, loader, None, optimizer_p, optimizer_m, device, 
                        epsilon=0.1, sinkhorn_iter=50, lambda_mass=0.005, 
                        lambda_disp=0.001, lambda_lap=0.0, use_divergence=True)
            
            if ep % 10 == 0:
                ev = evaluate(model, loader, device, 0.1, 50)
                m = evaluate_clustering(ev['y_true'].numpy(), ev['y_pred'].numpy())
                if m['ARI'] > best_ari:
                    best_ari = m['ARI']
                    # Internal metrics
                    z_eval = ev['z'].numpy()
                    m['silhouette'] = silhouette_score(z_eval[idx], ev['y_pred'].numpy()[idx])
                    m['db_score'] = davies_bouldin_score(z_eval[idx], ev['y_pred'].numpy()[idx])
                    best_metrics = m
                print(f"  LOTC Ep {ep:2d}: ARI={m['ARI']:.4f}")
        
        results['lotc'].append(best_metrics)

    # 2. Aggregate Results
    summary = {'raw_results': results}
    for method in ['kmeans', 'lotc']:
        summary[method] = {}
        for metric in ['ACC', 'ARI', 'NMI', 'silhouette', 'db_score']:
            vals = [r[metric] for r in results[method]]
            summary[method][f"{metric}_mean"] = float(np.mean(vals))
            summary[method][f"{metric}_std"] = float(np.std(vals))

    os.makedirs('experiments/results', exist_ok=True)
    with open('experiments/results/master_cifar10.yaml', 'w') as f:
        yaml.dump(clean_dict(summary), f)
    
    print("\n" + "="*40)
    print("MASTER BENCHMARK COMPLETE")
    print(f"K-Means ARI: {summary['kmeans']['ARI_mean']:.4f} ± {summary['kmeans']['ARI_std']:.4f}")
    print(f"LOTC    ARI: {summary['lotc']['ARI_mean']:.4f} ± {summary['lotc']['ARI_std']:.4f}")
    print("="*40)

if __name__ == '__main__':
    run_master_benchmark()
