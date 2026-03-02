import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from sklearn.cluster import KMeans

def subsample_dataset(X, y, ratio=0.01):
    """Subsamples class 9 to achieve a specific ratio vs class 0."""
    # class 0 has ~5000 samples in CIFAR-10
    # we want class 9 to have 5000 * ratio samples
    target_count = int(5000 * ratio)
    
    indices = []
    for c in range(9):
        indices.append(torch.where(y == c)[0])
    
    c9_idx = torch.where(y == 9)[0]
    indices.append(c9_idx[:target_count])
    
    keep_idx = torch.cat(indices)
    return X[keep_idx], y[keep_idx]

def run_experiment(ratio_name, ratio_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_full, y_full = get_image_dataset("cifar10", pretrained=True, device=device.type)
    X, y = subsample_dataset(X_full, y_full, ratio=ratio_val)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    input_dim = X.shape[1]
    num_classes = 10
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = kmeans.fit_predict(X.numpy())
    km_metrics = compute_all_metrics(y.numpy(), y_pred_km)
    
    # 2. LOTC
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.02, 'sinkhorn_iter': 100, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
        'training': {'epochs': 30, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05, 'lambda_cons': 0.0}
    }
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=cfg['training']['lr_mass'])

    for ep in range(1, cfg['training']['epochs'] + 1):
        train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                    epsilon=cfg['ot']['epsilon'], 
                    sinkhorn_iter=cfg['ot']['sinkhorn_iter'],
                    lambda_mass=cfg['reg']['lambda_mass'],
                    lambda_disp=cfg['reg']['lambda_disp'],
                    lambda_lap=cfg['reg']['lambda_lap'],
                    lambda_cons=0.0,
                    disp_type='l2',
                    use_divergence=True)

    eval_out = evaluate(model, eval_loader, device, cfg['ot']['epsilon'], cfg['ot']['sinkhorn_iter'])
    lotc_metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    
    return km_metrics, lotc_metrics

def main():
    ratios = {"1:50": 0.02, "1:100": 0.01}
    all_results = {}
    
    for name, val in ratios.items():
        print(f"\n--- Running Severe Imbalance {name} ---")
        km, lotc = run_experiment(name, val)
        print(f"K-Means | ARI: {km['ARI']:.3f}")
        print(f"LOTC    | ARI: {lotc['ARI']:.3f}")
        all_results[name] = {'kmeans': km, 'lotc': lotc}
        
    out_path = "experiments/results/severe_imbalance_summary.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(all_results, f)
    print("\nResults saved to", out_path)

if __name__ == "__main__":
    main()
