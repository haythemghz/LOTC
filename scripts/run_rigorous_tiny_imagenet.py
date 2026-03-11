import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from src.eval.statistics import calculate_statistics
from sklearn.cluster import KMeans

def run_single_seed(X, y, device, input_dim, num_classes, seed):
    # 1. K-Means
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = kmeans.fit_predict(X.cpu().numpy())
    km_metrics = compute_all_metrics(y.cpu().numpy(), y_pred_km)
    
    # 2. LOTC
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.1, 'sinkhorn_iter': 50, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.001, 'lambda_lap': 0.0, 'disp_type': 'l2'},
        'training': {'epochs': 30, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
    }
    
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=cfg['training']['lr_mass'])

    for ep in range(cfg['training']['epochs']):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Tiny-ImageNet features via Pretrained ResNet18...")
    X, y = get_image_dataset("tiny_imagenet", pretrained=True, device=device.type)
    
    input_dim = X.shape[1] # 512
    num_classes = 200
    n_seeds = 10
    
    results = {'kmeans': {'ACC': [], 'ARI': [], 'NMI': []},
               'lotc': {'ACC': [], 'ARI': [], 'NMI': []}}
    
    print(f"Running {n_seeds} seeds for Tiny-ImageNet (ImageNet-Subset)...")
    
    for i in range(n_seeds):
        seed = 42 + i
        km, lotc = run_single_seed(X, y, device, input_dim, num_classes, seed)
        
        for k in ['ACC', 'ARI', 'NMI']:
            results['kmeans'][k].append(km[k])
            results['lotc'][k].append(lotc[k])
            
        print(f"Seed {i+1}/{n_seeds} | KM ARI: {km['ARI']:.3f} | LOTC ARI: {lotc['ARI']:.3f}")
        
    final_stats = {
        'kmeans': {
            'ACC': calculate_statistics(results['kmeans']['ACC']),
            'ARI': calculate_statistics(results['kmeans']['ARI']),
            'NMI': calculate_statistics(results['kmeans']['NMI'])
        },
        'lotc': {
            'ACC': calculate_statistics(results['lotc']['ACC']),
            'ARI': calculate_statistics(results['lotc']['ARI']),
            'NMI': calculate_statistics(results['lotc']['NMI'])
        }
    }
    
    out_path = "experiments/results/rigorous_tiny_imagenet.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(final_stats, f)
        
    print(f"\nTiny-ImageNet rigorous results saved to {out_path}")

if __name__ == "__main__":
    main()
