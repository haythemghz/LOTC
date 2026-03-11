import os
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

def run_stability_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = get_image_dataset("cifar10", pretrained=True, device=device.type)
    
    n_seeds = 10 # Reduced to 10 for faster validation
    input_dim = X.shape[1]
    num_classes = 10
    
    km_ari = []
    lotc_ari = []
    
    print(f"Running {n_seeds} seeds...")
    
    for i in range(n_seeds):
        seed = i + 1000
        # 1. K-Means (Random Init)
        km = KMeans(n_clusters=num_classes, init='random', n_init=1, random_state=seed)
        y_km = km.fit_predict(X.cpu().numpy())
        km_metrics = compute_all_metrics(y.cpu().numpy(), y_km)
        km_ari.append(km_metrics['ARI'])
        
        # 2. LOTC (Random Proto Init)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        cfg = {
            'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.02, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
            'training': {'epochs': 15, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
        }
        model = build_model(cfg, input_dim).to(device)
        
        # Random Init Prototypes
        idx = torch.randperm(X.size(0))[:num_classes]
        model.prototypes.prototypes.data.copy_(X[idx])
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)
        
        loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=256, shuffle=True)
        eval_loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=256, shuffle=False)
        
        for _ in range(cfg['training']['epochs']):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                        epsilon=0.02, sinkhorn_iter=50, 
                        lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)
            
        eval_out = evaluate(model, eval_loader, device, 0.02, 50)
        lotc_metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
        lotc_ari.append(lotc_metrics['ARI'])
        
        if (i+1) % 10 == 0:
            print(f"Seed {i+1}/{n_seeds} - KM: {np.mean(km_ari):.3f}, LOTC: {np.mean(lotc_ari):.3f}")
            
    # Results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/stability_study.yaml", 'w') as f:
        yaml.dump({'km_ari': km_ari, 'lotc_ari': lotc_ari}, f)
        
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.boxplot([km_ari, lotc_ari], labels=['K-Means (Random)', 'LOTC (Random)'])
    plt.ylabel('ARI')
    plt.title(f'Initialization Stability (CIFAR-10, n={n_seeds} seeds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("paper/figures/initialization_stability.png")
    print("Stability plot saved to paper/figures/initialization_stability.png")

if __name__ == "__main__":
    run_stability_study()
