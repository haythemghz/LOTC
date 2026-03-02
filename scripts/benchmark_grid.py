import os
import torch
import time
import yaml
import numpy as np
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch

def benchmark_config(dataset_name, K, T, B, device):
    print(f"Benchmarking: K={K}, T={T}, B={B}...")
    
    # Load sample data (subsample for speed if N is large, but for scaling N should be realistic)
    X, y = get_image_dataset(dataset_name, pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)
    
    input_dim = X.shape[1]
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': K, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.05, 'sinkhorn_iter': T, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
        'training': {'epochs': 1, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
    }
    
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    # Reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.time()
    
    # Run 1 epoch
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
    train_epoch(model, loader, None, opt_proto, opt_mass, device,
                epsilon=cfg['ot']['epsilon'],
                sinkhorn_iter=cfg['ot']['sinkhorn_iter'],
                lambda_mass=cfg['reg']['lambda_mass'],
                lambda_disp=cfg['reg']['lambda_disp'],
                lambda_lap=cfg['reg']['lambda_lap'],
                lambda_cons=0.0,
                disp_type=cfg['reg']['disp_type'],
                use_divergence=cfg['ot']['use_divergence'])
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) # MB
        
    return {
        'K': K, 'T': T, 'B': B,
        'epoch_time': epoch_time,
        'peak_memory_mb': peak_mem
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = "stl10" # standardized for scaling
    
    K_vals = [10, 50, 100, 500, 1000]
    T_vals = [5, 10, 20, 50, 100]
    B_vals = [64, 128, 256, 512, 1024]
    
    results = {
        'scaling_vs_K': [],
        'scaling_vs_T': [],
        'scaling_vs_B': []
    }
    
    # 1. Scale vs K (fixed T=50, B=256)
    for K in K_vals:
        res = benchmark_config(dataset, K, 50, 256, device)
        results['scaling_vs_K'].append(res)
        
    # 2. Scale vs T (fixed K=10, B=256)
    for T in T_vals:
        res = benchmark_config(dataset, 10, T, 256, device)
        results['scaling_vs_T'].append(res)
        
    # 3. Scale vs B (fixed K=10, T=50)
    for B in B_vals:
        res = benchmark_config(dataset, 10, 50, B, device)
        results['scaling_vs_B'].append(res)
        
    out_path = "experiments/results/scaling_study.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(results, f)
        
    print(f"Scaling study complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
