import os
import torch
import numpy as np
import yaml
import time
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics

def run_cardinality_study(X, y, device):
    k_values = [10, 50, 100, 500]
    results = []
    
    input_dim = X.shape[1]
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True) # Large batch for high K
    eval_loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    for K in k_values:
        print(f"\nEvaluating K = {K}")
        cfg = {
            'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': K, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
            'training': {'epochs': 10, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
        }
        model = build_model(cfg, input_dim).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)
        
        start_time = time.time()
        for _ in range(cfg['training']['epochs']):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                        epsilon=0.05, sinkhorn_iter=50, 
                        lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start_time
        
        eval_out = evaluate(model, eval_loader, device, 0.05, 50)
        
        # Calculate assignment sparsity (percentage of elements < 1e-4 in P)
        P = eval_out['P']
        sparsity = (P < 1e-4).float().mean().item()
        
        results.append({
            'K': K,
            'runtime': elapsed,
            'sparsity': sparsity,
            'ot_cost': eval_out['ot_cost']
        })
        
    return results

def run_batch_size_study(X, y, device):
    K = 50
    b_k_ratios = [1, 2, 5, 10, 20] # B = K * ratio
    results = []
    
    input_dim = X.shape[1]
    dataset = torch.utils.data.TensorDataset(X, y)
    
    for ratio in b_k_ratios:
        B = int(K * ratio)
        print(f"\nEvaluating B/K = {ratio} (B = {B})")
        
        loader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)
        eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        cfg = {
            'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': K, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
            'training': {'epochs': 10, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
        }
        
        model = build_model(cfg, input_dim).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)
        
        for _ in range(cfg['training']['epochs']):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                        epsilon=0.05, sinkhorn_iter=50, 
                        lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)
            
        eval_out = evaluate(model, eval_loader, device, 0.05, 50)
        
        metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
        results.append({
            'ratio': ratio,
            'B': B,
            'ARI': metrics['ARI'],
            'ot_cost': eval_out['ot_cost']
        })
        
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = get_image_dataset("cifar10", pretrained=True, device=device.type)
    
    print("--- Running High Cardinality Study ---")
    cardinality_res = run_cardinality_study(X, y, device)
    
    print("\n--- Running Batch Size Scaling Validation ---")
    batch_res = run_batch_size_study(X, y, device)
    
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/phase3_scaling_limits.yaml", 'w') as f:
        yaml.dump({'cardinality': cardinality_res, 'batch_size': batch_res}, f)
        
    print("\nPhase 3 results saved to experiments/results/phase3_scaling_limits.yaml")

if __name__ == "__main__":
    main()
