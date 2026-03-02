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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load Fashion-MNIST (Embedded via Pretrained ResNet)
    print("Loading Fashion-MNIST features...")
    X, y = get_image_dataset("fmnist", pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    input_dim = X.shape[1]
    num_prototypes = 10
    
    epsilons = [0.01, 0.05, 0.1, 0.5]
    iters = [5, 10, 20, 50]
    
    results = {'epsilon_study': {}, 'iter_study': {}}

    # Base configuration
    base_cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': 10, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.001, 'lambda_lap': 0.0, 'disp_type': 'l2'},
        'training': {'epochs': 10, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01}
    }

    # 2. Epsilon Sensitivity Study (Fix Iter=50)
    print("\n--- Epsilon Sensitivity Study ---")
    for eps in epsilons:
        print(f"Testing epsilon={eps}...")
        model = build_model(base_cfg, input_dim).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
        
        t0 = time.time()
        for ep in range(1, base_cfg['training']['epochs'] + 1):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                        epsilon=eps, sinkhorn_iter=50,
                        lambda_mass=0.01, lambda_disp=0.001, lambda_lap=0.0, use_divergence=True)
        
        dt = time.time() - t0
        eval_out = evaluate(model, eval_loader, device, eps, 50)
        metrics = compute_all_metrics(y.numpy(), eval_out['y_pred'].numpy())
        results['epsilon_study'][eps] = {'ARI': float(metrics['ARI']), 'ACC': float(metrics['ACC']), 'time': dt}
        print(f"eps={eps} | ARI: {metrics['ARI']:.3f} | Time: {dt:.1f}s")

    # 3. Sinkhorn Iteration Study (Fix Epsilon=0.05)
    print("\n--- Sinkhorn Iteration Study ---")
    for t in iters:
        print(f"Testing sinkhorn_iter={t}...")
        model = build_model(base_cfg, input_dim).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
        
        t0 = time.time()
        for ep in range(1, base_cfg['training']['epochs'] + 1):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                        epsilon=0.05, sinkhorn_iter=t,
                        lambda_mass=0.01, lambda_disp=0.001, lambda_lap=0.0, use_divergence=True)
        
        dt = time.time() - t0
        eval_out = evaluate(model, eval_loader, device, 0.05, t)
        metrics = compute_all_metrics(y.numpy(), eval_out['y_pred'].numpy())
        results['iter_study'][t] = {'ARI': float(metrics['ARI']), 'ACC': float(metrics['ACC']), 'time': dt}
        print(f"T={t} | ARI: {metrics['ARI']:.3f} | Time: {dt:.1f}s")

    # Save results
    out_path = "experiments/results/sensitivity_study.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(results, f)

    print("\nSensitivity Study Complete. Results saved to", out_path)

if __name__ == "__main__":
    main()
