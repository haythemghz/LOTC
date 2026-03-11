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

def run_epsilon_sweep(X, y, device):
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    input_dim = X.shape[1]
    num_classes = 10
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    for eps in epsilons:
        print(f"Sweep ε = {eps}")
        cfg = {
            'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': eps, 'sinkhorn_iter': 100, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
            'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
        }
        model = build_model(cfg, input_dim).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)
        
        grad_variances = []
        
        for ep in range(cfg['training']['epochs']):
            metrics = train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                                epsilon=eps, sinkhorn_iter=100, 
                                lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)
            
            # Estimate gradient variance (simplified: norm of proto grads)
            with torch.no_grad():
                g_norm = model.prototypes.prototypes.grad.norm().item() if model.prototypes.prototypes.grad is not None else 0
                grad_variances.append(g_norm)
                
        eval_out = evaluate(model, eval_loader, device, eps, 100)
        metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
        
        results.append({
            'epsilon': eps,
            'ARI': metrics['ARI'],
            'grad_var': np.mean(grad_variances),
            'ot_cost': eval_out['ot_cost']
        })
        
    return results

def run_t_sweep(X, y, device):
    t_values = [5, 10, 20, 50, 100]
    results = []
    
    input_dim = X.shape[1]
    num_classes = 10
    eps = 0.02
    
    dataset = torch.utils.data.TensorDataset(X, y)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Train once with T=100
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': eps, 'sinkhorn_iter': 100, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
        'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
    }
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)
    
    for _ in range(cfg['training']['epochs']):
        train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                    epsilon=eps, sinkhorn_iter=100, 
                    lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)
        
    # Evaluate with different T
    for t in t_values:
        print(f"Evaluating T = {t}")
        eval_out = evaluate(model, eval_loader, device, eps, t)
        metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
        results.append({
            'T': t,
            'ARI': metrics['ARI'],
            'ot_cost': eval_out['ot_cost']
        })
        
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = get_image_dataset("cifar10", pretrained=True, device=device.type)
    
    print("\n--- Running ε Sweep ---")
    eps_res = run_epsilon_sweep(X, y, device)
    
    print("\n--- Running T Sweep ---")
    t_res = run_t_sweep(X, y, device)
    
    # Plotting
    os.makedirs("paper/figures", exist_ok=True)
    
    # Plot 1: ε Phase Diagram
    eps_vals = [r['epsilon'] for r in eps_res]
    ari_vals = [r['ARI'] for r in eps_res]
    var_vals = [r['grad_var'] for r in eps_res]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eps_vals, ari_vals, 'b-o', label='ARI (Performance)')
    ax1.set_xlabel('Regularization ε')
    ax1.set_ylabel('ARI', color='b')
    ax1.set_xscale('log')
    
    ax2 = ax1.twinx()
    ax2.plot(eps_vals, var_vals, 'r-s', label='Grad Variance')
    ax2.set_ylabel('Gradient Variance (Estimation)', color='r')
    
    plt.title("ε Phase Diagram: Performance vs. Stability")
    fig.tight_layout()
    plt.savefig("paper/figures/epsilon_phase_diagram.png")
    
    # Plot 2: T Convergence
    t_vals = [r['T'] for r in t_res]
    t_ari = [r['ARI'] for r in t_res]
    t_cost = [r['ot_cost'] for r in t_res]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t_vals, t_ari, 'g-o', label='ARI')
    ax1.set_xlabel('Sinkhorn Iterations T')
    ax1.set_ylabel('ARI', color='g')
    
    ax2 = ax1.twinx()
    ax2.plot(t_vals, t_cost, 'k-x', label='OT Cost')
    ax2.set_ylabel('OT Cost', color='k')
    
    plt.title("Sinkhorn Convergence: Performance vs. Iterations T")
    fig.tight_layout()
    plt.savefig("paper/figures/t_convergence_sweep.png")
    
    # Save raw data
    with open("experiments/results/priority2_theoretical_validation.yaml", 'w') as f:
        yaml.dump({'eps_sweep': eps_res, 't_sweep': t_res}, f)
        
    print("\nPriority 2 plots in paper/figures/")

if __name__ == "__main__":
    main()
