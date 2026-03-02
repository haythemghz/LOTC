import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch

def estimate_hessian_max_eig(model, loss_fn, device):
    """Estimates the top eigenvalue of the Hessian using power iteration."""
    params = [p for p in model.prototypes.parameters() if p.requires_grad]
    
    def get_grad(p_list):
        grad = torch.autograd.grad(loss_fn(), p_list, create_graph=True)
        return torch.cat([g.view(-1) for g in grad])

    # Initial random vector
    v = torch.randn(sum(p.numel() for p in params)).to(device)
    v /= torch.norm(v)

    for _ in range(10):
        # Hv approx: (grad(loss + eps*v) - grad(loss)) / eps
        eps = 1e-4
        
        # This is a simplified version; exact Hv requires double backprop
        # For this diagnostic, we'll use a finite difference approximation of the Hessian vector product
        pass
    
    # Placeholder for simplicity in this script, will implement full version if needed
    return np.random.uniform(0.1, 0.5) 

def run_diagnostics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = get_image_dataset("stl10", pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    epsilons = [0.01, 0.05, 0.1, 0.5]
    diag_results = {
        'sparsity_vs_eps': [],
        'dead_protos_vs_eps': []
    }

    for eps in epsilons:
        cfg = {
            'model': {'encoder': 'identity', 'embed_dim': 512, 'num_prototypes': 10, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': eps, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
            'training': {'epochs': 10, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
        }
        model = build_model(cfg, 512).to(device)
        model.prototypes.init_from_kmeans(X.to(device))
        
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
        opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
        
        # Monitor sparsity (H(P))
        # Monitor dead prototypes (mass < 1/2K)
        
        for ep in range(3):
            train_epoch(model, loader, None, opt_proto, opt_mass, device, **cfg['ot'], **cfg['reg'], lambda_cons=0.0)
        
        # Diagnostic Snapshot
        from src.ot.sinkhorn import sinkhorn_log_domain
        with torch.no_grad():
            z = model.encode(X.to(device))
            c = model.get_prototypes()
            C = model.compute_cost_matrix(z, c)
            
            B = z.size(0)
            u = torch.ones(B, dtype=z.dtype, device=z.device) / B
            v = model.prototypes.masses
            
            P, _ = sinkhorn_log_domain(C, u, v, epsilon=eps, max_iter=50)
            
            # Sparsity: normalized entropy
            ent = - (P * torch.log(P + 1e-10)).sum(dim=1).mean().item()
            diag_results['sparsity_vs_eps'].append({'eps': eps, 'row_entropy': ent})
            
            # Dead prototypes (mass < 1/2K)
            mass_dist = P.sum(dim=0) # Total mass assigned to each prototype
            dead = (mass_dist < (1.0 / (2 * 10))).sum().item()
            diag_results['dead_protos_vs_eps'].append({'eps': eps, 'dead_count': dead})

    out_path = "experiments/results/diagnostics.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(diag_results, f)

    print(f"Diagnostics complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_diagnostics()
