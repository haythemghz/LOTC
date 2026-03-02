"""
Synthetic benchmark v2: Two Moons manifold untangling.
Generates clusters_synthetic.png and loss_synthetic.png.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from src.models.encoders import MLPEncoder
from src.models.lotc_model import LOTCModel
from src.training.loops import train_epoch

def run_synthetic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_np, y_np = make_moons(n_samples=2000, noise=0.05, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32).to(device)
    y = torch.tensor(y_np, dtype=torch.long).to(device)
    
    enc = MLPEncoder(2, [32, 32], 2).to(device) # Embed to 2D for visualization
    model = LOTCModel(enc, 2, 2, cost_type='cosine', normalize=True).to(device)
    
    # Init prototypes from kmeans on initial embeddings
    with torch.no_grad():
        z = model.encode(X)
        model.prototypes.init_from_kmeans(z)
        
    optimizer_e = torch.optim.Adam(model.encoder.parameters(), lr=0.01)
    optimizer_p = torch.optim.Adam([model.prototypes.prototypes], lr=0.01)
    optimizer_m = torch.optim.Adam([model.prototypes.mass_logits], lr=0.01)
    
    losses = []
    ot_losses = []
    
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    
    print("Training Synthetic...")
    for ep in range(1, 101):
        res = train_epoch(model, loader, optimizer_e, optimizer_p, optimizer_m, device, 
                          epsilon=0.05, sinkhorn_iter=100, lambda_mass=0.01, 
                          lambda_disp=0.005, lambda_lap=0.0, use_divergence=True)
        losses.append(res['loss'])
        ot_losses.append(res['ot_loss'])
        if ep % 20 == 0:
            print(f"  Ep {ep:3d} | Loss: {res['loss']:.4f} | OT: {res['ot_loss']:.4f}")
            
    # Visualization
    model.eval()
    with torch.no_grad():
        z = model.encode(X).cpu().numpy()
        protos = model.get_prototypes().cpu().numpy()
        
    os.makedirs('paper/figures', exist_ok=True)
    
    # 1. Clusters
    plt.figure(figsize=(6, 6))
    plt.scatter(z[:, 0], z[:, 1], c=y_np, s=10, alpha=0.5, cmap='coolwarm')
    plt.scatter(protos[:, 0], protos[:, 1], c='black', marker='X', s=200, label='Prototypes')
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), '--', color='gray', alpha=0.3)
    plt.title("Learned Hyperspherical Clusters (Two Moons)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.2)
    plt.savefig('paper/figures/clusters_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Loss
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label='Total Loss')
    plt.plot(ot_losses, label='OT Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title("Synthetic Convergence")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('paper/figures/loss_synthetic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Figures saved to paper/figures/clusters_synthetic.png and loss_synthetic.png")

if __name__ == '__main__':
    run_synthetic()
