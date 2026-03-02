import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.ot.sinkhorn import sinkhorn_log_domain
from src.ot.costs import squared_euclidean_cost

def kmeans_loss_fn(X, prototypes):
    C = squared_euclidean_cost(X, prototypes)
    return C.min(dim=1)[0].mean()

def lotc_loss_fn(X, prototypes, epsilon=0.05):
    N = X.shape[0]
    K = prototypes.shape[0]
    u = torch.ones(N).to(X.device) / N
    v = torch.ones(K).to(X.device) / K
    C = squared_euclidean_cost(X, prototypes)
    _, cost = sinkhorn_log_domain(C, u, v, epsilon=epsilon, max_iter=50)
    return cost

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate synthetic 2D data (2 clusters)
    torch.manual_seed(42)
    X1 = torch.randn(50, 2) + torch.tensor([2.0, 2.0])
    X2 = torch.randn(50, 2) + torch.tensor([-2.0, -2.0])
    X = torch.cat([X1, X2]).to(device)
    
    # Prototypes: fix one, vary the other
    p1 = torch.tensor([2.0, 2.0]).to(device)
    
    # Grid of positions for p2
    x_range = np.linspace(-4, 4, 40)
    y_range = np.linspace(-4, 4, 40)
    XX, YY = np.meshgrid(x_range, y_range)
    
    Z_km = np.zeros_like(XX)
    Z_lotc = np.zeros_like(XX)
    
    print("Computing landscapes...")
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            p2 = torch.tensor([XX[i, j], YY[i, j]], dtype=torch.float32).to(device)
            prototypes = torch.stack([p1, p2])
            
            Z_km[i, j] = kmeans_loss_fn(X, prototypes).item()
            Z_lotc[i, j] = lotc_loss_fn(X, prototypes, epsilon=0.1).item()
            
    # Normalize for visualization
    Z_km = (Z_km - Z_km.min()) / (Z_km.max() - Z_km.min())
    Z_lotc = (Z_lotc - Z_lotc.min()) / (Z_lotc.max() - Z_lotc.min())
    
    # Plotting
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(XX, YY, Z_km, cmap='viridis', edgecolor='none')
    ax1.set_title("K-Means Landscape (Non-Differentiable)")
    ax1.set_zlabel("Normalized Loss")
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(XX, YY, Z_lotc, cmap='plasma', edgecolor='none')
    ax2.set_title("LOTC Landscape (Entropic Smoothness)")
    ax2.set_zlabel("Normalized Loss")
    
    plt.tight_layout()
    out_dir = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "loss_landscape.png"), dpi=300)
    print(f"Landscape saved to {os.path.join(out_dir, 'loss_landscape.png')}")

if __name__ == "__main__":
    main()
