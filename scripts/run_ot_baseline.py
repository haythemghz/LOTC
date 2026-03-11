import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from src.ot.sinkhorn import sinkhorn_log_domain

class OTKMeans:
    """Simple Mini-batch OT Clustering (mBOT) baseline."""
    def __init__(self, n_clusters, epsilon=0.05, n_iter=50):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.prototypes = None

    def fit(self, loader, device, epochs=10, lr=0.1):
        # Initialize prototypes from first batch
        X_init, _ = next(iter(loader))
        X_init = X_init.to(device)
        idx = torch.randperm(X_init.size(0))[:self.n_clusters]
        self.prototypes = torch.nn.Parameter(X_init[idx].clone().detach())
        
        optimizer = torch.optim.Adam([self.prototypes], lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                B = X_batch.size(0)
                
                # Compute cost matrix
                C = torch.cdist(X_batch, self.prototypes, p=2)**2
                
                # Uniform marginals
                a = torch.ones(B, device=device) / B
                b = torch.ones(self.n_clusters, device=device) / self.n_clusters
                
                # Sinkhorn
                P, _ = sinkhorn_log_domain(C, a, b, epsilon=self.epsilon, max_iter=self.n_iter)
                
                # OT Loss
                loss = torch.sum(P.detach() * C)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    def predict(self, X, device):
        C = torch.cdist(X.to(device), self.prototypes, p=2)**2
        return torch.argmin(C, dim=1).cpu()

def run_ot_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = get_image_dataset("cifar10", pretrained=True, device=device.type)
    
    loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=256, shuffle=True)
    
    print("Training OT-KMeans (mBOT) baseline...")
    model = OTKMeans(n_clusters=10, epsilon=0.05, n_iter=50)
    model.fit(loader, device, epochs=15, lr=0.01)
    
    y_pred = model.predict(X, device)
    metrics = compute_all_metrics(y.numpy(), y_pred.numpy())
    
    print("\nOT-KMeans Baseline Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/ot_baseline_cifar10.yaml", 'w') as f:
        yaml.dump(metrics, f)

if __name__ == "__main__":
    run_ot_baseline()
