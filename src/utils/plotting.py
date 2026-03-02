import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_2d_clusters(z, y_true, y_pred, prototypes, title, save_path=None):
    """
    Plots the 2D latent space with data points, true labels (colors), 
    predicted labels (markers), and prototype locations (stars).
    """
    plt.figure(figsize=(10, 8))
    
    # Use different colors for true clusters
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y_true, cmap='viridis', alpha=0.5, s=20)
    plt.colorbar(scatter, label='True Cluster')
    
    # Plot prototypes
    plt.scatter(prototypes[:, 0], prototypes[:, 1], c='red', marker='*', s=200, label='Prototypes', edgecolors='black')
    
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    plt.close()

def plot_loss_curve(metrics_history, save_path=None):
    """
    Plots multiple loss components over epochs.
    """
    epochs = range(1, len(metrics_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [m['total_loss'] for m in metrics_history], label='Total Loss', lw=2)
    plt.plot(epochs, [m['ot_cost'] for m in metrics_history], label='OT Cost', linestyle='--')
    
    plt.title("Training Dynamics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
