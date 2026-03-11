import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
import os

from src.data.synthetic import generate_blobs, generate_noisy_moons, generate_noisy_circles, generate_unbalanced_blobs
from src.data.datasets import InMemoryDataset
from src.data.real_world import get_mnist, get_fashion_mnist, get_cifar10, get_subsampled_dataset
from PIL import Image
from src.models.encoders import IdentityEncoder, MLPEncoder, ResNetEncoder
from src.models.lotc_model import LOTCModel
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering
from src.utils.plotting import plot_2d_clusters, plot_loss_curve
import json

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset(cfg: dict):
    dataset_name = cfg['data']['name']
    n_samples = cfg['data'].get('n_samples', 1000)
    use_consistency = cfg['training'].get('lambda_cons', 0.0) > 0 or cfg['training'].get('warmup_epochs', 0) > 0
    
    if dataset_name == 'blobs':
        X, y = generate_blobs(n_samples=n_samples, n_clusters=cfg['data']['n_clusters'], noise=cfg['data'].get('noise', 1.0))
        ds = InMemoryDataset(X, y, standardize=cfg['data'].get('standardize', True))
    elif dataset_name == 'moons':
        X, y = generate_noisy_moons(n_samples=n_samples, noise=cfg['data'].get('noise', 0.1))
        ds = InMemoryDataset(X, y, standardize=cfg['data'].get('standardize', True))
    elif dataset_name == 'circles':
        X, y = generate_noisy_circles(n_samples=n_samples, noise=cfg['data'].get('noise', 0.05))
        ds = InMemoryDataset(X, y, standardize=cfg['data'].get('standardize', True))
    elif dataset_name == 'unbalanced':
        X, y = generate_unbalanced_blobs(n_samples=n_samples)
        ds = InMemoryDataset(X, y, standardize=cfg['data'].get('standardize', True))
    elif dataset_name in ['mnist', 'fmnist', 'cifar10']:
        # If using consistency, we need raw PIL images for strong augmentations
        base_transform = (lambda x: x) if use_consistency else None
        
        if dataset_name == 'mnist':
            # MNIST is grayscale, might need specific handling but for now same
            ds = get_mnist(transform=base_transform) if use_consistency else get_mnist()
        elif dataset_name == 'fmnist':
            ds = get_fashion_mnist(transform=base_transform)
        else:
            ds = get_cifar10(transform=base_transform)
            
        ds = get_subsampled_dataset(ds, n_samples)
        
        if use_consistency:
            from src.data.real_world import MultiViewDataset, get_strong_transforms
            img_size = 32 if dataset_name == 'cifar10' else 28
            strong_transform = get_strong_transforms(img_size=img_size)
            ds = MultiViewDataset(ds, strong_transform)
        return ds
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return ds

def build_model(cfg: dict, input_dim: int):
    enc_type = cfg['model']['encoder']
    embed_dim = cfg['model']['embed_dim']
    K = cfg['model']['num_prototypes']

    if enc_type == 'identity':
        encoder = IdentityEncoder()
        embed_dim = input_dim
    elif enc_type == 'mlp':
        encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=cfg['model'].get('hidden_dims', [128, 128]),
            output_dim=embed_dim
        )
    elif enc_type in ('resnet', 'resnet18'):
        encoder = ResNetEncoder(output_dim=embed_dim, backbone='resnet18')
    elif enc_type == 'resnet50':
        encoder = ResNetEncoder(output_dim=embed_dim, backbone='resnet50')
    elif enc_type == 'dino':
        from src.models.encoders import DINOViTEncoder
        encoder = DINOViTEncoder(output_dim=embed_dim)
    else:
        raise ValueError(f"Unknown encoder: {enc_type}")

    model = LOTCModel(
        encoder=encoder,
        num_prototypes=K,
        embed_dim=embed_dim,
        cost_type=cfg['model'].get('cost_type', 'cosine'),
        normalize=cfg['model'].get('normalize', True)
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="LOTC Experiment Runner")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Results directory
    results_dir = os.path.join('experiments', 'results', cfg.get('name', 'latest'))
    os.makedirs(results_dir, exist_ok=True)
    
    if cfg.get('wandb', False):
        if HAS_WANDB:
            wandb.init(project="lotc", config=cfg, name=cfg.get('name', 'experiment'))
        else:
            print("Warning: wandb is not installed. Continuous logging disabled.")
            cfg['wandb'] = False
        
    # Data
    dataset = get_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    
    # Input dim detection
    if hasattr(dataset, 'X'):
        input_dim = dataset.X.shape[1]
    elif hasattr(dataset, 'base_dataset'):
        # For MultiView or Subset, look at the base element
        sample_x, _ = dataset.base_dataset[0]
        if isinstance(sample_x, Image.Image):
             # Hardcode/Detect for raw images
             input_dim = (3, sample_x.size[1], sample_x.size[0]) if sample_x.mode == 'RGB' else (1, sample_x.size[1], sample_x.size[0])
        else:
             input_dim = sample_x.numel() if cfg['model']['encoder'] == 'mlp' else sample_x.shape
    else:
        sample_x, _ = dataset[0]
        if isinstance(sample_x, Image.Image):
             input_dim = (3, sample_x.size[1], sample_x.size[0]) if sample_x.mode == 'RGB' else (1, sample_x.size[1], sample_x.size[0])
        else:
             input_dim = sample_x.numel() if cfg['model']['encoder'] == 'mlp' else sample_x.shape
            
    # Model
    model = build_model(cfg, input_dim=input_dim).to(device)
    
    # Warmup / Pre-training
    warmup_epochs = int(cfg['training'].get('warmup_epochs', 0))
    enc_params = list(model.encoder.parameters())
    if warmup_epochs > 0 and enc_params:
        print(f"Starting SSL Warmup for {warmup_epochs} epochs...")
        from src.training.loops import warmup_epoch
        opt_warmup = torch.optim.Adam(enc_params, lr=float(cfg['training'].get('lr_enc', 1e-3)))
        for w_ep in range(1, warmup_epochs + 1):
            w_metrics = warmup_epoch(model, loader, opt_warmup, device)
            print(f"Warmup Epoch {w_ep:02d} | Loss: {w_metrics['warmup_loss']:.4f}")

    # Init prototypes
    print("Initialising prototypes via K-Means...")
    total_samples = len(dataset)
    init_size = min(total_samples, 2000)
    init_loader = DataLoader(dataset, batch_size=init_size, shuffle=True)
    batch = next(iter(init_loader))
    # Handle multi-view batch (v1, v2, y)
    init_batch = batch[0] if isinstance(batch, (list, tuple)) and len(batch) == 3 else batch[0]
    init_batch = init_batch.to(device)
    
    with torch.no_grad():
        initial_z = model.encoder(init_batch)
        model.prototypes.init_from_kmeans(initial_z)
    
    # Optimizers
    if enc_params:
        opt_enc = torch.optim.Adam(enc_params, lr=float(cfg['training'].get('lr_enc', 1e-3)))
    else:
        opt_enc = None
        
    opt_proto = torch.optim.Adam([model.prototypes.prototypes], lr=float(cfg['training'].get('lr_proto', 1e-2)))
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=float(cfg['training'].get('lr_mass', 1e-2)))
    
    # Training Loop
    epochs = int(cfg['training']['epochs'])
    history = []
    
    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            model=model,
            dataloader=loader,
            optimizer_encoder=opt_enc,
            optimizer_prototypes=opt_proto,
            optimizer_masses=opt_mass,
            device=device,
            epsilon=float(cfg['ot']['epsilon']),
            sinkhorn_iter=int(cfg['ot']['sinkhorn_iter']),
            lambda_mass=float(cfg['reg']['lambda_mass']),
            lambda_disp=float(cfg['reg']['lambda_disp']),
            lambda_lap=float(cfg['reg'].get('lambda_lap', 0.0)),
            lambda_cons=float(cfg['training'].get('lambda_cons', 0.0)),
            disp_type=cfg['reg'].get('disp_type', 'l2'),
            use_divergence=cfg['ot'].get('use_divergence', False)
        )
        history.append(metrics)
        
        if epoch % cfg['training'].get('eval_every', 5) == 0 or epoch == epochs:
            # Full dataset eval
            eval_out = evaluate(
                model=model,
                dataloader=DataLoader(dataset, batch_size=512, shuffle=False),
                device=device,
                epsilon=cfg['ot']['epsilon'],
                sinkhorn_iter=cfg['ot']['sinkhorn_iter']
            )
            
            y_true = eval_out['y_true'].numpy()
            y_pred = eval_out['y_pred'].numpy()
            
            cluster_metrics = evaluate_clustering(y_true, y_pred)
            metrics.update(cluster_metrics)
            
            print(f"Epoch {epoch:03d} | Loss: {metrics['total_loss']:.4f} | OT: {metrics['ot_cost']:.4f} "
                  f"| ACC: {cluster_metrics['ACC']:.3f} | NMI: {cluster_metrics['NMI']:.3f} | ARI: {cluster_metrics['ARI']:.3f}")
            
        if cfg.get('wandb', False) and HAS_WANDB:
            wandb.log(metrics, step=epoch)
            
    # Final cleanup and saving
    print(f"Saving results to {results_dir}")
    
    # Final eval
    final_eval = evaluate(model, DataLoader(dataset, batch_size=512), device, cfg['ot']['epsilon'], cfg['ot']['sinkhorn_iter'])
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pt'))
    
    # Plot results if 2D
    sample_x, _ = dataset[0]
    if (hasattr(dataset, 'X') and dataset.X.shape[1] == 2) or cfg['model'].get('embed_dim') == 2:
        plot_2d_clusters(
            final_eval['z'].numpy(), 
            final_eval['y_true'].numpy(), 
            final_eval['y_pred'].numpy(), 
            model.prototypes.prototypes.detach().cpu().numpy(),
            f"LOTC Results: {cfg['data']['name']}",
            save_path=os.path.join(results_dir, 'clusters.png')
        )
        plot_loss_curve(history, save_path=os.path.join(results_dir, 'loss.png'))

if __name__ == '__main__':
    main()
