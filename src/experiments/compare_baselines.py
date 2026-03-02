
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import yaml
import argparse
import os
from tqdm import tqdm

from src.experiments.run_experiment import get_dataset, build_model
from src.training.loops import train_epoch, evaluate, warmup_epoch
from src.eval.metrics import evaluate_clustering as compute_all_metrics


def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(device)
            z = model.encoder(x)
            # Apply hyperspherical projection if model does it
            z = torch.nn.functional.normalize(z, dim=1)
            features.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def run_kmeans_comparison(features, y_true, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(features)
    metrics = compute_all_metrics(y_true, y_pred)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Dataset
    dataset = get_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    
    # Get input_dim (Synced with run_experiment.py)
    sample_batch = next(iter(loader))
    sample_x = sample_batch[0]
    if isinstance(sample_x, (list, tuple)):
        # MultiViewDataset returns (v1, v2) where each is a batch
        input_dim = sample_x[0][0].shape if cfg['model']['encoder'] != 'mlp' else sample_x[0][0].numel()
    else:
        # sample_x is a batch [B, ...]
        input_dim = sample_x[0].numel() if cfg['model']['encoder'] == 'mlp' else sample_x[0].shape
    
    model = build_model(cfg, input_dim).to(device)
    
    results = {}
    
    # 1. Warmup (Representation Learning)
    warmup_epochs = cfg['training'].get('warmup_epochs', 0)
    if warmup_epochs > 0:
        print(f"\n--- Stage 1: SSL Warmup ({warmup_epochs} epochs) ---")
        opt_warmup = torch.optim.Adam(model.encoder.parameters(), lr=float(cfg['training'].get('lr_enc', 1e-3)))
        for ep in range(1, warmup_epochs + 1):
            w_metrics = warmup_epoch(model, loader, opt_warmup, device)
            print(f"Warmup {ep:02d} | Loss: {w_metrics['warmup_loss']:.4f}", flush=True)
            
        # Eval K-means on warmed features
        feats, y_true = extract_features(model, eval_loader, device)
        km_metrics = run_kmeans_comparison(feats, y_true, cfg['model']['num_prototypes'])
        print(f"Post-Warmup K-Means | ACC: {km_metrics['ACC']:.3f} | ARI: {km_metrics['ARI']:.3f} | NMI: {km_metrics['NMI']:.3f}", flush=True)
        results['post_warmup_kmeans'] = km_metrics

    # 2. Initialize Prototypes
    print("\n--- Initialising LOTC Prototypes ---")
    feats, y_true = extract_features(model, eval_loader, device)
    model.prototypes.init_from_kmeans(torch.from_numpy(feats).to(device))
    
    # 3. OT Phase
    print(f"\n--- Stage 2: OT Clustering ({cfg['training']['epochs']} epochs) ---", flush=True)
    
    # 3.1 Initial Evaluation (Epoch 0)
    eval_zero = evaluate(model, eval_loader, device,
                         epsilon=cfg['ot']['epsilon'],
                         sinkhorn_iter=cfg['ot'].get('sinkhorn_iter', 50))
    metrics_zero = compute_all_metrics(eval_zero['y_true'].numpy(), eval_zero['y_pred'].numpy())
    print(f"Epoch 000 | ACC: {metrics_zero['ACC']:.3f} | ARI: {metrics_zero['ARI']:.3f} (Initialization Check)", flush=True)

    # 3.2 Optimizers with refined LR
    lr_enc_joint = float(cfg['training'].get('lr_enc', 1e-4)) / 10.0 # Aggressive stability
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc_joint)
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=float(cfg['training'].get('lr_proto', 1e-2)))
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=float(cfg['training'].get('lr_mass', 1e-2)))
    
    for ep in range(1, cfg['training']['epochs'] + 1):
        # Freeze encoder for the first 5 epochs to stabilize prototypes
        current_opt_enc = None if ep <= 5 else opt_enc
        
        m = train_epoch(model, loader, current_opt_enc, opt_proto, opt_mass, device, 
                        epsilon=cfg['ot']['epsilon'], 
                        sinkhorn_iter=cfg['ot'].get('sinkhorn_iter', 50),
                        lambda_mass=cfg['reg']['lambda_mass'],
                        lambda_disp=cfg['reg']['lambda_disp'],
                        lambda_lap=cfg['reg'].get('lambda_lap', 0.0),
                        lambda_cons=cfg['training'].get('lambda_cons', 0.1), # Reduced cons during OT
                        disp_type=cfg['reg'].get('disp_type', 'l2'),
                        use_divergence=cfg['ot'].get('use_divergence', False))
        
        if ep % cfg['training'].get('eval_every', 2) == 0:
            eval_out = evaluate(model, eval_loader, device,
                                epsilon=cfg['ot']['epsilon'],
                                sinkhorn_iter=cfg['ot'].get('sinkhorn_iter', 50))
            eval_metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
            enc_status = "Frozen" if ep <= 5 else "Tuned"
            print(f"Epoch {ep:03d} | Enc: {enc_status} | Loss: {m['total_loss']:.4f} | ACC: {eval_metrics['ACC']:.3f} | ARI: {eval_metrics['ARI']:.3f}", flush=True)

    # 4. Final Comparison
    print("\n--- Final Comparison ---", flush=True)
    final_out = evaluate(model, eval_loader, device,
                         epsilon=cfg['ot']['epsilon'],
                         sinkhorn_iter=cfg['ot'].get('sinkhorn_iter', 50))
    final_eval = compute_all_metrics(final_out['y_true'].numpy(), final_out['y_pred'].numpy())
    print(f"Final LOTC    | ACC: {final_eval['ACC']:.3f} | ARI: {final_eval['ARI']:.3f} | NMI: {final_eval['NMI']:.3f}", flush=True)
    
    feats_final, y_true = extract_features(model, eval_loader, device)
    km_final = run_kmeans_comparison(feats_final, y_true, cfg['model']['num_prototypes'])
    print(f"Final K-Means | ACC: {km_final['ACC']:.3f} | ARI: {km_final['ARI']:.3f} | NMI: {km_final['NMI']:.3f}")
    
    # Save results summary
    summary_path = os.path.join("experiments", "results", f"{cfg['name']}_comparison.yaml")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        yaml.dump({
            'config': cfg,
            'lotc_final': final_eval,
            'kmeans_final': km_final,
            'post_warmup_kmeans': results.get('post_warmup_kmeans', {})
        }, f)

if __name__ == "__main__":
    main()
