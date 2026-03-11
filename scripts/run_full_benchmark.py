#!/usr/bin/env python3
"""
Central Benchmark Runner for LOTC Paper Revision.

Runs ALL methods × ALL datasets × ALL backbones × N seeds.
Saves structured YAML results for each configuration.

Usage:
    python scripts/run_full_benchmark.py --seeds 10 --device cuda
    python scripts/run_full_benchmark.py --datasets cifar10 stl10 --backbones resnet50 dino_vits16
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.images import get_image_dataset
from src.data.text_datasets import get_text_dataset
from src.experiments.baselines import run_kmeans, run_dec, run_scan_proxy, run_p2ot_proxy, run_imsat_proxy
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_DATASETS = ['cifar10', 'cifar100', 'stl10', 'tiny_imagenet']
TEXT_DATASETS = ['20newsgroups']
BACKBONES = ['resnet18', 'resnet50', 'dino_vits16']

BASELINE_METHODS = ['kmeans', 'dec', 'scan_proxy', 'imsat_proxy']

LOTC_CONFIG = {
    'model': {
        'encoder': 'identity', 'embed_dim': 128, 'num_prototypes': 10,
        'cost_type': 'squared_euclidean'
    },
    'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
    'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01},
    'training': {'epochs': 10, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
}

DATASET_K = {
    'cifar10': 10, 'cifar100': 20, 'stl10': 10,
    'tiny_imagenet': 200, '20newsgroups': 20,
}


def run_lotc_on_features(X, y, K, seed, device):
    """Run LOTC on pre-extracted features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = LOTC_CONFIG.copy()
    cfg['model'] = {**cfg['model'], 'num_prototypes': K}

    input_dim = X.shape[1]
    model = build_model(cfg, input_dim).to(device)

    dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    # Init prototypes
    model.prototypes.init_from_kmeans(X.to(device))

    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.05)

    start_time = time.time()
    peak_mem = 0

    for epoch in range(cfg['training']['epochs']):
        train_epoch(model, loader, None, opt_proto, opt_mass, device,
                    epsilon=0.05, sinkhorn_iter=50,
                    lambda_mass=0.01, lambda_disp=0.01, lambda_lap=0.0)

        if device.type == 'cuda':
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e6)

    elapsed = time.time() - start_time
    if device.type == 'cuda':
        torch.cuda.synchronize()

    eval_out = evaluate(model, eval_loader, device, 0.05, 50)
    metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    metrics['method'] = 'LOTC'
    metrics['runtime_s'] = elapsed
    metrics['peak_mem_mb'] = peak_mem
    metrics['n_params'] = sum(p.numel() for p in model.parameters())

    # Track mass statistics
    masses = model.prototypes.masses.detach().cpu()
    metrics['mass_min'] = float(masses.min())
    metrics['mass_max'] = float(masses.max())
    metrics['mass_entropy'] = float(-(masses * (masses + 1e-10).log()).sum())

    return metrics


def run_baseline_on_features(method_name, X_np, y_np, K, seed, device):
    """Run a baseline method on NumPy features."""
    start_time = time.time()

    if method_name == 'kmeans':
        metrics = run_kmeans(X_np, y_np, K, seed=seed)
    elif method_name == 'dec':
        metrics = run_dec(X_np, y_np, K, seed=seed, device=device)
    elif method_name == 'scan_proxy':
        metrics = run_scan_proxy(X_np, y_np, K, seed=seed)
    elif method_name == 'imsat_proxy':
        metrics = run_imsat_proxy(X_np, y_np, K, seed=seed, device=device)
    elif method_name == 'p2ot_proxy':
        metrics = run_p2ot_proxy(X_np, y_np, K, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics['runtime_s'] = time.time() - start_time
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full LOTC Benchmark")
    parser.add_argument('--seeds', type=int, default=10, help="Number of seeds")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--datasets', nargs='+', default=IMAGE_DATASETS + TEXT_DATASETS)
    parser.add_argument('--backbones', nargs='+', default=BACKBONES)
    parser.add_argument('--methods', nargs='+', default=BASELINE_METHODS + ['lotc'])
    parser.add_argument('--output_dir', type=str, default='experiments/results/full_benchmark')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    all_results = {}

    for dataset_name in args.datasets:
        is_text = dataset_name in TEXT_DATASETS
        backbones_to_use = ['identity'] if is_text else args.backbones
        K = DATASET_K.get(dataset_name, 10)

        for backbone in backbones_to_use:
            config_key = f"{dataset_name}_{backbone}"
            print(f"\n{'='*60}")
            print(f"CONFIG: {config_key} | K={K} | Seeds={args.seeds}")
            print(f"{'='*60}")

            # Load features
            if is_text:
                X, y = get_text_dataset(dataset_name)
            else:
                X, y = get_image_dataset(dataset_name, pretrained=True,
                                         device=args.device, backbone=backbone)

            X_np = X.numpy()
            y_np = y.numpy()

            config_results = {}
            for method in args.methods:
                method_runs = []
                for seed in range(args.seeds):
                    print(f"  {method} | seed {seed}...", end=" ", flush=True)
                    try:
                        if method == 'lotc':
                            metrics = run_lotc_on_features(X, y, K, seed, device)
                        else:
                            metrics = run_baseline_on_features(method, X_np, y_np, K, seed, str(device))
                        method_runs.append(metrics)
                        print(f"ARI={metrics['ARI']:.4f}")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        method_runs.append({'error': str(e)})

                config_results[method] = method_runs

            all_results[config_key] = config_results

            # Save incrementally
            out_path = os.path.join(args.output_dir, f"{config_key}.yaml")
            with open(out_path, 'w') as f:
                yaml.dump(config_results, f, default_flow_style=False)
            print(f"  Saved → {out_path}")

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_results.yaml")
    with open(combined_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"\nAll results saved to {combined_path}")


if __name__ == '__main__':
    main()
