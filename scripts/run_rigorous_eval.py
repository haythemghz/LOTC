import os
import torch
import numpy as np
import yaml
import argparse
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from src.eval.statistics import paired_ttest, cohens_d
from sklearn.cluster import KMeans
from scipy import stats

def run_single_seed(dataset_name, seed, device, cfg):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    X, y = get_image_dataset(dataset_name, pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    input_dim = X.shape[1]
    num_classes = 20 if dataset_name == "cifar100" else 10
    
    seed_results = {}
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = kmeans.fit_predict(X.numpy())
    seed_results['kmeans'] = compute_all_metrics(y.numpy(), y_pred_km)
    
    # 2. LOTC
    model_cfg = cfg.copy()
    model_cfg['model']['num_prototypes'] = num_classes
    model = build_model(model_cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    enc_params = list(model.encoder.parameters())
    opt_enc = torch.optim.Adam(enc_params, lr=model_cfg['training']['lr_enc']) if enc_params else None
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=model_cfg['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=model_cfg['training']['lr_mass'])
    
    for ep in range(1, model_cfg['training']['epochs'] + 1):
        train_epoch(model, loader, opt_enc, opt_proto, opt_mass, device, 
                    **model_cfg['ot'], **model_cfg['reg'], 
                    lambda_cons=model_cfg['training']['lambda_cons'])
        
    eval_out = evaluate(model, eval_loader, device, model_cfg['ot']['epsilon'], model_cfg['ot']['sinkhorn_iter'])
    seed_results['lotc'] = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    
    return seed_results

def compute_stats(scores_lotc, scores_base):
    res = {}
    res['mean_lotc'] = np.mean(scores_lotc)
    res['std_lotc'] = np.std(scores_lotc)
    res['mean_base'] = np.mean(scores_base)
    res['std_base'] = np.std(scores_base)
    
    t_res = paired_ttest(scores_lotc, scores_base)
    res['p_value'] = t_res['p_value']
    res['cohens_d'] = cohens_d(scores_lotc, scores_base)
    
    # 95% Confidence Interval for the difference
    diff = scores_lotc - scores_base
    n = len(diff)
    se = stats.sem(diff)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    res['ci_95'] = [np.mean(diff) - h, np.mean(diff) + h]
    
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "stl10"], default="stl10")
    parser.add_argument("--n_seeds", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default high-quality configs per dataset
    configs = {
        "stl10": {
            'model': {'encoder': 'identity', 'embed_dim': 512, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.005, 'lambda_lap': 0.0, 'disp_type': 'collision'},
            'training': {'epochs': 30, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
        },
        "cifar100": {
            'model': {'encoder': 'identity', 'embed_dim': 512, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.02, 'sinkhorn_iter': 100, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
            'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.02, 'lambda_cons': 0.0}
        },
        "cifar10": {
            'model': {'encoder': 'identity', 'embed_dim': 512, 'cost_type': 'squared_euclidean'},
            'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
            'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.01, 'lambda_lap': 0.0, 'disp_type': 'collision'},
            'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
        }
    }
    
    cfg = configs[args.dataset]
    all_runs = []
    
    print(f"Starting rigorous evaluation for {args.dataset} over {args.n_seeds} seeds...")
    
    for s in range(args.n_seeds):
        seed = 42 + s
        print(f"Running seed {seed} ({s+1}/{args.n_seeds})...")
        res = run_single_seed(args.dataset, seed, device, cfg)
        all_runs.append(res)
        
    # Aggregate
    aggregated = {
        'dataset': args.dataset,
        'n_seeds': args.n_seeds,
        'seeds': [42 + i for i in range(args.n_seeds)],
        'metrics': {}
    }
    
    methods = ['kmeans', 'lotc']
    metric_keys = ['ARI', 'ACC', 'NMI']
    
    for m in methods:
        aggregated['metrics'][m] = {k: [run[m][k] for run in all_runs] for k in metric_keys}
    
    # Stats
    stats_results = {}
    for k in metric_keys:
        stats_results[k] = compute_stats(np.array(aggregated['metrics']['lotc'][k]), 
                                        np.array(aggregated['metrics']['kmeans'][k]))
        
    aggregated['stats_vs_kmeans'] = stats_results
    
    # Save
    out_path = f"experiments/results/{args.dataset}_rigorous.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(aggregated, f)
        
    print(f"\nResults for {args.dataset}:")
    for k in metric_keys:
        s = stats_results[k]
        print(f"{k:3s} | LOTC: {s['mean_lotc']:.4f} ± {s['std_lotc']:.4f} | K-Means: {s['mean_base']:.4f} ± {s['std_base']:.4f}")
        print(f"      p-val: {s['p_value']:.4e} | Cohen's d: {s['cohens_d']:.3f} | 95% CI: [{s['ci_95'][0]:.4f}, {s['ci_95'][1]:.4f}]")

if __name__ == "__main__":
    main()
