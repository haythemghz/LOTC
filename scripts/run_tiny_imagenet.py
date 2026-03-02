import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from sklearn.cluster import KMeans

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading Tiny-ImageNet features via Pretrained ResNet18...")
    # This will download/load ~400MB
    X, y = get_image_dataset("tiny_imagenet", pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    input_dim = X.shape[1] # 512
    num_classes = 200
    
    results = {}

    # 1. Baseline: K-Means
    print("\n--- Baseline: K-Means ---")
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = kmeans.fit_predict(X.numpy())
    km_metrics = compute_all_metrics(y.numpy(), y_pred_km)
    print(f"K-Means | ACC: {km_metrics['ACC']:.3f} | ARI: {km_metrics['ARI']:.3f} | NMI: {km_metrics['NMI']:.3f}")
    results['kmeans'] = {'metrics': km_metrics}

    # Base Config for LOTC
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.1, 'sinkhorn_iter': 50, 'use_divergence': True},
        'reg': {'lambda_mass': 0.01, 'lambda_disp': 0.001, 'lambda_lap': 0.0, 'disp_type': 'l2'},
        'training': {'epochs': 30, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.01, 'lambda_cons': 0.0}
    }

    # 2. LOTC
    print("\n--- LOTC ---")
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    enc_params = list(model.encoder.parameters())
    if len(enc_params) > 0:
        opt_enc = torch.optim.Adam(enc_params, lr=cfg['training']['lr_enc'])
    else:
        opt_enc = None
        
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg['training']['lr_proto'])
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=cfg['training']['lr_mass'])

    for ep in range(1, cfg['training']['epochs'] + 1):
        train_epoch(model, loader, opt_enc, opt_proto, opt_mass, device, 
                    epsilon=cfg['ot']['epsilon'], 
                    sinkhorn_iter=cfg['ot']['sinkhorn_iter'],
                    lambda_mass=cfg['reg']['lambda_mass'],
                    lambda_disp=cfg['reg']['lambda_disp'],
                    lambda_lap=cfg['reg']['lambda_lap'],
                    lambda_cons=0.0,
                    disp_type='l2',
                    use_divergence=True)

    eval_out = evaluate(model, eval_loader, device, cfg['ot']['epsilon'], cfg['ot']['sinkhorn_iter'])
    metrics_lotc = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    
    print(f"LOTC | ACC: {metrics_lotc['ACC']:.3f} | ARI: {metrics_lotc['ARI']:.3f} | NMI: {metrics_lotc['NMI']:.3f}")
    results['lotc'] = {'metrics': metrics_lotc}

    # Output to YAML
    out_path = "experiments/results/tiny_imagenet_summary.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(results, f)

    print("\nExperiment Complete. Results saved to", out_path)

if __name__ == "__main__":
    main()
