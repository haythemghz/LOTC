import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from src.data.images import get_image_dataset
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate, warmup_epoch
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from src.data.datasets import MultiViewDataset
from sklearn.cluster import KMeans

def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            y = batch[-1]
            z = model.encoder(x.to(device))
            z = torch.nn.functional.normalize(z, dim=1)
            features.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def compute_kl_divergence(pred_masses, true_masses):
    """Computes KL divergence between learned masses and true proportions."""
    # Ensure they sum to 1 and avoid log(0)
    pred_masses = np.clip(pred_masses, 1e-8, 1.0)
    pred_masses /= pred_masses.sum()
    true_masses = np.clip(true_masses, 1e-8, 1.0)
    true_masses /= true_masses.sum()
    return np.sum(true_masses * np.log(true_masses / pred_masses))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Load Imbalanced CIFAR-10 Dataset
    print("Loading imbalanced_cifar10...")
    # This automatically computes resnet features via extract_pretrained_features
    # if pretrained=True is supported. Oh wait, `get_dataset` from config handles true images?
    # Our data pipeline in src/data/images.py can do pretrained=True.
    X, y = get_image_dataset("imbalanced_cifar10", pretrained=True, device=device.type)
    dataset = torch.utils.data.TensorDataset(X, y)
    
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    true_proportions = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    input_dim = X.shape[1] # usually 512 for resnet18 features
    
    results = {}

    # 2. Baseline 1: K-Means (post-warmup/pretrained features)
    print("\n--- Baseline 1: K-Means ---")
    kmeans = KMeans(n_clusters=6, random_state=seed, n_init=10)
    y_pred_km = kmeans.fit_predict(X.numpy())
    km_metrics = compute_all_metrics(y.numpy(), y_pred_km)
    
    # K-Means mass explicitly calculated by label counts
    km_counts = np.bincount(y_pred_km, minlength=6)
    km_masses = km_counts / km_counts.sum()
    # Note: K-means labels are invariant to permutations. To compute true mass KL, we need to align the labels or sort the masses.
    # True masses are sorted descending. So we sort predicted masses descending to match.
    km_masses_sorted = np.sort(km_masses)[::-1]
    km_kl = compute_kl_divergence(km_masses_sorted, true_proportions)
    
    print(f"K-Means | ACC: {km_metrics['ACC']:.3f} | ARI: {km_metrics['ARI']:.3f} | KL: {km_kl:.3f}")
    results['kmeans'] = {'metrics': km_metrics, 'kl_mass': float(km_kl)}

    # Base Config for LOTC
    cfg = {
        'model': {
            'encoder': 'mlp', 
            'embed_dim': 128,  # Projection to lower dim
            'num_prototypes': 6, 
            'cost_type': 'squared_euclidean'
        },
        'ot': {
            'epsilon': 0.02, # Sharper assignments
            'sinkhorn_iter': 100, 
            'use_divergence': True
        },
        'reg': {
            'lambda_mass': 0.005, # Weaker mass entropy to allow imbalance
            'lambda_disp': 0.01,   # Stronger dispersion to prevent prototype bunching
            'lambda_lap': 0.0, 
            'disp_type': 'collision' # Use repulsive force
        },
        'training': {
            'epochs': 50, 
            'lr_enc': 0.0005, 
            'lr_proto': 0.001, 
            'lr_mass': 0.02, 
            'lambda_cons': 0.5  # Add consistency strength
        }
    }

    # Helper function to run LOTC
    def run_lotc(variant, lr_mass):
        print(f"\n--- LOTC Variant: {variant} ---")
        torch.manual_seed(seed)
        model = build_model(cfg, input_dim).to(device)
        
        # Initialise prototypes from ENCODED data to match dimensions
        with torch.no_grad():
            z = model.encoder(X.to(device))
            model.prototypes.init_from_kmeans(z)
        
        enc_params = list(model.encoder.parameters())
        if len(enc_params) > 0:
            opt_enc = torch.optim.Adam(enc_params, lr=cfg['training']['lr_enc'])
        else:
            opt_enc = None
            
        opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=cfg['training']['lr_proto'])
        
        if lr_mass > 0:
            opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=lr_mass)
        else:
            opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.0) # Effectively no updates

        if cfg['training']['lambda_cons'] > 0:
            train_loader = DataLoader(MultiViewDataset(X, y), batch_size=256, shuffle=True)
        else:
            train_loader = DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=256, shuffle=True)

        for ep in range(1, cfg['training']['epochs'] + 1):
            stats = train_epoch(model, train_loader, opt_enc, opt_proto, opt_mass, device, 
                        epsilon=cfg['ot']['epsilon'], 
                        sinkhorn_iter=cfg['ot']['sinkhorn_iter'],
                        lambda_mass=cfg['reg']['lambda_mass'] if lr_mass > 0 else 0.0,
                        lambda_disp=cfg['reg']['lambda_disp'],
                        lambda_lap=cfg['reg']['lambda_lap'],
                        lambda_cons=cfg['training']['lambda_cons'],
                        disp_type=cfg['reg']['disp_type'],
                        use_divergence=True)
            if ep % 10 == 0:
                print(f"  Epoch {ep:2d} | Loss: {stats.get('loss', 0):.4f} | OT: {stats.get('ot_loss', 0):.4f} | Cons: {stats.get('cons_loss', 0):.4f}")

        eval_out = evaluate(model, eval_loader, device, cfg['ot']['epsilon'], cfg['ot']['sinkhorn_iter'])
        metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
        
        # Calculate KL Divergence
        pred_masses = model.prototypes.masses.detach().cpu().numpy()
        pred_masses_sorted = np.sort(pred_masses)[::-1]
        kl = compute_kl_divergence(pred_masses_sorted, true_proportions)
        
        print(f"{variant} | ACC: {metrics['ACC']:.3f} | ARI: {metrics['ARI']:.3f} | NMI: {metrics['NMI']:.3f} | KL: {kl:.3f}")
        return metrics, float(kl)

    # 3. Fixed-Mass LOTC (Uniform)
    metrics_fixed, kl_fixed = run_lotc("Fixed Mass (Uniform)", lr_mass=0.0)
    results['lotc_fixed'] = {'metrics': metrics_fixed, 'kl_mass': kl_fixed}

    # 4. Learned-Mass LOTC
    metrics_learned, kl_learned = run_lotc("Learned Mass", lr_mass=0.05)
    results['lotc_learned'] = {'metrics': metrics_learned, 'kl_mass': kl_learned}

    # 5. DEC Baseline
    print("\n--- DEC Baseline ---")
    try:
        from src.eval.deep_baselines import run_dec_baseline
        dec_metrics = run_dec_baseline(X.numpy(), y.numpy(), n_clusters=6)
        
        dec_counts = np.bincount(dec_metrics.get('y_pred', np.zeros_like(y.numpy())), minlength=6)
        dec_masses = dec_counts / max(dec_counts.sum(), 1)
        dec_mass_sorted = np.sort(dec_masses)[::-1]
        dec_kl = compute_kl_divergence(dec_mass_sorted, true_proportions)
        
        print(f"DEC | ACC: {dec_metrics['ACC']:.3f} | ARI: {dec_metrics['ARI']:.3f} | KL: {dec_kl:.3f}")
        results['dec'] = {'metrics': dec_metrics, 'kl_mass': float(dec_kl)}
    except Exception as e:
        print(f"Skipping DEC: {e}")

    # Output to YAML
    out_path = "experiments/results/imbalanced_cifar10_summary.yaml"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(results, f)

    print("\nExperiment Complete. Results saved to", out_path)

if __name__ == "__main__":
    main()
