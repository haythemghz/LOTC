import os
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from sklearn.preprocessing import StandardScaler
from src.experiments.run_experiment import build_model
from src.training.loops import train_epoch, evaluate
from src.eval.metrics import evaluate_clustering as compute_all_metrics
from sklearn.cluster import KMeans

def get_random_features(device, batch_size=256):
    # Load CIFAR-10
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = torchvision.datasets.CIFAR10(root="data/images", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load completely random ResNet18 (untrained, weights=None)
    resnet = models.resnet18(weights=None)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()
    
    all_feats, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            feats = resnet(X_batch.to(device))
            all_feats.append(feats.cpu())
            all_labels.append(y_batch)
            
    features = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    scaler = StandardScaler()
    features_np = scaler.fit_transform(features.numpy())
    return torch.tensor(features_np, dtype=torch.float32), labels.long()

def run_nossl_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extracting random features from untrained ResNet18...")
    X, y = get_random_features(device)
    
    input_dim = X.shape[1]
    num_classes = 10
    seed = 42
    
    # K-Means
    print("Running K-Means on random features...")
    km = KMeans(n_clusters=num_classes, random_state=seed, n_init=10)
    y_pred_km = km.fit_predict(X.numpy())
    km_metrics = compute_all_metrics(y.numpy(), y_pred_km)
    
    # LOTC
    print("Running LOTC on random features...")
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cfg = {
        'model': {'encoder': 'identity', 'embed_dim': input_dim, 'num_prototypes': num_classes, 'cost_type': 'squared_euclidean'},
        'ot': {'epsilon': 0.05, 'sinkhorn_iter': 50, 'use_divergence': True},
        'reg': {'lambda_mass': 0.0, 'lambda_disp': 0.001},
        'training': {'epochs': 20, 'lr_enc': 0.001, 'lr_proto': 0.01, 'lr_mass': 0.05}
    }
    
    model = build_model(cfg, input_dim).to(device)
    model.prototypes.init_from_kmeans(X.to(device))
    
    opt_proto = torch.optim.Adam(model.prototypes.parameters(), lr=0.01)
    # Fixed mass for fair comparison with standard KM
    opt_mass = torch.optim.Adam([model.prototypes.mass_logits], lr=0.00) 
    
    for _ in range(cfg['training']['epochs']):
        train_epoch(model, loader, None, opt_proto, opt_mass, device, 
                    epsilon=0.05, sinkhorn_iter=50, 
                    lambda_mass=0.0, lambda_disp=0.001, lambda_lap=0.0)
                    
    eval_out = evaluate(model, eval_loader, device, 0.05, 50)
    lotc_metrics = compute_all_metrics(eval_out['y_true'].numpy(), eval_out['y_pred'].numpy())
    
    results = {'kmeans': km_metrics, 'lotc': lotc_metrics}
    
    print("\nResults (No-SSL Random Backbone):")
    print(f"K-Means | ARI: {km_metrics['ARI']:.4f}")
    print(f"LOTC    | ARI: {lotc_metrics['ARI']:.4f}")
    
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/phase3_nossl.yaml", 'w') as f:
        yaml.dump(results, f)
        
    print("Saved to experiments/results/phase3_nossl.yaml")

if __name__ == "__main__":
    run_nossl_study()
