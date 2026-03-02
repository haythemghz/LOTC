import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_encoder: torch.optim.Optimizer | None,
    optimizer_prototypes: torch.optim.Optimizer,
    optimizer_masses: torch.optim.Optimizer,
    device: torch.device,
    epsilon: float,
    sinkhorn_iter: int,
    lambda_mass: float,
    lambda_disp: float,
    lambda_lap: float,
    lambda_cons: float = 0.0,
    disp_type: str = 'collision',
    W_graph: torch.Tensor | None = None,
    use_divergence: bool = False,
    mass_prior: torch.Tensor | None = None,
    grad_clip: float = 5.0
) -> Dict[str, float]:
    """
    Trains the LOTC model for a single epoch.
    Supports consistency regularization if dataloader returns (v1, v2, y).
    """
    model.train()
    
    epoch_metrics = {
        'loss': 0.0,
        'ot_loss': 0.0,
        'reg_mass': 0.0,
        'reg_disp': 0.0,
        'cons_loss': 0.0
    }
    
    from src.models.consistency import AssignmentConsistencyLoss
    cons_criterion = AssignmentConsistencyLoss() if lambda_cons > 0 else None
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle (x, y) or (v1, v2, y)
        if len(batch) == 3:
            v1, v2, _ = batch
            v1, v2 = v1.to(device), v2.to(device)
            multi_view = True
        else:
            x, _ = batch
            x = x.to(device)
            multi_view = False
            
        if W_graph is not None:
            W_graph = W_graph.to(device)
            
        # Zero gradients
        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad()
        optimizer_prototypes.zero_grad()
        optimizer_masses.zero_grad()
        
        # Forward pass(es)
        if multi_view and cons_criterion is not None:
            out1 = model(v1, epsilon=epsilon, sinkhorn_iter=sinkhorn_iter, 
                         lambda_mass=lambda_mass, lambda_disp=lambda_disp, 
                         lambda_lap=lambda_lap, disp_type=disp_type, 
                         W_graph=W_graph, use_divergence=use_divergence,
                         mass_prior=mass_prior)
            out2 = model(v2, epsilon=epsilon, sinkhorn_iter=sinkhorn_iter, 
                         lambda_mass=lambda_mass, lambda_disp=lambda_disp, 
                         lambda_lap=lambda_lap, disp_type=disp_type, 
                         W_graph=W_graph, use_divergence=use_divergence,
                         mass_prior=mass_prior)
            
            # Use total_loss from model (OT + regs) — averaged over views
            # Regularizers are computed on shared prototypes, so average both
            base_loss = 0.5 * (out1['total_loss'] + out2['total_loss'])
            
            # Consistency loss on soft assignment agreement
            c_loss = cons_criterion(out1['soft_assignments'], out2['soft_assignments'])
            
            loss = base_loss + lambda_cons * c_loss
            
            metrics_to_log = {
                'loss': loss.item(),
                'ot_loss': 0.5 * (out1['ot_cost'].item() + out2['ot_cost'].item()),
                'reg_mass': out1['reg_mass'].item(),
                'reg_disp': out1['reg_disp'].item(),
                'cons_loss': c_loss.item()
            }
        else:
            if multi_view:
                x = v1  # Use first view if no consistency
            out = model(x, epsilon=epsilon, sinkhorn_iter=sinkhorn_iter, 
                        lambda_mass=lambda_mass, lambda_disp=lambda_disp, 
                        lambda_lap=lambda_lap, disp_type=disp_type, 
                        W_graph=W_graph, use_divergence=use_divergence,
                        mass_prior=mass_prior)
            loss = out['total_loss']
            metrics_to_log = {
                'loss': loss.item(),
                'ot_loss': out['ot_cost'].item(),
                'reg_mass': out['reg_mass'].item(),
                'reg_disp': out['reg_disp'].item(),
                'cons_loss': 0.0
            }
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        if optimizer_encoder is not None:
            optimizer_encoder.step()
        optimizer_prototypes.step()
        optimizer_masses.step()
        
        # Accumulate metrics
        for k in epoch_metrics:
            epoch_metrics[k] += metrics_to_log[k]
        
    num_batches = max(len(dataloader), 1)
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
        
    return epoch_metrics

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, 
             epsilon: float, sinkhorn_iter: int) -> Dict[str, Any]:
    """
    Evaluates the model and computes full dataset assignments.
    Uses model.encode() and model.get_prototypes() for consistent normalization.
    """
    model.eval()
    
    all_z = []
    all_y = []
    
    for batch in dataloader:
        if len(batch) == 3:
            x1, x2, y = batch
            x = x1  # Use first view for evaluation
        else:
            x, y = batch
            
        x = x.to(device)
        z = model.encode(x)  # Uses same normalization as training
        all_z.append(z.cpu())
        all_y.append(y.cpu())
        
    all_z = torch.cat(all_z, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0)
    
    # Get normalized prototypes (same as training)
    c = model.get_prototypes()
    masses = model.prototypes.masses
    
    C = model.compute_cost_matrix(all_z, c)
    N = all_z.size(0)
    u = torch.ones(N, dtype=all_z.dtype, device=device) / N
    
    from src.ot.sinkhorn import sinkhorn_log_domain
    P, ot_cost = sinkhorn_log_domain(C, u, masses, epsilon=epsilon, max_iter=sinkhorn_iter)
    
    hard_assignments = torch.argmax(P, dim=1).cpu()
    
    return {
        'z': all_z.cpu(),
        'y_true': all_y,
        'y_pred': hard_assignments,
        'P': P.cpu(),
        'ot_cost': ot_cost.item()
    }

def warmup_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer_encoder: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.5
) -> Dict[str, float]:
    """
    SimCLR-style pre-training epoch.
    """
    model.train()
    from src.models.consistency import ContrastiveLoss
    criterion = ContrastiveLoss(temperature=temperature)
    
    epoch_loss = 0.0
    for batch in dataloader:
        # Expect MultiViewDataset (v1, v2, y)
        if len(batch) == 3:
            v1, v2, _ = batch
            v1, v2 = v1.to(device), v2.to(device)
        else:
            # Fallback for standard dataset
            x, _ = batch
            v1 = v2 = x.to(device)
            
        optimizer_encoder.zero_grad()
        
        z1 = model.encoder(v1)
        z2 = model.encoder(v2)
        
        loss = criterion(z1, z2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=5.0)
        optimizer_encoder.step()
        
        epoch_loss += loss.item()
        
    return {'warmup_loss': epoch_loss / max(len(dataloader), 1)}
