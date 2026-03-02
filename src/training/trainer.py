"""
LOTC Trainer: training loop with logging, checkpointing, and evaluation.

Supports both full-OT and mini-batch modes, separate optimizer groups
for encoder/prototypes/masses, and optional W&B or TensorBoard logging.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models.lotc_model import LOTCModel
from ..models.encoders import IdentityEncoder, MLPEncoder, CNNEncoder
from ..models.prototypes import PrototypeModule
from ..eval.metrics import compute_all_metrics
from ..utils.helpers import seed_everything, get_device, to_numpy
from .config import ExperimentConfig


class LOTCTrainer:
    """Training loop for the LOTC model.

    Handles model construction from config, optimizer setup, training,
    evaluation, logging, and checkpointing.

    Args:
        config: Experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = self._resolve_device()
        self.model: LOTCModel | None = None
        self.optimizer = None
        self.scheduler = None
        self.logger = None
        self.train_log: list[dict[str, Any]] = []

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            return get_device(prefer_cuda=True)
        return torch.device(self.config.device)

    def build_model(self, input_dim: int) -> LOTCModel:
        """Construct the LOTC model from config.

        Args:
            input_dim: Dimensionality of raw input features.

        Returns:
            Initialized LOTCModel on the configured device.
        """
        cfg = self.config
        enc_cfg = cfg.encoder
        ot_cfg = cfg.ot
        reg_cfg = cfg.reg
        tr_cfg = cfg.training

        # Build encoder
        if enc_cfg.type == "identity":
            encoder = IdentityEncoder(input_dim)
        elif enc_cfg.type == "mlp":
            encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dims=enc_cfg.hidden_dims,
                output_dim=enc_cfg.output_dim,
                dropout=enc_cfg.dropout,
                batch_norm=enc_cfg.batch_norm,
            )
        elif enc_cfg.type == "cnn":
            encoder = CNNEncoder(
                output_dim=enc_cfg.output_dim,
                pretrained=enc_cfg.pretrained,
                freeze_backbone=enc_cfg.freeze_backbone,
                in_channels=cfg.data.in_channels,
                image_size=cfg.data.image_size,
            )
        else:
            raise ValueError(f"Unknown encoder type: {enc_cfg.type}")

        model = LOTCModel(
            encoder=encoder,
            K=cfg.data.n_clusters,
            cost_fn=ot_cfg.cost_fn,
            epsilon=ot_cfg.epsilon,
            sinkhorn_iter=ot_cfg.sinkhorn_iter,
            sinkhorn_tol=ot_cfg.sinkhorn_tol,
            learn_masses=tr_cfg.learn_masses,
            lambda_mass=reg_cfg.lambda_mass,
            lambda_disp=reg_cfg.lambda_disp,
            lambda_lap=reg_cfg.lambda_lap,
            disp_mode=reg_cfg.disp_mode,
            lap_k=reg_cfg.lap_k,
        )
        self.model = model.to(self.device)
        return self.model

    def init_prototypes(self, data: torch.Tensor) -> None:
        """Initialize prototypes from data.

        Args:
            data: Data tensor for initialization (embedding-space if using
                an encoder, raw features otherwise).
        """
        assert self.model is not None, "Build model first"
        cfg = self.config.training
        if cfg.init_method == "kmeans":
            self.model.prototype_module.init_from_kmeans(data, seed=self.config.seed)
        elif cfg.init_method == "random":
            self.model.prototype_module.init_random_sample(data, seed=self.config.seed)
        else:
            raise ValueError(f"Unknown init method: {cfg.init_method}")

    def setup_optimizer(self) -> None:
        """Create optimizer with separate parameter groups."""
        assert self.model is not None
        tr = self.config.training

        param_groups = []
        # Encoder parameters (if trainable)
        enc_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        if enc_params:
            param_groups.append({"params": enc_params, "lr": tr.lr_encoder})

        # Prototype locations
        param_groups.append({
            "params": [self.model.prototype_module.prototypes],
            "lr": tr.lr_prototypes,
        })

        # Mass logits (if learnable)
        if tr.learn_masses:
            param_groups.append({
                "params": [self.model.prototype_module.mass_logits],
                "lr": tr.lr_masses,
            })

        # Mahalanobis metric (if applicable)
        if self.model.metric_L is not None:
            param_groups.append({
                "params": [self.model.metric_L],
                "lr": tr.lr_encoder,
            })

        if tr.optimizer == "adam":
            self.optimizer = Adam(param_groups, weight_decay=tr.weight_decay)
        elif tr.optimizer == "sgd":
            self.optimizer = SGD(param_groups, weight_decay=tr.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {tr.optimizer}")

        # Scheduler
        if tr.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=tr.epochs)
        elif tr.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=tr.scheduler_step_size, gamma=tr.scheduler_gamma
            )

    def train(
        self,
        X: torch.Tensor,
        y_true: torch.Tensor | None = None,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            X: Training data, shape ``(n, d)`` or ``(n, C, H, W)`` for images.
            y_true: Optional ground-truth labels for metric logging.
            X_val: Optional validation data.
            y_val: Optional validation labels.

        Returns:
            Dictionary of training history (per-epoch losses and metrics).
        """
        assert self.model is not None
        tr = self.config.training
        self.train_log = []

        # DataLoader for minibatch mode
        dataset = TensorDataset(X)
        if y_true is not None:
            dataset = TensorDataset(X, y_true)
        loader = DataLoader(
            dataset, batch_size=tr.batch_size, shuffle=True, drop_last=False
        )

        self.model.train()

        for epoch in range(1, tr.epochs + 1):
            epoch_loss = 0.0
            epoch_ot = 0.0
            epoch_reg = 0.0
            n_batches = 0

            if tr.mode == "minibatch":
                for batch in loader:
                    X_b = batch[0].to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(X_b)
                    loss = out["total_loss"]
                    loss.backward()

                    if tr.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), tr.grad_clip)

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    epoch_ot += out["ot_cost"].item()
                    epoch_reg += out["reg_loss"].item()
                    n_batches += 1

            else:  # full OT
                X_full = X.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(X_full)
                loss = out["total_loss"]
                loss.backward()

                if tr.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), tr.grad_clip)

                self.optimizer.step()
                epoch_loss = loss.item()
                epoch_ot = out["ot_cost"].item()
                epoch_reg = out["reg_loss"].item()
                n_batches = 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / n_batches
            avg_ot = epoch_ot / n_batches
            avg_reg = epoch_reg / n_batches

            log_entry: dict[str, Any] = {
                "epoch": epoch,
                "loss": avg_loss,
                "ot_cost": avg_ot,
                "reg_loss": avg_reg,
            }

            # Evaluate metrics periodically
            if epoch % tr.log_every == 0 or epoch == tr.epochs:
                if y_true is not None:
                    self.model.eval()
                    with torch.no_grad():
                        preds = self.model.get_hard_assignments(X.to(self.device))
                    metrics = compute_all_metrics(
                        to_numpy(y_true), to_numpy(preds), X_np=to_numpy(X) if X.dim() == 2 else None
                    )
                    log_entry.update(metrics)
                    self.model.train()

            self.train_log.append(log_entry)

            # Checkpoint
            if epoch % tr.save_every == 0:
                self._save_checkpoint(epoch)

        # Final checkpoint
        self._save_checkpoint(tr.epochs, tag="final")

        return {"history": self.train_log}

    def _save_checkpoint(self, epoch: int, tag: str | None = None) -> None:
        """Save model checkpoint."""
        out_dir = Path(self.config.output_dir) / self.config.name
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"checkpoint_epoch{epoch}.pt" if tag is None else f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
            },
            out_dir / fname,
        )

    def save_results(self, extra: dict[str, Any] | None = None) -> Path:
        """Save training log and final metrics to JSON.

        Returns:
            Path to the saved results file.
        """
        out_dir = Path(self.config.output_dir) / self.config.name
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "config": self.config.to_dict(),
            "history": self.train_log,
        }
        if extra:
            results.update(extra)
        path = out_dir / "results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path
