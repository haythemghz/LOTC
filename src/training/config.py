"""
Configuration system for LOTC experiments.

Uses Python dataclasses for structured, typed configuration with YAML
serialization/deserialization support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset configuration."""
    name: str = "blobs"
    n_samples: int = 1000
    n_features: int = 2
    n_clusters: int = 5
    noise: float = 0.0
    data_dir: str = "data"
    # Image-specific
    image_size: int = 28
    in_channels: int = 1
    # Pretrained features
    use_pretrained_features: bool = False


@dataclass
class EncoderConfig:
    """Encoder network configuration."""
    type: str = "identity"  # "identity", "mlp", "cnn"
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    output_dim: int = 64
    dropout: float = 0.0
    batch_norm: bool = True
    pretrained: bool = False
    freeze_backbone: bool = False


@dataclass
class OTConfig:
    """Optimal transport configuration."""
    epsilon: float = 0.1
    sinkhorn_iter: int = 50
    sinkhorn_tol: float = 0.0
    cost_fn: str = "sqeuclidean"  # "sqeuclidean", "cosine", "mahalanobis"


@dataclass
class RegConfig:
    """Regularization configuration."""
    lambda_mass: float = 0.0
    lambda_disp: float = 0.0
    lambda_lap: float = 0.0
    disp_mode: str = "l2"  # "l2", "collision"
    lap_k: int = 5


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 200
    batch_size: int = 256
    lr_encoder: float = 1e-3
    lr_prototypes: float = 1e-3
    lr_masses: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # "adam", "sgd"
    scheduler: str = "none"  # "none", "cosine", "step"
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    learn_masses: bool = True
    init_method: str = "kmeans"  # "kmeans", "random"
    mode: str = "minibatch"  # "full", "minibatch"
    grad_clip: float = 0.0
    # Logging
    log_every: int = 10
    save_every: int = 50
    log_backend: str = "tensorboard"  # "tensorboard", "wandb", "none"


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    seed: int = 42
    n_seeds: int = 10
    device: str = "auto"  # "auto", "cpu", "cuda"
    output_dir: str = "experiments/results"

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated ``ExperimentConfig``.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        """Recursively build config from a dictionary."""
        data = DataConfig(**d.get("data", {}))
        encoder = EncoderConfig(**d.get("encoder", {}))
        ot = OTConfig(**d.get("ot", {}))
        reg = RegConfig(**d.get("reg", {}))
        training = TrainingConfig(**d.get("training", {}))

        top_keys = {
            k: v for k, v in d.items()
            if k not in ("data", "encoder", "ot", "reg", "training")
        }
        return cls(data=data, encoder=encoder, ot=ot, reg=reg, training=training, **top_keys)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)
