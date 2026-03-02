"""
Utility helpers: seeding, device management, and common operations.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python ``random``, ``numpy``, ``torch`` (CPU & CUDA),
    and configures CuDNN for deterministic behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_cuda: If True, use CUDA when available.

    Returns:
        ``torch.device`` for the selected hardware.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()
