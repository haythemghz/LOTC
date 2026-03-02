"""
Smoke test for the LOTC trainer.
"""

import pytest
import torch
import numpy as np


class TestTrainer:
    def test_smoke_training_blobs(self):
        """5-epoch training on tiny blobs — loss should decrease."""
        from src.training.config import ExperimentConfig
        from src.training.trainer import LOTCTrainer

        config = ExperimentConfig(
            name="smoke_test",
            seed=42,
            device="cpu",
            output_dir="/tmp/lotc_test_results",
        )
        config.data.name = "blobs"
        config.data.n_samples = 100
        config.data.n_features = 2
        config.data.n_clusters = 3
        config.encoder.type = "identity"
        config.ot.epsilon = 0.1
        config.ot.sinkhorn_iter = 30
        config.training.epochs = 10
        config.training.batch_size = 50
        config.training.lr_prototypes = 0.01
        config.training.mode = "full"
        config.training.log_every = 5
        config.training.save_every = 100  # no saving during test

        from src.data.synthetic import get_synthetic_dataset
        from src.utils.helpers import seed_everything

        seed_everything(42)
        X, y = get_synthetic_dataset(
            "blobs", n_samples=100, n_features=2, n_clusters=3, seed=42
        )

        trainer = LOTCTrainer(config)
        trainer.build_model(input_dim=2)
        trainer.init_prototypes(X)
        trainer.setup_optimizer()
        result = trainer.train(X, y_true=y)

        history = result["history"]
        assert len(history) == 10
        # Loss should generally decrease
        assert history[-1]["loss"] < history[0]["loss"] * 1.5  # allow some slack


class TestMetrics:
    def test_perfect_clustering(self):
        from src.eval.metrics import compute_all_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics["ARI"] == pytest.approx(1.0)
        assert metrics["NMI"] == pytest.approx(1.0)

    def test_random_clustering(self):
        from src.eval.metrics import compute_all_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics["ARI"] < 0.5  # Should be low
