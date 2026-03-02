"""
Unit tests for the unified LOTCModel.
"""

import pytest
import torch
import numpy as np


class TestLOTCModel:
    def _make_model(self, n_features=5, K=3, cost_fn="sqeuclidean"):
        from src.models.encoders import IdentityEncoder
        from src.models.lotc_model import LOTCModel
        encoder = IdentityEncoder(input_dim=n_features)
        model = LOTCModel(
            encoder=encoder,
            K=K,
            cost_fn=cost_fn,
            epsilon=0.1,
            sinkhorn_iter=30,
            sinkhorn_tol=0.0,
            learn_masses=True,
            lambda_mass=0.01,
            lambda_disp=0.001,
        )
        return model

    def test_forward_output_shape(self):
        model = self._make_model(n_features=5, K=3)
        X = torch.randn(20, 5)
        out = model(X)
        assert out["P"].shape == (20, 3)
        assert out["C"].shape == (20, 3)
        assert out["Z"].shape == (20, 5)
        assert out["hard_assignments"].shape == (20,)
        assert out["ot_cost"].dim() == 0
        assert out["reg_loss"].dim() == 0
        assert out["total_loss"].dim() == 0

    def test_hard_assignments_valid(self):
        model = self._make_model(K=4)
        X = torch.randn(50, 5)
        labels = model.get_hard_assignments(X)
        assert labels.min() >= 0
        assert labels.max() < 4

    def test_soft_assignments_sum_to_one(self):
        model = self._make_model(K=3)
        X = torch.randn(15, 5)
        soft = model.get_soft_assignments(X)
        row_sums = soft.sum(dim=1)
        np.testing.assert_allclose(row_sums.numpy(), 1.0, atol=1e-5)

    def test_backward_pass(self):
        model = self._make_model(K=3)
        X = torch.randn(20, 5)
        out = model(X)
        out["total_loss"].backward()
        # Check gradients exist on prototypes and mass logits
        assert model.prototype_module.prototypes.grad is not None
        assert model.prototype_module.mass_logits.grad is not None

    def test_cosine_cost_fn(self):
        model = self._make_model(K=3, cost_fn="cosine")
        X = torch.randn(15, 5)
        out = model(X)
        assert out["P"].shape == (15, 3)
        assert torch.isfinite(out["total_loss"])

    def test_mlp_encoder(self):
        from src.models.encoders import MLPEncoder
        from src.models.lotc_model import LOTCModel
        encoder = MLPEncoder(input_dim=10, hidden_dims=[32, 16], output_dim=8)
        model = LOTCModel(encoder=encoder, K=3, epsilon=0.1, sinkhorn_iter=20)
        X = torch.randn(30, 10)
        out = model(X)
        assert out["Z"].shape == (30, 8)
        assert out["P"].shape == (30, 3)
        out["total_loss"].backward()
