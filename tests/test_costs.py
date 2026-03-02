"""
Unit tests for cost functions.
"""

import pytest
import torch
import numpy as np


class TestSquaredEuclideanCost:
    def test_shape(self):
        from src.ot.costs import squared_euclidean_cost
        Z = torch.randn(20, 5)
        C_proto = torch.randn(3, 5)
        cost = squared_euclidean_cost(Z, C_proto)
        assert cost.shape == (20, 3)

    def test_nonnegative(self):
        from src.ot.costs import squared_euclidean_cost
        Z = torch.randn(10, 4)
        C_proto = torch.randn(5, 4)
        cost = squared_euclidean_cost(Z, C_proto)
        assert (cost >= -1e-6).all()

    def test_zero_self_distance(self):
        from src.ot.costs import squared_euclidean_cost
        Z = torch.randn(5, 3)
        cost = squared_euclidean_cost(Z, Z)
        diag = cost.diag()
        np.testing.assert_allclose(diag.detach().numpy(), 0.0, atol=1e-5)

    def test_gradient_flow(self):
        from src.ot.costs import squared_euclidean_cost
        Z = torch.randn(10, 3)
        C_proto = torch.randn(4, 3, requires_grad=True)
        cost = squared_euclidean_cost(Z, C_proto)
        cost.sum().backward()
        assert C_proto.grad is not None
        assert C_proto.grad.abs().sum() > 0


class TestCosineCost:
    def test_shape(self):
        from src.ot.costs import cosine_cost
        Z = torch.randn(15, 8)
        C_proto = torch.randn(4, 8)
        cost = cosine_cost(Z, C_proto)
        assert cost.shape == (15, 4)

    def test_range(self):
        from src.ot.costs import cosine_cost
        Z = torch.randn(10, 5)
        C_proto = torch.randn(3, 5)
        cost = cosine_cost(Z, C_proto)
        assert (cost >= -1e-6).all()
        assert (cost <= 2.0 + 1e-6).all()


class TestMahalanobisCost:
    def test_shape_and_gradient(self):
        from src.ot.costs import mahalanobis_cost
        d = 4
        Z = torch.randn(10, d)
        C_proto = torch.randn(3, d)
        M = torch.eye(d, requires_grad=True)
        cost = mahalanobis_cost(Z, C_proto, M)
        assert cost.shape == (10, 3)
        cost.sum().backward()
        assert M.grad is not None

    def test_identity_equals_euclidean(self):
        """With M=I (identity Cholesky), Mahalanobis should equal squared Euclidean."""
        from src.ot.costs import mahalanobis_cost, squared_euclidean_cost
        torch.manual_seed(10)
        d = 5
        Z = torch.randn(8, d)
        C_proto = torch.randn(3, d)
        M = torch.eye(d)
        cost_mah = mahalanobis_cost(Z, C_proto, M)
        cost_euc = squared_euclidean_cost(Z, C_proto)
        np.testing.assert_allclose(
            cost_mah.detach().numpy(), cost_euc.detach().numpy(), atol=1e-5
        )
