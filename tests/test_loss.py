from __future__ import annotations

import pytest
import torch

from torchdeq.loss import fp_correction, jac_reg, power_method
from torchdeq.loss.correction import register_weight_func


class TestFPCorrection:
    def test_exp_weights(self):
        x = [torch.randn(4, 8) for _ in range(3)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss = fp_correction(crit, (x, y), weight_func='exp')
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_linear_weights(self):
        x = [torch.randn(4, 8) for _ in range(3)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss = fp_correction(crit, (x, y), weight_func='linear')
        assert loss.item() > 0

    def test_const_weights(self):
        x = [torch.randn(4, 8) for _ in range(3)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss = fp_correction(crit, (x, y), weight_func='const')
        assert loss.item() > 0

    def test_return_loss_values(self):
        x = [torch.randn(4, 8) for _ in range(3)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss, values = fp_correction(crit, (x, y), weight_func='exp', return_loss_values=True)
        assert len(values) == 3
        assert all(v > 0 for v in values)

    def test_register_custom_weight_func(self):
        def my_weight(n, k, **kwargs):
            return 1.0
        register_weight_func("my_weight", my_weight)
        x = [torch.randn(4, 8)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss = fp_correction(crit, (x, y), weight_func='my_weight')
        assert loss.item() > 0


class TestJacReg:
    def test_basic(self):
        z = torch.randn(2, 4, requires_grad=True)
        f = 0.5 * z
        loss = jac_reg(f, z, vecs=1)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_multiple_vecs(self):
        z = torch.randn(2, 4, requires_grad=True)
        f = 0.5 * z
        loss = jac_reg(f, z, vecs=3)
        assert loss.item() >= 0


class TestPowerMethod:
    def test_basic(self):
        z = torch.randn(2, 4, requires_grad=True)
        f = 0.5 * z
        evec, evalue = power_method(f, z, n_iters=50)
        assert evec.shape == z.shape
        assert evalue.shape == (2,)
        # For f=0.5z, spectral radius should be ~0.5
        assert (evalue - 0.5).abs().max() < 0.1
