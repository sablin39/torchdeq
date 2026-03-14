from __future__ import annotations

import pytest
import torch
import torch.nn as nn

HAS_COMPILE = hasattr(torch, "compile")


@pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile not available")
class TestCompileSmoke:
    def test_phantom_grad_forward(self):
        from torchdeq.grad import backward_factory
        func = backward_factory(grad_type=2, tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)
        result = func(trainer, f, z)
        assert len(result) == 1

    def test_deq_eval_mode(self):
        from torchdeq import get_deq
        deq = get_deq(core="sliced", f_max_iter=5)
        deq.eval()
        linear = nn.Linear(4, 4)
        with torch.no_grad():
            linear.weight.mul_(0.1)
        z_init = torch.zeros(2, 4)
        with torch.no_grad():
            z_out, info = deq(linear, z_init)
        assert len(z_out) == 1

    def test_dropout_compile(self):
        from torchdeq.dropout import VariationalDropout
        m = VariationalDropout(dropout=0.3)
        m.train()
        x = torch.ones(4, 8)
        out = m(x)
        assert out.shape == x.shape

    def test_fp_correction_compile(self):
        from torchdeq.loss import fp_correction
        x = [torch.randn(4, 8)]
        y = torch.randn(4, 8)
        crit = lambda x, y: ((x - y) ** 2).mean()
        loss = fp_correction(crit, (x, y), weight_func='exp')
        assert loss.item() > 0
