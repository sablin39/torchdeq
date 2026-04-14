from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq import get_deq, reset_deq
from torchdeq.norm import apply_norm
from torchdeq.loss import fp_correction


class SimpleDEQFunc(nn.Module):
    """A simple 2-layer MLP for testing DEQ training."""
    def __init__(self, dim=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        # Scale down weights for contraction but keep large enough to learn
        with torch.no_grad():
            self.fc1.weight.mul_(0.3)
            self.fc2.weight.mul_(0.3)

    def forward(self, z):
        return self.act(self.fc2(self.act(self.fc1(z))))


class TestEndToEnd:
    def test_train_loss_decreases(self):
        torch.manual_seed(42)
        dim = 8
        deq = get_deq(core="sliced", f_max_iter=12, grad=[1])
        func = SimpleDEQFunc(dim)

        target = torch.randn(4, dim) * 0.3
        optimizer = torch.optim.Adam(func.parameters(), lr=5e-3)

        initial_loss = None
        final_loss = None

        for step in range(100):
            optimizer.zero_grad()
            deq.train()

            z_init = torch.zeros(4, dim)
            z_out, info = deq(func, z_init)

            loss = ((z_out[-1] - target) ** 2).mean()

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_train_with_fp_correction(self):
        torch.manual_seed(42)
        dim = 16
        deq = get_deq(core="sliced", f_max_iter=8, n_states=2, grad=[1])
        func = SimpleDEQFunc(dim)
        apply_norm(func, norm_type="weight_norm")

        target = torch.randn(4, dim)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
        crit = nn.MSELoss()

        for step in range(10):
            optimizer.zero_grad()
            reset_deq(func)
            deq.train()

            z_init = torch.zeros(4, dim)
            z_out, info = deq(func, z_init)

            loss = fp_correction(crit, (z_out, target))
            loss.backward()
            optimizer.step()

        # Just verify it runs without error and produces a loss
        assert loss.item() >= 0

    def test_eval_after_train(self):
        torch.manual_seed(42)
        dim = 8
        deq = get_deq(core="sliced", f_max_iter=10)
        func = SimpleDEQFunc(dim)

        deq.train()
        z_init = torch.zeros(2, dim)
        z_out_train, _ = deq(func, z_init)

        deq.eval()
        with torch.no_grad():
            z_out_eval, info = deq(func, z_init)

        assert z_out_eval[0].shape == (2, dim)
        assert 'abs_lowest' in info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAIntegration:
    @pytest.mark.parametrize("core", ["sliced", "indexing"])
    def test_mem_gc_training_under_cuda_autocast(self, core):
        torch.manual_seed(42)
        dim = 8
        deq = get_deq(
            core=core,
            f_max_iter=6,
            b_max_iter=6,
            grad=[1],
            mem_gc=True,
        )
        deq.train()
        func = SimpleDEQFunc(dim).cuda()
        target = torch.randn(4, dim, device="cuda")
        z_init = torch.zeros(4, dim, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_out, info = deq(func, z_init)
            loss = ((z_out[-1] - target) ** 2).mean()
        loss.backward()

        assert torch.isfinite(loss).item()
        assert torch.isfinite(info["abs_lowest"]).all().item()
        assert func.fc1.weight.grad is not None
        assert torch.isfinite(func.fc1.weight.grad).all().item()
        assert func.fc2.weight.grad is not None
        assert torch.isfinite(func.fc2.weight.grad).all().item()
