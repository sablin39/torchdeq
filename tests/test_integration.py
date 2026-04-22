from __future__ import annotations

import torch
import torch.nn as nn

from torchdeq import get_deq, reset_deq
from torchdeq.loss import fp_correction
from torchdeq.norm import apply_norm


class SimpleDEQFunc(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        with torch.no_grad():
            self.fc1.weight.mul_(0.2)
            self.fc2.weight.mul_(0.2)
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.fc1(z))
        return 0.5 * self.act(self.fc2(hidden)) + 0.1


def _grad_norm(module: nn.Module) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


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

        for step in range(40):
            optimizer.zero_grad()
            deq.train()
            z_init = torch.zeros(4, dim)
            z_out, _ = deq(func, z_init)
            loss = ((z_out[-1] - target) ** 2).mean()
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        assert initial_loss is not None
        assert final_loss is not None
        assert final_loss < initial_loss

    def test_train_with_fp_correction(self):
        torch.manual_seed(42)
        dim = 16
        deq = get_deq(core="sliced", f_max_iter=8, n_states=2, grad=[1])
        func = SimpleDEQFunc(dim)
        apply_norm(func, norm_type="weight_norm")
        target = torch.randn(4, dim)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        for _ in range(8):
            optimizer.zero_grad()
            reset_deq(func)
            deq.train()
            z_init = torch.zeros(4, dim)
            z_out, _ = deq(func, z_init)
            loss = fp_correction(criterion, (z_out, target))
            loss.backward()
            optimizer.step()

        assert loss.item() >= 0

    def test_compile_solver_phantom_grad_training_smoke(self):
        torch.manual_seed(7)
        dim = 8
        deq = get_deq(core="sliced", f_max_iter=6, grad=[1], compile_fn=True)
        func = SimpleDEQFunc(dim)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
        z_init = torch.zeros(4, dim)

        for _ in range(3):
            optimizer.zero_grad()
            deq.train()
            z_out, info = deq(
                func,
                z_init,
                solver_kwargs={'f_max_iter': torch.tensor(6), 'max_iter_bound': 6},
            )
            loss = z_out[-1].pow(2).mean()
            loss.backward()
            optimizer.step()

        assert torch.isfinite(loss)
        assert _grad_norm(func) > 0
        assert info['abs_trace'].shape == (4, 7)

    def test_compile_solver_ift_training_smoke(self):
        torch.manual_seed(9)
        dim = 8
        deq = get_deq(core="sliced", f_max_iter=6, grad=[1], ift=True, compile_fn=True)
        func = SimpleDEQFunc(dim)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
        z_init = torch.zeros(4, dim)

        optimizer.zero_grad()
        deq.train()
        z_out, _ = deq(
            func,
            z_init,
            solver_kwargs={'f_max_iter': torch.tensor(6), 'max_iter_bound': 6},
        )
        loss = z_out[-1].pow(2).mean()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        assert _grad_norm(func) > 0

    def test_train_to_eval_transition_with_compile_solver(self):
        torch.manual_seed(11)
        dim = 8
        deq = get_deq(core="sliced", f_max_iter=6, eval_f_max_iter=5, grad=[1], compile_fn=True)
        func = SimpleDEQFunc(dim)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-2)
        z_init = torch.zeros(4, dim)

        optimizer.zero_grad()
        deq.train()
        z_out, train_info = deq(
            func,
            z_init,
            solver_kwargs={'f_max_iter': torch.tensor(6), 'max_iter_bound': 6},
        )
        train_loss = z_out[-1].pow(2).mean()
        train_loss.backward()
        optimizer.step()

        deq.eval()
        with torch.no_grad():
            eval_out, eval_info = deq(func, z_init)

        assert torch.isfinite(train_loss)
        assert eval_out[0].shape == (4, dim)
        assert train_info['abs_trace'].shape == (4, 7)
        assert eval_info['abs_trace'].shape == (4, 6)
        assert 'sradius' in eval_info
