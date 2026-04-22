from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq import get_deq
from torchdeq.grad import backward_factory, make_pair
from torchdeq.solver.fp_iter import fixed_point_iter
from torchdeq.solver.stat import SolverStat


class AffineContraction(nn.Module):
    def __init__(self, dim: int = 4, scale: float = 0.2, bias: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scale = scale
        self.bias = bias
        with torch.no_grad():
            self.linear.weight.mul_(0.1)
            self.linear.bias.fill_(bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.scale * self.linear(z) + self.bias


class TestMakePair:
    def test_equal_length(self):
        assert make_pair([1, 2, 3], [4, 5, 6]) == [4, 5, 6]

    def test_broadcast(self):
        assert make_pair([1, 2, 3], [7]) == [7, 7, 7]

    def test_mismatch_raises(self):
        with pytest.raises(ValueError):
            make_pair([1, 2, 3], [4, 5])


class TestPhantomGrad:
    def test_1step(self):
        func = backward_factory(grad_type=1, tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)

        result = func(trainer, f, z)

        assert len(result) == 1
        assert result[0].requires_grad

    def test_kstep(self):
        func = backward_factory(grad_type=3, tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)

        result = func(trainer, f, z)

        assert len(result) == 1

    def test_sup_gap(self):
        func = backward_factory(grad_type=6, sup_gap=2, tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)

        result = func(trainer, f, z)

        assert len(result) == 3

    def test_sup_loc(self):
        func = backward_factory(grad_type=5, sup_loc=[2, 4], tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)

        result = func(trainer, f, z)

        assert len(result) == 3

    def test_tau_damping(self):
        func = backward_factory(grad_type=1, tau=0.5)
        trainer = nn.Module()
        f = lambda z, tau=1.0: tau * (0.5 * z + 1.0) + (1 - tau) * z
        z = torch.tensor([0.0], requires_grad=True)

        result = func(trainer, f, z)

        assert len(result) == 1


class TestIFTGrad:
    def test_ift_produces_gradients(self):
        func = backward_factory(
            grad_type='ift',
            hook_ift=False,
            b_solver=fixed_point_iter,
            b_solver_kwargs=dict(max_iter=10, tol=1e-6, stop_mode='abs'),
        )

        linear = nn.Linear(4, 4, bias=False)
        trainer = nn.Module()
        trainer.hook = None

        def f(z):
            return 0.1 * linear(z)

        z = torch.randn(2, 4)
        result = func(trainer, f, z)
        loss = result[0].sum()
        loss.backward()

        assert len(result) == 1
        assert linear.weight.grad is not None

    @pytest.mark.parametrize("hook_ift", [False, True])
    def test_backward_solver_kwargs_do_not_receive_compile_solver_args(self, hook_ift: bool):
        seen_kwargs: list[dict] = []

        def tracking_solver(func, x0, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return func(torch.zeros_like(x0)), [], SolverStat()

        deq = get_deq(
            core="sliced",
            f_max_iter=6,
            grad=[1],
            ift=not hook_ift,
            hook_ift=hook_ift,
            compile_fn=True,
        )
        deq.produce_grad[-1] = backward_factory(
            grad_type='ift',
            hook_ift=hook_ift,
            b_solver=tracking_solver,
            b_solver_kwargs=dict(max_iter=5, tol=1e-6, stop_mode='abs'),
        )
        deq.train()
        func = AffineContraction()
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(
            func,
            z_init,
            solver_kwargs={'f_max_iter': torch.tensor(6), 'max_iter_bound': 6},
        )
        z_out[-1].sum().backward()

        assert seen_kwargs
        assert all('compile_solver' not in kwargs for kwargs in seen_kwargs)
        assert all('max_iter_bound' not in kwargs for kwargs in seen_kwargs)
        assert func.linear.weight.grad is not None
        assert func.linear.weight.grad.abs().sum() > 0

    def test_invalid_grad_type_raises(self):
        with pytest.raises(ValueError):
            backward_factory(grad_type=-1)
