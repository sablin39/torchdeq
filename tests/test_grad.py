from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq.grad import backward_factory, make_pair


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
        assert callable(func)

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
        assert len(result) == 3  # steps 2, 4, 6

    def test_sup_loc(self):
        func = backward_factory(grad_type=5, sup_loc=[2, 4], tau=1.0)
        trainer = nn.Module()
        f = lambda z, tau=1.0: 0.5 * z + 1.0
        z = torch.tensor([0.0], requires_grad=True)
        result = func(trainer, f, z)
        assert len(result) == 3  # loc 2, loc 4, final

    def test_tau_damping(self):
        func = backward_factory(grad_type=1, tau=0.5)
        trainer = nn.Module()
        f = lambda z, tau=1.0: tau * (0.5 * z + 1.0) + (1 - tau) * z
        z = torch.tensor([0.0], requires_grad=True)
        result = func(trainer, f, z)
        assert len(result) == 1


class TestIFTGrad:
    def test_ift_produces_gradients(self):
        from torchdeq.solver.fp_iter import fixed_point_iter

        func = backward_factory(
            grad_type='ift', hook_ift=False,
            b_solver=fixed_point_iter,
            b_solver_kwargs=dict(max_iter=10, tol=1e-6, stop_mode='abs')
        )

        linear = nn.Linear(4, 4, bias=False)
        trainer = nn.Module()
        trainer.hook = None

        def f(z):
            return 0.1 * linear(z)

        z = torch.randn(2, 4)
        result = func(trainer, f, z)
        assert len(result) == 1
        loss = result[0].sum()
        loss.backward()
        assert linear.weight.grad is not None

    def test_invalid_grad_type_raises(self):
        with pytest.raises(ValueError):
            backward_factory(grad_type=-1)
