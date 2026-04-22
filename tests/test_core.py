from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq import get_deq, register_deq, reset_deq
from torchdeq.core import DEQIndexing, DEQSliced
from torchdeq.utils.config import DEQConfig

HAS_WHILE_LOOP = hasattr(torch, "while_loop")


class AffineContraction(nn.Module):
    def __init__(self, dim: int = 8, scale: float = 0.2, bias: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.scale = scale
        self.bias = bias
        with torch.no_grad():
            self.linear.weight.mul_(0.1)
            self.linear.bias.fill_(bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.scale * self.linear(z) + self.bias


class SlowResidualContraction(nn.Module):
    def __init__(self, dim: int = 8, residual: float = 0.95, drive: float = 0.05) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.residual = residual
        self.drive = drive
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(1.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.residual * z + self.drive * self.linear.bias.view(1, -1)


def _assert_nonzero_parameter_gradients(module: nn.Module) -> None:
    grads = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert grads
    assert sum(grad.abs().sum().item() for grad in grads) > 0


class TestGetDEQ:
    def test_default_creates_sliced(self):
        deq = get_deq()
        assert isinstance(deq, DEQSliced)

    def test_indexing_core(self):
        deq = get_deq(core="indexing")
        assert isinstance(deq, DEQIndexing)

    def test_from_dict(self):
        deq = get_deq({"core": "sliced", "f_max_iter": 20})
        assert isinstance(deq, DEQSliced)
        assert deq.f_max_iter == 20

    def test_from_config(self):
        config = DEQConfig(core="indexing", f_max_iter=30)
        deq = get_deq(config)
        assert isinstance(deq, DEQIndexing)
        assert deq.f_max_iter == 30

    def test_unknown_core_raises(self):
        with pytest.raises(KeyError):
            get_deq(core="nonexistent")

    def test_compile_fn_sets_compile_solver_flag(self):
        deq = get_deq(core="sliced", compile_fn=True)
        assert deq._compile_solver is True
        assert deq._compile_fn is False


class TestRegisterDEQ:
    def test_register_custom(self):
        class CustomDEQ(DEQSliced):
            pass

        register_deq("custom_test", CustomDEQ)
        deq = get_deq(core="custom_test")
        assert isinstance(deq, CustomDEQ)


class TestForwardBehavior:
    def test_sliced_train_forward(self):
        deq = get_deq(core="sliced", f_max_iter=8)
        deq.train()
        func = AffineContraction()
        z_init = torch.zeros(2, 8)

        z_out, info = deq(func, z_init)

        assert len(z_out) >= 1
        assert z_out[0].shape == (2, 8)
        assert 'abs_lowest' in info

    def test_indexing_eval_forward(self):
        deq = get_deq(core="indexing", f_max_iter=8)
        deq.eval()
        func = AffineContraction()
        z_init = torch.zeros(2, 8)

        with torch.no_grad():
            z_out, info = deq(func, z_init)

        assert len(z_out) == 1
        assert z_out[0].shape == (2, 8)
        assert 'abs_trace' in info

    def test_sliced_phantom_grad_outputs_and_gradients(self):
        deq = get_deq(core="sliced", f_max_iter=6, n_states=2, grad=[1])
        deq.train()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(func, z_init)
        loss = sum(each.sum() for each in z_out)
        loss.backward()

        assert len(z_out) == 2
        assert all(each.shape == (2, 4) for each in z_out)
        _assert_nonzero_parameter_gradients(func)

    def test_sliced_sup_gap_outputs_match_sample_count(self):
        deq = get_deq(core="sliced", f_max_iter=6, n_states=2, grad=[4], sup_gap=2)
        deq.train()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(func, z_init)

        assert len(z_out) == 4
        assert all(each.shape == (2, 4) for each in z_out)

    def test_sliced_sup_loc_outputs_match_sample_count(self):
        deq = get_deq(core="sliced", f_max_iter=6, n_states=2, grad=[4], sup_loc=[2])
        deq.train()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(func, z_init)

        assert len(z_out) == 4
        assert all(each.shape == (2, 4) for each in z_out)

    def test_indexing_phantom_grad_outputs_and_gradients(self):
        deq = get_deq(core="indexing", f_max_iter=6, f_tol=1e-12, n_states=2, grad=[1])
        deq.train()
        func = SlowResidualContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, info = deq(func, z_init)
        loss = sum(each.sum() for each in z_out)
        loss.backward()

        assert len(z_out) == 2
        assert info['nstep'].min().item() == 6
        _assert_nonzero_parameter_gradients(func)

    def test_indexing_explicit_indexing_outputs(self):
        deq = get_deq(core="indexing", f_max_iter=6, f_tol=1e-12, n_states=1, indexing=[2, 4], grad=[1])
        deq.train()
        func = SlowResidualContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, info = deq(func, z_init)

        assert len(z_out) == 3
        assert info['nstep'].min().item() == 6
        assert all(each.shape == (2, 4) for each in z_out)


class TestImplicitGradients:
    @pytest.mark.parametrize("compile_fn", [False, True])
    def test_sliced_ift_gradients(self, compile_fn: bool):
        if compile_fn and not HAS_WHILE_LOOP:
            pytest.skip("torch.while_loop not available")

        deq = get_deq(core="sliced", f_max_iter=6, grad=[1], ift=True, compile_fn=compile_fn)
        deq.train()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)
        solver_kwargs = {'f_max_iter': torch.tensor(6), 'max_iter_bound': 6} if compile_fn else {}

        z_out, _ = deq(func, z_init, solver_kwargs=solver_kwargs)
        z_out[-1].sum().backward()

        assert len(z_out) == 1
        _assert_nonzero_parameter_gradients(func)

    @pytest.mark.parametrize("compile_fn", [False, True])
    def test_sliced_hook_ift_gradients(self, compile_fn: bool):
        if compile_fn and not HAS_WHILE_LOOP:
            pytest.skip("torch.while_loop not available")

        deq = get_deq(core="sliced", f_max_iter=6, grad=[1], hook_ift=True, compile_fn=compile_fn)
        deq.train()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)
        solver_kwargs = {'f_max_iter': torch.tensor(6), 'max_iter_bound': 6} if compile_fn else {}

        z_out, _ = deq(func, z_init, solver_kwargs=solver_kwargs)
        z_out[-1].sum().backward()

        assert len(z_out) == 1
        _assert_nonzero_parameter_gradients(func)

    def test_indexing_ift_gradients(self):
        deq = get_deq(core="indexing", f_max_iter=6, f_tol=1e-12, grad=[1], ift=True)
        deq.train()
        func = SlowResidualContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(func, z_init)
        z_out[-1].sum().backward()

        assert len(z_out) == 1
        _assert_nonzero_parameter_gradients(func)

    def test_indexing_hook_ift_gradients(self):
        deq = get_deq(core="indexing", f_max_iter=6, f_tol=1e-12, grad=[1], hook_ift=True)
        deq.train()
        func = SlowResidualContraction(dim=4)
        z_init = torch.zeros(2, 4)

        z_out, _ = deq(func, z_init)
        z_out[-1].sum().backward()

        assert len(z_out) == 1
        _assert_nonzero_parameter_gradients(func)


@pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
class TestCompileSolverEval:
    @pytest.mark.parametrize("core", ["sliced", "indexing"])
    def test_eval_forward_sradius_and_info(self, core: str):
        deq = get_deq(core=core, f_max_iter=6, eval_f_max_iter=5, compile_fn=True)
        deq.eval()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)

        with torch.no_grad():
            z_out, info = deq(func, z_init, sradius_mode=True)

        assert len(z_out) == 1
        assert z_out[0].shape == (2, 4)
        assert info['abs_trace'].shape == (2, 6)
        assert info['rel_trace'].shape == (2, 6)
        assert info['sradius'].shape[0] == 2

    def test_eval_factor_route_uses_compile_solver_budget(self):
        deq = get_deq(core="sliced", f_max_iter=4, eval_factor=1.5, compile_fn=True)
        deq.eval()
        func = AffineContraction(dim=4)
        z_init = torch.zeros(2, 4)

        with torch.no_grad():
            z_out, info = deq(func, z_init)

        assert len(z_out) == 1
        assert z_out[0].shape == (2, 4)
        assert info['abs_trace'].shape == (2, 7)
        assert info['rel_trace'].shape == (2, 7)


class TestResetDEQ:
    def test_reset_runs_without_error(self):
        model = nn.Sequential(nn.Linear(4, 4))
        reset_deq(model)
