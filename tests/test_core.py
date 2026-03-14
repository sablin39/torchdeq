from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq import get_deq, reset_deq, register_deq
from torchdeq.utils.config import DEQConfig
from torchdeq.core import DEQIndexing, DEQSliced


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


class TestRegisterDEQ:
    def test_register_custom(self):
        class CustomDEQ(DEQSliced):
            pass
        register_deq("custom_test", CustomDEQ)
        deq = get_deq(core="custom_test")
        assert isinstance(deq, CustomDEQ)


class TestDEQForward:
    def _make_func(self):
        linear = nn.Linear(8, 8)
        with torch.no_grad():
            linear.weight.mul_(0.1)  # ensure contraction
        return linear

    def test_sliced_train_forward(self):
        deq = get_deq(core="sliced", f_max_iter=10)
        deq.train()
        func = self._make_func()
        z_init = torch.zeros(2, 8)
        z_out, info = deq(func, z_init)
        assert isinstance(z_out, list)
        assert len(z_out) >= 1
        assert z_out[0].shape == (2, 8)

    def test_sliced_eval_forward(self):
        deq = get_deq(core="sliced", f_max_iter=10)
        deq.eval()
        func = self._make_func()
        z_init = torch.zeros(2, 8)
        with torch.no_grad():
            z_out, info = deq(func, z_init)
        assert len(z_out) == 1
        assert z_out[0].shape == (2, 8)

    def test_indexing_train_forward(self):
        deq = get_deq(core="indexing", f_max_iter=10)
        deq.train()
        func = self._make_func()
        z_init = torch.zeros(2, 8)
        z_out, info = deq(func, z_init)
        assert isinstance(z_out, list)
        assert len(z_out) >= 1

    def test_indexing_eval_forward(self):
        deq = get_deq(core="indexing", f_max_iter=10)
        deq.eval()
        func = self._make_func()
        z_init = torch.zeros(2, 8)
        with torch.no_grad():
            z_out, info = deq(func, z_init)
        assert len(z_out) == 1


class TestGradientFlow:
    def test_gradients_flow_to_params(self):
        deq = get_deq(core="indexing", f_max_iter=5, grad=[1])
        deq.train()

        linear = nn.Linear(4, 4)
        with torch.no_grad():
            linear.weight.mul_(0.1)

        z_init = torch.randn(2, 4)
        z_out, info = deq(linear, z_init)
        loss = z_out[-1].sum()
        loss.backward()
        assert linear.weight.grad is not None
        assert linear.weight.grad.abs().sum() > 0


class TestResetDEQ:
    def test_reset_runs_without_error(self):
        model = nn.Sequential(nn.Linear(4, 4))
        reset_deq(model)  # should not raise
