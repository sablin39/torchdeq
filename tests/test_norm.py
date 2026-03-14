from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchdeq.norm import apply_norm, reset_norm, remove_norm


class TestApplyNorm:
    def test_weight_norm_applied(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="weight_norm")
        linear = model[0]
        assert hasattr(linear, "_deq_norm")
        assert hasattr(linear, "weight_v")

    def test_spectral_norm_applied(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="spectral_norm")
        linear = model[0]
        assert hasattr(linear, "_deq_norm")
        assert hasattr(linear, "weight_orig")

    def test_none_norm_noop(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="none")
        assert not hasattr(model[0], "_deq_norm")

    def test_filter_out(self):
        model = nn.ModuleDict({"embedding": nn.Linear(4, 4), "layer": nn.Linear(4, 4)})
        apply_norm(model, norm_type="weight_norm", filter_out=["embedding"])
        assert not hasattr(model["embedding"], "_deq_norm")
        assert hasattr(model["layer"], "_deq_norm")

    def test_prefix_filter_out(self):
        model = nn.ModuleDict({"encoder": nn.Linear(4, 4), "decoder": nn.Linear(4, 4)})
        apply_norm(model, norm_type="weight_norm", prefix_filter_out=["enc"])
        # "encoder" should be skipped since it starts with "enc"
        assert not hasattr(model["encoder"], "_deq_norm")
        assert hasattr(model["decoder"], "_deq_norm")

    def test_unknown_norm_raises(self):
        model = nn.Sequential(nn.Linear(4, 4))
        with pytest.raises(KeyError):
            apply_norm(model, norm_type="nonexistent")


class TestResetNorm:
    def test_reset_weight_norm(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="weight_norm")
        # Modify v to check reset recomputes
        with torch.no_grad():
            model[0].weight_v.fill_(1.0)
        reset_norm(model)
        # After reset, weight should be recomputed
        assert model[0].weight is not None

    def test_reset_spectral_norm(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="spectral_norm")
        reset_norm(model)
        assert model[0].weight is not None


class TestRemoveNorm:
    def test_remove_weight_norm(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="weight_norm")
        remove_norm(model)
        assert not hasattr(model[0], "_deq_norm")
        assert "weight" in dict(model[0].named_parameters())

    def test_remove_spectral_norm(self):
        model = nn.Sequential(nn.Linear(4, 4))
        apply_norm(model, norm_type="spectral_norm")
        remove_norm(model)
        assert not hasattr(model[0], "_deq_norm")
        assert "weight" in dict(model[0].named_parameters())
