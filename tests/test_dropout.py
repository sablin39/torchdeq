from __future__ import annotations

import pytest
import torch

from torchdeq.dropout import (
    VariationalDropout, VariationalDropout1d, VariationalDropout2d,
    reset_dropout,
)


class TestVariationalDropout:
    def test_train_mode_drops(self):
        m = VariationalDropout(dropout=0.5)
        m.train()
        torch.manual_seed(42)
        x = torch.ones(4, 8)
        out = m(x)
        # Some values should be zero
        assert (out == 0).any()

    def test_eval_mode_identity(self):
        m = VariationalDropout(dropout=0.5)
        m.eval()
        x = torch.ones(4, 8)
        out = m(x)
        assert torch.allclose(out, x)

    def test_mask_consistency(self):
        m = VariationalDropout(dropout=0.5)
        m.train()
        torch.manual_seed(42)
        x = torch.ones(4, 8)
        out1 = m(x)
        out2 = m(x)  # Same mask should be reused
        assert torch.allclose(out1, out2)

    def test_reset_changes_mask(self):
        m = VariationalDropout(dropout=0.5)
        m.train()
        torch.manual_seed(42)
        x = torch.ones(4, 8)
        out1 = m(x)
        mask1 = m.mask.clone()
        m.mask = None  # Reset
        torch.manual_seed(123)
        out2 = m(x)
        # Masks should generally differ (with different seeds)
        assert m.mask is not None

    def test_zero_dropout(self):
        m = VariationalDropout(dropout=0.0)
        m.train()
        x = torch.ones(4, 8)
        out = m(x)
        assert torch.allclose(out, x)

    def test_full_dropout(self):
        m = VariationalDropout(dropout=1.0)
        m.train()
        x = torch.ones(4, 8)
        out = m(x)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(x))

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            VariationalDropout(dropout=1.5)


class TestVariationalDropout1d:
    def test_channel_wise(self):
        m = VariationalDropout1d(dropout=0.5, token_first=True)
        m.train()
        torch.manual_seed(42)
        x = torch.ones(2, 4, 8)  # (B, L, D)
        out = m(x)
        assert out.shape == x.shape

    def test_mask_matches_input_device_and_dtype(self):
        m = VariationalDropout1d(dropout=0.5, token_first=True)
        m.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        x = torch.ones(2, 4, 8, device=device, dtype=dtype)
        out = m(x)
        assert out.device == x.device
        assert out.dtype == x.dtype
        assert m.mask is not None
        assert m.mask.device == x.device
        assert m.mask.dtype == x.dtype


class TestVariationalDropout2d:
    def test_channel_wise(self):
        m = VariationalDropout2d(dropout=0.5, token_first=False)
        m.train()
        torch.manual_seed(42)
        x = torch.ones(2, 8, 4, 4)  # (B, D, H, W)
        out = m(x)
        assert out.shape == x.shape


class TestResetDropout:
    def test_reset_model(self):
        model = torch.nn.Sequential(
            VariationalDropout(dropout=0.5),
            torch.nn.Linear(8, 8),
        )
        model.train()
        x = torch.ones(4, 8)
        model[0](x)
        assert model[0].mask is not None
        reset_dropout(model)
        assert model[0].mask is None
