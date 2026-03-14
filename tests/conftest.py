from __future__ import annotations

import pytest
import torch

from torchdeq.utils.config import DEQConfig


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def default_config():
    return DEQConfig()


@pytest.fixture
def contraction_half():
    """f(z) = 0.5z + 1, fixed point at z=2."""
    def f(z, **kwargs):
        return 0.5 * z + 1.0
    return f


@pytest.fixture
def contraction_cos():
    """f(z) = cos(z), fixed point near z=0.7391."""
    def f(z, **kwargs):
        return torch.cos(z)
    return f


@pytest.fixture
def batched_contraction():
    """Batched f(z) = 0.5z + 1 for batch of 4, dim 8."""
    def f(z, **kwargs):
        return 0.5 * z + 1.0
    return f
