from __future__ import annotations

import argparse
import warnings

import pytest

from torchdeq.utils.config import DEQConfig


class TestDEQConfigDefaults:
    def test_default_construction(self):
        config = DEQConfig()
        assert config.f_solver == "fixed_point_iter"
        assert config.b_solver == "fixed_point_iter"
        assert config.f_max_iter == 40
        assert config.b_max_iter == 40
        assert config.f_tol == 1e-3
        assert config.b_tol == 1e-6
        assert config.core == "sliced"
        assert config.grad == [1]
        assert config.indexing == []
        assert config.sup_loc == []
        assert config.mem_gc is False
        assert config.norm_type == "weight_norm"

    def test_custom_construction(self):
        config = DEQConfig(f_max_iter=100, f_tol=1e-5, core="indexing")
        assert config.f_max_iter == 100
        assert config.f_tol == 1e-5
        assert config.core == "indexing"


class TestDEQConfigFromArgs:
    def test_from_none(self):
        config = DEQConfig.from_args(None)
        assert config.f_max_iter == 40

    def test_from_dict(self):
        config = DEQConfig.from_args({"f_max_iter": 100, "f_tol": 1e-5})
        assert config.f_max_iter == 100
        assert config.f_tol == 1e-5

    def test_from_namespace(self):
        ns = argparse.Namespace(f_max_iter=20, f_tol=1e-4, unknown_key="ignored")
        config = DEQConfig.from_args(ns)
        assert config.f_max_iter == 20
        assert config.f_tol == 1e-4

    def test_from_deq_config(self):
        original = DEQConfig(f_max_iter=50)
        config = DEQConfig.from_args(original)
        assert config.f_max_iter == 50

    def test_overrides(self):
        config = DEQConfig.from_args({"f_max_iter": 10}, f_max_iter=20)
        assert config.f_max_iter == 20

    def test_from_args_preserves_mem_gc(self):
        config = DEQConfig.from_args({"mem_gc": True})
        assert config.mem_gc is True

    def test_filters_unknown_keys(self):
        config = DEQConfig.from_args({"f_max_iter": 10, "unknown_thing": 42})
        assert config.f_max_iter == 10
        assert not hasattr(config, "unknown_thing")


class TestDEQConfigValidation:
    def test_f_max_iter_validation(self):
        with pytest.raises(ValueError, match="f_max_iter"):
            DEQConfig(f_max_iter=0)

    def test_f_tol_validation(self):
        with pytest.raises(ValueError, match="f_tol"):
            DEQConfig(f_tol=-1.0)

    def test_b_tol_validation(self):
        with pytest.raises(ValueError, match="b_tol"):
            DEQConfig(b_tol=0.0)

    def test_grad_int_auto_convert(self):
        config = DEQConfig(grad=3)
        assert config.grad == [3]


class TestDEQConfigLegacy:
    def test_get_deprecated(self):
        config = DEQConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = config.get("f_max_iter")
            assert val == 40
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_update_deprecated(self):
        config = DEQConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.update(f_max_iter=100)
            assert config.f_max_iter == 100
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_get_default(self):
        config = DEQConfig()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            val = config.get("nonexistent", "default_val")
            assert val == "default_val"
