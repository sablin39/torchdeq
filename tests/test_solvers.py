from __future__ import annotations

import pytest
import torch

from torchdeq.solver import get_solver, register_solver
from torchdeq.solver.fp_iter import fixed_point_iter, simple_fixed_point_iter
from torchdeq.solver.anderson import anderson_solver
from torchdeq.solver.broyden import broyden_solver


class TestGetSolver:
    def test_get_known_solvers(self):
        for name in ["anderson", "broyden", "fixed_point_iter", "simple_fixed_point_iter"]:
            solver = get_solver(name)
            assert callable(solver)

    def test_get_unknown_solver_raises(self):
        with pytest.raises(KeyError):
            get_solver("nonexistent_solver")

    def test_register_and_get(self):
        def my_solver(func, x0, **kwargs):
            return x0, [], {}
        register_solver("my_solver", my_solver)
        assert get_solver("my_solver") is my_solver


class TestFixedPointIter:
    def test_converges_half(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = fixed_point_iter(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-5

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = fixed_point_iter(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4

    def test_batched(self, batched_contraction):
        x0 = torch.zeros(4, 8)
        z_star, _, info = fixed_point_iter(batched_contraction, x0, max_iter=100, tol=1e-6)
        assert z_star.shape == (4, 8)
        assert (z_star - batched_contraction(z_star)).abs().max() < 1e-5

    def test_early_stop(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = fixed_point_iter(contraction_half, x0, max_iter=1000, tol=1e-3)
        # Should stop well before 1000 iterations
        assert info['nstep'].max() < 1000

    def test_indexing(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, trajectory, info = fixed_point_iter(
            contraction_half, x0, max_iter=40, tol=1e-6, indexing=[10, 20, 30]
        )
        assert len(trajectory) >= 1

    def test_stat_keys(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        _, _, info = fixed_point_iter(contraction_half, x0, max_iter=40)
        for key in ['abs_lowest', 'rel_lowest', 'abs_trace', 'rel_trace', 'nstep']:
            assert key in info


class TestSimpleFixedPointIter:
    def test_converges(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = simple_fixed_point_iter(contraction_half, x0, max_iter=100)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-5


class TestAndersonSolver:
    def test_converges_half(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = anderson_solver(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-4

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = anderson_solver(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4


class TestBroydenSolver:
    def test_converges_half(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = broyden_solver(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-4

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, info = broyden_solver(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4
