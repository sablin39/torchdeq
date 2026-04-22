from __future__ import annotations

import pytest
import torch

from torchdeq.solver import get_solver, register_solver
from torchdeq.solver.anderson import anderson_solver
from torchdeq.solver.broyden import broyden_solver
from torchdeq.solver.fp_iter import fixed_point_iter, simple_fixed_point_iter

HAS_WHILE_LOOP = hasattr(torch, "while_loop")


def _assert_tensor_close(left: torch.Tensor, right: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6):
    torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def _assert_trajectory_equal(left: list[torch.Tensor], right: list[torch.Tensor]):
    assert len(left) == len(right)
    for left_state, right_state in zip(left, right):
        _assert_tensor_close(left_state, right_state)


def _assert_info_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]):
    for key in ['abs_lowest', 'rel_lowest', 'nstep', 'abs_trace', 'rel_trace']:
        _assert_tensor_close(left[key], right[key])


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
        z_star, _, _ = fixed_point_iter(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-5

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = fixed_point_iter(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4

    def test_batched(self, batched_contraction):
        x0 = torch.zeros(4, 8)
        z_star, _, _ = fixed_point_iter(batched_contraction, x0, max_iter=100, tol=1e-6)
        assert z_star.shape == (4, 8)
        assert (z_star - batched_contraction(z_star)).abs().max() < 1e-5

    def test_early_stop(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        _, _, info = fixed_point_iter(contraction_half, x0, max_iter=1000, tol=1e-3)
        assert info['nstep'].max() < 1000

    def test_indexing(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        _, trajectory, _ = fixed_point_iter(
            contraction_half, x0, max_iter=40, tol=1e-6, indexing=[10, 20, 30], return_final=True
        )
        assert len(trajectory) == 3

    def test_stat_keys(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        _, _, info = fixed_point_iter(contraction_half, x0, max_iter=40)
        for key in ['abs_lowest', 'rel_lowest', 'abs_trace', 'rel_trace', 'nstep']:
            assert key in info


@pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
class TestFixedPointIterCompileParity:
    def test_return_final_abs_mode_matches_eager(self, contraction_half):
        x0 = torch.zeros(2, 4)
        eager = fixed_point_iter(contraction_half, x0, max_iter=12, tol=1e-6, return_final=True)
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=12,
                tol=1e-6,
                return_final=True,
                compile_solver=True,
            )

        _assert_tensor_close(eager[0], compiled[0])
        _assert_trajectory_equal(eager[1], compiled[1])
        _assert_info_equal(eager[2], compiled[2])

    def test_return_final_rel_mode_matches_eager(self, contraction_cos):
        x0 = torch.zeros(2, 3)
        eager = fixed_point_iter(
            contraction_cos,
            x0,
            max_iter=16,
            tol=1e-4,
            stop_mode='rel',
            return_final=True,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_cos,
                x0,
                max_iter=16,
                tol=1e-4,
                stop_mode='rel',
                return_final=True,
                compile_solver=True,
            )

        _assert_tensor_close(eager[0], compiled[0], atol=1e-5, rtol=1e-5)
        _assert_trajectory_equal(eager[1], compiled[1])
        _assert_info_equal(eager[2], compiled[2])

    def test_early_stop_abs_mode_matches_eager(self, contraction_half):
        x0 = torch.zeros(3, 2)
        eager = fixed_point_iter(contraction_half, x0, max_iter=20, tol=1e-3, return_final=False)
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=torch.tensor(20),
                tol=1e-3,
                return_final=False,
                compile_solver=True,
                max_iter_bound=20,
            )

        _assert_tensor_close(eager[0], compiled[0])
        _assert_tensor_close(eager[2]['abs_lowest'], compiled[2]['abs_lowest'])
        _assert_tensor_close(eager[2]['rel_lowest'], compiled[2]['rel_lowest'])
        _assert_tensor_close(eager[2]['nstep'], compiled[2]['nstep'])
        _assert_tensor_close(eager[2]['abs_trace'], compiled[2]['abs_trace'])
        _assert_tensor_close(eager[2]['rel_trace'], compiled[2]['rel_trace'])

    def test_early_stop_rel_mode_matches_eager(self, contraction_cos):
        x0 = torch.zeros(1, 3)
        eager = fixed_point_iter(
            contraction_cos,
            x0,
            max_iter=25,
            tol=1e-4,
            stop_mode='rel',
            return_final=False,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_cos,
                x0,
                max_iter=torch.tensor(25),
                tol=1e-4,
                stop_mode='rel',
                return_final=False,
                compile_solver=True,
                max_iter_bound=25,
            )

        _assert_tensor_close(eager[0], compiled[0], atol=1e-5, rtol=1e-5)
        _assert_tensor_close(eager[2]['abs_lowest'], compiled[2]['abs_lowest'], atol=1e-5, rtol=1e-5)
        _assert_tensor_close(eager[2]['rel_lowest'], compiled[2]['rel_lowest'], atol=1e-5, rtol=1e-5)
        _assert_tensor_close(eager[2]['nstep'], compiled[2]['nstep'])
        _assert_tensor_close(eager[2]['abs_trace'], compiled[2]['abs_trace'], atol=1e-5, rtol=1e-5)
        _assert_tensor_close(eager[2]['rel_trace'], compiled[2]['rel_trace'], atol=1e-5, rtol=1e-5)

    def test_non_unit_tau_matches_eager(self, contraction_half):
        x0 = torch.zeros(2, 2)
        eager = fixed_point_iter(contraction_half, x0, max_iter=10, tau=0.5, return_final=True)
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=torch.tensor(10),
                tau=0.5,
                return_final=True,
                compile_solver=True,
                max_iter_bound=10,
            )
        _assert_tensor_close(eager[0], compiled[0])
        _assert_tensor_close(eager[2]['abs_lowest'], compiled[2]['abs_lowest'])
        _assert_tensor_close(eager[2]['rel_lowest'], compiled[2]['rel_lowest'])
        _assert_tensor_close(eager[2]['nstep'], compiled[2]['nstep'])
        _assert_tensor_close(eager[2]['abs_trace'], compiled[2]['abs_trace'])
        _assert_tensor_close(eager[2]['rel_trace'], compiled[2]['rel_trace'])


class TestFixedPointIterIndexing:
    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_exact_capture_case_matches_eager(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        eager = fixed_point_iter(
            contraction_half,
            x0,
            max_iter=40,
            tol=1e-6,
            indexing=[10, 20, 30],
            return_final=True,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=40,
                tol=1e-6,
                indexing=[10, 20, 30],
                return_final=True,
                compile_solver=True,
            )

        _assert_tensor_close(eager[0], compiled[0])
        _assert_trajectory_equal(eager[1], compiled[1])
        _assert_info_equal(eager[2], compiled[2])

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_unordered_duplicate_indexing_normalizes(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        eager = fixed_point_iter(
            contraction_half,
            x0,
            max_iter=40,
            tol=1e-6,
            indexing=[30, 10, 10, 20],
            return_final=True,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=40,
                tol=1e-6,
                indexing=[30, 10, 10, 20],
                return_final=True,
                compile_solver=True,
            )
        _assert_trajectory_equal(eager[1], compiled[1])
        _assert_info_equal(eager[2], compiled[2])

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_no_hit_fallback_by_budget(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        with torch.no_grad():
            z_star, trajectory, _ = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=20,
                tol=1e-6,
                indexing=[50],
                return_final=True,
                compile_solver=True,
            )
        assert len(trajectory) == 1
        _assert_tensor_close(trajectory[0], z_star)

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_no_hit_fallback_by_early_stop(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        eager = fixed_point_iter(
            contraction_half,
            x0,
            max_iter=20,
            tol=1e-2,
            indexing=[10],
            return_final=False,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=20,
                tol=1e-2,
                indexing=[10],
                return_final=False,
                compile_solver=True,
            )

        assert len(eager[1]) == 1
        assert len(compiled[1]) == 1
        _assert_trajectory_equal(eager[1], compiled[1])

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_partial_hit_early_stop(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        eager = fixed_point_iter(
            contraction_half,
            x0,
            max_iter=20,
            tol=2e-1,
            indexing=[2, 4, 8],
            return_final=False,
        )
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=20,
                tol=2e-1,
                indexing=[2, 4, 8],
                return_final=False,
                compile_solver=True,
            )

        _assert_trajectory_equal(eager[1], compiled[1])
        _assert_info_equal(eager[2], compiled[2])

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_tensor_depth_with_indexing_raises(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        with torch.no_grad(), pytest.raises(ValueError, match="does not support tensor max_iter together with indexing"):
            fixed_point_iter(
                contraction_half,
                x0,
                max_iter=torch.tensor(12),
                tol=1e-6,
                indexing=[2, 4],
                compile_solver=True,
                max_iter_bound=12,
            )


@pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
class TestFixedPointIterDynamicDepth:
    @pytest.mark.parametrize("depth", [8, 10, 12, 14])
    def test_dynamic_depth_matches_eager_summary_stats(self, contraction_half, depth: int):
        x0 = torch.zeros(2, 3)
        eager = fixed_point_iter(contraction_half, x0, max_iter=depth, tol=1e-6, return_final=True)
        with torch.no_grad():
            compiled = fixed_point_iter(
                contraction_half,
                x0,
                max_iter=torch.tensor(depth),
                tol=1e-6,
                return_final=True,
                compile_solver=True,
                max_iter_bound=16,
            )

        _assert_tensor_close(eager[0], compiled[0])
        _assert_tensor_close(eager[2]['abs_lowest'], compiled[2]['abs_lowest'])
        _assert_tensor_close(eager[2]['rel_lowest'], compiled[2]['rel_lowest'])
        _assert_tensor_close(eager[2]['nstep'], compiled[2]['nstep'])
        assert compiled[2]['abs_trace'].shape == (2, 17)
        assert compiled[2]['rel_trace'].shape == (2, 17)
        _assert_tensor_close(compiled[2]['abs_trace'][:, : depth + 1], eager[2]['abs_trace'])
        _assert_tensor_close(compiled[2]['rel_trace'][:, : depth + 1], eager[2]['rel_trace'])

    def test_missing_bound_raises(self, contraction_half):
        x0 = torch.zeros(1, 1)
        with torch.no_grad(), pytest.raises(ValueError, match="requires max_iter_bound"):
            fixed_point_iter(
                contraction_half,
                x0,
                max_iter=torch.tensor(10),
                compile_solver=True,
            )


class TestSimpleFixedPointIter:
    def test_converges(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = simple_fixed_point_iter(contraction_half, x0, max_iter=100)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-5


class TestAndersonSolver:
    def test_converges_half(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = anderson_solver(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-4

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = anderson_solver(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4


class TestBroydenSolver:
    def test_converges_half(self, contraction_half):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = broyden_solver(contraction_half, x0, max_iter=100, tol=1e-6)
        assert (z_star - contraction_half(z_star)).abs().max() < 1e-4

    def test_converges_cos(self, contraction_cos):
        x0 = torch.tensor([[0.0]])
        z_star, _, _ = broyden_solver(contraction_cos, x0, max_iter=200, tol=1e-6)
        assert (z_star - contraction_cos(z_star)).abs().max() < 1e-4
