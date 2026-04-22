from __future__ import annotations

import pytest
import torch

from torchdeq.solver.fp_iter import fixed_point_iter, simple_fixed_point_iter

HAS_COMPILE = hasattr(torch, "compile")
HAS_WHILE_LOOP = hasattr(torch, "while_loop")

try:
    from torch._dynamo.testing import CompileCounter
except Exception:  # pragma: no cover - defensive import guard
    CompileCounter = None


def _contraction(z: torch.Tensor, **kwargs) -> torch.Tensor:
    return 0.5 * z + 1.0


@pytest.mark.skipif(not HAS_COMPILE or CompileCounter is None, reason="torch.compile testing helpers unavailable")
class TestCompileStability:
    def test_fixed_point_iter_eager_branch_is_stable_across_python_depths(self):
        counter = CompileCounter()

        def caller(x: torch.Tensor, max_iter: int) -> torch.Tensor:
            z_star, _, _ = fixed_point_iter(
                _contraction,
                x,
                max_iter=max_iter,
                tol=1e-6,
                compile_solver=False,
            )
            return z_star + x

        compiled = torch.compile(caller, backend=counter, dynamic=True)
        x = torch.zeros(2, 3)

        compiled(x, 8)
        first_frame_count = counter.frame_count
        compiled(x, 12)

        assert first_frame_count >= 1
        assert counter.frame_count == first_frame_count

    def test_simple_fixed_point_iter_eager_branch_is_stable_across_python_depths(self):
        counter = CompileCounter()

        def caller(x: torch.Tensor, max_iter: int) -> torch.Tensor:
            z_star, _, _ = simple_fixed_point_iter(
                _contraction,
                x,
                max_iter=max_iter,
            )
            return z_star + x

        compiled = torch.compile(caller, backend=counter, dynamic=True)
        x = torch.zeros(2, 3)

        compiled(x, 4)
        first_frame_count = counter.frame_count
        compiled(x, 7)

        assert first_frame_count >= 1
        assert counter.frame_count == first_frame_count

    @pytest.mark.skipif(not HAS_WHILE_LOOP, reason="torch.while_loop not available")
    def test_fixed_point_iter_compile_branch_is_stable_across_tensor_depths(self):
        counter = CompileCounter()

        def caller(x: torch.Tensor, max_iter: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                z_star, _, _ = fixed_point_iter(
                    _contraction,
                    x,
                    max_iter=max_iter,
                    tol=1e-6,
                    return_final=True,
                    compile_solver=True,
                    max_iter_bound=16,
                )
            return z_star + x

        compiled = torch.compile(caller, backend=counter)
        x = torch.zeros(2, 3)

        compiled(x, torch.tensor(8))
        first_frame_count = counter.frame_count
        compiled(x, torch.tensor(10))
        compiled(x, torch.tensor(14))

        assert first_frame_count >= 1
        assert counter.frame_count == first_frame_count


@pytest.mark.skipif(not HAS_COMPILE, reason="torch.compile not available")
class TestCompileSmoke:
    def test_dropout_smoke(self):
        from torchdeq.dropout import VariationalDropout

        module = VariationalDropout(dropout=0.3)
        module.train()
        x = torch.ones(4, 8)

        out = module(x)

        assert out.shape == x.shape

    def test_fp_correction_smoke(self):
        from torchdeq.loss import fp_correction

        x = [torch.randn(4, 8)]
        y = torch.randn(4, 8)
        criterion = lambda left, right: ((left - right) ** 2).mean()

        loss = fp_correction(criterion, (x, y), weight_func='exp')

        assert loss.item() > 0
