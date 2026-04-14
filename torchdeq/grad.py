from __future__ import annotations

"""
The `torchdeq.grad` module offers a factory function, `backward_factory`,
which is designed to facilitate the customization of various differentiation methods during the backward pass.

This function is integral to the construction of the backward computational graph in the DEQ class,
as it is invoked multiple times to generate gradient functors.

While the backward_factory function is a powerful tool, it is generally not recommended for direct use outside of the library.
Instead, users should primarily interact with the DEQ class via the `torch.core` entry point for most DEQ computations.
This approach ensures the appropriate and efficient use of the library's features.
"""
import torch
from torch import autograd
from typing import Callable

from .utils.mem import (
    _capture_autocast_state,
    _recompute_with_autocast,
    mem_gc as checkpoint_mem_gc,
)


__all__ = ['backward_factory']


def make_pair(target: list, source: list) -> list:
    """
    Aligns the argument sequence between target and source.

    Args:
        target: The target list for alignment.
        source: The source list for alignment.

    Returns:
        The aligned source.

    Raises:
        ValueError: If the length of source is neither 1 nor equal to the length of target.
    """
    if len(target) == len(source):
        return source
    elif len(source) == 1:
        return [source[0] for _ in range(len(target))]
    else:
        raise ValueError('Unable to align the arg squence!')


def backward_factory(
        grad_type: str | int = 'ift',
        hook_ift: bool = False,
        b_solver: Callable | None = None,
        b_solver_kwargs: dict | None = None,
        sup_gap: int = -1,
        sup_loc: list[int] | None = None,
        tau: float = 1.0,
        **grad_factory_kwargs,
) -> Callable:
    """
    Factory for the backward pass of implicit deep learning.

    Args:
        grad_type: Gradient type. ``'ift'`` for IFT or an int for PhantomGrad. Default ``'ift'``.
        hook_ift: Use hook-based O(1) memory IFT. Default False.
        b_solver: Solver for IFT backward pass. Default None.
        b_solver_kwargs: Backward solver keyword arguments. Default None.
        sup_gap: Sampling gap for PhantomGrad trajectories. Default -1.
        sup_loc: Sampling locations for PhantomGrad. Default None.
        tau: Damping factor for PhantomGrad. Default 1.0.
        grad_factory_kwargs: Extra arguments are ignored.

    Returns:
        A gradient functor for implicit deep learning.
    """
    if b_solver_kwargs is None:
        b_solver_kwargs = {}

    def replay_with_mem_gc(
        func: Callable,
        z_pred: torch.Tensor,
        *,
        tau: float,
        use_mem_gc: bool,
        func_params,
    ) -> torch.Tensor:
        if not use_mem_gc:
            return func(z_pred, tau=tau)

        if func_params is None:
            raise ValueError(
                "PhantomGrad mem_gc replay requires func_params from the original nn.Module."
            )

        return checkpoint_mem_gc(
            lambda z: func(z, tau=tau),
            (z_pred,),
            params=func_params,
        )

    # IFT grad
    if grad_type == 'ift':
        if hook_ift:
            # IFT via Pytorch hook mechanism
            @torch.compiler.disable
            def hook_ift_grad(
                trainer: torch.nn.Module,
                func: Callable,
                z_pred: torch.Tensor,
                writer: Callable | None = None,
                **kwargs,
            ) -> list[torch.Tensor]:
                z_pred = z_pred.requires_grad_()
                new_z_pred = func(z_pred)        # 1-step grad for df/dtheta

                def backward_hook(grad: torch.Tensor) -> torch.Tensor:
                    if trainer.hook is not None:
                        trainer.hook.remove()    # To avoid infinite loop
                    grad_star, _, info = b_solver(
                            lambda y: autograd.grad(new_z_pred, z_pred, y, retain_graph=True)[0] + grad,
                            torch.zeros_like(grad), **b_solver_kwargs
                            )
                    if writer:
                        writer(info)
                    return grad_star
                trainer.hook = new_z_pred.register_hook(backward_hook)

                return [new_z_pred]
            return hook_ift_grad
        else:
            # IFT via torch.autograd.Function
            class IFTGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, func: Callable, z_pred: torch.Tensor, writer: Callable | None) -> torch.Tensor:
                    ctx.func, ctx.writer = func, writer
                    ctx.autocast_state = _capture_autocast_state()
                    ctx.save_for_backward(z_pred.detach())
                    return z_pred

                @staticmethod
                @torch.compiler.disable
                def backward(ctx, grad: torch.Tensor) -> tuple[None, torch.Tensor, None]:
                    func, writer = ctx.func, ctx.writer
                    z_pred, = ctx.saved_tensors

                    h = z_pred.clone().detach().requires_grad_()
                    with torch.enable_grad():
                        f = _recompute_with_autocast(func, ctx.autocast_state, h)

                    grad_f = lambda x: autograd.grad(f, h, x, retain_graph=True)[0] + grad
                    grad_star, _, info = b_solver(
                            grad_f, torch.zeros_like(grad), **b_solver_kwargs
                            )
                    if writer:
                        writer(info)

                    return (None, grad_star, None)

            def func_ift(
                _: torch.nn.Module,
                func: Callable,
                z_pred: torch.Tensor,
                writer: Callable | None = None,
                **kwargs,
            ) -> list[torch.Tensor]:
                new_z_pred = func(z_pred)       # 1-step grad for df/dtheta
                return [IFTGrad.apply(func, new_z_pred, writer)]
            return func_ift

    # Phantom Grad
    else:
        if not (type(grad_type) is int and grad_type >= 1):
            raise ValueError(f'grad_type must be "ift" or a positive int, got {grad_type}')
        n_grad_step = grad_type

        if sup_gap > 0:
            def sup_gap_grad_func(
                _: torch.nn.Module,
                func: Callable,
                z_pred: torch.Tensor,
                **kwargs,
            ) -> list[torch.Tensor]:
                z_out: list[torch.Tensor] = []
                use_mem_gc = kwargs.get("mem_gc", False)
                func_params = kwargs.get("func_params")
                for i in range(n_grad_step):
                    z_pred = replay_with_mem_gc(
                        func,
                        z_pred,
                        tau=tau,
                        use_mem_gc=use_mem_gc,
                        func_params=func_params,
                    )
                    if (i+1) % sup_gap == 0:
                        z_out.append(z_pred)

                return z_out
            return sup_gap_grad_func
        elif sup_loc:
            def sup_loc_grad_func(
                _: torch.nn.Module,
                func: Callable,
                z_pred: torch.Tensor,
                **kwargs,
            ) -> list[torch.Tensor]:
                z_out: list[torch.Tensor] = []
                use_mem_gc = kwargs.get("mem_gc", False)
                func_params = kwargs.get("func_params")
                for i in range(n_grad_step):
                    z_pred = replay_with_mem_gc(
                        func,
                        z_pred,
                        tau=tau,
                        use_mem_gc=use_mem_gc,
                        func_params=func_params,
                    )
                    if i+1 in sup_loc:
                        z_out.append(z_pred)
                z_out.append(z_pred)

                return z_out
            return sup_loc_grad_func
        else:
            def grad_func(
                _: torch.nn.Module,
                func: Callable,
                z_pred: torch.Tensor,
                **kwargs,
            ) -> list[torch.Tensor]:
                use_mem_gc = kwargs.get("mem_gc", False)
                func_params = kwargs.get("func_params")
                for _ in range(n_grad_step):
                    z_pred = replay_with_mem_gc(
                        func,
                        z_pred,
                        tau=tau,
                        use_mem_gc=use_mem_gc,
                        func_params=func_params,
                    )

                return [z_pred]
            return grad_func
