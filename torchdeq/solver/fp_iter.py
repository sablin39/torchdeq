from __future__ import annotations

import torch
import torch.nn.functional as F

from .stat import SolverStat
from .utils import (
    batch_flatten,
)

__all__ = ['fixed_point_iter', 'simple_fixed_point_iter']


def _normalize_indexing(indexing):
    if not indexing:
        return None
    return sorted({int(index) for index in indexing})


def _max_iter_to_int(max_iter):
    if torch.is_tensor(max_iter):
        if max_iter.ndim != 0:
            raise ValueError(f"max_iter tensor must be scalar, got shape={tuple(max_iter.shape)}")
        return int(max_iter.item())
    return int(max_iter)


def _update_trace_buffer(trace_buffer: torch.Tensor, step: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    mask = F.one_hot(step.to(torch.int64), num_classes=trace_buffer.shape[1]).to(trace_buffer.dtype)
    return trace_buffer * (1 - mask.unsqueeze(0)) + values.unsqueeze(1) * mask.unsqueeze(0)


@torch.compiler.disable(recursive=False, reason="Solver dispatch stays outside Dynamo")
def _dispatch_fixed_point_iter(
        func, x0,
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None,
        tau=1.0, return_final=False,
        compile_solver=False,
        max_iter_bound=None,
        **kwargs):
    if compile_solver:
        return _fixed_point_iter_compile(
            func,
            x0,
            max_iter=max_iter,
            tol=tol,
            stop_mode=stop_mode,
            indexing=indexing,
            tau=tau,
            return_final=return_final,
            max_iter_bound=max_iter_bound,
            **kwargs,
        )

    return _fixed_point_iter_eager(
        func,
        x0,
        max_iter=max_iter,
        tol=tol,
        stop_mode=stop_mode,
        indexing=indexing,
        tau=tau,
        return_final=return_final,
        **kwargs,
    )


@torch.compiler.disable(recursive=False, reason="Simple solver dispatch stays outside Dynamo")
def _dispatch_simple_fixed_point_iter(
        func, x0,
        max_iter=50, tau=1.0,
        indexing=None,
        **kwargs):
    return _simple_fixed_point_iter_eager(
        func,
        x0,
        max_iter=max_iter,
        tau=tau,
        indexing=indexing,
        **kwargs,
    )


@torch.compiler.disable(recursive=False, reason="Keep solver-step Python control flow out of Dynamo")
def _fixed_point_iter_eager(
        func, x0,
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None,
        tau=1.0, return_final=False,
        **kwargs):
    """
    Implements the fixed-point iteration solver for solving a system of nonlinear equations.
    
    Args:
        func (callable): The function for which we seek a fixed point.
        x0 (torch.Tensor): The initial guess for the root.
        max_iter (int, optional): The maximum number of iterations. Default: 50.
        tol (float, optional): The convergence criterion. Default: 1e-3.
        stop_mode (str, optional): The stopping criterion. Can be either 'abs' or 'rel'. Default: 'abs'.
        indexing (list, optional): List of iteration indices at which to store the solution. Default: None.
        tau (float, optional): Damping factor. It is used to control the step size in the direction of the solution. Default: 1.0.
        return_final (bool, optional): If True, run all steps and returns the final solution instead of the one with smallest residual. Default: False.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: Fixed point solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]: 
                A dict containing solver statistics in a batch.
                Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
    
    Examples:
        >>> f = lambda z: torch.cos(z)                  # Function for which we seek a fixed point
        >>> z0 = torch.tensor(0.0)                      # Initial estimate
        >>> z_star, _, _ = fixed_point_iter(f, z0)      # Run Fixed Point iterations.
        >>> print((z_star - f(z_star)).norm(p=1))       # Print the numerical error
    """
    # Check input batch size
    bsz = x0.shape[0] if x0.dim() >= 2 else x0.nelement()

    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    # Initialize solver statistics
    init_loss = torch.tensor(1e8, device=x0.device).repeat(bsz)
    trace_dict = {
        'abs': [init_loss.clone()],
        'rel': [init_loss.clone()],
    }
    lowest_dict = {
        'abs': init_loss.clone(),
        'rel': init_loss.clone(),
    }
    lowest_step_dict = {
        'abs': torch.zeros(bsz, device=x0.device),
        'rel': torch.zeros(bsz, device=x0.device),
    }
    lowest_xest = x0

    indexing = _normalize_indexing(indexing)
    indexing_list = []
    
    max_iter_int = _max_iter_to_int(max_iter)
    fx = x = x0
    for k in range(max_iter_int):
        x = fx
        fx = tau * func(x) + (1 - tau) * x
        
        # Calculate the absolute and relative differences# Update the state based on the new estimate
        gx = fx - x
        abs_diff = batch_flatten(gx).norm(dim=1)
        rel_diff = abs_diff / (batch_flatten(fx).norm(dim=1) + 1e-9)

        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        abs_is_lowest = (abs_diff < lowest_dict['abs']) | return_final
        rel_is_lowest = (rel_diff < lowest_dict['rel']) | return_final
        update_mask = abs_is_lowest if stop_mode == 'abs' else rel_is_lowest
        aligned_mask = update_mask.view(update_mask.shape[0], *([1] * (fx.dim() - 1)))

        lowest_xest = torch.where(aligned_mask, fx, lowest_xest).clone().detach()
        lowest_dict['abs'] = torch.where(abs_is_lowest, abs_diff, lowest_dict['abs'])
        lowest_dict['rel'] = torch.where(rel_is_lowest, rel_diff, lowest_dict['rel'])
        step_tensor = torch.zeros_like(lowest_step_dict['abs']) + (k + 1)
        lowest_step_dict['abs'] = torch.where(abs_is_lowest, step_tensor, lowest_step_dict['abs'])
        lowest_step_dict['rel'] = torch.where(rel_is_lowest, step_tensor, lowest_step_dict['rel'])

         # If indexing is enabled, store the solution at the specified indices
        if indexing and (k+1) in indexing:
            indexing_list.append(lowest_xest)

        # If the difference is smaller than the given tolerance, terminate the loop early
        if not return_final and trace_dict[stop_mode][-1].max() < tol:
            for _ in range(max_iter_int - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    info = SolverStat(
        abs_lowest=lowest_dict['abs'],
        rel_lowest=lowest_dict['rel'],
        abs_trace=torch.stack(trace_dict['abs'], dim=1),
        rel_trace=torch.stack(trace_dict['rel'], dim=1),
        nstep=lowest_step_dict[stop_mode],
    )
    return lowest_xest, indexing_list, info


def _fixed_point_iter_compile(
        func, x0,
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None,
        tau=1.0, return_final=False,
        max_iter_bound=None,
        **kwargs):
    if torch.is_grad_enabled():
        return _fixed_point_iter_eager(
            func,
            x0,
            max_iter=max_iter,
            tol=tol,
            stop_mode=stop_mode,
            indexing=indexing,
            tau=tau,
            return_final=return_final,
            **kwargs,
        )

    if not torch.is_tensor(max_iter):
        return _fixed_point_iter_eager(
            func,
            x0,
            max_iter=max_iter,
            tol=tol,
            stop_mode=stop_mode,
            indexing=indexing,
            tau=tau,
            return_final=return_final,
            **kwargs,
        )

    if max_iter.ndim != 0:
        raise ValueError(f"compile solver requires scalar tensor max_iter, got shape={tuple(max_iter.shape)}")

    indexing = _normalize_indexing(indexing)
    if indexing is not None:
        raise ValueError(
            "compile solver does not support tensor max_iter together with indexing; "
            "use eager mode or a static Python int max_iter instead."
        )

    if max_iter_bound is None:
        raise ValueError("compile solver requires max_iter_bound when max_iter is a tensor")
    max_iter_bound = int(max_iter_bound)
    if max_iter_bound < 1:
        raise ValueError(f"max_iter_bound must be >= 1, got {max_iter_bound}")

    device = x0.device
    max_iter = max_iter.to(device=device, dtype=torch.int64)
    tol_tensor = torch.as_tensor(float(tol), device=device, dtype=torch.float32)
    init_loss = torch.tensor(1e8, device=device, dtype=torch.float32)
    bsz = x0.shape[0] if x0.dim() >= 2 else x0.nelement()

    abs_trace = init_loss.repeat(bsz, max_iter_bound + 1)
    rel_trace = init_loss.repeat(bsz, max_iter_bound + 1)
    abs_lowest = init_loss.repeat(bsz)
    rel_lowest = init_loss.repeat(bsz)
    lowest_xest = x0.clone()
    nstep = torch.zeros(bsz, device=device, dtype=torch.float32)
    k0 = torch.zeros((), device=device, dtype=torch.int64)
    x0_clone = x0.clone()
    fx0 = x0.clone()
    current_stop = init_loss.clone()

    def cond_fn(k, x, fx, lowest_xest, abs_lowest, rel_lowest, nstep, current_stop, abs_trace, rel_trace):
        if return_final:
            return k < max_iter
        return (k < max_iter) & (current_stop >= tol_tensor)

    def body_fn(k, x, fx, lowest_xest, abs_lowest, rel_lowest, nstep, current_stop, abs_trace, rel_trace):
        x_new = fx.clone()
        fx_new = tau * func(x_new) + (1 - tau) * x_new

        gx = fx_new - x_new
        abs_diff = batch_flatten(gx).norm(dim=1).to(torch.float32)
        rel_diff = abs_diff / (batch_flatten(fx_new).norm(dim=1).to(torch.float32) + 1e-9)

        step = k + 1
        step_float = torch.zeros_like(nstep) + step.to(torch.float32)
        abs_trace_new = _update_trace_buffer(abs_trace, step, abs_diff)
        rel_trace_new = _update_trace_buffer(rel_trace, step, rel_diff)

        if return_final:
            lowest_xest_new = fx_new.clone()
            abs_lowest_new = abs_diff.clone()
            rel_lowest_new = rel_diff.clone()
            nstep_new = step_float
        else:
            abs_is_lowest = abs_diff < abs_lowest
            rel_is_lowest = rel_diff < rel_lowest
            update_mask = abs_is_lowest if stop_mode == 'abs' else rel_is_lowest
            lowest_xest_new = torch.where(
                update_mask.view(update_mask.shape[0], *([1] * (fx_new.dim() - 1))),
                fx_new,
                lowest_xest,
            ).clone()
            abs_lowest_new = torch.where(abs_is_lowest, abs_diff, abs_lowest)
            rel_lowest_new = torch.where(rel_is_lowest, rel_diff, rel_lowest)
            nstep_new = torch.where(
                update_mask,
                step_float,
                nstep,
            )

        stop_metric = abs_diff if stop_mode == 'abs' else rel_diff
        current_stop_new = stop_metric.max()

        return (
            step,
            x_new,
            fx_new,
            lowest_xest_new,
            abs_lowest_new,
            rel_lowest_new,
            nstep_new,
            current_stop_new,
            abs_trace_new,
            rel_trace_new,
        )

    (
        final_k,
        _final_x,
        _final_fx,
        lowest_xest,
        abs_lowest,
        rel_lowest,
        nstep,
        _current_stop,
        abs_trace,
        rel_trace,
    ) = torch.while_loop(
        cond_fn,
        body_fn,
        (
            k0,
            x0_clone,
            fx0,
            lowest_xest,
            abs_lowest,
            rel_lowest,
            nstep,
            current_stop,
            abs_trace,
            rel_trace,
        ),
    )

    trace_positions = torch.arange(max_iter_bound + 1, device=device, dtype=torch.int64)
    fill_mask = trace_positions.unsqueeze(0) > final_k
    abs_trace = torch.where(fill_mask, abs_lowest.unsqueeze(1), abs_trace)
    rel_trace = torch.where(fill_mask, rel_lowest.unsqueeze(1), rel_trace)

    info = SolverStat(
        abs_lowest=abs_lowest,
        rel_lowest=rel_lowest,
        abs_trace=abs_trace,
        rel_trace=rel_trace,
        nstep=nstep,
    )
    return lowest_xest, [], info


def fixed_point_iter(func, x0, 
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None, 
        tau=1.0, return_final=False, 
        compile_solver=False,
        max_iter_bound=None,
        **kwargs):
    """
    Implements the fixed-point iteration solver for solving a system of nonlinear equations.
    """
    return _dispatch_fixed_point_iter(
        func,
        x0,
        max_iter=max_iter,
        tol=tol,
        stop_mode=stop_mode,
        indexing=indexing,
        tau=tau,
        return_final=return_final,
        compile_solver=compile_solver,
        max_iter_bound=max_iter_bound,
        **kwargs,
    )


@torch.compiler.disable(recursive=False, reason="Keep solver-step Python control flow out of Dynamo")
def _simple_fixed_point_iter_eager(func, x0, 
        max_iter=50, tau=1.0,
        indexing=None, 
        **kwargs):
    """
    Implements a simplified fixed-point solver for solving a system of nonlinear equations.
    
    Speeds up by removing statistics monitoring.

    Args:
        func (callable): The function for which the fixed point is to be computed.
        x0 (torch.Tensor): The initial guess for the fixed point.
        max_iter (int, optional): The maximum number of iterations. Default: 50.
        tau (float, optional): Damping factor to control the step size in the solution direction. Default: 1.0.
        indexing (list, optional): List of iteration indices at which to store the solution. Default: None.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: The approximate solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]: 
                A dummy dict for solver statistics. All values are initialized as -1 of tensor shape (1, 1).

    Examples:
        >>> f = lambda z: torch.cos(z)                      # Function for which we seek a fixed point
        >>> z0 = torch.tensor(0.0)                          # Initial estimate
        >>> z_star, _, _ = simple_fixed_point_iter(f, z0)   # Run fixed point iterations
        >>> print((z_star - f(z_star)).norm(p=1))           # Print the numerical error
    """
    indexing = _normalize_indexing(indexing)
    indexing_list = []
    
    fx = x = x0
    for k in range(_max_iter_to_int(max_iter)):
        x = fx
        fx = func(x, tau=tau)

         # If indexing is enabled, store the solution at the specified indices
        if indexing and (k+1) in indexing:
            indexing_list.append(fx)
    lowest_xest = fx

    diff = fx - x
    flat_diff = diff.flatten(start_dim=1) if diff.dim() >= 2 else diff.view(diff.nelement(), 1)
    flat_fx = fx.flatten(start_dim=1) if fx.dim() >= 2 else fx.view(fx.nelement(), 1)
    abs_lowest = flat_diff.norm(dim=1)
    rel_lowest = abs_lowest / (flat_fx.norm(dim=1) + 1e-8)
    nstep = torch.full((abs_lowest.shape[0],), float(_max_iter_to_int(max_iter)), device=fx.device)
    info = SolverStat(
        abs_lowest=abs_lowest,
        rel_lowest=rel_lowest,
        abs_trace=abs_lowest.unsqueeze(1),
        rel_trace=rel_lowest.unsqueeze(1),
        nstep=nstep,
    )
    return lowest_xest, indexing_list, info


def simple_fixed_point_iter(func, x0, 
        max_iter=50, tau=1.0,
        indexing=None, 
        **kwargs):
    return _dispatch_simple_fixed_point_iter(
        func,
        x0,
        max_iter=max_iter,
        tau=tau,
        indexing=indexing,
        **kwargs,
    )
