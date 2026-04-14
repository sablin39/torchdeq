from __future__ import annotations

"""
The DEQ models are a class of implicit models that solve for fixed points to make predictions.
This module provides the core classes and functions for implementing Deep Equilibrium (DEQ) models in PyTorch.

The main classes in this module are `DEQBase`, `DEQIndexing`, and `DEQSliced`.
`DEQBase` is the base class for DEQ models, and `DEQIndexing` and `DEQSliced` are two specific implementations of DEQ models
that use different strategies for applying gradients during training.

The module also provides utility functions for creating and manipulating DEQ models, such as `get_deq` for creating a DEQ model based on command line arguments,
`register_deq` for registering a new DEQ model class, and `reset_deq` for resetting the normalization and dropout layers of a DEQ model.

Example:
    To create a DEQ model, you can use the `get_deq` function:

    >>> deq = get_deq(args)

    To reset the normalization and dropout layers of a DEQ model, you can use the `reset_deq` function:

    >>> deq_layer = DEQLayer(args)          # A Pytorch Module used in the f of z* = f(z*, x).
    >>> reset_deq(deq_layer)
"""

import numpy as np

import torch
import torch.nn as nn

from .solver import get_solver
from .grad import make_pair, backward_factory
from .loss import power_method
from .utils.config import DEQConfig
from .utils.layer_utils import deq_decorator
from .utils.mem import DEQGradCkpt, _capture_autocast_state, _recompute_with_autocast

from .norm import reset_norm
from .dropout import reset_dropout


__all__ = ['get_deq', 'reset_deq', 'register_deq', 'DEQSliced', 'DEQIndexing']


class DEQBase(nn.Module):
    """
    Base class for Deep Equilibrium (DEQ) model.

    This class is not intended to be directly instantiated as the actual DEQ module.
    Instead, you should create an instance of a subclass of this class.

    Args:
        args: Configuration for the DEQ model (argparse.Namespace, dict, DEQConfig, or None).
        **kwargs: Additional keyword arguments to override configuration values.
    """
    def __init__(self, args=None, **kwargs):
        super(DEQBase, self).__init__()

        self.args = DEQConfig.from_args(args, **kwargs)

        self.f_solver = get_solver(self.args.f_solver)
        self.b_solver = get_solver(self.args.b_solver)

        self.no_stat = self.args.f_solver == 'simple_fixed_point_iter'

        self.f_max_iter = self.args.f_max_iter
        self.b_max_iter = self.args.b_max_iter

        self.f_tol = self.args.f_tol
        self.b_tol = self.args.b_tol

        self.f_stop_mode = self.args.f_stop_mode
        self.b_stop_mode = self.args.b_stop_mode

        self.eval_f_max_iter = (
            self.args.eval_f_max_iter if self.args.eval_f_max_iter > 0
            else int(self.f_max_iter * self.args.eval_factor)
        )

        self.hook = None
        self._compile_fn = False
        self._compiled_func_cache = None

    def _maybe_compile(self, func):
        """
        Optionally wraps the user's function in ``torch.compile()`` for eval mode.
        Cached so compilation only happens once.
        """
        if not self._compile_fn or self.training:
            return func
        if self._compiled_func_cache is None:
            self._compiled_func_cache = torch.compile(func)
        return self._compiled_func_cache

    def _sradius(self, deq_func, z_star):
        """
        Estimates the spectral radius using the power method.

        Args:
            deq_func (callable): The DEQ function.
            z_star (torch.Tensor): The fixed point solution.

        Returns:
            float: The spectral radius.
        """
        autocast_state = _capture_autocast_state()
        with torch.enable_grad():
            new_z_star = _recompute_with_autocast(
                deq_func,
                autocast_state,
                z_star.requires_grad_(),
            )
        _, sradius = power_method(new_z_star, z_star, n_iters=100)

        return sradius

    def _resolve_replay_params(self, func):
        if not self.args.mem_gc:
            return None

        if not isinstance(func, nn.Module):
            raise TypeError(
                "DEQ mem_gc requires func to be an nn.Module. Wrap the DEQ function "
                "in a module or call mem_gc(..., params=...) manually."
            )

        return tuple(DEQGradCkpt.fetch_params(func))

    def _solve_fixed_point(
            self, deq_func, z_init,
            f_max_iter=None,
            solver_kwargs=None,
            **kwargs
            ):
        """
        Solves for the fixed point. Must be overridden in subclasses.
        """
        raise NotImplementedError

    def forward(
            self, func, z_init,
            solver_kwargs=None,
            sradius_mode=False,
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ. Must be overridden in subclasses.
        """
        raise NotImplementedError


class DEQIndexing(DEQBase):
    """
    DEQ computational graph that samples fixed point states at specific indices.

    For `DEQIndexing`, it defines a computational graph with tracked gradients by indexing the internal solver
    states and applying the gradient function to the sampled states.
    This is equivalent to attaching the gradient function aside the full solver computational graph.
    The maximum number of DEQ function calls is defined by ``args.f_max_iter``.

    Args:
        args: Configuration for the DEQ model (argparse.Namespace, dict, DEQConfig, or None).
        **kwargs: Additional keyword arguments to override configuration values.
    """
    def __init__(self, args=None, **kwargs):
        super(DEQIndexing, self).__init__(args=args, **kwargs)

        # Preprocess arguments.
        grad = self.args.grad
        if isinstance(grad, int):
            if grad < 1:
                raise ValueError('The minimal gradient step is 1!')
            grad = [grad]

        if type(grad) not in (list, tuple):
            raise ValueError(f'grad must be a list or tuple, got {type(grad)}')

        '''Define gradient functions through the backward factory.'''

        # First compute the f_max_iter indexing where we add corrections.
        self.indexing = self._compute_f_iter(self.f_max_iter)

        # By default, we use the same phantom grad for all correction losses.
        indexing_pg = make_pair(self.indexing, grad)
        produce_grad = [
                backward_factory(grad_type=pg,
                                 tau=self.args.tau,
                                 sup_gap=self.args.sup_gap,
                                 sup_loc=self.args.sup_loc
                                 ) for pg in indexing_pg
                ]

        # Enabling ift will replace the last gradient function by Implicit Differentiation.
        if self.args.ift or self.args.hook_ift:
            produce_grad[-1] = backward_factory(
                grad_type='ift', hook_ift=self.args.hook_ift, b_solver=self.b_solver,
                b_solver_kwargs=dict(max_iter=self.b_max_iter, tol=self.b_tol, stop_mode=self.b_stop_mode)
                )

        self.produce_grad = produce_grad

    def _compute_f_iter(self, f_max_iter):
        """
        Computes the steps for sampling internal solver states.
        """
        if self.args.n_states > 1:
            n_states = max(min(f_max_iter, self.args.n_states), 1)
            delta = int(f_max_iter // n_states)
            if f_max_iter % n_states == 0:
                return [(k+1)*delta for k in range(n_states)]
            else:
                return [f_max_iter-(n_states-k-1)*delta for k in range(n_states)]
        else:
            return [*self.args.indexing, f_max_iter]

    def _solve_fixed_point(
               self, deq_func, z_init,
               f_max_iter=None,
               indexing=None,
               solver_kwargs=None,
               **kwargs
               ):
        """
        Solves for the fixed point using the DEQ solver.
        """
        solver_kwargs = {k:v for k, v in solver_kwargs.items() if k != 'f_max_iter'}
        indexing = indexing if self.training else None

        with torch.no_grad():
            z_star, trajectory, info = self.f_solver(
                    deq_func, x0=z_init, max_iter=f_max_iter,
                    tol=self.f_tol, stop_mode=self.f_stop_mode, indexing=indexing,
                    **solver_kwargs
                    )

        return z_star, trajectory, info

    def forward(
            self, func, z_init,
            solver_kwargs=None,
            sradius_mode=False,
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ.
        """
        replay_params = self._resolve_replay_params(func)
        func = self._maybe_compile(func)
        deq_func, z_init = deq_decorator(func, z_init, no_stat=self.no_stat)

        if solver_kwargs is None:
            solver_kwargs = dict()

        if self.training:
            if type(solver_kwargs.get('f_max_iter', None)) in [int, float]:
                indexing = self._compute_f_iter(solver_kwargs['f_max_iter'])
            else:
                indexing = self.indexing

            _, trajectory, info = self._solve_fixed_point(deq_func, z_init,
                    f_max_iter=solver_kwargs.get('f_max_iter', self.f_max_iter), indexing=indexing, solver_kwargs=solver_kwargs)

            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_pred = deq_func.detach(z_pred)
                z_out += produce_grad(
                    self,
                    deq_func,
                    z_pred,
                    writer=backward_writer,
                    mem_gc=self.args.mem_gc,
                    func_params=replay_params,
                )

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            z_star, _, info = self._solve_fixed_point(deq_func, z_init,
                    f_max_iter=solver_kwargs.get('f_max_iter', self.eval_f_max_iter), solver_kwargs=solver_kwargs)

            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


class DEQSliced(DEQBase):
    """
    DEQ computational graph that slices the full solver trajectory to apply gradients.

    For `DEQSliced`, it slices the full solver steps into several smaller graphs (w/o grad).
    The gradient function will be applied to the returned state of each subgraph.
    Then a new fixed point solver will resume from the output of the gradient function.
    This is equivalent to inserting the gradient function into the full solver computational graph.
    The maximum number of DEQ function calls is defined by, for example, ``args.f_max_iter + args.n_states * args.grad``.

    Args:
        args: Configuration for the DEQ model (argparse.Namespace, dict, DEQConfig, or None).
        **kwargs: Additional keyword arguments to override configuration values.
    """
    def __init__(self, args=None, **kwargs):
        super(DEQSliced, self).__init__(args, **kwargs)

        # Preprocess arguments.
        grad = self.args.grad
        if isinstance(grad, int):
            if grad < 1:
                raise ValueError('The minimal gradient step is 1!')
            grad = [grad]

        if type(grad) not in (list, tuple):
            raise ValueError(f'grad must be a list or tuple, got {type(grad)}')

        '''Define gradient functions through the backward factory.'''

        # First compute the f_max_iter indexing where we add corrections.
        self.indexing = self._compute_f_iter(self.f_max_iter)

        # By default, we use the same phantom grad for all correction losses.
        indexing_pg = make_pair(self.indexing, grad)
        produce_grad = [
                backward_factory(grad_type=pg,
                                 tau=self.args.tau,
                                 sup_gap=self.args.sup_gap,
                                 sup_loc=self.args.sup_loc
                                 ) for pg in indexing_pg
                ]

        # Enabling ift will replace the last gradient function by Implicit Differentiation.
        if self.args.ift or self.args.hook_ift:
            produce_grad[-1] = backward_factory(
                grad_type='ift', hook_ift=self.args.hook_ift, b_solver=self.b_solver,
                b_solver_kwargs=dict(max_iter=self.b_max_iter, tol=self.b_tol, stop_mode=self.b_stop_mode)
                )

        self.produce_grad = produce_grad

    def _compute_f_iter(self, f_max_iter):
        """
        Computes the steps for sampling internal solver states.
        """
        if self.args.n_states > 1:
            return [int(f_max_iter // self.args.n_states) for _ in range(self.args.n_states)]
        else:
            return np.diff([0, *self.args.indexing, f_max_iter]).tolist()

    def _solve_fixed_point(
            self, deq_func, z_init,
            f_max_iter=None,
            solver_kwargs=None,
            **kwargs
            ):
        """
        Solves for the fixed point using the DEQ solver.
        """
        solver_kwargs = {k:v for k, v in solver_kwargs.items() if k != 'f_max_iter'}

        with torch.no_grad():
            z_star, _, info = self.f_solver(
                    deq_func, x0=z_init, max_iter=f_max_iter,
                    tol=self.f_tol, stop_mode=self.f_stop_mode,
                    **solver_kwargs
                    )

        return z_star, info

    def forward(
            self, func, z_star,
            solver_kwargs=None,
            sradius_mode=False,
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ.
        """
        replay_params = self._resolve_replay_params(func)
        func = self._maybe_compile(func)
        deq_func, z_star = deq_decorator(func, z_star, no_stat=self.no_stat)

        if solver_kwargs is None:
            solver_kwargs = dict()

        if self.training:
            if type(solver_kwargs.get('f_max_iter', None)) in [int, float]:
                indexing = self._compute_f_iter(solver_kwargs['f_max_iter'])
            else:
                indexing = self.indexing

            z_out = []
            for f_max_iter, produce_grad in zip(indexing, self.produce_grad):
                z_star, info = self._solve_fixed_point(deq_func, z_star, f_max_iter=f_max_iter, solver_kwargs=solver_kwargs)
                z_star = deq_func.detach(z_star)
                z_out += produce_grad(
                    self,
                    deq_func,
                    z_star,
                    writer=backward_writer,
                    mem_gc=self.args.mem_gc,
                    func_params=replay_params,
                )
                z_star = z_out[-1]

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            z_star, info = self._solve_fixed_point(deq_func, z_star,
                    f_max_iter=solver_kwargs.get('f_max_iter', self.eval_f_max_iter), solver_kwargs=solver_kwargs)

            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


_core = {
    'indexing': DEQIndexing,
    'sliced': DEQSliced,
    }


def register_deq(deq_type, core):
    """
    Registers a user-defined DEQ class for the get_deq function.

    Args:
        deq_type (str): The type of DEQ model to register.
        core (type): The class defining the DEQ model.

    Example:
        >>> register_deq('custom', CustomDEQ)
    """
    _core[deq_type] = core


def get_deq(args=None, compile_fn: bool = False, **kwargs):
    """
    Factory function to generate an instance of a DEQ model based on configuration.

    Args:
        args: Configuration (argparse.Namespace, dict, DEQConfig, or None). Default None.
        compile_fn: If True, wrap the user's function in ``torch.compile()`` during eval mode
            for faster solver iterations. Default False.
        **kwargs: Additional keyword arguments to override config.

    Returns:
        DEQBase (torch.nn.Module): DEQ module.

    Example:
        >>> deq = get_deq(core='sliced')
        >>> deq = get_deq(f_max_iter=20, compile_fn=True)
    """
    config = DEQConfig.from_args(args, **kwargs)

    if config.core not in _core:
        raise KeyError(f"Unknown DEQ core '{config.core}'. Registered: {list(_core.keys())}")

    deq = _core[config.core](config)
    deq._compile_fn = compile_fn
    return deq


def reset_deq(model):
    """
    Resets the normalization and dropout layers of the given DEQ model.

    Args:
        model (torch.nn.Module): The DEQ model to reset.

    Example:
        >>> deq_layer = DEQLayer(args)
        >>> reset_deq(deq_layer)
    """
    reset_norm(model)
    reset_dropout(model)
