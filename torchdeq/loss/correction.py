from __future__ import annotations

import torch
from typing import Callable


__all__ = ['fp_correction', 'register_weight_func']


def _linear(n: int, k: int, gamma: float = 0.9, bias: float = 0.0, **kwargs) -> float:
    return 1 - (n-k-1) / n * gamma + bias


def _exp(n: int, k: int, gamma: float = 0.8, **kwargs) -> float:
    return gamma ** (n-k-1)


def _const(n: int, k: int, c: float = 1.0, **kwargs) -> float:
    return c


_weight_func_dict: dict[str, Callable] = {
        'exp': _exp,
        'linear': _linear,
        'const': _const
        }


def _get_weight_func(name: str) -> Callable:
    if name not in _weight_func_dict:
        raise KeyError(f"Unknown weight function '{name}'. Available: {list(_weight_func_dict.keys())}")
    return _weight_func_dict[name]


def register_weight_func(name: str, func: Callable) -> None:
    """
    Registers a new weight function for fixed point correction.

    Args:
        name: Identifier to associate with the new weight function.
        func: The weight function to register, mapping (n, k) to a float value.
    """
    if not callable(func):
        raise ValueError(f"func must be callable, got {type(func)}")
    _weight_func_dict[name] = func


def _align_list(args: tuple | list) -> tuple[list, int]:
    max_len = 0
    out_args: list = []
    for each in args:
        if type(each) not in (list, tuple):
            each = [each]
        out_args.append(each)
        max_len = max(len(each), max_len)
    return out_args, max_len


def _get_idx(args: list, idx: int) -> list:
    return [each[idx%len(each)] for each in args]


def fp_correction(
        crit: Callable,
        args: tuple | list,
        weight_func: str = 'exp',
        return_loss_values: bool = False,
        **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
    """
    Computes fixed-point correction for stabilizing Deep Equilibrium (DEQ) models.

    Args:
        crit: Loss function.
        args: List of arguments to pass to the criterion.
        weight_func: Name of the weight function to use. Default 'exp'.
        return_loss_values: Whether to return the loss values. Default False.
        **kwargs: Additional keyword arguments for the weight function.

    Returns:
        The computed loss, and optionally a list of individual loss values.
    """
    args, max_len = _align_list(args)
    weight_func_fn = _get_weight_func(weight_func)

    loss = 0.0
    loss_list: list[float] = []
    for i in range(max_len):
        i_weight = weight_func_fn(max_len, i, **kwargs)
        i_loss = crit(*_get_idx(args, i))
        loss += i_weight * i_loss

        if return_loss_values:
            loss_list.append(i_loss.item())

    if return_loss_values:
        return loss, loss_list
    else:
        return loss
