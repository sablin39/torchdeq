from __future__ import annotations

"""
Credit to https://github.com/locuslab/deq/blob/master/lib/jacobian.py
"""
import math
import torch


__all__ = ['jac_reg', 'power_method']

@torch.compiler.disable
def jac_reg(f0: torch.Tensor, z0: torch.Tensor, vecs: int = 1, create_graph: bool = True) -> torch.Tensor:
    """
    Estimates tr(J^TJ)=tr(JJ^T) via Hutchinson estimator.

    Args:
        f0: Output of the function f (whose J is to be analyzed).
        z0: Input to the function f.
        vecs: Number of random Gaussian vectors to use. Default 1.
        create_graph: Whether to create backward graph. Default True.

    Returns:
        A 1x1 tensor encoding the (shape-normalized) jacobian loss.
    """
    result = 0
    for i in range(vecs):
        v = torch.randn(*z0.shape).to(z0)
        vJ = torch.autograd.grad(f0, z0, v, retain_graph=True, create_graph=create_graph)[0]
        result += vJ.norm()**2
    return result / vecs / math.prod(z0.shape)


def power_method(f0: torch.Tensor, z0: torch.Tensor, n_iters: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the spectral radius of J using power method.

    Args:
        f0: Output of the function f (whose J is to be analyzed).
        z0: Input to the function f.
        n_iters: Number of power method iterations. Default 100.

    Returns:
        Tuple of (largest eigenvector, largest abs eigenvalue).
    """
    evector = torch.randn_like(z0)
    bsz = evector.shape[0]
    for i in range(n_iters):
        vTJ = torch.autograd.grad(f0, z0, evector, retain_graph=(i < n_iters-1), create_graph=False)[0]
        evalue = (vTJ * evector).reshape(bsz, -1).sum(1) / (evector * evector).reshape(bsz, -1).sum(1)
        evector = (vTJ.reshape(bsz, -1) / vTJ.reshape(bsz, -1).norm(dim=1, keepdim=True)).reshape_as(z0)
    return (evector, torch.abs(evalue))
