from __future__ import annotations

"""
A module containing several implementations of variational dropout.

Variational dropout is a type of dropout where a single dropout mask is generated once per sample
and applied consistently across all solver steps in the sample. This is particularly effective when used with
implicit models, as it counters overfitting while preserving the dynamics.

This module provides variational dropout for 1d, 2d, and 3d inputs, with both channel-wise and token-wise options.
"""

import torch
import torch.nn as nn


__all__ = ['VariationalDropout', 
           'VariationalDropout1d', 'VariationalDropout2d', 'VariationalDropout3d', 
           'VariationalDropToken1d', 'VariationalDropToken2d', 'VariationalDropToken3d',
           'init_dropout', 'reset_dropout']


class _VariationalDropoutNd(nn.Module):
    """
    Abstract base class for Variational Dropout layers.

    The concrete subclasses should implement the `reset_mask` method to define the mask behavior 
    specific to the dimensionality of the input tensor.

    Args:
        dropout (float): Dropout probability between 0 and 1. Default: 0.5.

    Shape:
        - Input: Tensor of any shape.
        - Output: Tensor of the same shape as input.

    Note:
        Raises ValueError if dropout probability is not in [0,1].
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(dropout))

        self.dropout = dropout
        self.mask = None

    def _make_mask(self, x, *shape):
        if self.dropout == 1.0:
            return x.new_zeros(shape).requires_grad_(False)

        mask = x.new_empty(shape)
        mask.bernoulli_(1 - self.dropout)
        mask.div_(1 - self.dropout)
        return mask.requires_grad_(False)

    def reset_mask(self, x):
        """
        Resets the dropout mask. 
        Subclasses should implement this method according to the dimensionality of the input tensor.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the variational dropout layer.

        If the layer is in training mode and the dropout probability is greater than 0, 
        applies the same dropout mask to all inputs in the batch. Otherwise, returns the input unchanged.
        
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying variational dropout.
        """
        if not self.training or self.dropout == 0.:
            return x

        if self.mask is None:
            self.reset_mask(x)
        mask = self.mask.expand_as(x).to(x.device)  # Make sure the dimension matches
        return mask * x


class VariationalDropout(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zeros some of the elements of the input tensor 
    with probability 'dropout' using a mask tensor sampled from a Bernoulli distribution.

    The same mask is used for each input in a training iteration. (for fixed point convergence)
    This random mask is reset at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5.

    Shape:
        - Input: Tensor of any shape.
        - Output: Tensor of the same shape as input.

    Examples:
        >>> m = VariationalDropout(dropout=0.5)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)
    """
    def __init__(self, dropout=0.5):
        super().__init__(dropout)

    def reset_mask(self, x):
        self.mask = self._make_mask(x, *x.shape)


class VariationalDropout1d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire channel/feature dimension of the input 1d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The channel/feature dimension of 1d tensor is the :math:`*` slice of :math:`(B, L, *)` 
    for token_first=True, or :math:`(B, *, L)` for token_first=False.

    The same mask is used for each input in a training iteration. (for fixed point convergence)
    This random mask is reset at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expects input tensor in shape :math:`(B, L, D)`,
                                       otherwise expects :math:`(B, D, L)`. Here, `B` is batch size, 
                                       `L` is sequence length, and `D` is feature dimension.
                                       Default: False.
                                       
    Shape:
        - Input: :math:`(B, L, D)` or :math:`(B, D, L)`.
        - Output: :math:`(B, L, D)` or :math:`(B, D, L)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, L, D)
            B, _, D = x.shape
            self.mask = self._make_mask(x, B, 1, D)
        else:
            # Dimension (B, D, L)
            B, D, _ = x.shape
            self.mask = self._make_mask(x, B, D, 1)


class VariationalDropout2d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire channel/feature dimension of the input 2d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The channel/feature dimension of 2d tensor is the :math:`*` of :math:`(B, H, W, *)` 
    for token_first=True, or :math:`(B, *, H, W)` for token_first=False.

    During the fixed point solving, a fixed mask will be applied until convergence.
    Reset this random mask at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expect input tensor in shape :math:`(B, H, W, D)`,
                                        otherwise expect :math:`(B, D, H, W)`. Here, `B` is batch size, 
                                        and `D` is feature dimension. Default: False                                     
    
    Shape:
        - Input: :math:`(B, H, W, D)` or :math:`(B, D, H, W)`.
        - Output: :math:`(B, H, W, D)` or :math:`(B, D, H, W)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, H, W, D)
            B, _, _, D = x.shape
            self.mask = self._make_mask(x, B, 1, 1, D)
        else:
            # Dimension (B, D, H, W)
            B, D, _, _ = x.shape
            self.mask = self._make_mask(x, B, D, 1, 1)


class VariationalDropout3d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire channel/feature dimension of the input 3d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The channel/feature dimension of 3d tensor is the :math:`*` slice of :math:`(B, T, H, W, *)` 
    for token_first=True, or :math:`(B, *, T, H, W)` for token_first=False.

    During the fixed point solving, a fixed mask will be applied until convergence.
    Reset this random mask at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expect input tensor in shape :math:`(B, T, H, W, D)`,
                                        otherwise expect :math:`(B, D, T, H, W)`. Here, `B` is batch size, 
                                        and `D` is feature dimension. Default: False
    
    Shape:
        - Input: :math:`(B, T, H, W, D)` or :math:`(B, D, T, H, W)`.
        - Output: :math:`(B, T, H, W, D)` or :math:`(B, D, T, H, W)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, T, H, W, D)
            B, _, _, _, D = x.shape
            self.mask = self._make_mask(x, B, 1, 1, 1, D)
        else:
            # Dimension (B, D, T, H, W)
            B, D, _, _, _ = x.shape
            self.mask = self._make_mask(x, B, D, 1, 1, 1)


class VariationalDropToken1d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire token/sequence dimension of the input 1d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The token/sequence dimension of 1d tensor is the :math:`*` slice of :math:`(B, *, L)` 
    for token_first=True, or :math:`(B, D, *)` for token_first=False.

    During the fixed point solving, a fixed mask will be applied until convergence.
    Reset this random mask at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expect input tensor in shape :math:`(B, L, D)`,
                                        otherwise expect :math:`(B, D, L)`. Here, `B` is batch size, 
                                        and `D` is feature dimension. Default: False
                                        
    Shape:
        - Input: :math:`(B, L, D)` or :math:`(B, D, L)`.
        - Output: :math:`(B, L, D)` or :math:`(B, D, L)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, L, D)
            B, L, _ = x.shape
            self.mask = self._make_mask(x, B, L, 1)
        else:
            # Dimension (B, D, L)
            B, _, L = x.shape
            self.mask = self._make_mask(x, B, 1, L)


class VariationalDropToken2d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.

    During training, randomly zero out the entire token/sequence dimension of the input 2d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The token/sequence dimension of 2d tensor is the :math:`*` slice of :math:`(B, H, W, *)` 
    for token_first=True, or :math:`(B, *, H, W)` for token_first=False.

    During the fixed point solving, a fixed mask will be applied until convergence.
    Reset this random mask at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expect input tensor in shape :math:`(B, H, W, D)`,
                                        otherwise expect :math:`(B, D, H, W)`. Here, `B` is batch size, 
                                        and `D` is feature dimension. Default: False
                                        
    Shape:
        - Input: :math:`(B, H, W, D)` or :math:`(B, D, H, W)`.
        - Output: :math:`(B, H, W, D)` or :math:`(B, D, H, W)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, H, W, D)
            B, H, W, _ = x.shape
            self.mask = self._make_mask(x, B, H, W, 1)
        else:
            # Dimension (B, D, H, W)
            B, _, H, W = x.shape
            self.mask = self._make_mask(x, B, 1, H, W)


class VariationalDropToken3d(_VariationalDropoutNd):
    """
    Applies Variational Dropout to the input tensor.
    
    During training, randomly zero out the entire token/sequence dimension of the input 3d tensor 
    with probability 'dropout' using a mask tensor sample from a Bernoulli distribution.

    The token/sequence dimension of 3d tensor is the :math:`*` slice of :math:`(B, T, H, W, *)` 
    for token_first=True, or :math:`(B, *, T, H, W)` for token_first=False.

    During the fixed point solving, a fixed mask will be applied until convergence.
    Reset this random mask at the beginning of the next training iteration using `reset_dropout`.

    Args:
        dropout (float, optional): The probability of an element to be zeroed. Default: 0.5
        token_first (bool, optional): If True, expect input tensor in shape :math:`(B, T, H, W, D)`,
                                        otherwise expect :math:`(B, D, T, H, W)`.  Here, `B` is batch size, 
                                        and `D` is feature dimension. Default: False
    
    Shape:
        - Input: :math:`(B, T, H, W, D)` or :math:`(B, D, T, H, W)`.
        - Output: :math:`(B, T, H, W, D)` or :math:`(B, D, T, H, W)` (same shape as input).
    """
    def __init__(self, dropout=0.5, token_first=True):
        super().__init__(dropout)
        self.token_first = token_first

    def reset_mask(self, x):
        if self.token_first:
            # Dimension (B, T, H, W, D)
            B, T, H, W, _ = x.shape
            self.mask = self._make_mask(x, B, T, H, W, 1)
        else:
            # Dimension (B, D, T, H, W)
            B, _, T, H, W = x.shape
            self.mask = self._make_mask(x, B, 1, T, H, W)


def reset_dropout(model: torch.nn.Module) -> None:
    """
    Resets the dropout mask for all variational dropout layers in the model
    at the beginning of a training iteration.

    Args:
        model: A DEQ layer in which the dropout masks should be reset.
    """
    for module in model.modules():
        if isinstance(module, _VariationalDropoutNd):
            module.mask = None


def init_dropout(model: torch.nn.Module, x: torch.Tensor) -> None:
    """
    Initializes variational dropout masks for all variational dropout layers in
    ``model`` using ``x`` as the representative input shape.

    This is useful before running a compile-stable solver loop because
    ``torch.while_loop`` cannot capture the lazy ``self.mask = ...`` mutation
    that normally happens on the first dropout forward call.
    """
    for module in model.modules():
        if isinstance(module, _VariationalDropoutNd):
            module.reset_mask(x)
            module.mask = module.mask.to(device=x.device)
