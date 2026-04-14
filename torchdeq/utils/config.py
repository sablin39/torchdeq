from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, fields
from typing import Literal


@dataclass
class DEQConfig:
    """
    Single source of truth for all DEQ configuration.

    Can be constructed directly, or from an argparse.Namespace / dict / legacy DEQConfig
    via the ``from_args`` classmethod.
    """

    # Solver
    f_solver: str = field(default="fixed_point_iter", metadata={"help": "Forward solver to use."})
    b_solver: str = field(default="fixed_point_iter", metadata={"help": "Backward solver to use."})
    f_max_iter: int = field(default=40, metadata={"help": "Max NFE for forward solver."})
    b_max_iter: int = field(default=40, metadata={"help": "Max NFE for backward solver."})
    f_tol: float = field(default=1e-3, metadata={"help": "Forward pass solver stopping criterion."})
    b_tol: float = field(default=1e-6, metadata={"help": "Backward pass solver stopping criterion."})
    f_stop_mode: Literal["abs", "rel"] = field(default="abs", metadata={"help": "Forward pass convergence stop mode."})
    b_stop_mode: Literal["abs", "rel"] = field(default="abs", metadata={"help": "Backward pass convergence stop mode."})
    eval_factor: float = field(default=1.0, metadata={"help": "Factor to scale f_max_iter at test time."})
    eval_f_max_iter: int = field(default=0, metadata={"help": "Directly set f_max_iter at test time."})

    # Norm
    norm_type: Literal["none", "weight_norm", "spectral_norm"] = field(
        default="weight_norm", metadata={"help": "Normalization type for DEQ."}
    )
    norm_no_scale: bool = field(default=False, metadata={"help": "Remove learnable target_norm."})
    norm_clip: bool = field(default=False, metadata={"help": "Clip the scale value."})
    norm_clip_value: float = field(default=1.0, metadata={"help": "Upper bound for renormalization factor."})
    norm_target_norm: float = field(default=1.0, metadata={"help": "Pre-defined target norm."})
    sn_n_power_iters: int = field(default=1, metadata={"help": "Power iterations for spectral radius estimation."})

    # Training / DEQ core
    core: Literal["indexing", "sliced"] = field(default="sliced", metadata={"help": "Training container for DEQ."})
    ift: bool = field(default=False, metadata={"help": "Use implicit differentiation."})
    hook_ift: bool = field(default=False, metadata={"help": "Use hook-based O(1) memory IFT."})
    n_states: int = field(default=1, metadata={"help": "Number of loss terms."})
    indexing: list[int] = field(default_factory=list, metadata={"help": "Index solver states for FP correction."})
    gamma: float = field(default=0.8, metadata={"help": "Strength of fixed point correction."})
    grad: list[int] = field(default_factory=lambda: [1], metadata={"help": "Steps of Phantom Grad."})
    tau: float = field(default=1.0, metadata={"help": "Damping factor for Phantom Grad."})
    sup_gap: int = field(default=-1, metadata={"help": "Sampling gap along PhantomGrad trajectories."})
    sup_loc: list[int] = field(default_factory=list, metadata={"help": "Sampling locations for PhantomGrad."})
    mem_gc: bool = field(default=False, metadata={"help": "Use activation recompute for DEQ replay paths."})

    # Regularization
    jac_loss_weight: float = field(default=0.0, metadata={"help": "Jacobian regularization loss weight."})
    jac_loss_freq: float = field(default=0.0, metadata={"help": "Frequency of jacobian regularization."})
    jac_incremental: int = field(default=0, metadata={"help": "Increment jac_loss_weight by 0.1 after N steps."})

    # Monitoring
    sradius_mode: bool = field(default=False, metadata={"help": "Monitor spectral radius during validation."})

    def __post_init__(self) -> None:
        if isinstance(self.grad, int):
            self.grad = [self.grad]
        if self.f_max_iter < 1:
            raise ValueError(f"f_max_iter must be >= 1, got {self.f_max_iter}")
        if self.b_max_iter < 1:
            raise ValueError(f"b_max_iter must be >= 1, got {self.b_max_iter}")
        if self.f_tol <= 0:
            raise ValueError(f"f_tol must be > 0, got {self.f_tol}")
        if self.b_tol <= 0:
            raise ValueError(f"b_tol must be > 0, got {self.b_tol}")

    @classmethod
    def from_args(cls, args: argparse.Namespace | dict | DEQConfig | None = None, **overrides) -> DEQConfig:
        """
        Build a DEQConfig from various input types, with keyword overrides.

        Accepts ``argparse.Namespace``, ``dict``, an existing ``DEQConfig``, or ``None``.
        Unknown keys are silently filtered out so that passing a full argparse namespace
        (which may contain non-DEQ keys) works without error.
        """
        if args is None:
            raw = {}
        elif isinstance(args, DEQConfig):
            from dataclasses import asdict
            raw = asdict(args)
        elif isinstance(args, dict):
            raw = dict(args)
        elif isinstance(args, argparse.Namespace):
            raw = vars(args)
        else:
            warnings.warn(f"Unrecognized config type: {type(args)}. Attempting vars().")
            raw = vars(args)

        raw.update(overrides)

        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}

        return cls(**filtered)

    # Backward compatibility shims
    def get(self, key: str, default=None):
        """Deprecated dict-style access. Use attribute access instead."""
        warnings.warn(
            "DEQConfig.get() is deprecated. Use attribute access (e.g., config.f_solver) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key, default)

    def update(self, **kwargs) -> None:
        """Deprecated dict-style update. Pass kwargs to from_args() or the constructor."""
        warnings.warn(
            "DEQConfig.update() is deprecated. Pass kwargs to DEQConfig() or DEQConfig.from_args() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        valid_keys = {f.name for f in fields(self)}
        for k, v in kwargs.items():
            if k in valid_keys:
                setattr(self, k, v)
