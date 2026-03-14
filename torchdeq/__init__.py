from __future__ import annotations

__version__ = "0.2.0"

from .utils.arg_utils import add_deq_args
from .utils.config import DEQConfig

from .core import register_deq, get_deq, reset_deq

from .solver import *
from .norm import *
from .dropout import *

from .loss import *
from .utils import *
