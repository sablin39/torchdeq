from __future__ import annotations

from dataclasses import fields
from typing import get_args, get_origin

from .config import DEQConfig


def add_deq_args(parser):
    """
    Decorate an argument parser with all DEQ-related arguments.

    Arguments are derived automatically from :class:`DEQConfig` fields,
    so the parser always stays in sync with the configuration dataclass.

    Args:
        parser (argparse.ArgumentParser): The parser to decorate.
    """
    for f in fields(DEQConfig):
        name = f"--{f.name}"
        help_text = f.metadata.get("help", "")
        default = f.default if f.default is not f.default_factory else None  # type: ignore[attr-defined]

        # Handle fields with default_factory
        if default is None:
            try:
                default = f.default_factory()  # type: ignore[misc]
            except TypeError:
                default = None

        origin = get_origin(f.type)
        type_args = get_args(f.type)

        if f.type is bool or f.type == "bool":
            # Boolean fields are store_true flags
            if not default:
                parser.add_argument(name, action="store_true", default=False, help=help_text)
            else:
                parser.add_argument(name, action="store_true", default=True, help=help_text)
        elif origin is list or (isinstance(f.type, str) and f.type.startswith("list")):
            # List fields use nargs='+'
            inner_type = type_args[0] if type_args else int
            if isinstance(inner_type, str):
                inner_type = int
            parser.add_argument(name, type=inner_type, nargs="+", default=default, help=help_text)
        elif f.type is int or f.type == "int":
            parser.add_argument(name, type=int, default=default, help=help_text)
        elif f.type is float or f.type == "float":
            parser.add_argument(name, type=float, default=default, help=help_text)
        elif f.type is str or f.type == "str":
            parser.add_argument(name, type=str, default=default, help=help_text)
        else:
            # Literal or other types — treat as str
            parser.add_argument(name, type=str, default=default, help=help_text)
