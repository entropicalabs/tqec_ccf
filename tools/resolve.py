"""Re-export of :mod:`tqec.compile.conditional.resolve`, kept for existing scripts."""

from tqec.compile.conditional.resolve import (
    resolve_if_else,
    resolve_if_else_by_measurement,
    resolve_if_else_by_order,
)

__all__ = ["resolve_if_else", "resolve_if_else_by_measurement", "resolve_if_else_by_order"]
