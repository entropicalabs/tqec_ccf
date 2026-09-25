"""Conditional-cube compilation support.

Provides the data structures used to compile a :class:`BlockGraph` containing
conditional cubes into a single Stim text circuit annotated with ``IF/ELSE``
blocks plus XOR-normalized ``rec[-K]`` references.

See ``loom-weave/PPD_tqec_conditional_circuits.md`` for the design.
"""

from tqec.compile.conditional.circuit import (
    CircuitEntry,
    ConditionalCircuit,
    IfBlock,
    remap_entry_qubit_indices,
)
from tqec.compile.conditional.frame import ConditionId, PauliFrameTracker

# Note: ``resolve_condition_recs`` is intentionally not re-exported here to
# avoid a circular import via ``tqec.compile.blocks.layers.atomic.layout``.
# Import it directly from ``tqec.compile.conditional.condition_recs``.

__all__ = [
    "CircuitEntry",
    "ConditionId",
    "ConditionalCircuit",
    "IfBlock",
    "PauliFrameTracker",
    "remap_entry_qubit_indices",
]
