"""Compile-time Pauli frame tracker for conditional-cube compilation.

The tracker holds, per qubit per timestep, the set of branch conditions whose
outcome affects the qubit's effective Pauli at that point in the schedule.
At detector / observable emission, the active frame is consulted and the
relevant ``rec[-K]`` indices are XOR-appended so the emitted record list is
branch-independent.

Stage 1 stub: identity Clifford propagation only.  A frame registered on
qubit ``q`` at timestep ``t`` stays attached to ``q`` until a later
``register_measurement`` call overrides it.  Real Clifford propagation
(``CX``, ``CZ``, ``H``, ``S``, ...) is wired in a later stage when the
emission code starts driving the tracker through actual gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NewType

from tqec.circuit.qubit import GridQubit


ConditionId = NewType("ConditionId", int)
"""Stable identifier for a branch condition.

Each conditional cube in the block graph contributes one ``ConditionId``.  The
ID is the index assigned to that cube's condition by the compilation pass; it
serves as the dictionary key everywhere the tracker needs to look up ``rec[-K]``
offsets for a branch outcome.
"""


class PauliFrameTracker:
    """Per-qubit branch-condition frame across the compilation schedule.

    The tracker is a thin bookkeeping layer.  It does **not** know anything
    about Stim instructions; the emission code drives it via the
    ``register_measurement`` / ``propagate_through`` API and consumes its
    output via ``frame_at`` / ``xor_records_for``.
    """

    def __init__(self) -> None:
        self._frames: dict[tuple[GridQubit, int], frozenset[ConditionId]] = {}

    def register_measurement(
        self,
        qubit: GridQubit,
        timestep: int,
        conditions: frozenset[ConditionId],
    ) -> None:
        """Mark ``qubit`` at ``timestep`` as carrying the given branch conditions.

        Subsequent ``frame_at(qubit, t >= timestep)`` queries return at least
        ``conditions``.  Use empty ``conditions`` to clear a previously
        registered frame at this exact ``(qubit, timestep)``.
        """
        if conditions:
            self._frames[(qubit, timestep)] = conditions
        else:
            self._frames.pop((qubit, timestep), None)

    def propagate_through(
        self,
        gate_name: str,
        qubits: list[GridQubit],
        timestep: int,
    ) -> None:
        """Propagate frames through a Clifford gate.

        Stage 1 stub: no-op.  When the emission code starts feeding the tracker
        every gate it emits, this method will gain a Clifford propagation table
        (``CX``: X1->X1X2 / Z2->Z1Z2 / etc.) so frames follow Pauli flow.
        """
        del gate_name, qubits, timestep

    def frame_at(self, qubit: GridQubit, timestep: int) -> frozenset[ConditionId]:
        """Return the union of branch conditions affecting ``qubit`` at ``timestep``."""
        active: frozenset[ConditionId] = frozenset()
        for (q, t), conds in self._frames.items():
            if q == qubit and t <= timestep:
                active = active | conds
        return active

    def xor_records_for(
        self,
        qubit: GridQubit,
        timestep: int,
        condition_to_rec: Mapping[ConditionId, int],
    ) -> list[int]:
        """Return ``rec[-K]`` offsets to XOR into a detector referencing ``qubit`` at ``timestep``.

        Args:
            qubit: physical qubit the detector references.
            timestep: schedule timestep of the reference.
            condition_to_rec: mapping from each known ``ConditionId`` to the
                ``rec[-K]`` offset (negative integer) of the measurement that
                produced that condition's outcome.

        Returns:
            ordered list of ``rec[-K]`` offsets; deterministic across runs
            because conditions are sorted before lookup.
        """
        conds = sorted(self.frame_at(qubit, timestep))
        return [condition_to_rec[c] for c in conds if c in condition_to_rec]

    def __repr__(self) -> str:
        return f"PauliFrameTracker({len(self._frames)} entries)"
