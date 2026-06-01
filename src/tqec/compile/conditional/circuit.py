"""IF/ELSE-capable circuit container.

Vanilla ``stim.Circuit`` (1.15) has no notion of an ``IF`` block, so the
conditional-cube compilation path emits its output via this thin wrapper.
``ConditionalCircuit`` mirrors a useful subset of ``stim.Circuit.append``
and adds :meth:`append_if` for explicit ``IF(rec[-k]) { ... } ELSE { ... }``
blocks.  :meth:`to_stim_text` serialises to the Stim text dialect used by
``loom-weave``'s parser; :meth:`to_stim_circuit_strict` round-trips back to
``stim.Circuit`` and raises if any ``IfBlock`` is still present.

The :func:`remap_entry_qubit_indices` helper rewrites qubit-target indices on
a :class:`CircuitEntry` (recursing into :class:`IfBlock` bodies) — used by the
tree-level assembly that joins per-leaf conditional circuits against a shared
global qubit map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Union

import stim

if TYPE_CHECKING:
    from tqec.circuit.qubit_map import QubitMap


@dataclass
class IfBlock:
    """An ``IF(rec[-k]) { ... } ELSE { ... }`` block.

    ``condition_rec`` is the negative ``rec`` offset of the branch-selecting
    measurement at the point the block is emitted.  ``else_body`` is optional;
    when absent, only the ``IF`` arm is rendered.
    """

    condition_rec: int
    then_body: list[CircuitEntry] = field(default_factory=list)
    else_body: list[CircuitEntry] | None = None

    def append(self, entry: CircuitEntry) -> None:
        self.then_body.append(entry)

    def append_else(self, entry: CircuitEntry) -> None:
        if self.else_body is None:
            self.else_body = []
        self.else_body.append(entry)


CircuitEntry = Union[stim.CircuitInstruction, "IfBlock"]


class ConditionalCircuit:
    """Mutable container holding ``stim.CircuitInstruction`` + ``IfBlock`` entries."""

    def __init__(
        self,
        entries: list[CircuitEntry] | None = None,
        qubit_map: "QubitMap | None" = None,
    ) -> None:
        self._entries: list[CircuitEntry] = list(entries) if entries else []
        self._qubit_map = qubit_map

    @property
    def qubit_map(self) -> "QubitMap | None":
        """Local qubit map of the circuit, when known. Set by
        :meth:`LayoutLayer.to_conditional_circuit`; ``None`` for hand-built
        instances. Tree-level assembly uses this to remap local qubit indices to
        the global qubit map.
        """
        return self._qubit_map

    def set_qubit_map(self, qubit_map: "QubitMap") -> None:
        self._qubit_map = qubit_map

    @property
    def entries(self) -> list[CircuitEntry]:
        return self._entries

    def __iter__(self) -> Iterator[CircuitEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def append(
        self,
        name: str,
        targets: list[stim.GateTarget] | list[int] | None = None,
        args: list[float] | None = None,
    ) -> None:
        """Append a plain Stim instruction.  Mirrors ``stim.Circuit.append``."""
        ts = targets if targets is not None else []
        ag = args if args is not None else []
        self._entries.append(stim.CircuitInstruction(name, ts, ag))

    def append_instruction(self, inst: stim.CircuitInstruction) -> None:
        self._entries.append(inst)

    def append_if(self, if_block: IfBlock) -> None:
        self._entries.append(if_block)

    def append_instruction_or_if(self, entry: CircuitEntry) -> None:
        """Append either a plain :class:`stim.CircuitInstruction` or an
        :class:`IfBlock`, dispatching on type. Convenience for callers iterating
        a mixed sequence."""
        self._entries.append(entry)

    def extend(self, entries: list[CircuitEntry]) -> None:
        self._entries.extend(entries)

    def to_stim_text(self) -> str:
        """Serialise to Stim text with ``IF(rec[-k]) { ... } ELSE { ... }`` blocks."""
        lines: list[str] = []
        _render(self._entries, lines, indent=0)
        return "\n".join(lines)

    def to_stim_circuit_strict(self) -> stim.Circuit:
        """Round-trip to vanilla ``stim.Circuit``; raise if any ``IfBlock`` remains.

        Useful for unit tests on circuits that happen to have no surviving
        conditional branches (everything was XOR-normalised away).
        """
        circuit = stim.Circuit()
        for entry in self._entries:
            if isinstance(entry, IfBlock):
                raise ValueError(
                    "Cannot convert ConditionalCircuit to stim.Circuit: "
                    "an IfBlock is still present."
                )
            circuit.append(entry)
        return circuit

    def __repr__(self) -> str:
        return f"ConditionalCircuit({len(self._entries)} entries)"


def remap_entry_qubit_indices(
    entry: CircuitEntry, qubit_index_remap: dict[int, int]
) -> CircuitEntry:
    """Return a copy of ``entry`` with every qubit-target index remapped.

    Recurses into :class:`IfBlock` bodies. ``stim.CircuitInstruction`` with no
    qubit targets is returned as-is. Non-qubit targets (measurement records,
    args) are preserved unchanged.
    """
    if isinstance(entry, IfBlock):
        return IfBlock(
            condition_rec=entry.condition_rec,
            then_body=[remap_entry_qubit_indices(e, qubit_index_remap) for e in entry.then_body],
            else_body=(
                [remap_entry_qubit_indices(e, qubit_index_remap) for e in entry.else_body]
                if entry.else_body is not None
                else None
            ),
        )
    new_targets: list[stim.GateTarget] = []
    for t in entry.targets_copy():
        if t.is_qubit_target and t.qubit_value is not None:
            new_targets.append(stim.GateTarget(qubit_index_remap[t.qubit_value]))
        else:
            new_targets.append(t)
    return stim.CircuitInstruction(entry.name, new_targets, list(entry.gate_args_copy()))


def _render(entries: list[CircuitEntry], lines: list[str], indent: int) -> None:
    pad = "  " * indent
    for entry in entries:
        if isinstance(entry, IfBlock):
            lines.append(f"{pad}IF(rec[{entry.condition_rec}]) {{")
            _render(entry.then_body, lines, indent + 1)
            if entry.else_body is not None:
                lines.append(f"{pad}}} ELSE {{")
                _render(entry.else_body, lines, indent + 1)
            lines.append(f"{pad}}}")
        else:
            text = str(entry).strip()
            for line in text.splitlines():
                lines.append(f"{pad}{line}")

