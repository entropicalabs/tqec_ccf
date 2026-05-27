"""IF/ELSE-capable circuit container.

Vanilla ``stim.Circuit`` (1.15) has no notion of an ``IF`` block, so the
conditional-cube compilation path emits its output via this thin wrapper.
``ConditionalCircuit`` mirrors a useful subset of ``stim.Circuit.append``
and adds :meth:`append_if` for explicit ``IF(rec[-k]) { ... } ELSE { ... }``
blocks.  :meth:`to_stim_text` serialises to the Stim text dialect used by
``loom-weave``'s parser; :meth:`to_stim_circuit_strict` round-trips back to
``stim.Circuit`` and raises if any ``IfBlock`` is still present.

A small ``branch_diff`` helper is provided to turn two parallel
``stim.Circuit`` instances representing the same time-slice in two branches
into a single sequence of ``stim.CircuitInstruction`` plus ``IfBlock`` entries
suitable for feeding back into ``ConditionalCircuit.extend``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Union

import stim


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

    def __init__(self, entries: list[CircuitEntry] | None = None) -> None:
        self._entries: list[CircuitEntry] = list(entries) if entries else []

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


def branch_diff(
    branch_zero: stim.Circuit,
    branch_one: stim.Circuit,
    condition_rec: int,
) -> list[CircuitEntry]:
    """Diff two parallel branch circuits into a single entry sequence.

    Walks both circuits instruction-by-instruction.  Identical instructions
    (same name, args, target list) emit once unconditionally.  Differing
    instructions get wrapped in a single ``IfBlock`` whose ``condition_rec``
    selects ``branch_one`` over ``branch_zero``.  Differing runs are
    collected greedily until the next identical instruction is encountered.

    The two circuits MUST have the same number of top-level instructions
    (verified by ``ConditionalBlock`` Equal Measurement Count + identical
    layer count assumptions).  ``REPEAT`` / nested blocks are not supported
    in this stage.
    """
    if len(branch_zero) != len(branch_one):
        raise ValueError(
            f"branch_diff: instruction count mismatch ({len(branch_zero)} "
            f"vs {len(branch_one)}).  Equal Measurement Count violated?"
        )

    out: list[CircuitEntry] = []
    pending_zero: list[stim.CircuitInstruction] = []
    pending_one: list[stim.CircuitInstruction] = []

    def flush() -> None:
        if not pending_zero and not pending_one:
            return
        block = IfBlock(condition_rec=condition_rec)
        block.then_body = list(pending_one)
        block.else_body = list(pending_zero)
        out.append(block)
        pending_zero.clear()
        pending_one.clear()

    for inst_zero, inst_one in zip(branch_zero, branch_one):
        if not isinstance(inst_zero, stim.CircuitInstruction) or not isinstance(
            inst_one, stim.CircuitInstruction
        ):
            raise ValueError(
                "branch_diff does not support REPEAT or nested blocks "
                "in branch circuits at this stage."
            )
        if _instructions_equal(inst_zero, inst_one):
            flush()
            out.append(inst_zero)
        else:
            pending_zero.append(inst_zero)
            pending_one.append(inst_one)
    flush()
    return out


def _instructions_equal(
    a: stim.CircuitInstruction, b: stim.CircuitInstruction
) -> bool:
    if a.name != b.name:
        return False
    if list(a.gate_args_copy()) != list(b.gate_args_copy()):
        return False
    return list(a.targets_copy()) == list(b.targets_copy())
