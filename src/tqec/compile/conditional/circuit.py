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

    Uses ``difflib.SequenceMatcher`` to align the two circuits at the
    instruction level.  Matching blocks emit once unconditionally; unmatched
    spans (which may have different lengths in the two branches) collapse
    into a single ``IfBlock`` whose ``then_body`` carries the ``branch_one``
    span and ``else_body`` the ``branch_zero`` span.  This handles cases
    where ``Equal Measurement Count`` holds but branch-only annotations
    (e.g. boundary detectors) differ in count.

    ``REPEAT`` / nested blocks are not supported in this stage.
    """
    import difflib

    zero_instrs: list[stim.CircuitInstruction] = []
    for inst in branch_zero:
        if not isinstance(inst, stim.CircuitInstruction):
            raise ValueError(
                "branch_diff does not support REPEAT or nested blocks "
                "in branch circuits at this stage."
            )
        zero_instrs.append(inst)
    one_instrs: list[stim.CircuitInstruction] = []
    for inst in branch_one:
        if not isinstance(inst, stim.CircuitInstruction):
            raise ValueError(
                "branch_diff does not support REPEAT or nested blocks "
                "in branch circuits at this stage."
            )
        one_instrs.append(inst)

    zero_keys = [_instruction_key(i) for i in zero_instrs]
    one_keys = [_instruction_key(i) for i in one_instrs]
    matcher = difflib.SequenceMatcher(a=zero_keys, b=one_keys, autojunk=False)

    out: list[CircuitEntry] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(zero_instrs[i1:i2])
        else:
            block = IfBlock(condition_rec=condition_rec)
            block.then_body = list(one_instrs[j1:j2])
            block.else_body = list(zero_instrs[i1:i2])
            out.append(block)
    return out


def _instruction_key(inst: stim.CircuitInstruction) -> tuple:
    return (
        inst.name,
        tuple(inst.gate_args_copy()),
        tuple((t.value, t.is_qubit_target, t.is_measurement_record_target)
              for t in inst.targets_copy()),
    )


def _instructions_equal(
    a: stim.CircuitInstruction, b: stim.CircuitInstruction
) -> bool:
    if a.name != b.name:
        return False
    if list(a.gate_args_copy()) != list(b.gate_args_copy()):
        return False
    return list(a.targets_copy()) == list(b.targets_copy())
