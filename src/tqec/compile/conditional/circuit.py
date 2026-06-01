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
            continue
        zero_span = zero_instrs[i1:i2]
        one_span = one_instrs[j1:j2]
        split = _split_unequal_span(zero_span, one_span, condition_rec)
        if split is not None:
            out.extend(split)
            continue
        block = IfBlock(condition_rec=condition_rec)
        block.then_body = list(one_span)
        block.else_body = list(zero_span)
        out.append(block)
    return out


_PER_QUBIT_GATES: frozenset[str] = frozenset(
    {
        "I",
        "X",
        "Y",
        "Z",
        "H",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "S_DAG",
        "C_XYZ",
        "C_ZYX",
        "R",
        "RX",
        "RY",
        "RZ",
    }
)


def _check_per_qubit_span(
    span: list[stim.CircuitInstruction],
) -> bool:
    """Return True iff every instruction in ``span`` is a per-qubit gate
    with no parameter args and only qubit targets."""
    for inst in span:
        if inst.name not in _PER_QUBIT_GATES:
            return False
        if list(inst.gate_args_copy()):
            return False
        if not all(t.is_qubit_target for t in inst.targets_copy()):
            return False
    return True


def _qubits_of(inst: stim.CircuitInstruction) -> list[int]:
    return [t.qubit_value for t in inst.targets_copy()]


def _aggregate_by_name(
    span: list[stim.CircuitInstruction],
) -> dict[str, list[int]]:
    """Union qubit targets per gate name across ``span``, first-seen order."""
    by_name: dict[str, list[int]] = {}
    seen: dict[str, set[int]] = {}
    for inst in span:
        bucket = by_name.setdefault(inst.name, [])
        seen_set = seen.setdefault(inst.name, set())
        for q in _qubits_of(inst):
            if q not in seen_set:
                bucket.append(q)
                seen_set.add(q)
    return by_name


def _name_of_qubit_in(
    span: list[stim.CircuitInstruction], qubit: int
) -> str | None:
    """Return the gate name applied to ``qubit`` in ``span``, if any."""
    for inst in span:
        if qubit in _qubits_of(inst):
            return inst.name
    return None


def _split_unequal_span(
    zero_span: list[stim.CircuitInstruction],
    one_span: list[stim.CircuitInstruction],
    condition_rec: int,
) -> list[CircuitEntry] | None:
    """Per-target diff that preserves the structured side's instruction shape.

    Walk the side with more instructions (the "structured" side); for each
    instruction, emit common targets unconditionally and wrap only the
    divergent targets in an ``IfBlock``.  When resolved per branch,
    adjacent same-name pieces are auto-merged by ``stim.Circuit`` so the
    baseline instruction structure is reproduced exactly.

    Returns ``None`` if either span violates the per-qubit-gate
    preconditions (caller falls back to whole-span IF/ELSE wrap).
    Measurement gates are excluded by :data:`_PER_QUBIT_GATES`.
    """
    if not zero_span and not one_span:
        return []
    if not _check_per_qubit_span(zero_span) or not _check_per_qubit_span(one_span):
        return None

    # Identify structured (S) and merged (M) sides.  If one_span has at
    # least as many instructions as zero_span we drive by one_span; else
    # by zero_span and flip then/else.
    drive_by_one = len(one_span) >= len(zero_span)
    s_span = one_span if drive_by_one else zero_span
    m_span = zero_span if drive_by_one else one_span

    m_targets_by_name = _aggregate_by_name(m_span)

    # Track which m-qubits have been attributed to some IF block so we
    # can dump leftover m-only qubits into the first IF for their name.
    m_attributed: dict[str, set[int]] = {n: set() for n in m_targets_by_name}

    out: list[CircuitEntry] = []
    # Walk structured side; first pass records placeholders for IF blocks
    # so we can fold leftover m-only qubits at the end.
    if_block_indices_by_name: dict[str, int] = {}

    for s_inst in s_span:
        name = s_inst.name
        s_qs = _qubits_of(s_inst)
        m_qs_same_name = m_targets_by_name.get(name, [])
        m_qs_same_name_set = set(m_qs_same_name)
        common = [q for q in s_qs if q in m_qs_same_name_set]
        only_s = [q for q in s_qs if q not in m_qs_same_name_set]
        for q in common:
            m_attributed[name].add(q)
        if common:
            out.append(stim.CircuitInstruction(name, common, []))
        if only_s:
            # qubits only on s side under this name -- they may match m
            # under a *different* name (name-flip).  Build the matching
            # ELSE entry per source-name in m_span.
            block = IfBlock(condition_rec=condition_rec)
            block.then_body = [stim.CircuitInstruction(name, only_s, [])]
            # Group only_s qubits by the name they have in m_span (if any).
            by_m_name: dict[str, list[int]] = {}
            unmatched: list[int] = []
            for q in only_s:
                mn = _name_of_qubit_in(m_span, q)
                if mn is None:
                    unmatched.append(q)
                else:
                    by_m_name.setdefault(mn, []).append(q)
                    m_attributed.setdefault(mn, set()).add(q)
            else_body: list[CircuitEntry] = []
            for mn, qs in by_m_name.items():
                else_body.append(stim.CircuitInstruction(mn, qs, []))
            block.else_body = else_body
            # If a qubit is in s only (truly inserted), it has no else
            # counterpart — leave else_body as collected (possibly None).
            out.append(block)
            if_block_indices_by_name.setdefault(name, len(out) - 1)

    # Now handle m-only qubits (present in m_span but not in any s_inst
    # of the same name).  They become an extra ELSE arm — attribute to
    # the first IF block under the same name; if none exists, create a
    # trailing IF block with empty then_body.
    for mn, qubits in m_targets_by_name.items():
        leftover = [q for q in qubits if q not in m_attributed.get(mn, set())]
        if not leftover:
            continue
        # Find an IF block under mn or any IF block; else create one.
        target_idx = if_block_indices_by_name.get(mn)
        if target_idx is None:
            # No matching IF — append a new IfBlock with empty then.
            block = IfBlock(condition_rec=condition_rec)
            block.else_body = [stim.CircuitInstruction(mn, leftover, [])]
            out.append(block)
            continue
        block = out[target_idx]
        assert isinstance(block, IfBlock)
        new_else: list[CircuitEntry] = list(block.else_body or [])
        # Merge into an existing same-name else entry if present.
        merged = False
        for k, entry in enumerate(new_else):
            if isinstance(entry, stim.CircuitInstruction) and entry.name == mn:
                merged_qubits = _qubits_of(entry) + leftover
                new_else[k] = stim.CircuitInstruction(mn, merged_qubits, [])
                merged = True
                break
        if not merged:
            new_else.append(stim.CircuitInstruction(mn, leftover, []))
        block.else_body = new_else

    # If we drove by zero_span (merged-larger case), swap then/else.
    if not drive_by_one:
        for entry in out:
            if isinstance(entry, IfBlock):
                entry.then_body, entry.else_body = (
                    entry.else_body or [],
                    entry.then_body or None,
                )

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
