"""IF/ELSE-capable circuit container.

Vanilla ``stim.Circuit`` (1.15) has no notion of an ``IF`` block, so the
conditional-cube compilation path emits its output via this thin wrapper.
``ConditionalCircuit`` mirrors a useful subset of ``stim.Circuit.append``
and adds :meth:`append_if` for explicit ``IF(rec[-k]) { ... } ELSE { ... }``
blocks.  :meth:`to_stim_text` serialises to Stim text extended with those
blocks; :meth:`to_stim_circuit_strict` round-trips back to
``stim.Circuit`` and raises if any ``IfBlock`` is still present.

The :func:`remap_entry_qubit_indices` helper rewrites qubit-target indices on
a :class:`CircuitEntry` (recursing into :class:`IfBlock` bodies) — used by the
tree-level assembly that joins per-leaf conditional circuits against a shared
global qubit map.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import stim

if TYPE_CHECKING:
    from tqec.circuit.qubit_map import QubitMap


@dataclass
class IfBlock:
    """An ``IF(rec[-k]^rec[-j]^...) { ... } ELSE { ... }`` block.

    ``condition_recs`` lists the measurements whose XOR selects the
    ``then_body`` branch, in one of two forms:

    * negative ``rec`` offsets, relative to the point the block is emitted,
      rendered as given;
    * non-negative *absolute* measurement indices (0 for the first measurement
      of the circuit being rendered). They are converted to ``rec`` offsets
      when the enclosing :class:`ConditionalCircuit` is rendered, from the
      number of measurements that precede the block there. The compiler uses
      this form: one condition is shared by every ``IF`` of a conditional
      cube, and each of them sits after a different number of measurements.

    The two forms cannot be mixed within one block. A single-element list
    renders as ``IF(rec[-k])``; multiple offsets render as
    ``IF(rec[-a]^rec[-b]^...)``. ``else_body`` is optional; when absent, only
    the ``IF`` arm is rendered.
    """

    condition_recs: list[int]
    then_body: list[CircuitEntry] = field(default_factory=list)
    else_body: list[CircuitEntry] | None = None

    def __post_init__(self) -> None:
        if not self.condition_recs:
            raise ValueError("IfBlock requires at least one condition rec offset.")
        if len({r < 0 for r in self.condition_recs}) > 1:
            raise ValueError(
                "IfBlock condition_recs mixes relative rec offsets (negative) with "
                f"absolute measurement indices (non-negative): {self.condition_recs}."
            )

    def append(self, entry: CircuitEntry) -> None:
        """Append an entry to the ``then_body`` (``IF`` arm)."""
        self.then_body.append(entry)

    def append_else(self, entry: CircuitEntry) -> None:
        """Append an entry to the ``else_body`` (``ELSE`` arm), creating it if absent."""
        if self.else_body is None:
            self.else_body = []
        self.else_body.append(entry)


CircuitEntry = Union[stim.CircuitInstruction, "IfBlock"]

Condition = tuple[int, ...]
"""An ``IF`` condition: the sorted absolute indices of the measurements it XORs.

Absolute indices count from 0, the circuit's first measurement. Every ``IF``
block a conditional cube emits reads the same measurements, so it has the same
:data:`Condition` wherever it sits, unlike its rendered ``rec`` offsets.
"""


class ConditionalCircuit:
    """Mutable container holding ``stim.CircuitInstruction`` + ``IfBlock`` entries."""

    def __init__(
        self,
        entries: list[CircuitEntry] | None = None,
        qubit_map: QubitMap | None = None,
    ) -> None:
        """Create a conditional circuit.

        Args:
            entries: the initial entries, copied. Defaults to no entries.
            qubit_map: the qubit map the entries' targets refer to, if any.

        """
        self._entries: list[CircuitEntry] = list(entries) if entries else []
        self._qubit_map = qubit_map

    @property
    def qubit_map(self) -> QubitMap | None:
        """Local qubit map of the circuit, when known.

        Set by
        :meth:`LayoutLayer.to_conditional_circuit`; ``None`` for hand-built
        instances. Tree-level assembly uses this to remap local qubit indices to
        the global qubit map.
        """
        return self._qubit_map

    def set_qubit_map(self, qubit_map: QubitMap) -> None:
        """Set the local qubit map used for tree-level index remapping."""
        self._qubit_map = qubit_map

    @property
    def entries(self) -> list[CircuitEntry]:
        """Return the underlying list of circuit entries."""
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
        """Append an already-built :class:`stim.CircuitInstruction`."""
        self._entries.append(inst)

    def append_if(self, if_block: IfBlock) -> None:
        """Append an :class:`IfBlock` entry."""
        self._entries.append(if_block)

    def append_instruction_or_if(self, entry: CircuitEntry) -> None:
        """Append a plain instruction or an :class:`IfBlock`.

        Append either a plain :class:`stim.CircuitInstruction` or an
        :class:`IfBlock`, dispatching on type. Convenience for callers iterating
        a mixed sequence.
        """
        self._entries.append(entry)

    def extend(self, entries: list[CircuitEntry]) -> None:
        """Append every entry in ``entries`` in order."""
        self._entries.extend(entries)

    def to_stim_text(self) -> str:
        """Serialise to Stim text with ``IF(rec[-k]) { ... } ELSE { ... }`` blocks."""
        lines: list[str] = []
        _render(self._entries, lines, indent=0, measurements_before=0)
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

    @property
    def conditions(self) -> list[Condition]:
        """Return the distinct conditions of the ``IF`` blocks, in order of first use.

        Blocks nested in either arm of another block are included.
        """
        found: dict[Condition, None] = {}

        def visit(block: IfBlock, measurements_before: int) -> int | None:
            found.setdefault(_absolute_condition(block, measurements_before))
            return None

        _walk_conditions(self._entries, 0, visit)
        return list(found)

    def resolve(self, outcomes: Mapping[Condition, int]) -> stim.Circuit:
        """Return the ``stim.Circuit`` of one branch, picking an arm of every ``IF``.

        Every ``rec`` offset stays valid in the result: the two arms of a block
        perform the same measurements (the Equal Measurement Count invariant),
        so the instructions around it see the same records whichever arm is
        kept.

        Args:
            outcomes: for each condition (see :attr:`conditions`), the parity
                of the measurements it reads: ``1`` keeps the ``IF`` arm of
                every block with that condition, ``0`` the ``ELSE`` arm (or
                nothing, for a block without one).

        Returns:
            the circuit of the chosen branch, without any ``IF`` block.

        Raises:
            ValueError: if a condition the circuit reaches is missing from
                ``outcomes``, or an outcome is neither 0 nor 1.

        """
        circuit = stim.Circuit()

        def choose(block: IfBlock, measurements_before: int) -> int | None:
            condition = _absolute_condition(block, measurements_before)
            if condition not in outcomes:
                raise ValueError(f"No outcome given for the IF condition on {list(condition)}.")
            outcome = outcomes[condition]
            if outcome not in (0, 1):
                raise ValueError(
                    f"The outcome of the IF condition on {list(condition)} must be 0 "
                    f"or 1, got {outcome}."
                )
            return outcome

        _walk_conditions(self._entries, 0, choose, circuit.append)
        return circuit

    def __repr__(self) -> str:
        return f"ConditionalCircuit({len(self._entries)} entries)"


def _absolute_condition(block: IfBlock, measurements_before: int) -> Condition:
    """Return the :data:`Condition` of ``block``, emitted after ``measurements_before``."""
    if block.condition_recs[0] < 0:
        return tuple(sorted(measurements_before + r for r in block.condition_recs))
    return tuple(sorted(block.condition_recs))


def _walk_conditions(
    entries: list[CircuitEntry],
    measurements_before: int,
    visit: Callable[[IfBlock, int], int | None],
    emit: Callable[[stim.CircuitInstruction], None] | None = None,
) -> int:
    """Walk ``entries`` in program order, calling ``visit`` on every ``IF`` block.

    ``visit`` receives a block and the number of measurements before it, and
    returns the arm to descend into (``1`` for ``IF``, ``0`` for ``ELSE``), or
    ``None`` to descend into both. ``emit`` receives every instruction on the
    path walked. Returns the number of measurements after ``entries``.
    """
    for entry in entries:
        if isinstance(entry, IfBlock):
            arm = visit(entry, measurements_before)
            if arm in (None, 1):
                _walk_conditions(entry.then_body, measurements_before, visit, emit)
            if arm in (None, 0) and entry.else_body is not None:
                _walk_conditions(entry.else_body, measurements_before, visit, emit)
        elif emit is not None:
            emit(entry)
        measurements_before += len(_measured_qubits(entry))
    return measurements_before


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
            condition_recs=list(entry.condition_recs),
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
    return stim.CircuitInstruction(
        entry.name, new_targets, list(entry.gate_args_copy()), tag=entry.tag
    )


def _measured_qubits(entry: CircuitEntry) -> list[int]:
    """Return the qubits ``entry`` measures, one per measurement, in record order.

    For an :class:`IfBlock` this is the sequence of either branch, which the
    Equal Measurement Count and Canonical Emission Order assumptions make
    identical: every later ``rec`` offset, and the branch-zero measurement
    records the detectors of both branches are built from, depend on it.

    Raises:
        ValueError: if the two branches of an :class:`IfBlock` measure a
            different number of qubits, or different qubits in some position.

    """
    if isinstance(entry, IfBlock):
        then_qubits = [q for e in entry.then_body for q in _measured_qubits(e)]
        if entry.else_body is not None:
            else_qubits = [q for e in entry.else_body for q in _measured_qubits(e)]
            if else_qubits != then_qubits:
                raise ValueError(
                    "The two branches of an IfBlock do not measure the same qubits "
                    f"in the same order ({then_qubits} vs {else_qubits}), so no later "
                    "rec offset can be branch-independent."
                )
        return then_qubits
    circuit = stim.Circuit()
    circuit.append(entry)
    if circuit.num_measurements == 0:
        return []
    qubits = [t.qubit_value for t in entry.targets_copy() if t.is_qubit_target]
    if len(qubits) != circuit.num_measurements:
        # A multi-qubit measurement (e.g. MPP): its records are not per qubit,
        # so only the count is meaningful. Represent each record by -1.
        return [-1] * circuit.num_measurements
    return [q for q in qubits if q is not None]


def _rendered_condition(condition_recs: list[int], measurements_before: int) -> list[int]:
    """Return ``condition_recs`` as ``rec`` offsets at a point after ``measurements_before``."""
    if condition_recs[0] < 0:
        return list(condition_recs)
    late = [index for index in condition_recs if index >= measurements_before]
    if late:
        raise ValueError(
            f"An IF condition refers to measurement(s) {late}, but only "
            f"{measurements_before} measurements precede the IF block."
        )
    return [index - measurements_before for index in condition_recs]


def _render(
    entries: list[CircuitEntry], lines: list[str], indent: int, measurements_before: int
) -> int:
    """Render ``entries`` into ``lines`` and return the measurements performed so far."""
    pad = "  " * indent
    for entry in entries:
        if isinstance(entry, IfBlock):
            condition = _rendered_condition(entry.condition_recs, measurements_before)
            cond = "^".join(f"rec[{r}]" for r in condition)
            lines.append(f"{pad}IF({cond}) {{")
            _render(entry.then_body, lines, indent + 1, measurements_before)
            if entry.else_body is not None:
                lines.append(f"{pad}}} ELSE {{")
                _render(entry.else_body, lines, indent + 1, measurements_before)
            lines.append(f"{pad}}}")
        else:
            text = str(entry).strip()
            lines.extend(f"{pad}{line}" for line in text.splitlines())
        measurements_before += len(_measured_qubits(entry))
    return measurements_before
