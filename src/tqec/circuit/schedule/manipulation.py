"""Defines functions to modify or merge :class:`.ScheduledCircuit` instances.

This module implement a few central functions for the :mod:`tqec` library:

- :func:`remove_duplicate_instructions` to remove some instructions appearing
  twice in a single moment (most of the time due to data qubit
  reset/measurements that are defined by each plaquette, even on qubits shared
  with other plaquettes, leading to duplicates).
- :func:`merge_scheduled_circuits` that merge several
  :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit` instances into one.
- :func:`relabel_circuits_qubit_indices` to prepare several
  :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit` before merging them.
  This function is called internally by :func:`merge_scheduled_circuits` but
  might be useful at other places and so is kept public.

"""

from __future__ import annotations

import functools
import itertools
import operator
import warnings
from collections.abc import Iterable, Mapping, Sequence

import stim

from tqec.circuit.moment import Moment
from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.circuit.schedule.schedule import Schedule
from tqec.compile.conditional.circuit import CircuitEntry, IfBlock
from tqec.utils.exceptions import TQECError, TQECWarning
from tqec.utils.position import BlockPosition2D


class _ScheduledCircuits:
    def __init__(self, circuits: list[ScheduledCircuit], global_qubit_map: QubitMap) -> None:
        """Represent a collection of :class:`.ScheduledCircuit` instances.

        This class aims at providing accessors for several compatible instances
        of :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit`. It allows
        to iterate on gates globally, for all the managed instances of
        :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit`, and implement
        a few other accessor methods to help with the task of merging multiple
        :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit` together.

        Args:
            circuits: the instances that should be managed. Note that the
                instances provided here have to be "compatible" with each
                other.
            global_qubit_map: a unique qubit map that can be used to map qubits
                to indices for all the provided ``circuits``.

        """
        # We might need to remap qubits to avoid index collision on several
        # circuits.
        self._circuits = circuits
        self._global_qubit_map = global_qubit_map
        self._iterators = [circuit.scheduled_moments for circuit in self._circuits]
        self._current_moments = [next(it, None) for it in self._iterators]

    def has_pending_moment(self) -> bool:
        """Check if any of the managed instances has a pending moment.

        Any moment that has not been collected by using collect_moment is considered to be pending.

        Any moment that has not been collected by using ``collect_moment`` is considered to be
        pending.

        """
        return any(self._has_pending_moment(i) for i in range(len(self._circuits)))

    def _has_pending_moment(self, index: int) -> bool:
        """Check if the managed instance at the given index has a pending operation."""
        return self._current_moments[index] is not None

    def _peek_scheduled_moment(self, index: int) -> tuple[int, Moment]:
        """Recover **without collecting** pending operations for the instance at the given index."""
        ret = self._current_moments[index]
        assert ret is not None
        return ret

    def _pop_scheduled_moment(self, index: int) -> tuple[int, Moment]:
        """Recover and mark as collected the pending moment for the instance at the given index.

        Raises:
            AssertionError: ``if not self.has_pending_operation(index)``.

        """
        ret = self._current_moments[index]
        if ret is None:
            raise TQECError(
                "Trying to pop a Moment instance from a ScheduledCircuit with "
                "all its moments already collected."
            )
        self._current_moments[index] = next(self._iterators[index], None)
        return ret

    @property
    def number_of_circuits(self) -> int:
        return len(self._circuits)

    def collect_moments_at_minimum_schedule(self) -> tuple[int, list[Moment]]:
        """Collect all the moments that can be collected.

        This method collects and returns a list of all the moments that should
        be scheduled next.

        Returns:
            a list of :class:`~tqec.circuit.moment.Moment` instances that should
            be added next to the QEC circuit.

        """
        assert self.has_pending_moment()
        circuit_indices_organised_by_schedule: dict[int, list[int]] = dict()
        for circuit_index in range(self.number_of_circuits):
            if not self._has_pending_moment(circuit_index):
                continue
            schedule, _ = self._peek_scheduled_moment(circuit_index)
            circuit_indices_organised_by_schedule.setdefault(schedule, list()).append(circuit_index)

        minimum_schedule = min(circuit_indices_organised_by_schedule.keys())
        moments_to_return: list[Moment] = list()
        for circuit_index in circuit_indices_organised_by_schedule[minimum_schedule]:
            _, moment = self._pop_scheduled_moment(circuit_index)
            moments_to_return.append(moment)
        return minimum_schedule, moments_to_return

    @property
    def q2i(self) -> dict[GridQubit, int]:
        return self._global_qubit_map.q2i


def _sort_target_groups(
    targets: Iterable[list[stim.GateTarget]],
) -> list[list[stim.GateTarget]]:
    def _sort_key(target_group: list[stim.GateTarget]) -> tuple[int, ...]:
        return tuple(t.value for t in target_group)

    return sorted(targets, key=_sort_key)


_CEOStaged = tuple[
    BlockPosition2D, tuple[int, ...], str, tuple[float, ...], list[stim.GateTarget]
]


def _stage_ceo_entries(
    merged_instructions: list[stim.CircuitInstruction],
    qubit_to_block: Mapping[GridQubit, BlockPosition2D],
    global_i2q: Mapping[int, GridQubit],
) -> tuple[list[stim.CircuitInstruction], list[_CEOStaged]]:
    """Split a moment's merged instructions into (passthrough, CEO-sorted staged groups).

    Instructions with no qubit targets, or any target qubit not owned by a known block,
    pass through unchanged (preserving their pre-CEO order). Remaining instructions
    contribute one staged entry per target group, sorted by ``(block.y, block.x, qids)``.
    """
    passthrough: list[stim.CircuitInstruction] = []
    ceo_entries: list[_CEOStaged] = []
    for inst in merged_instructions:
        groups = inst.target_groups()
        if not groups:
            passthrough.append(inst)
            continue
        if not all(t.qubit_value is not None for grp in groups for t in grp):
            passthrough.append(inst)
            continue
        args = tuple(inst.gate_args_copy())
        staged: list[_CEOStaged] = []
        eligible = True
        for grp in groups:
            qids = tuple(t.qubit_value for t in grp)
            first_q = global_i2q.get(qids[0])
            block_pos = qubit_to_block.get(first_q) if first_q is not None else None
            if block_pos is None:
                eligible = False
                break
            staged.append((block_pos, qids, inst.name, args, list(grp)))
        if eligible:
            ceo_entries.extend(staged)
        else:
            passthrough.append(inst)
    ceo_entries.sort(key=lambda e: (e[0].y, e[0].x, e[1]))
    return passthrough, ceo_entries


def _ceo_entry_signature(entry: _CEOStaged) -> tuple:
    return (entry[0], entry[1], entry[2], entry[3], tuple(t.value for t in entry[4]))


def _instruction_signature(inst: stim.CircuitInstruction) -> tuple:
    return (inst.name, tuple(inst.gate_args_copy()), [
        (t.value, t.is_qubit_target, t.is_measurement_record_target)
        for t in inst.targets_copy()
    ])


def _emit_moment_with_ceo(
    merged_instructions: list[stim.CircuitInstruction],
    qubit_to_block: Mapping[GridQubit, BlockPosition2D],
    global_i2q: Mapping[int, GridQubit],
    *,
    branch_merged_instructions: list[stim.CircuitInstruction] | None = None,
    condition_rec: int | None = None,
) -> list[CircuitEntry]:
    """Produce a moment's instruction stream in Canonical Emission Order.

    Returns a list of :class:`stim.CircuitInstruction` (and, when
    ``branch_merged_instructions`` is supplied, :class:`IfBlock`) entries in CEO
    order. The non-conditional path returns only plain instructions and is
    byte-equivalent to the pre-refactor emitter.

    When ``branch_merged_instructions`` is provided alongside ``condition_rec``:
    the staged CEO slots of both branches are walked in parallel; identical
    slots collapse into plain instructions exactly as in the single-branch path,
    and divergent slots emit an :class:`IfBlock` with the second-branch
    fragment as ``then_body`` and the first-branch fragment as ``else_body``
    (convention: first arg = branch-zero, second arg = branch-one).

    Cross-cube CEO slot ownership note: a two-qubit gate spanning a spatial
    pipe is assigned to the first qubit's owning block (see
    :func:`_stage_ceo_entries`). Per the commit-145dc902 guard, no spatial
    pipe touches a conditional cube, so a conditional CEO slot never spans two
    cubes.
    """
    passthrough_z, ceo_z = _stage_ceo_entries(merged_instructions, qubit_to_block, global_i2q)

    if branch_merged_instructions is None:
        result: list[CircuitEntry] = list(passthrough_z)
        i = 0
        while i < len(ceo_z):
            name = ceo_z[i][2]
            args = ceo_z[i][3]
            flat: list[stim.GateTarget] = []
            while i < len(ceo_z) and ceo_z[i][2] == name and ceo_z[i][3] == args:
                flat.extend(ceo_z[i][4])
                i += 1
            result.append(stim.CircuitInstruction(name, flat, list(args)))
        return result

    if condition_rec is None:
        raise TQECError(
            "_emit_moment_with_ceo: branch_merged_instructions requires condition_rec."
        )
    passthrough_o, ceo_o = _stage_ceo_entries(
        branch_merged_instructions, qubit_to_block, global_i2q
    )
    if [_instruction_signature(p) for p in passthrough_z] != [
        _instruction_signature(p) for p in passthrough_o
    ]:
        raise TQECError(
            "_emit_moment_with_ceo: passthrough instructions differ between branches; "
            "Equal Emission Order assumes non-block-owned instructions are identical."
        )
    if len(ceo_z) != len(ceo_o):
        raise TQECError(
            "_emit_moment_with_ceo: CEO slot count differs between branches "
            f"(branch-zero={len(ceo_z)}, branch-one={len(ceo_o)}); Equal Measurement Count + CEO violated."
        )

    result = list(passthrough_z)
    i = 0
    while i < len(ceo_z):
        if _ceo_entry_signature(ceo_z[i]) == _ceo_entry_signature(ceo_o[i]):
            name = ceo_z[i][2]
            args = ceo_z[i][3]
            flat = []
            while (
                i < len(ceo_z)
                and ceo_z[i][2] == name
                and ceo_z[i][3] == args
                and _ceo_entry_signature(ceo_z[i]) == _ceo_entry_signature(ceo_o[i])
            ):
                flat.extend(ceo_z[i][4])
                i += 1
            result.append(stim.CircuitInstruction(name, flat, list(args)))
        else:
            # Batch consecutive divergent slots sharing per-branch
            # (name, args) signatures into a single IfBlock with merged
            # targets. Common pattern: a basis-swap across an ancilla run
            # (M/MX over qubits 4, 5, 6, ...) — CEO sorts these adjacently,
            # so per-slot IF/ELSE wraps collapse to one IF/ELSE per run.
            z_name = ceo_z[i][2]
            z_args = ceo_z[i][3]
            o_name = ceo_o[i][2]
            o_args = ceo_o[i][3]
            z_targets: list[stim.GateTarget] = []
            o_targets: list[stim.GateTarget] = []
            while (
                i < len(ceo_z)
                and _ceo_entry_signature(ceo_z[i]) != _ceo_entry_signature(ceo_o[i])
                and ceo_z[i][2] == z_name
                and ceo_z[i][3] == z_args
                and ceo_o[i][2] == o_name
                and ceo_o[i][3] == o_args
            ):
                z_targets.extend(ceo_z[i][4])
                o_targets.extend(ceo_o[i][4])
                i += 1
            zero_inst = stim.CircuitInstruction(z_name, z_targets, list(z_args))
            one_inst = stim.CircuitInstruction(o_name, o_targets, list(o_args))
            result.append(
                IfBlock(
                    condition_rec=condition_rec,
                    then_body=[one_inst],
                    else_body=[zero_inst],
                )
            )
    return result


def remove_duplicate_instructions(
    instructions: list[stim.CircuitInstruction],
    mergeable_instruction_names: frozenset[str],
) -> list[stim.CircuitInstruction]:
    """Remove all the duplicate instructions from the given list.

    Note:
        This function guarantees the following post-conditions on the returned
        results:

        - Instructions with a name that is not in ``mergeable_instruction_names``
          are returned at the front of the returned list, in the same relative
          ordering as provided in ``instructions`` input.
        - Instructions with a name that is in ``mergeable_instruction_names``
          will be returned after all the instructions with a name that does not
          appear in ``mergeable_instruction_names``. The order in which these
          instructions are returned is not guaranteed and can change between
          executions.

    Warning:
        this function **does not keep instruction ordering**. It is intended to
        be used with input ``instructions`` that, once de-duplication has been
        applied, form a valid moment, which means that each instruction can be
        executed in parallel, and so their order in the returned list does not
        matter.

        If that is not your case, take extra care to the output of this
        function as it will likely introduce hard-to-debug issues. To prevent
        such potential misuse, this function checks for such cases and outputs
        a warning if it happens.

    Returns:
        a list containing a copy of the ``stim.CircuitInstruction`` instances
        from the given instructions but without any duplicate.

    """
    # Separate mergeable operations from non-mergeable ones.
    mergeable_operations: dict[tuple[str, tuple[float, ...]], set[tuple[stim.GateTarget, ...]]] = {}
    final_operations: list[stim.CircuitInstruction] = list()
    for inst in instructions:
        if inst.name in mergeable_instruction_names:
            # Mergeable operations are automatically merged thanks to
            # the use of a set here.
            mergeable_operations.setdefault(
                (inst.name, tuple(inst.gate_args_copy())), set()
            ).update(tuple(group) for group in inst.target_groups())
        else:
            final_operations.append(inst)
    # Add the merged operations into the final ones
    final_operations.extend(
        stim.CircuitInstruction(
            name,
            functools.reduce(operator.iadd, _sort_target_groups([list(t) for t in targets]), []),
            args,
        )
        for (name, args), targets in mergeable_operations.items()
    )
    # Warn if the output instructions do not form a valid moment, as this is
    # likely a misuse of this function.
    circuit = stim.Circuit()
    for instr in final_operations:
        circuit.append(instr)
    try:
        Moment.check_is_valid_moment(circuit)
    except TQECError as e:
        warnings.warn(
            "The instructions obtained at the end of the "
            "`remove_duplicate_instructions` function do not form a valid "
            "moment. You are likely misusing the function. Final instructions "
            "obtained and gathered into a single stim.Circuit: "
            f"\n{circuit}\nReason:\n{e}",
            TQECWarning,
        )
    return final_operations


def merge_instructions(
    instructions: list[stim.CircuitInstruction],
) -> list[stim.CircuitInstruction]:
    """Merge instructions with the same name and arguments.

    Returns:
        a list containing a copy of the ``stim.CircuitInstruction`` instances
        from the given instructions but merged.

    """
    instructions_merger: dict[tuple[str, tuple[float, ...]], list[list[stim.GateTarget]]] = {}
    for instruction in instructions:
        args = tuple(instruction.gate_args_copy())
        instructions_merger.setdefault((instruction.name, args), []).extend(
            instruction.target_groups()
        )
    return [
        stim.CircuitInstruction(name, functools.reduce(operator.iadd, targets, []), args)
        for (name, args), targets in instructions_merger.items()
    ]


def merge_scheduled_circuits(
    circuits: list[ScheduledCircuit],
    global_qubit_map: QubitMap,
    mergeable_instructions: Iterable[str] = (),
    qubit_to_block: Mapping[GridQubit, BlockPosition2D] | None = None,
) -> ScheduledCircuit:
    """Merge several :class:`.ScheduledCircuit` instances into one instance.

    This function takes several **compatible** scheduled circuits as input and
    merge them, respecting their schedules, into a unique
    :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit` instance that will
    then be returned to the caller.

    The provided circuits should be compatible between each other. Compatible
    circuits are circuits that can all be described with a unique global qubit
    map. In other words, if two circuits from the list of compatible circuits
    use the same qubit index, that should mean that they use the same qubit.
    You can obtain compatible circuits by using
    :func:`relabel_circuits_qubit_indices`.

    Args:
        circuits: **compatible** circuits to merge.
        global_qubit_map: global qubit map for all the provided ``circuits``.
        mergeable_instructions: a list of instruction names that are considered
            mergeable. Duplicate instructions with a name in this list will be
            merged into a single instruction.
        qubit_to_block: optional mapping from ``GridQubit`` to owning
            ``BlockPosition2D`` (cube position in the enclosing ``LayoutLayer``).
            Plumbed in for the upcoming Canonical Emission Order pass; currently
            unused. ``None`` preserves baseline behaviour.

    Returns:
        a circuit representing the merged scheduled circuits given as input.

    """
    scheduled_circuits = _ScheduledCircuits(circuits, global_qubit_map)

    all_moments: list[Moment] = []
    all_schedules = Schedule()
    global_i2q = QubitMap({i: q for q, i in scheduled_circuits.q2i.items()})

    while scheduled_circuits.has_pending_moment():
        schedule, moments = scheduled_circuits.collect_moments_at_minimum_schedule()
        # Flatten the moments into a list of operations to perform some modifications
        instructions: list[stim.CircuitInstruction] = functools.reduce(
            operator.iadd, (list(moment.instructions) for moment in moments), []
        )
        # Avoid duplicated operations. Any operation that have the Plaquette.get_mergeable_tag() tag
        # is considered mergeable, and can be removed if another operation in the list
        # is considered equal (and has the mergeable tag).
        deduplicated_instructions = remove_duplicate_instructions(
            instructions,
            mergeable_instruction_names=frozenset(mergeable_instructions),
        )
        merged_instructions = merge_instructions(deduplicated_instructions)
        circuit = stim.Circuit()
        if qubit_to_block is None:
            for inst in merged_instructions:
                circuit.append(
                    inst.name,
                    functools.reduce(operator.iadd, _sort_target_groups(inst.target_groups()), []),
                    inst.gate_args_copy(),
                )
        else:
            entries = _emit_moment_with_ceo(
                merged_instructions, qubit_to_block, global_i2q.i2q
            )
            for entry in entries:
                if isinstance(entry, IfBlock):
                    raise NotImplementedError(
                        "merge_scheduled_circuits: IfBlock emission requires per-branch "
                        "input wiring (Stage A commit 3)."
                    )
                circuit.append(entry)
        all_moments.append(Moment(circuit))
        all_schedules.append(schedule)

    return ScheduledCircuit(all_moments, all_schedules, global_i2q, _avoid_checks=True)


def merge_scheduled_circuits_per_branch(
    zero_circuits: list[ScheduledCircuit],
    one_circuits: list[ScheduledCircuit],
    global_qubit_map: QubitMap,
    *,
    condition_rec: int,
    mergeable_instructions: Iterable[str] = (),
    qubit_to_block: Mapping[GridQubit, BlockPosition2D],
) -> tuple[list[list[CircuitEntry]], Schedule]:
    """Merge two parallel branches of :class:`.ScheduledCircuit` instances into a per-moment
    stream of :class:`CircuitEntry` values, weaving :class:`IfBlock` at CEO slots that
    differ between the branches.

    Both branches must share ``global_qubit_map`` and produce moments at identical
    schedules (enforced by Equal Measurement Count + Canonical Emission Order).
    The two branch lists may have different per-plaquette circuits only at slots
    owned by a conditional cube; everywhere else they are byte-identical and
    collapse to plain :class:`stim.CircuitInstruction` entries.

    Args:
        zero_circuits: branch-zero per-plaquette scheduled circuits, in the same
            order as the per-plaquette walk of the layer.
        one_circuits: branch-one parallel list; same length and same per-slot
            qubit footprint as ``zero_circuits``.
        global_qubit_map: shared qubit map for both branches.
        condition_rec: ``stim`` record offset that drives the woven IF/ELSE.
            Convention: ``then_body`` runs when the condition is one,
            ``else_body`` runs when it is zero.
        mergeable_instructions: as in :func:`merge_scheduled_circuits`.
        qubit_to_block: required; CEO needs block ownership to align slots
            between branches.

    Returns:
        ``(moments_entries, schedule)`` where ``moments_entries[i]`` is the
        ``list[CircuitEntry]`` for the moment at ``schedule[i]``. Callers
        assemble the result into a :class:`~tqec.compile.conditional.circuit.ConditionalCircuit`
        (or unwrap to a plain ``stim.Circuit`` when no ``IfBlock`` surfaces).

    Raises:
        TQECError: if the two branches diverge in moment count, schedule, or
            number of CEO slots at a moment.

    """
    sched_z = _ScheduledCircuits(zero_circuits, global_qubit_map)
    sched_o = _ScheduledCircuits(one_circuits, global_qubit_map)
    global_i2q = QubitMap({i: q for q, i in global_qubit_map.q2i.items()})
    mergeable = frozenset(mergeable_instructions)

    moments_entries: list[list[CircuitEntry]] = []
    schedule_out = Schedule()

    while sched_z.has_pending_moment() or sched_o.has_pending_moment():
        if not (sched_z.has_pending_moment() and sched_o.has_pending_moment()):
            raise TQECError(
                "merge_scheduled_circuits_per_branch: branches disagree on moment "
                "count; Equal Measurement Count violated."
            )
        schedule_z, moments_z = sched_z.collect_moments_at_minimum_schedule()
        schedule_o, moments_o = sched_o.collect_moments_at_minimum_schedule()
        if schedule_z != schedule_o:
            raise TQECError(
                "merge_scheduled_circuits_per_branch: schedule mismatch between "
                f"branches (zero={schedule_z}, one={schedule_o})."
            )
        instructions_z = functools.reduce(
            operator.iadd, (list(m.instructions) for m in moments_z), []
        )
        instructions_o = functools.reduce(
            operator.iadd, (list(m.instructions) for m in moments_o), []
        )
        merged_z = merge_instructions(
            remove_duplicate_instructions(instructions_z, mergeable)
        )
        merged_o = merge_instructions(
            remove_duplicate_instructions(instructions_o, mergeable)
        )
        entries = _emit_moment_with_ceo(
            merged_z,
            qubit_to_block,
            global_i2q.i2q,
            branch_merged_instructions=merged_o,
            condition_rec=condition_rec,
        )
        moments_entries.append(entries)
        schedule_out.append(schedule_z)
    return moments_entries, schedule_out


def relabel_circuits_qubit_indices(
    circuits: Sequence[ScheduledCircuit],
) -> tuple[list[ScheduledCircuit], QubitMap]:
    """Relabel the qubit indices of the provided circuits to avoid collision.

    When several :class:`~tqec.circuit.schedule.circuit.ScheduledCircuit` are
    constructed without a global knowledge of all the qubits, qubit indices used
    by each instance likely overlap. This is an issue when we try to merge such
    circuits because one index might represent a different qubit depending on
    the circuit it is used in.

    This function takes a sequence of circuits and relabel their qubits to avoid
    such collisions.

    Warning:
        all the qubit targets used in each of the provided circuits should have
        a corresponding entry in the circuit qubit map for this function to work
        correctly. If that is not the case, a ``KeyError`` will be raised.

    Raises:
        KeyError: if any of the provided circuit contains a qubit target that is
            not present in its qubit map.

    Args:
        circuits: circuit instances to remap. This parameter is not mutated by
            this function and is only used in read-only mode.

    Returns:
        the same circuits with updated qubit indices as well as the global qubit
        indices map that has been used. Qubits in the returned global qubit map
        are assigned to an index such that:

        1. the sequence of indices is ``range(0, len(qubit_map))``.
        2. qubits are assigned indices in sorted order.

    """
    # First, get a global qubit index map.
    # Using itertools to avoid the edge case `len(circuits) == 0`
    needed_qubits = frozenset(itertools.chain.from_iterable([c.qubits for c in circuits]))
    global_qubit_map = QubitMap.from_qubits(sorted(needed_qubits))
    global_q2i = global_qubit_map.q2i
    # Then, get the remapped circuits. Note that map_qubit_indices should
    # have approximately the same runtime cost whatever the value of inplace
    # so we ask for a new instance to avoid keeping a reference to the given
    # circuits.
    relabeled_circuits: list[ScheduledCircuit] = []
    for circuit in circuits:
        local_indices_to_global_indices = {
            local_index: global_q2i[q] for local_index, q in circuit.qubit_map.items()
        }
        relabeled_circuits.append(
            circuit.map_qubit_indices(local_indices_to_global_indices, inplace=False)
        )
    return relabeled_circuits, global_qubit_map
