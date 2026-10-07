"""Apply a :class:`~tqec.utils.noise_model.NoiseModel` to a :class:`ConditionalCircuit`.

:meth:`NoiseModel.noisy_circuit <tqec.utils.noise_model.NoiseModel.noisy_circuit>`
works moment by moment: every operation gets the noise of its rule, part of it
during the moment and part after it, and every qubit no operation touches gets
idling noise. In a conditional circuit an ``IF`` block sits inside a moment, so
what happens in that moment depends on the branch. :func:`noisy_conditional_circuit`
noises each arm of a block in place and puts the noise that follows the moment
in an ``IF`` of its own wherever the two arms disagree on it: the noise after an
arm's operations, and the idling of a qubit only one arm touches.

Resolving the result to a branch gives the same noise as resolving first and
applying :meth:`~tqec.utils.noise_model.NoiseModel.noisy_circuit`, up to the
order of the noise channels that follow a moment, which are Pauli channels and
so commute.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Set

import stim

from tqec.compile.conditional.circuit import (
    CircuitEntry,
    ConditionalCircuit,
    IfBlock,
    _measured_qubits,
)
from tqec.utils.noise_model import (
    NoiseModel,
    _split_targets_if_needed,
    occurs_in_classical_control_system,
)

_After = defaultdict[tuple[str, float], stim.Circuit]


def noisy_conditional_circuit(
    noise_model: NoiseModel,
    circuit: ConditionalCircuit,
    *,
    system_qubits: Set[int] | None = None,
    noiseless_qubits: Mapping[int, Set[int]] | None = None,
) -> ConditionalCircuit:
    """Return a noisy version of ``circuit``, keeping its ``IF``/``ELSE`` blocks.

    Args:
        noise_model: the noise model to apply.
        circuit: the circuit to layer noise over. Its ``IF`` blocks must sit
            inside a moment (contain no ``TICK``), the blocks acting on qubits
            in one moment must share their condition, and a block acting on
            qubits must not hold another block.
        system_qubits: all the qubits of the circuit, the ones eligible for
            idling noise. Defaults to every qubit index below the highest one
            used, in either arm of any block. A compiled circuit declares all
            its qubits up front with ``QUBIT_COORDS``, so each of its branches
            uses the same qubits; for a circuit where a qubit only appears in
            one arm, pass the qubits explicitly for every branch to idle the
            same ones as :meth:`~tqec.utils.noise_model.NoiseModel.noisy_circuit`
            on that branch alone would.
        noiseless_qubits: for some moments, the qubits to leave untouched in
            that moment, as in
            :meth:`~tqec.utils.noise_model.NoiseModel.noisy_circuit`. Keys index
            the moments of ``circuit``, the spans between its top-level
            ``TICK`` instructions.

    Returns:
        the noisy circuit, with the qubit map of ``circuit``.

    Raises:
        NotImplementedError: if ``circuit`` breaks one of the requirements above.
        ValueError: if an operation has no noise rule, or a moment operates on
            a qubit twice, as for
            :meth:`~tqec.utils.noise_model.NoiseModel.noisy_circuit`.

    """
    noiseless = noiseless_qubits if noiseless_qubits is not None else {}
    if system_qubits is None:
        system_qubits = set(range(_num_qubits(circuit.entries)))
    result = ConditionalCircuit(qubit_map=circuit.qubit_map)
    measurements_before = 0
    for index, moment in enumerate(_moments(circuit.entries)):
        if result.entries:
            result.append("TICK")
        immune = frozenset(noiseless.get(index, ()))
        result.extend(
            _noisy_moment(noise_model, moment, system_qubits, immune, measurements_before)
        )
        measurements_before += sum(len(_measured_qubits(entry)) for entry in moment)
    return result


def _num_qubits(entries: list[CircuitEntry]) -> int:
    """Return the number of qubits ``entries`` use, counting both arms of every block."""
    flat = stim.Circuit()
    for entry in _flatten_both_arms(entries):
        flat.append(entry)
    return flat.num_qubits


def _flatten_both_arms(entries: list[CircuitEntry]) -> Iterator[stim.CircuitInstruction]:
    for entry in entries:
        if isinstance(entry, IfBlock):
            yield from _flatten_both_arms(entry.then_body)
            yield from _flatten_both_arms(entry.else_body or [])
        else:
            yield entry


def _moments(entries: list[CircuitEntry]) -> Iterator[list[CircuitEntry]]:
    """Split ``entries`` at their ``TICK`` instructions, like ``noisy_circuit`` does."""
    moment: list[CircuitEntry] = []
    for entry in entries:
        if isinstance(entry, IfBlock):
            if any(inst.name == "TICK" for inst in _flatten_both_arms([entry])):
                raise NotImplementedError(
                    "Cannot add noise to an IF block that spans several moments (holds a TICK)."
                )
            moment.append(entry)
        elif entry.name == "TICK":
            yield moment
            moment = []
        else:
            moment.append(entry)
    if moment:
        yield moment


def _acts_on_qubits(block: IfBlock) -> bool:
    """Return whether an arm of ``block`` holds a quantum operation."""
    return any(not occurs_in_classical_control_system(op) for op in _flatten_both_arms([block]))


def _noisy_moment(
    noise_model: NoiseModel,
    moment: list[CircuitEntry],
    system_qubits: Set[int],
    immune: frozenset[int],
    measurements_before: int,
) -> list[CircuitEntry]:
    """Return the noisy version of one moment, with its ``IF`` blocks noised in place."""
    condition: list[int] | None = None
    measured = measurements_before
    for entry in moment:
        if isinstance(entry, IfBlock) and _acts_on_qubits(entry):
            if any(isinstance(e, IfBlock) for e in [*entry.then_body, *(entry.else_body or [])]):
                raise NotImplementedError(
                    "Cannot add noise to an IF block acting on qubits that holds another IF block."
                )
            # Keep the block's own order: a reader may key a block by its first rec.
            absolute = [measured + r if r < 0 else r for r in entry.condition_recs]
            if condition is not None and sorted(absolute) != sorted(condition):
                raise NotImplementedError(
                    "Cannot add noise to a moment holding IF blocks with different "
                    f"conditions ({condition} and {absolute}) that act on qubits."
                )
            if condition is None:
                condition = absolute
        measured += len(_measured_qubits(entry))

    out: list[CircuitEntry] = []
    during = stim.Circuit()
    shared_after: _After = defaultdict(stim.Circuit)
    arm_after: dict[int, _After] = {1: defaultdict(stim.Circuit), 0: defaultdict(stim.Circuit)}
    shared_ops: list[stim.CircuitInstruction] = []
    arm_ops: dict[int, list[stim.CircuitInstruction]] = {1: [], 0: []}
    for entry in moment:
        if isinstance(entry, IfBlock) and _acts_on_qubits(entry):
            out.extend(during)
            during = stim.Circuit()
            arms: dict[int, stim.Circuit] = {}
            for arm, body in ((1, entry.then_body), (0, entry.else_body)):
                arms[arm] = stim.Circuit()
                for inst in body or []:
                    assert isinstance(inst, stim.CircuitInstruction)
                    for op in _split_targets_if_needed(inst, immune_qubits=immune):
                        arm_ops[arm].append(op)
                        _append_noisy_op(noise_model, op, arms[arm], arm_after[arm], immune)
            out.append(
                IfBlock(
                    condition_recs=list(entry.condition_recs),
                    then_body=list(arms[1]),
                    else_body=list(arms[0]) if entry.else_body is not None else None,
                )
            )
        elif isinstance(entry, IfBlock):
            out.extend(during)
            during = stim.Circuit()
            out.append(entry)
        else:
            for op in _split_targets_if_needed(entry, immune_qubits=immune):
                shared_ops.append(op)
                _append_noisy_op(noise_model, op, during, shared_after, immune)
    out.extend(during)

    out.extend(_merged_after(shared_after))
    if condition is not None:
        _append_if_arms_differ(
            out, condition, _merged_after(arm_after[1]), _merged_after(arm_after[0])
        )

    idle: dict[int, stim.Circuit] = {}
    for arm in (1, 0) if condition is not None else (1,):
        idle[arm] = stim.Circuit()
        noise_model._append_idle_error(
            moment_split_ops=shared_ops + arm_ops[arm],
            out=idle[arm],
            system_qubits=system_qubits,
            immune_qubits=immune,
        )
    if condition is None or idle[1] == idle[0]:
        out.extend(idle[1])
    else:
        _append_if_arms_differ(out, condition, list(idle[1]), list(idle[0]))
    return out


def _append_noisy_op(
    noise_model: NoiseModel,
    op: stim.CircuitInstruction,
    during: stim.Circuit,
    after: _After,
    immune: frozenset[int],
) -> None:
    rule = noise_model._noise_rule_for_split_operation(split_op=op)
    if rule is None:
        during.append(op)
    else:
        rule._append_noisy_version_of(
            split_op=op, out_during_moment=during, after_moments=after, immune_qubits=immune
        )


def _merged_after(after: _After) -> list[stim.CircuitInstruction]:
    merged = stim.Circuit()
    for key in sorted(after):
        merged += after[key]
    return list(merged)


def _append_if_arms_differ(
    out: list[CircuitEntry],
    condition: list[int],
    then_noise: list[stim.CircuitInstruction],
    else_noise: list[stim.CircuitInstruction],
) -> None:
    """Append the noise channels both arms share, then an ``IF`` for the rest.

    The channels follow a moment and are Pauli channels, so they commute and can
    be split per target (per pair, for a two-qubit channel) and regrouped.
    """
    then_pieces, else_pieces = Counter(_pieces(then_noise)), Counter(_pieces(else_noise))
    shared = then_pieces & else_pieces
    out.extend(_regrouped(shared))
    then_body = _regrouped(then_pieces - shared)
    else_body = _regrouped(else_pieces - shared)
    if then_body or else_body:
        out.append(
            IfBlock(
                condition_recs=list(condition),
                then_body=then_body,
                else_body=else_body or None,
            )
        )


_Piece = tuple[str, tuple[float, ...], tuple[int, ...]]


def _pieces(noise: list[stim.CircuitInstruction]) -> Iterator[_Piece]:
    """Split noise channels into one piece per target, or per pair of targets."""
    for inst in noise:
        arity = 2 if stim.gate_data(inst.name).is_two_qubit_gate else 1
        targets = [t.value for t in inst.targets_copy()]
        for i in range(0, len(targets), arity):
            yield inst.name, tuple(inst.gate_args_copy()), tuple(targets[i : i + arity])


def _regrouped(pieces: Counter[_Piece]) -> list[stim.CircuitInstruction]:
    """Rebuild noise channels from pieces, one instruction per channel and argument."""
    grouped: dict[tuple[str, tuple[float, ...]], list[int]] = {}
    for (name, args, targets), count in sorted(pieces.items()):
        grouped.setdefault((name, args), []).extend(list(targets) * count)
    return [
        stim.CircuitInstruction(name, targets, list(args))
        for (name, args), targets in grouped.items()
    ]
