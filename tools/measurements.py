"""Map measurement records of a vanilla ``stim.Circuit`` to qubit coord + tick.

Given a resolved (branch-free) ``stim.Circuit``, walk it once to build a table
mapping each measurement index to the coordinate of the measured qubit
(``QUBIT_COORDS``) and the tick (number of ``TICK`` instructions seen so far).
A relative ``rec[-k]`` reference at some point resolves to the absolute index
``num_measurements_seen - k``, then into that table.

To inspect an ``IF/ELSE``-annotated circuit, first pick a branch with
:func:`tools.resolve.resolve_if_else` to obtain a vanilla circuit, then use these
helpers. Working on a resolved circuit avoids the double-counting pitfall of
counting both arms of an ``IF/ELSE`` block.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import stim


@dataclass(frozen=True)
class MeasurementInfo:
    """Where/when a single measurement record was produced.

    Attributes:
        coord: the coordinate of the measured qubit, from the matching
            ``QUBIT_COORDS`` instruction, or ``None`` if none was declared.
            For a multi-qubit product measurement (``MPP``) this is the first
            qubit of the product.
        tick: the number of ``TICK`` instructions emitted before the
            measurement -- a temporal index for the measurement.

    """

    coord: tuple[float, ...] | None
    tick: int


def _records_with_coords(targets, coords):
    """Yield the coord for each measurement record produced by ``targets``.

    Combiner targets (``*``) join qubits into one product measurement, which
    produces a single record attributed to the first qubit of the product.
    """
    prev_combiner = False
    for t in targets:
        if t.is_combiner:
            prev_combiner = True
            continue
        if not prev_combiner:
            yield coords.get(t.qubit_value)
        prev_combiner = False


def build_measurement_map(circuit: stim.Circuit) -> list[MeasurementInfo]:
    """Return ``MeasurementInfo`` for every measurement record, in order.

    Args:
        circuit: a vanilla (branch-free) ``stim.Circuit``. ``REPEAT`` blocks are
            expanded automatically.

    Returns:
        A list indexed by measurement record number.

    """
    coords: dict[int, tuple[float, ...]] = {}
    tick = 0
    out: list[MeasurementInfo] = []
    for inst in circuit.flattened():
        name = inst.name
        if name == "QUBIT_COORDS":
            args = tuple(inst.gate_args_copy())
            for t in inst.targets_copy():
                coords[t.qubit_value] = args
        elif name == "TICK":
            tick += 1
        elif stim.gate_data(name).produces_measurements:
            out.extend(
                MeasurementInfo(coord, tick)
                for coord in _records_with_coords(inst.targets_copy(), coords)
            )
    return out


def locate_record(
    measurement_map: list[MeasurementInfo],
    num_measurements_before: int,
    rec_offset: int,
) -> MeasurementInfo:
    """Resolve a relative ``rec[rec_offset]`` reference to a ``MeasurementInfo``.

    Args:
        measurement_map: output of :func:`build_measurement_map`.
        num_measurements_before: number of measurement records emitted before
            the referencing instruction (its measurement frame).
        rec_offset: the (negative) offset inside ``rec[...]``.

    Returns:
        The ``MeasurementInfo`` for the referenced measurement.

    """
    idx = num_measurements_before + rec_offset
    if not 0 <= idx < len(measurement_map):
        raise IndexError(
            f"rec[{rec_offset}] at frame {num_measurements_before} -> index "
            f"{idx} is out of range [0, {len(measurement_map)})."
        )
    return measurement_map[idx]


def observable_records(
    circuit: stim.Circuit, observable_index: int = 0
) -> list[MeasurementInfo]:
    """Return the net measurements of an observable in a vanilla circuit.

    Accumulates every ``OBSERVABLE_INCLUDE(observable_index)`` in the circuit and
    XORs them (a measurement included an even number of times cancels), then maps
    the surviving records to ``MeasurementInfo``.

    Args:
        circuit: a vanilla (branch-free) ``stim.Circuit``.
        observable_index: the observable id to extract (the ``OBSERVABLE_INCLUDE``
            argument).

    Returns:
        The ``MeasurementInfo`` of each measurement in the observable's support.

    """
    coords: dict[int, tuple[float, ...]] = {}
    tick = 0
    mmap: list[MeasurementInfo] = []
    counts: Counter[int] = Counter()
    for inst in circuit.flattened():
        name = inst.name
        if name == "QUBIT_COORDS":
            args = tuple(inst.gate_args_copy())
            for t in inst.targets_copy():
                coords[t.qubit_value] = args
        elif name == "TICK":
            tick += 1
        elif name == "OBSERVABLE_INCLUDE":
            if int(inst.gate_args_copy()[0]) == observable_index:
                for t in inst.targets_copy():
                    if t.is_measurement_record_target:
                        counts[len(mmap) + t.value] += 1
        elif stim.gate_data(name).produces_measurements:
            mmap.extend(
                MeasurementInfo(coord, tick)
                for coord in _records_with_coords(inst.targets_copy(), coords)
            )
    return [mmap[i] for i in sorted(i for i, n in counts.items() if n % 2 == 1)]
