"""Validation tests for the native Y-cube circuit construction.

These exercise the standard surface-code round + detector machinery
independently of the Y transition, by building a plain memory experiment and
checking it is a well-formed, full-distance surface code.
"""

from __future__ import annotations

import stim

import pytest

from tqec.compile.specs.library.generators._ycube_circuit import (
    memory_experiment_circuit,
    standard_round,
    _Builder,
    _bulk_detectors,
    _first_round_detectors,
)
from tqec.compile.specs.library.generators.ycube import xtop_qubit_patch
from tqec.utils.enums import Basis


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_memory_experiment_detectors_are_deterministic(distance: int, basis: Basis) -> None:
    """A noiseless memory experiment must have only deterministic detectors,
    i.e. its detector error model builds without error."""
    circuit = memory_experiment_circuit(distance, rounds=distance, basis=basis)
    # Raises ValueError if any detector/observable is non-deterministic.
    circuit.detector_error_model(decompose_errors=False)


def _memory_experiment_with_logical(distance: int, rounds: int, basis: Basis) -> stim.Circuit:
    patch = xtop_qubit_patch(distance)
    b = _Builder()
    b.allocate(set(patch.data_qubits) | {s.ancilla for s in patch.stabilizers})
    init = {d: basis for d in patch.data_qubits}
    tags = [f"r{i}" for i in range(rounds)]
    standard_round(b, patch, tags[0], init_data_basis=init)
    _first_round_detectors(b, patch, tags[0], init)
    for i in range(1, rounds):
        standard_round(b, patch, tags[i])
        _bulk_detectors(b, patch, tags[i - 1], tags[i])
    final_tag = "final"
    data = sorted(patch.data_qubits, key=lambda p: (p[1], p[0]))
    b.measure(f"M{basis.value}", data, final_tag)
    for s in patch.stabilizers:
        if s.basis != basis:
            continue
        dd = [x for x in s.ordered_data if x is not None]
        b.detector(
            [b.rec(tags[-1], s.ancilla)] + [b.rec(final_tag, x) for x in dd],
            (s.ancilla[0], s.ancilla[1], 1),
        )
    # xtop has X boundaries top/bottom -> X logical is vertical (column x==1);
    # Z boundaries left/right -> Z logical is horizontal (row y==1).
    logical = [q for q in data if q[0] == 1] if basis == Basis.X else [q for q in data if q[1] == 1]
    b.circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(b.rec(final_tag, q) - b.num_measurements) for q in logical],
        0,
    )
    return b.circuit


def _with_depolarizing_noise(circuit: stim.Circuit, p: float = 0.001) -> stim.Circuit:
    all_q = sorted(
        {t.value for inst in circuit for t in inst.targets_copy() if t.is_qubit_target}
    )
    out = stim.Circuit()
    for inst in circuit:
        out.append(inst)
        if inst.name == "TICK":
            out.append("DEPOLARIZE1", all_q, p)
    return out


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_memory_experiment_has_full_code_distance(distance: int, basis: Basis) -> None:
    """The memory experiment must be a genuine distance-``d`` code: the shortest
    graphlike logical error has weight ``d``."""
    circuit = _with_depolarizing_noise(
        _memory_experiment_with_logical(distance, distance, basis)
    )
    circuit.detector_error_model(decompose_errors=False)
    assert len(circuit.shortest_graphlike_error()) == distance
