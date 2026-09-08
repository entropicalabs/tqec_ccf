"""Tests for the state-injection encoder.

The encoder is cross-checked against the vendored ``noncliff`` implementation
(``tests/_vendor/noncliff``), which is the construction it ports, and against the
patch geometry tqec's own memory rounds use.
"""

from __future__ import annotations

import pytest
import stim

from tests._vendor.noncliff.injection import injection as oracle_injection
from tqec.compile.specs.library.generators.injection import (
    INJECTION_ENCODER_MOMENTS,
    injection_encoder_circuit,
)
from tqec.compile.specs.library.generators.ycube import xtop_qubit_patch
from tqec.utils.exceptions import TQECError

_DISTANCES = (3, 5, 7, 9)

_Coord = tuple[int, int]
_Moment = dict[str, frozenset[tuple[_Coord, ...]]]


def _moments(circuit: stim.Circuit) -> list[_Moment]:
    """Split ``circuit`` into moments of coordinate-addressed operands.

    Each moment maps an instruction name to the set of operand tuples it acts
    on: a one-element tuple per single-qubit target, and an ordered
    ``(control, target)`` pair per two-qubit gate. Empty moments are dropped, so
    the comparison is insensitive to where a circuit chooses to put its ``TICK``
    boundaries relative to its ``QUBIT_COORDS`` block.
    """
    coords = {
        qubit: (round(c[0]), round(c[1]))
        for qubit, c in circuit.get_final_qubit_coordinates().items()
    }
    moments: list[_Moment] = []
    current: dict[str, set[tuple[_Coord, ...]]] = {}
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            if current:
                moments.append({name: frozenset(ops) for name, ops in current.items()})
            current = {}
            continue
        if instruction.name == "QUBIT_COORDS":
            continue
        targets = [coords[t.qubit_value] for t in instruction.targets_copy()]
        step = 2 if stim.gate_data(instruction.name).is_two_qubit_gate else 1
        operands = {tuple(targets[i : i + step]) for i in range(0, len(targets), step)}
        current.setdefault(instruction.name, set()).update(operands)
    if current:
        moments.append({name: frozenset(ops) for name, ops in current.items()})
    return moments


def _patch_qubits(distance: int) -> set[_Coord]:
    patch = xtop_qubit_patch(distance)
    return set(patch.data_qubits) | {s.ancilla for s in patch.stabilizers}


def _logical_operator(distance: int, basis: str, transposed: bool) -> list[tuple[_Coord, str]]:
    """Return the logical ``X`` column or ``Z`` row through the centre of the patch."""
    data = xtop_qubit_patch(distance).data_qubits
    axis = 0 if basis == "X" else 1
    line = [c for c in data if c[axis] == distance]
    if transposed:
        line = [(y, x) for x, y in line]
    return [(c, basis) for c in line]


def _expectation(circuit: stim.Circuit, operator: list[tuple[_Coord, str]]) -> float:
    """Return the expectation of ``operator`` on the state ``circuit`` prepares."""
    coords = {
        (round(c[0]), round(c[1])): qubit
        for qubit, c in circuit.get_final_qubit_coordinates().items()
    }
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    pauli = stim.PauliString(simulator.num_qubits)
    for coord, basis in operator:
        pauli[coords[coord]] = basis
    return simulator.peek_observable_expectation(pauli)


def _without_proxy(circuit: stim.Circuit) -> stim.Circuit:
    """Return ``circuit`` without the proxy gate, so it prepares a plain ``|+>``."""
    stripped = stim.Circuit()
    for instruction in circuit.flattened():
        if instruction.name != "S_DAG":
            stripped.append(instruction)
    return stripped


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_matches_oracle(distance: int) -> None:
    assert _moments(injection_encoder_circuit(distance)) == _moments(oracle_injection(distance))


@pytest.mark.parametrize("distance", _DISTANCES)
@pytest.mark.parametrize("transposed", [False, True])
def test_encoder_prepares_every_stabilizer(distance: int, transposed: bool) -> None:
    circuit = injection_encoder_circuit(distance, transposed=transposed)
    for stabilizer in xtop_qubit_patch(distance, transposed).stabilizers:
        support = [c for c in stabilizer.ordered_data if c is not None]
        operator = [(c, stabilizer.basis.value) for c in support]
        assert _expectation(circuit, operator) == 1.0, (
            f"stabilizer at {stabilizer.ancilla} is not prepared in its +1 eigenstate"
        )


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_declares_the_whole_patch(distance: int) -> None:
    circuit = injection_encoder_circuit(distance)
    declared = {(round(c[0]), round(c[1])) for c in circuit.get_final_qubit_coordinates().values()}
    assert declared == _patch_qubits(distance)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_only_touches_data_qubits(distance: int) -> None:
    circuit = injection_encoder_circuit(distance)
    coords = {
        qubit: (round(c[0]), round(c[1]))
        for qubit, c in circuit.get_final_qubit_coordinates().items()
    }
    touched = {
        coords[target.qubit_value]
        for instruction in circuit.flattened()
        if instruction.name not in ("QUBIT_COORDS", "TICK")
        for target in instruction.targets_copy()
    }
    assert touched == set(xtop_qubit_patch(distance).data_qubits)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_performs_no_measurement(distance: int) -> None:
    assert injection_encoder_circuit(distance).num_measurements == 0


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_moment_count(distance: int) -> None:
    circuit = injection_encoder_circuit(distance)
    k = (distance - 1) // 2
    assert circuit.num_ticks + 1 == INJECTION_ENCODER_MOMENTS.integer_eval(k)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_transposed_encoder_is_the_reflected_encoder(distance: int) -> None:
    reflected = [
        {
            name: frozenset(tuple((y, x) for x, y in operand) for operand in operands)
            for name, operands in moment.items()
        }
        for moment in _moments(injection_encoder_circuit(distance))
    ]
    assert _moments(injection_encoder_circuit(distance, transposed=True)) == reflected


@pytest.mark.parametrize("distance", _DISTANCES)
@pytest.mark.parametrize("transposed", [False, True])
def test_encoder_injects_the_proxy_state(distance: int, transposed: bool) -> None:
    # S_DAG |+> = |-i>, which has no X or Z component and is a Y eigenstate.
    circuit = injection_encoder_circuit(distance, transposed=transposed)
    logical_x = _logical_operator(distance, "X", transposed)
    logical_z = _logical_operator(distance, "Z", transposed)
    assert _expectation(circuit, logical_x) == 0.0
    assert _expectation(circuit, logical_z) == 0.0
    # Without the proxy gate the same encoder prepares the logical |+>.
    assert _expectation(_without_proxy(circuit), logical_x) == 1.0


@pytest.mark.parametrize("distance", _DISTANCES)
def test_proxy_gate_acts_on_the_centre_data_qubit(distance: int) -> None:
    circuit = injection_encoder_circuit(distance)
    coords = circuit.get_final_qubit_coordinates()
    proxied = [
        coords[target.qubit_value]
        for instruction in circuit.flattened()
        if instruction.name == "S_DAG"
        for target in instruction.targets_copy()
    ]
    assert proxied == [[float(distance), float(distance)]]


def test_proxy_false_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="stim does not support"):
        injection_encoder_circuit(3, proxy=False)


@pytest.mark.parametrize("distance", [-1, 0, 2, 4, 6])
def test_invalid_distance_is_rejected(distance: int) -> None:
    with pytest.raises(TQECError, match="odd and at least 3"):
        injection_encoder_circuit(distance)
