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
from tqec.utils.injection_state import INJECTION_STATES

_DISTANCES = (3, 5, 7, 9)
_STATES = tuple(INJECTION_STATES)

# The logical expectation each state prepares, as (basis, value). "T"/"T_DAG"
# match "i"/"-i" because they compile as the same Clifford stand-in.
_LOGICAL_EXPECTATION: dict[str, tuple[str, float]] = {
    "0": ("Z", 1.0),
    "1": ("Z", -1.0),
    "+": ("X", 1.0),
    "-": ("X", -1.0),
    "i": ("Y", 1.0),
    "-i": ("Y", -1.0),
    "T": ("Y", 1.0),
    "T_DAG": ("Y", -1.0),
}

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


def _logical_pauli(distance: int, basis: str, transposed: bool) -> stim.PauliString:
    """Return a logical ``X``, ``Z`` or ``Y`` representative as a Pauli string."""
    circuit = injection_encoder_circuit(distance, transposed=transposed)
    coords = {
        (round(c[0]), round(c[1])): qubit
        for qubit, c in circuit.get_final_qubit_coordinates().items()
    }

    def pauli(operator: list[tuple[_Coord, str]]) -> stim.PauliString:
        out = stim.PauliString(circuit.num_qubits)
        for coord, single in operator:
            out[coords[coord]] = single
        return out

    if basis in ("X", "Z"):
        return pauli(_logical_operator(distance, basis, transposed))
    # X_L and Z_L overlap only on the centre qubit, so their product is a logical
    # Y there; the factor of i makes the anticommuting product Hermitian.
    return (
        pauli(_logical_operator(distance, "X", transposed))
        * pauli(_logical_operator(distance, "Z", transposed))
        * 1j
    )


@pytest.mark.parametrize("distance", _DISTANCES)
def test_encoder_matches_oracle(distance: int) -> None:
    # The oracle applies S_DAG to the centre qubit, which is the "-i" state.
    assert _moments(injection_encoder_circuit(distance, state="-i")) == _moments(
        oracle_injection(distance)
    )


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
@pytest.mark.parametrize("state", _STATES)
def test_encoder_moment_count_is_state_independent(distance: int, state: str) -> None:
    # The gate moment is emitted even for the states needing no gate, as ``I``,
    # so ``scalable_num_moments`` can stay a constant.
    circuit = injection_encoder_circuit(distance, state=state)
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
@pytest.mark.parametrize("state", _STATES)
def test_encoder_injects_the_requested_state(distance: int, transposed: bool, state: str) -> None:
    circuit = injection_encoder_circuit(distance, transposed=transposed, state=state)
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    basis, value = _LOGICAL_EXPECTATION[state]
    assert simulator.peek_observable_expectation(
        _logical_pauli(distance, basis, transposed)
    ) == pytest.approx(value)
    # The two complementary logical operators are maximally uncertain.
    for other in ("X", "Y", "Z"):
        if other != basis:
            assert simulator.peek_observable_expectation(
                _logical_pauli(distance, other, transposed)
            ) == pytest.approx(0.0)


@pytest.mark.parametrize("distance", _DISTANCES)
@pytest.mark.parametrize("state", _STATES)
def test_reset_and_gate_act_on_the_centre_data_qubit(distance: int, state: str) -> None:
    reset, gate, _ = INJECTION_STATES[state]
    circuit = injection_encoder_circuit(distance, state=state)
    coords = circuit.get_final_qubit_coordinates()
    centre = [float(distance), float(distance)]
    instructions = [i for i in circuit.flattened() if i.name not in ("QUBIT_COORDS", "TICK")]
    # The encoder opens by preparing the centre qubit alone, then rotating it,
    # before any other qubit is touched.
    assert instructions[0].name == reset
    assert [coords[t.qubit_value] for t in instructions[0].targets_copy()] == [centre]
    assert instructions[1].name == gate
    assert [coords[t.qubit_value] for t in instructions[1].targets_copy()] == [centre]


@pytest.mark.parametrize("distance", _DISTANCES)
@pytest.mark.parametrize("state", _STATES)
def test_only_a_non_clifford_state_is_tagged(distance: int, state: str) -> None:
    stands_in_for = INJECTION_STATES[state][2]
    circuit = injection_encoder_circuit(distance, state=state)
    tags = [i.tag for i in circuit.flattened() if i.tag]
    assert tags == ([stands_in_for] if stands_in_for else [])


@pytest.mark.parametrize("state", ["T", "T_DAG"])
def test_a_tagged_encoder_round_trips_through_stim_text(state: str) -> None:
    # The tag is what the text emitter keys on, so it has to survive stim's own
    # serialisation.
    circuit = injection_encoder_circuit(5, state=state)
    assert stim.Circuit(str(circuit)) == circuit


@pytest.mark.parametrize("state", ["Q", "", "I", "t", "t_dag", "0i", "+i"])
def test_unknown_state_is_rejected(state: str) -> None:
    # Matching is exact and case-sensitive on purpose: "I" is the INJECTION cube
    # *kind*, so it must not also name a state.
    with pytest.raises(TQECError, match="Unknown injected state"):
        injection_encoder_circuit(3, state=state)


@pytest.mark.parametrize("distance", [-1, 0, 2, 4, 6])
def test_invalid_distance_is_rejected(distance: int) -> None:
    with pytest.raises(TQECError, match="odd and at least 3"):
        injection_encoder_circuit(distance)
