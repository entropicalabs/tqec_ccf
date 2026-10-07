"""Noise on a ``ConditionalCircuit``: every branch gets the noise it would on its own."""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
import stim

from tests.compile.conditional._conditions import add_condition_source
from tests.compile.conditional.branch_circuit_test import (
    _CONDITIONAL,
    _injection_above_a_conditional_cube,
    _injection_beside_a_conditional_cube,
)
from tests.compile.conditional.branching_tree_test import _branching_tree
from tqec.compile.compile import compile_block_graph
from tqec.compile.conditional.circuit import ConditionalCircuit, IfBlock
from tqec.compile.conditional.noise import noisy_conditional_circuit
from tqec.compile.conditional.resolve import resolve_if_else_by_measurement
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D

_NOISE_MODELS = [
    pytest.param(NoiseModel.uniform_depolarizing(1e-3), id="uniform_depolarizing"),
    pytest.param(
        NoiseModel(
            idle_depolarization=1e-4,
            additional_depolarization_waiting_for_m_or_r=2e-3,
            any_clifford_1q_rule=NoiseModel.si1000(1e-3).any_clifford_1q_rule,
            any_clifford_2q_rule=NoiseModel.si1000(1e-3).any_clifford_2q_rule,
            measure_rules=NoiseModel.uniform_depolarizing(1e-3).measure_rules,
            gate_rules=NoiseModel.uniform_depolarizing(1e-3).gate_rules,
        ),
        id="with_waiting_for_m_or_r",
    ),
    pytest.param(
        NoiseModel(
            idle_depolarization=1e-3,
            additional_depolarization_waiting_for_m_or_r=1e-3,
            any_clifford_1q_rule=NoiseModel.uniform_depolarizing(1e-3).any_clifford_1q_rule,
            any_clifford_2q_rule=NoiseModel.uniform_depolarizing(1e-3).any_clifford_2q_rule,
            measure_rules=NoiseModel.uniform_depolarizing(1e-3).measure_rules,
            gate_rules=NoiseModel.uniform_depolarizing(1e-3).gate_rules,
        ),
        id="two_idle_channels_alike",
    ),
]


def _canonical(circuit: stim.Circuit) -> list[tuple[list[str], Counter[str]]]:
    """Return each moment's operations in order, and its noise channels as a multiset.

    The noise channels that follow a moment act on distinct qubits and commute, so
    their order is not part of the circuit's meaning; each channel is split per
    qubit (per pair, for a two-qubit channel) so that grouping does not matter
    either.
    """
    moments: list[tuple[list[str], Counter[str]]] = [([], Counter())]
    for inst in circuit.flattened():
        if inst.name == "TICK":
            moments.append(([], Counter()))
        elif inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"):
            arity = 2 if inst.name == "DEPOLARIZE2" else 1
            targets = [t.value for t in inst.targets_copy()]
            for i in range(0, len(targets), arity):
                moments[-1][1][f"{inst.name}{inst.gate_args_copy()} {targets[i : i + arity]}"] += 1
        else:
            moments[-1][0].extend(str(piece) for piece in _per_target(inst))
    return moments


def _per_target(inst: stim.CircuitInstruction) -> list[stim.CircuitInstruction]:
    if stim.gate_data(inst.name).is_two_qubit_gate:
        targets = inst.targets_copy()
        return [
            stim.CircuitInstruction(inst.name, targets[i : i + 2], inst.gate_args_copy())
            for i in range(0, len(targets), 2)
        ]
    if stim.gate_data(inst.name).is_unitary or inst.name in ("M", "MX", "MY", "R", "RX", "RY"):
        return [
            stim.CircuitInstruction(inst.name, [t], inst.gate_args_copy())
            for t in inst.targets_copy()
        ]
    return [inst]


def _hand_built() -> ConditionalCircuit:
    """Two moments whose IF arms differ in gates, measurements and touched qubits."""
    c = ConditionalCircuit()
    c.append("R", [0, 1, 2, 3, 4])
    c.append("M", [5])
    c.append("TICK")
    c.append("H", [0])
    c.append_if(
        IfBlock(
            condition_recs=[0],
            then_body=[stim.CircuitInstruction("CX", [1, 2])],
            else_body=[stim.CircuitInstruction("H", [1])],
        )
    )
    c.append_if(IfBlock(condition_recs=[-1], then_body=[stim.CircuitInstruction("S", [3])]))
    c.append("TICK")
    c.append("M", [0])
    c.append_if(
        IfBlock(
            condition_recs=[0],
            then_body=[stim.CircuitInstruction("MX", [1, 2])],
            else_body=[stim.CircuitInstruction("M", [1, 2])],
        )
    )
    # A block of annotations only, on another condition: no noise of its own.
    c.append_if(
        IfBlock(
            condition_recs=[1],
            then_body=[stim.CircuitInstruction("DETECTOR", [stim.target_rec(-1)])],
            else_body=[stim.CircuitInstruction("DETECTOR", [stim.target_rec(-2)])],
        )
    )
    c.append("TICK")
    c.append("R", [3])
    return c


@pytest.mark.parametrize("noise_model", _NOISE_MODELS)
@pytest.mark.parametrize("noiseless_qubits", [None, {1: {0, 1}}])
def test_every_branch_of_the_noisy_circuit_is_the_noisy_branch(
    noise_model: NoiseModel, noiseless_qubits: dict[int, set[int]] | None
) -> None:
    circuit = _hand_built()
    noisy = noisy_conditional_circuit(noise_model, circuit, noiseless_qubits=noiseless_qubits)
    assert noisy.conditions == circuit.conditions
    for bits in itertools.product((0, 1), repeat=len(circuit.conditions)):
        outcomes = dict(zip(circuit.conditions, bits, strict=True))
        expected = noise_model.noisy_circuit(
            circuit.resolve(outcomes), noiseless_qubits=noiseless_qubits
        )
        assert _canonical(noisy.resolve(outcomes)) == _canonical(expected)


def test_noise_the_arms_disagree_on_follows_the_moment_in_an_if() -> None:
    noisy = noisy_conditional_circuit(
        NoiseModel.uniform_depolarizing(0.1), _hand_built()
    ).to_stim_text()
    assert (
        "\n".join(
            [
                "IF(rec[-1]) {",
                "  DEPOLARIZE1(0.1) 3",
                "  DEPOLARIZE2(0.1) 1 2",
                "} ELSE {",
                "  DEPOLARIZE1(0.1) 1",
                "}",
                "DEPOLARIZE1(0.1) 4 5",
                "IF(rec[-1]) {",
                "} ELSE {",
                "  DEPOLARIZE1(0.1) 2 3",
                "}",
            ]
        )
        in noisy
    )


def test_a_qubit_only_one_arm_uses_idles_as_given_by_system_qubits() -> None:
    # Qubit 2 is the highest index and only the IF arm touches it, so the
    # resolved ELSE branch does not know it: pass the qubits explicitly.
    c = ConditionalCircuit()
    c.append("M", [0, 1])
    c.append("TICK")
    c.append_if(IfBlock(condition_recs=[1, 0], then_body=[stim.CircuitInstruction("H", [2])]))
    c.append("H", [1])
    noise_model = NoiseModel.uniform_depolarizing(1e-3)
    for system_qubits in ({0, 1, 2}, {0, 1}):
        noisy = noisy_conditional_circuit(noise_model, c, system_qubits=system_qubits)
        for bit in (0, 1):
            outcomes = {(0, 1): bit}
            expected = noise_model.noisy_circuit(c.resolve(outcomes), system_qubits=system_qubits)
            assert _canonical(noisy.resolve(outcomes)) == _canonical(expected)


def test_the_noise_if_keeps_the_order_of_its_block_condition() -> None:
    # A text reader keys a block by the first measurement its condition names,
    # so the noise IF must name them in the same order as the block it follows.
    c = ConditionalCircuit()
    c.append("M", [0, 1, 2])
    c.append("TICK")
    c.append_if(
        IfBlock(
            condition_recs=[2, 0],
            then_body=[stim.CircuitInstruction("CX", [3, 4])],
            else_body=[stim.CircuitInstruction("H", [3])],
        )
    )
    noise_model = NoiseModel.uniform_depolarizing(1e-3)
    qubits = set(range(5))
    text = noisy_conditional_circuit(noise_model, c, system_qubits=qubits).to_stim_text()
    assert text.count("IF(rec[-1]^rec[-3])") == 3
    for bit in (0, 1):
        resolved = resolve_if_else_by_measurement(text, {2: bit})
        expected = noise_model.noisy_circuit(c.resolve({(0, 2): bit}), system_qubits=qubits)
        assert _canonical(resolved) == _canonical(expected)


def test_an_if_block_inside_an_arm_acting_on_qubits_is_rejected() -> None:
    c = ConditionalCircuit()
    c.append("M", [0, 1])
    inner = IfBlock(condition_recs=[1], then_body=[stim.CircuitInstruction("X", [3])])
    c.append_if(IfBlock(condition_recs=[0], then_body=[stim.CircuitInstruction("H", [2]), inner]))
    with pytest.raises(NotImplementedError, match="holds another IF block"):
        noisy_conditional_circuit(NoiseModel.uniform_depolarizing(1e-3), c)


def test_a_tick_inside_an_if_block_is_rejected() -> None:
    c = ConditionalCircuit()
    c.append("M", [0])
    c.append_if(
        IfBlock(
            condition_recs=[0],
            then_body=[stim.CircuitInstruction("H", [1]), stim.CircuitInstruction("TICK", [])],
        )
    )
    with pytest.raises(NotImplementedError, match="TICK"):
        noisy_conditional_circuit(NoiseModel.uniform_depolarizing(1e-3), c)


def test_two_conditions_acting_on_qubits_in_one_moment_are_rejected() -> None:
    c = ConditionalCircuit()
    c.append("M", [0, 1])
    c.append_if(IfBlock(condition_recs=[0], then_body=[stim.CircuitInstruction("H", [2])]))
    c.append_if(IfBlock(condition_recs=[1], then_body=[stim.CircuitInstruction("H", [3])]))
    with pytest.raises(NotImplementedError, match="different conditions"):
        noisy_conditional_circuit(NoiseModel.uniform_depolarizing(1e-3), c)


def _single_conditional_graph(pair_name: str) -> BlockGraph:
    g = BlockGraph(f"Conditional {pair_name}")
    p0, p1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    g.add_cube(p0, ConditionalLeafCubeKind[pair_name].value[0])
    g.add_cube(p1, ConditionalLeafCubeKind[pair_name], condition=add_condition_source(g))
    g.add_pipe(p0, p1)
    return g


@pytest.mark.parametrize(
    "graph",
    [
        *(
            pytest.param(_single_conditional_graph(name), id=name)
            for name in ("XZZ_XZX", "ZXX_ZXZ", "XZX_XZZ", "ZXZ_ZXX")
        ),
        pytest.param(_branching_tree(2), id="branching_tree_2"),
    ],
)
def test_noisy_stim_text_resolves_to_each_noisy_branch_circuit(graph: BlockGraph) -> None:
    noise_model = NoiseModel.uniform_depolarizing(1e-3)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    text = compiled.generate_stim_text(1, noise_model=noise_model)
    _, cube_conditions, _ = compiled._compile_conditional(1)
    cubes = sorted(cube_conditions, key=lambda p: p.z)
    for bits in itertools.product((0, 1), repeat=len(cubes)):
        outcomes = {min(cube_conditions[c]): b for c, b in zip(cubes, bits, strict=True)}
        branches = {compiled._block_graph_position(c): b for c, b in zip(cubes, bits, strict=True)}
        resolved = resolve_if_else_by_measurement(text, outcomes)
        expected = compiled.generate_branch_circuit(1, branches, noise_model=noise_model)
        assert _canonical(resolved) == _canonical(expected)
        assert resolved.detector_error_model() == expected.detector_error_model()


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("branch", [0, 1])
@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(_injection_beside_a_conditional_cube(), id="beside"),
        pytest.param(_injection_above_a_conditional_cube(), id="above"),
    ],
)
def test_noiseless_injection_in_noisy_stim_text(graph: BlockGraph, branch: int, k: int) -> None:
    noise_model = NoiseModel.uniform_depolarizing(1e-3)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    text = compiled.generate_stim_text(k, noise_model=noise_model, noiseless_injection=True)
    (condition,) = compiled._compile_conditional(k)[1].values()
    resolved = resolve_if_else_by_measurement(text, {min(condition): branch})
    expected = compiled.generate_branch_circuit(
        k, {_CONDITIONAL: branch}, noise_model=noise_model, noiseless_injection=True
    )
    assert _canonical(resolved) == _canonical(expected)


def test_noisy_stim_text_keeps_the_non_clifford_gate() -> None:
    graph = _injection_beside_a_conditional_cube(state="T")
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    lines = compiled.generate_stim_text(
        1, noise_model=NoiseModel.uniform_depolarizing(1e-3)
    ).splitlines()
    (index,) = [i for i, line in enumerate(lines) if line.split()[:1] == ["T"]]
    assert lines[index + 1].startswith("DEPOLARIZE1(0.001)")
    assert not any("S[T]" in line for line in lines)
