"""``TopologicalComputationGraph.generate_branch_circuit``: one branch as a stim circuit."""

from __future__ import annotations

import pytest

from tests.compile.conditional.resolve_test import _assert_circuits_equivalent_modulo_detector_order
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.compile.graph import TopologicalComputationGraph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import BlockPosition3D, Position3D

_CONDITIONAL = Position3D(1, 0, 2)


def _injection_beside_a_conditional_cube(state: str = "+") -> BlockGraph:
    """Build an injection column beside a column ending in a conditional cube."""
    graph = BlockGraph("Injection beside a conditional cube")
    graph.add_cube(Position3D(0, 0, 0), "I", state=state)
    graph.add_cube(Position3D(0, 0, 1), "ZXZ")
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    graph.add_cube(Position3D(1, 0, 0), "ZXZ")
    graph.add_cube(Position3D(1, 0, 1), "ZXZ")
    graph.add_pipe(Position3D(1, 0, 0), Position3D(1, 0, 1))
    graph.add_cube(Position3D(2, 0, 1), "ZXZ")  # a lone cube the condition reads
    lone = ZXNode(Position3D(2, 0, 1), Basis.Z)
    condition = CorrelationSurface(span=frozenset([ZXEdge(lone, lone)]))
    graph.add_cube(_CONDITIONAL, ConditionalLeafCubeKind.ZXZ_ZXX, condition=condition)
    graph.add_pipe(Position3D(1, 0, 1), _CONDITIONAL)
    return graph


def _branch_compiled_on_its_own(graph: BlockGraph, branch: int) -> TopologicalComputationGraph:
    """Compile ``graph`` with its conditional cube replaced by the kind of ``branch``."""
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    ((position, block),) = compiled._conditional_blocks.items()
    compiled._blocks[position] = block.block_if_one if branch else block.block_if_zero
    compiled._conditional_blocks = {}
    return compiled


@pytest.mark.parametrize("branch", [0, 1])
@pytest.mark.parametrize("dz", [0, 3])
def test_branch_circuit_keyed_by_block_graph_position(branch: int, dz: int) -> None:
    # A graph that does not start at z = 0 is shifted down when compiled; the
    # branches are still keyed by the positions the user built it with.
    graph = _injection_beside_a_conditional_cube().shift_by(dz=dz)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    circuit = compiled.generate_branch_circuit(1, {_CONDITIONAL.shift_by(dz=dz): branch})
    reference = _branch_compiled_on_its_own(graph, branch).generate_stim_circuit(1)
    _assert_circuits_equivalent_modulo_detector_order(circuit.flattened(), reference.flattened())


@pytest.mark.parametrize("branch", [0, 1])
def test_noiseless_injection_in_a_branch_circuit(branch: int) -> None:
    graph = _injection_beside_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    circuit = compiled.generate_branch_circuit(
        1, {_CONDITIONAL: branch}, noise_model=noise, noiseless_injection=True
    )
    reference = _branch_compiled_on_its_own(graph, branch).generate_stim_circuit(
        1, noise_model=noise, noiseless_injection=True
    )
    _assert_circuits_equivalent_modulo_detector_order(circuit.flattened(), reference.flattened())
    noisy = compiled.generate_branch_circuit(1, {_CONDITIONAL: branch}, noise_model=noise)
    assert str(circuit).count("DEPOLARIZE") < str(noisy).count("DEPOLARIZE")


def _injection_above_a_conditional_cube(state: str = "+") -> BlockGraph:
    """Build a graph whose injection encoder runs after the conditional cube."""
    graph = _injection_beside_a_conditional_cube(state)
    top = Position3D(0, 0, 4)
    graph.add_cube(Position3D(0, 0, 3), "I", state=state)
    graph.add_cube(top, "ZXZ")
    graph.add_pipe(Position3D(0, 0, 3), top)
    return graph


def test_noiseless_injection_after_a_conditional_cube_is_not_supported() -> None:
    # A repeated round precedes the encoder, as on the plain path.
    graph = _injection_above_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(NotImplementedError, match="repeated round precedes it"):
        compiled.generate_branch_circuit(
            1,
            {_CONDITIONAL: 0},
            noise_model=NoiseModel.uniform_depolarizing(1e-3),
            noiseless_injection=True,
        )


def test_two_cubes_reading_the_same_measurements_take_the_same_branch() -> None:
    graph = _injection_beside_a_conditional_cube()
    # A second column, whose conditional cube reads the same lone cube.
    column = [Position3D(4, 0, z) for z in range(3)]
    for position in column:
        graph.add_cube(position, "ZXZ")
    graph.add_pipe(column[0], column[1])
    graph.add_pipe(column[1], column[2])
    second = Position3D(4, 0, 3)
    lone = ZXNode(Position3D(2, 0, 1), Basis.Z)
    condition = CorrelationSurface(span=frozenset([ZXEdge(lone, lone)]))
    graph.add_cube(second, ConditionalLeafCubeKind.ZXZ_ZXX, condition=condition)
    graph.add_pipe(column[2], second)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    compiled.generate_branch_circuit(1, {_CONDITIONAL: 1, second: 1})
    with pytest.raises(TQECError, match="same branch"):
        compiled.generate_branch_circuit(1, {_CONDITIONAL: 1, second: 0})


def test_a_cube_given_a_branch_twice_is_rejected() -> None:
    graph = _injection_beside_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    as_block_position = BlockPosition3D(_CONDITIONAL.x, _CONDITIONAL.y, _CONDITIONAL.z)
    with pytest.raises(TQECError, match="twice"):
        compiled.generate_branch_circuit(1, {_CONDITIONAL: 0, as_block_position: 1})


@pytest.mark.parametrize(
    ("branches", "match"),
    [
        ({}, "no branch given"),
        ({_CONDITIONAL: 2}, "must be 0 or 1"),
        ({_CONDITIONAL: 0, Position3D(1, 0, 1): 0}, "no conditional cube at"),
    ],
)
def test_branch_circuit_rejects_bad_branches(branches: dict[Position3D, int], match: str) -> None:
    graph = _injection_beside_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match=match):
        compiled.generate_branch_circuit(1, branches)


def test_branch_circuit_needs_a_conditional_cube() -> None:
    graph = BlockGraph("Memory")
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match="generate_stim_circuit"):
        compiled.generate_branch_circuit(1, {})


def test_branch_circuit_refuses_a_non_clifford_state() -> None:
    graph = _injection_beside_a_conditional_cube(state="T")
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match="generate_stim_text"):
        compiled.generate_branch_circuit(1, {_CONDITIONAL: 0})
