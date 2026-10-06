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
from tqec.utils.position import Position3D

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
