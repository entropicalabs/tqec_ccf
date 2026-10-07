"""``TopologicalComputationGraph.generate_branch_circuit``: one branch as a stim circuit."""

from __future__ import annotations

import itertools

import pytest

from tests.compile.conditional._conditions import add_condition_source
from tests.compile.conditional.conditional_observable_test import (
    _build_graph as _conditional_observable_graph,
)
from tests.compile.conditional.resolve_test import _assert_circuits_equivalent_modulo_detector_order
from tqec.compile.blocks.positioning import LayoutPosition3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.conditional.resolve import resolve_if_else_by_measurement
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.compile.graph import TopologicalComputationGraph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import (
    ConditionalCorrelationSurface,
    CorrelationSurface,
    ZXEdge,
    ZXNode,
)
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


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("branch", [0, 1])
def test_noiseless_injection_after_a_conditional_cube_and_a_repeated_round(
    k: int, branch: int
) -> None:
    # The branch circuit writes every repetition out, so the encoder's moments
    # are counted past each of them. A noisy encoder makes the column distance
    # 1; exempting it restores 2k + 1, which no misplaced moment index does.
    graph = _injection_above_a_conditional_cube(state="0")
    top = BlockGraph("The column above the conditional cube")
    top.add_cube(Position3D(0, 0, 3), "I", state="0")
    top.add_cube(Position3D(0, 0, 4), "ZXZ")
    top.add_pipe(Position3D(0, 0, 3), Position3D(0, 0, 4))
    compiled = compile_block_graph(
        graph, FIXED_BULK_CONVENTION, observables=top.find_correlation_surfaces()
    )
    noise = NoiseModel.uniform_depolarizing(1e-3)
    for noiseless, distance in ((False, 1), (True, 2 * k + 1)):
        circuit = compiled.generate_branch_circuit(
            k, {_CONDITIONAL: branch}, noise_model=noise, noiseless_injection=noiseless
        )
        error = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=False)
        assert len(error) == distance


def _add_conditional_column(
    graph: BlockGraph, x: int, top: int, lone: list[Position3D]
) -> Position3D:
    """Add a column at ``x`` ending at ``top`` in a cube conditioned on the ``lone`` cubes."""
    column = [Position3D(x, 0, z) for z in range(top)]
    for position in column:
        graph.add_cube(position, "ZXZ")
    for below, above in itertools.pairwise(column):
        graph.add_pipe(below, above)
    nodes = [ZXNode(p, Basis.Z) for p in lone]
    condition = CorrelationSurface(span=frozenset(ZXEdge(n, n) for n in nodes))
    conditional = Position3D(x, 0, top)
    graph.add_cube(conditional, ConditionalLeafCubeKind.ZXZ_ZXX, condition=condition)
    graph.add_pipe(column[-1], conditional)
    return conditional


def test_two_cubes_reading_the_same_measurements_take_the_same_branch() -> None:
    graph = _injection_beside_a_conditional_cube()
    second = _add_conditional_column(graph, 4, 3, [Position3D(2, 0, 1)])
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    compiled.generate_branch_circuit(1, {_CONDITIONAL: 1, second: 1})
    with pytest.raises(TQECError, match=r"XOR of the conditions of the cube\(s\) at \(1,0,2\)"):
        compiled.generate_branch_circuit(1, {_CONDITIONAL: 1, second: 0})


def test_a_condition_that_is_the_xor_of_two_others_takes_the_xor_of_their_outcomes() -> None:
    compiled = compile_block_graph(
        _injection_beside_a_conditional_cube(), FIXED_BULK_CONVENTION, observables=None
    )
    a, b, c = (LayoutPosition3D.from_block_position(BlockPosition3D(x, 0, x)) for x in (1, 2, 3))
    cube_conditions = {a: [3], b: [5, 7], c: [3, 5, 7]}
    observable = (3, 5, 7)  # read by c, and by a and b together
    for bit_a, bit_b in itertools.product((0, 1), repeat=2):
        outcomes = compiled._condition_outcomes(
            cube_conditions, {a: bit_a, b: bit_b, c: bit_a ^ bit_b}, [(3,), observable]
        )
        assert outcomes == {(3,): bit_a, observable: bit_a ^ bit_b}
        with pytest.raises(
            TQECError, match=rf"cube\(s\) at \(1,0,1\), \(2,0,2\), .* theirs, {bit_a ^ bit_b}"
        ):
            compiled._condition_outcomes(
                cube_conditions, {a: bit_a, b: bit_b, c: 1 - (bit_a ^ bit_b)}, []
            )
    with pytest.raises(TQECError, match=r"\[5\], which are not an XOR"):
        compiled._condition_outcomes(cube_conditions, {a: 0, b: 0, c: 0}, [(5,)])


def test_branch_circuits_compile_once_for_the_same_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _injection_beside_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    branches = [{_CONDITIONAL: 0}, {_CONDITIONAL: 1}, {_CONDITIONAL: 0}]
    alone = [compiled.generate_branch_circuit(1, b, noise_model=noise) for b in branches]

    compile_conditional = compiled._compile_conditional
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return compile_conditional(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(compiled, "_compile_conditional", counted)
    together = compiled.generate_branch_circuits(1, branches, noise_model=noise)
    assert len(calls) == 1
    assert together == alone
    assert together[0] != together[1]
    assert compiled.generate_branch_circuits(1, []) == []
    assert len(calls) == 1


def test_a_branch_keyed_by_anything_but_a_position_is_rejected() -> None:
    graph = _injection_beside_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match="Position3D"):
        compiled.generate_branch_circuit(1, {(1, 0, 2): 0})  # type: ignore[dict-item]


def test_a_conditional_observable_resolves_in_each_branch_circuit() -> None:
    # Its condition is the conditional cube's: each branch fixes it, and keeps
    # the observable deterministic.
    graph, observable = _conditional_observable_graph()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=[observable])
    text = compiled.generate_stim_text(1)
    ((position, condition),) = compiled._compile_conditional(1)[1].items()
    cube = compiled._block_graph_position(position)
    for branch in (0, 1):
        circuit = compiled.generate_branch_circuit(1, {cube: branch})
        assert circuit.num_observables == 1
        resolved = resolve_if_else_by_measurement(text, {min(condition): branch})
        assert circuit.detector_error_model() == resolved.detector_error_model()


def test_a_conditional_observable_without_a_conditional_cube_points_to_stim_text() -> None:
    # No cube fixes the observable's condition, and generate_stim_circuit
    # would leave the observable out, so the error must not send users there.
    graph, observable = _surface_anchored_conditional_observable_graph()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=[observable])
    with pytest.raises(TQECError, match="use generate_stim_text"):
        compiled.generate_branch_circuit(1, {})


def test_a_moment_count_mismatch_raises_instead_of_exempting_the_wrong_moments() -> None:
    graph = _injection_above_a_conditional_cube()
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION, observables=None)
    circuit, _, tree = compiled._compile_conditional(2)
    moments = 1 + sum(1 for e in circuit.entries if getattr(e, "name", None) == "TICK")
    assert compiled._injection_noiseless_qubits(tree, 2, flattened=True, moments=moments)
    with pytest.raises(TQECError, match="accounts for"):
        compiled._injection_noiseless_qubits(tree, 2, flattened=True, moments=moments + 1)


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


def _surface_anchored_conditional_observable_graph() -> tuple[
    BlockGraph, ConditionalCorrelationSurface
]:
    """Build a plain graph whose observable is conditioned on a surface no cube carries."""
    b1, b2, b3, c = (
        Position3D(0, 0, 0),
        Position3D(0, 0, 1),
        Position3D(0, 0, 2),
        Position3D(1, 0, 2),
    )
    graph = BlockGraph("surface_anchored")
    for position in (b1, b2, b3, c):
        graph.add_cube(position, "ZXZ")
    graph.add_pipe(b1, b2)
    graph.add_pipe(b2, b3)
    graph.add_pipe(b3, c)
    condition = add_condition_source(graph)
    spine = {
        ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(b2, Basis.Z)),
        ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b3, Basis.Z)),
    }
    off = CorrelationSurface(span=frozenset(spine))
    on = CorrelationSurface(
        span=frozenset(spine | {ZXEdge(u=ZXNode(b3, Basis.Z), v=ZXNode(c, Basis.Z))})
    )
    return graph, ConditionalCorrelationSurface(
        conditions=(condition,), resolutions={(False,): off, (True,): on}
    )
