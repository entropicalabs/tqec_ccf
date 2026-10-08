"""``observables="auto"`` on graphs with conditional cubes or disconnected parts."""

from __future__ import annotations

import itertools

import pytest

from tests.compile.conditional.branch_circuit_test import (
    _add_conditional_column,
    _injection_beside_a_conditional_cube,
)
from tests.compile.conditional.branching_tree_test import _branching_tree
from tests.compile.conditional.conditional_observable_test import _build_graph
from tqec.compile import compile as compile_module
from tqec.compile.compile import (
    _branch_independent_correlation_surfaces,
    _resolve_conditional_cubes,
    compile_block_graph,
)
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, find_correlation_surfaces
from tqec.utils.position import Position3D

_GRAPHS = [
    pytest.param(_branching_tree(2), 0, id="branching_tree"),
    pytest.param(_injection_beside_a_conditional_cube("0"), 2, id="injection_beside"),
    pytest.param(_build_graph()[0], 2, id="conditional_observable"),
]


def _touches_a_conditional_cube(surface: CorrelationSurface, graph: BlockGraph) -> bool:
    conditional = {cube.position for cube in graph.cubes if cube.is_conditional}
    return bool(surface.positions & conditional)


@pytest.mark.parametrize(("graph", "expected"), _GRAPHS)
def test_auto_observables_avoid_conditional_cubes(graph: BlockGraph, expected: int) -> None:
    with pytest.warns(UserWarning, match="cross a conditional cube"):
        surfaces = _branch_independent_correlation_surfaces(graph)
    assert len(surfaces) == expected
    assert not any(_touches_a_conditional_cube(s, graph) for s in surfaces)


@pytest.mark.parametrize(("graph", "expected"), _GRAPHS)
def test_auto_observables_are_deterministic_in_every_branch(
    graph: BlockGraph, expected: int
) -> None:
    with pytest.warns(UserWarning, match="cross a conditional cube"):
        compiled = compile_block_graph(graph)
    cubes = [cube.position for cube in graph.cubes if cube.is_conditional]
    for bits in itertools.product((0, 1), repeat=len(cubes)):
        circuit = compiled.generate_branch_circuit(1, dict(zip(cubes, bits, strict=True)))
        assert circuit.num_observables == expected
        circuit.detector_error_model()  # raises on a non-deterministic observable


def test_a_surface_hidden_as_the_xor_of_two_crossing_ones_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Of a generating set, the members that avoid the conditional cubes need not
    # generate every surface that does: here both cross, but their XOR avoids.
    graph = _build_graph()[0]
    found = find_correlation_surfaces(_resolve_conditional_cubes(graph, 0).to_zx_graph())
    crossing = next(s for s in found if _touches_a_conditional_cube(s, graph))
    avoiding = next(s for s in found if not _touches_a_conditional_cube(s, graph))
    monkeypatch.setattr(
        compile_module,
        "find_correlation_surfaces",
        lambda _: [crossing, crossing ^ avoiding],
    )
    with pytest.warns(UserWarning, match="leaves out 1"):
        assert _branch_independent_correlation_surfaces(graph) == [avoiding]


def test_an_avoiding_surface_hidden_behind_two_pivots_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _injection_beside_a_conditional_cube("0")
    _add_conditional_column(graph, 4, 3, [Position3D(2, 0, 1)])
    found = find_correlation_surfaces(_resolve_conditional_cubes(graph, 0).to_zx_graph())
    first, second = (s for s in found if _touches_a_conditional_cube(s, graph))
    avoiding = next(s for s in found if not _touches_a_conditional_cube(s, graph))
    monkeypatch.setattr(
        compile_module,
        "find_correlation_surfaces",
        lambda _: [first, second, first ^ second ^ avoiding],
    )
    with pytest.warns(UserWarning, match="leaves out 2"):
        assert _branch_independent_correlation_surfaces(graph) == [avoiding]


def _plain_disconnected_graphs() -> list[tuple[BlockGraph, int]]:
    column_and_lone_cube = BlockGraph("A column and a lone cube")
    column_and_y_cap = BlockGraph("A column beside a Y-capped column")
    for graph in (column_and_lone_cube, column_and_y_cap):
        graph.add_cube(Position3D(0, 0, 0), "ZXZ")
        graph.add_cube(Position3D(0, 0, 1), "ZXZ")
        graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    column_and_lone_cube.add_cube(Position3D(2, 0, 0), "ZXZ")
    column_and_y_cap.add_cube(Position3D(2, 0, 0), "ZXZ")
    column_and_y_cap.add_cube(Position3D(2, 0, 1), "Y")
    column_and_y_cap.add_pipe(Position3D(2, 0, 0), Position3D(2, 0, 1))
    # A Y readout is random, so the capped column has no deterministic surface,
    # which used to leave the whole graph with none.
    return [(column_and_lone_cube, 2), (column_and_y_cap, 1)]


@pytest.mark.parametrize(("graph", "expected"), _plain_disconnected_graphs())
def test_auto_observables_of_a_plain_disconnected_graph(graph: BlockGraph, expected: int) -> None:
    circuit = compile_block_graph(graph).generate_stim_circuit(1)
    assert circuit.num_observables == expected
    circuit.detector_error_model()  # raises on a non-deterministic observable
