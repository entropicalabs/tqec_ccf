import pytest

from tqec.computation.block_graph import BlockGraph
from tqec.computation.open_graph import fill_ports_for_minimal_simulation
from tqec.gallery.cnot import cnot
from tqec.utils.position import Position3D


@pytest.mark.parametrize("search_small_area_observables", [True, False])
def test_fill_ports_for_minimal_simulation(search_small_area_observables: bool) -> None:
    graph = cnot()
    filled_graphs = fill_ports_for_minimal_simulation(graph, search_small_area_observables)
    assert len(filled_graphs) == 2
    g1, g2 = filled_graphs
    assert not g1.graph.is_open
    assert not g2.graph.is_open
    assert set(g2.stabilizers) == {"ZIZI", "ZZIZ"}
    assert set(g2.get_external_stabilizers()) == {"ZZII", "ZIZZ"}
    if search_small_area_observables:
        assert set(g1.stabilizers) == {"IXIX", "XIXX"}
        assert set(g1.get_external_stabilizers()) == {"IIXX", "XXIX"}
    else:
        assert set(g1.stabilizers) == {"XXXI", "XIXX"}
        assert set(g1.get_external_stabilizers()) == {"XXXI", "XXIX"}


@pytest.mark.parametrize("search_small_area_observables", [False, True])
def test_a_component_without_ports_does_not_disturb_port_filling(
    search_small_area_observables: bool,
) -> None:
    # The closed column's surfaces are the identity on every port: they certify
    # nothing about the ports and must neither crash the search nor be mistaken
    # for a stabilizer.
    graph = BlockGraph("A ported column beside a closed one")
    graph.add_cube(Position3D(0, 0, 0), "ZXZ")
    graph.add_cube(Position3D(0, 0, -1), "PORT", "in")
    graph.add_cube(Position3D(0, 0, 1), "PORT", "out")
    graph.add_pipe(Position3D(0, 0, -1), Position3D(0, 0, 0))
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    graph.add_cube(Position3D(2, 0, 0), "ZXZ")
    graph.add_cube(Position3D(2, 0, 1), "ZXZ")
    graph.add_pipe(Position3D(2, 0, 0), Position3D(2, 0, 1))
    filled = fill_ports_for_minimal_simulation(graph, search_small_area_observables)
    stabilizers = sorted(s for fg in filled for s in fg.stabilizers)
    assert stabilizers == ["XX", "ZZ"]
    for fg in filled:
        assert all(o.positions <= {Position3D(0, 0, z) for z in (-1, 0, 1)} for o in fg.observables)
