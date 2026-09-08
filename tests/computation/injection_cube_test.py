"""Tests for the ``INJECTION`` cube kind at the block-graph level."""

from __future__ import annotations

import pathlib

import pytest
from pyzx.utils import VertexType

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import Cube, LeafCubeKind, cube_kind_from_string
from tqec.interop.collada._correlation import CorrelationSurfaceTransformationHelper
from tqec.interop.collada._geometry import BlockGeometries
from tqec.interop.color import TQECColor
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Direction3D, Position3D

_ORIGIN = Position3D(0, 0, 0)
_ABOVE = Position3D(0, 0, 1)


def _injection_column(above: str = "ZXX") -> BlockGraph:
    """Return an injection cube capping a temporal pipe into a memory cube."""
    graph = BlockGraph("injection column")
    graph.add_cube(_ORIGIN, "I")
    graph.add_cube(_ABOVE, above)
    graph.add_pipe(_ORIGIN, _ABOVE)
    return graph


@pytest.mark.parametrize("string", ["I", "i", "INJECTION", "injection"])
def test_kind_from_string(string: str) -> None:
    assert cube_kind_from_string(string) is LeafCubeKind.INJECTION
    assert LeafCubeKind.from_str(string) is LeafCubeKind.INJECTION


def test_kind_string_representation() -> None:
    assert str(LeafCubeKind.INJECTION) == "I"


def test_cube_is_injection_cube() -> None:
    cube = Cube(_ORIGIN, LeafCubeKind.INJECTION)
    assert cube.is_injection_cube
    assert not cube.is_y_cube
    assert not cube.is_port
    assert not cube.is_zx_cube
    assert not Cube(_ORIGIN, LeafCubeKind.Y_HALF_CUBE).is_injection_cube


def test_proxy_defaults_to_true() -> None:
    assert Cube(_ORIGIN, LeafCubeKind.INJECTION).proxy
    assert not Cube(_ORIGIN, LeafCubeKind.INJECTION, proxy=False).proxy


def test_proxy_is_rejected_on_other_kinds() -> None:
    with pytest.raises(TQECError, match="Only an injection cube can have a proxy flag"):
        Cube(_ORIGIN, LeafCubeKind.Y_HALF_CUBE, proxy=False)


@pytest.mark.parametrize("proxy", [True, False])
def test_cube_dict_round_trip_keeps_proxy(proxy: bool) -> None:
    cube = Cube(_ORIGIN, LeafCubeKind.INJECTION, proxy=proxy)
    assert Cube.from_dict(cube.to_dict()) == cube


def test_graph_with_an_injection_column_is_valid() -> None:
    graph = _injection_column()
    graph.validate()
    assert str(graph.pipes[0].kind) == "ZXO"


@pytest.mark.parametrize("above", ["ZXX", "ZXZ", "XZX", "XZZ"])
def test_injection_column_validates_under_either_orientation(above: str) -> None:
    _injection_column(above).validate()


def test_graph_add_cube_passes_proxy_through() -> None:
    graph = BlockGraph("proxy")
    graph.add_cube(_ORIGIN, "I", proxy=False)
    assert not graph[_ORIGIN].proxy


def test_injection_cube_needs_a_pipe() -> None:
    graph = BlockGraph("bare")
    graph.add_cube(_ORIGIN, "I")
    with pytest.raises(TQECError, match="does not have exactly one pipe connected"):
        graph.validate()


def test_injection_cube_rejects_a_second_pipe() -> None:
    graph = BlockGraph("two pipes")
    graph.add_cube(Position3D(0, 0, 1), "I")
    graph.add_cube(Position3D(0, 0, 0), "ZXX")
    graph.add_cube(Position3D(0, 0, 2), "ZXX")
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    graph.add_pipe(Position3D(0, 0, 1), Position3D(0, 0, 2))
    with pytest.raises(TQECError, match="does not have exactly one pipe connected"):
        graph.validate()


def test_injection_cube_rejects_a_horizontal_pipe() -> None:
    graph = BlockGraph("horizontal")
    graph.add_cube(_ORIGIN, "I")
    graph.add_cube(Position3D(1, 0, 0), "ZXZ")
    graph.add_pipe(_ORIGIN, Position3D(1, 0, 0))
    with pytest.raises(TQECError, match="non-timelike pipe"):
        graph.validate()


def test_injection_cube_rejects_a_pipe_below_it() -> None:
    graph = BlockGraph("capped from above")
    graph.add_cube(_ORIGIN, "ZXX")
    graph.add_cube(_ABOVE, "I")
    graph.add_pipe(_ORIGIN, _ABOVE)
    with pytest.raises(TQECError, match="pipe must go up"):
        graph.validate()


def test_every_face_is_magenta() -> None:
    faces = BlockGeometries().get_geometry(LeafCubeKind.INJECTION)
    # A full cube: three axes, two faces each.
    assert len(faces) == 6
    assert {face.color for face in faces} == {TQECColor.INJECTION}
    assert TQECColor.INJECTION.rgba.to_hex() == "#ff00ff"


def test_dae_round_trip(tmp_path: pathlib.Path) -> None:
    graph = _injection_column()
    path = tmp_path / "injection.dae"
    graph.to_dae_file(path)
    read_back = BlockGraph.from_dae_file(path, graph_name=graph.name)
    assert {(c.position, c.kind) for c in read_back.cubes} == {
        (c.position, c.kind) for c in graph.cubes
    }
    read_back.validate()


def test_zx_graph_conversion_treats_injection_as_a_boundary() -> None:
    positioned = _injection_column().to_zx_graph()
    types = {positioned.positions[v]: positioned.g.type(v) for v in positioned.g.vertices()}
    assert types[_ORIGIN] == VertexType.BOUNDARY


def test_correlation_surface_renders_through_the_injection_cube() -> None:
    graph = _injection_column()
    surface = graph.find_correlation_surfaces()[0]
    assert _ORIGIN in surface.positions
    helper = CorrelationSurfaceTransformationHelper(graph, pipe_length=2.0)
    # The injection cube's faces carry no basis, so the generic ZXCube geometry
    # does not apply; the piece is drawn from the incident pipe's plane instead.
    # Regression: this used to trip an ``assert isinstance(kind, ZXCube)``.
    pieces = helper.get_transformations_for_correlation_surface(surface)
    assert len(pieces) == 3
    assert {basis for basis, _ in pieces} == {surface.bases_at(_ORIGIN).pop()}


def test_view_as_html_with_a_correlation_surface() -> None:
    graph = _injection_column()
    surface = graph.find_correlation_surfaces()[0]
    html = graph.view_as_html(show_correlation_surface=surface, write_html_filepath=None)
    assert str(html)


def _proxy_column() -> BlockGraph:
    graph = BlockGraph("proxy false")
    graph.add_cube(_ORIGIN, "I", proxy=False)
    graph.add_cube(_ABOVE, "ZXX")
    graph.add_pipe(_ORIGIN, _ABOVE)
    graph.validate()
    return graph


def test_proxy_survives_a_shift() -> None:
    # Regression: ``compile_block_graph`` shifts the graph to z >= 0, and
    # ``shift_by`` used to rebuild each cube field by field, resetting ``proxy``.
    shifted = _proxy_column().shift_by(dz=5)
    assert not shifted[Position3D(0, 0, 5)].proxy


def test_proxy_survives_a_rotation() -> None:
    rotated = _proxy_column().rotate(rotation_axis=Direction3D.Z, num_90_degree_rotation=1)
    assert not next(cube for cube in rotated.cubes if cube.is_injection_cube).proxy


def test_proxy_survives_a_dict_round_trip() -> None:
    graph = _proxy_column()
    assert not BlockGraph.from_dict(graph.to_dict())[_ORIGIN].proxy
    # A graph serialised before ``proxy`` existed still reads back.
    data = graph.to_dict()
    for cube in data["cubes"]:
        cube.pop("proxy", None)
    assert BlockGraph.from_dict(data)[_ORIGIN].proxy


def test_insert_cube_preserves_every_attribute() -> None:
    graph = BlockGraph("insert")
    cube = Cube(_ORIGIN, LeafCubeKind.INJECTION, proxy=False)
    assert graph.insert_cube(cube) == _ORIGIN
    assert graph[_ORIGIN] == cube


def test_insert_cube_rejects_a_duplicate_position() -> None:
    graph = BlockGraph("insert")
    graph.insert_cube(Cube(_ORIGIN, LeafCubeKind.INJECTION))
    with pytest.raises(TQECError, match="Cube already exists"):
        graph.insert_cube(Cube(_ORIGIN, LeafCubeKind.INJECTION))


def test_insert_cube_rejects_a_duplicate_port_label() -> None:
    graph = BlockGraph("insert")
    graph.insert_cube(Cube(_ORIGIN, LeafCubeKind.PORT, "p"))
    with pytest.raises(TQECError, match="already a port with the same label"):
        graph.insert_cube(Cube(_ABOVE, LeafCubeKind.PORT, "p"))
