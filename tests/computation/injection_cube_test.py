"""Tests for the ``INJECTION`` cube kind at the block-graph level."""

from __future__ import annotations

import pathlib

import pytest
from pyzx.utils import VertexType

from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import Cube, LeafCubeKind, cube_kind_from_string
from tqec.interop.collada._geometry import BlockGeometries
from tqec.interop.color import TQECColor
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D

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
