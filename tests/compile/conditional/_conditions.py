"""A condition that reads real measurements, for conditional-cube fixtures."""

from __future__ import annotations

from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def add_condition_source(g: BlockGraph, z: int = 0) -> CorrelationSurface:
    """Add a lone ``ZXZ`` cube beside ``g`` at height ``z`` and return a condition on it.

    The cube has no pipe, so its data qubits are read out at the end of its
    layer and a single ``Z`` node on it reads real measurements. Conditioning on
    the cube directly below a conditional cube would not: the temporal pipe
    between them carries those data qubits on instead of measuring them.
    """
    x = max(cube.position.x for cube in g.cubes) + 1
    source = Position3D(x, 0, z)
    g.add_cube(source, "ZXZ")
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(source, Basis.Z), ZXNode(source, Basis.Z))])
    )
