"""Verify ConditionalBlock metadata survives graph -> tree plumbing."""

from __future__ import annotations

from tqec.compile.blocks.block import ConditionalBlock
from tqec.compile.specs.base import CubeSpec
from tqec.compile.specs.library.fixed_bulk import FixedBulkCubeBuilder
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.plaquette.compilation.base import IdentityPlaquetteCompiler
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction


def _condition() -> CorrelationSurface:
    p = Position3D(0, 0, 0)
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p, Basis.Z), ZXNode(p, Basis.Z))])
    )


def test_conditional_block_registered_on_graph() -> None:
    from tqec.compile.blocks.positioning import LayoutPosition3D
    from tqec.compile.graph import TopologicalComputationGraph
    from tqec.compile.observables.fixed_bulk_builder import (
        FIXED_BULK_OBSERVABLE_BUILDER as DEFAULT_OBSERVABLE_BUILDER,
    )
    from tqec.utils.position import BlockPosition3D
    from tqec.utils.scale import PhysicalQubitScalable2D

    builder = FixedBulkCubeBuilder(IdentityPlaquetteCompiler)
    cond = _condition()
    block = builder(
        CubeSpec(kind=ConditionalLeafCubeKind.XZX_XZZ, condition=cond),
        LinearFunction(2, -1),
    )
    assert isinstance(block, ConditionalBlock)

    graph = TopologicalComputationGraph(
        scalable_qubit_shape=block.scalable_shape,
        observable_builder=DEFAULT_OBSERVABLE_BUILDER,
    )
    graph.add_cube(BlockPosition3D(0, 0, 0), block)
    tree = graph.to_layer_tree()
    expected_pos = LayoutPosition3D.from_block_position(BlockPosition3D(0, 0, 0))
    assert expected_pos in tree.conditional_blocks
    assert tree.conditional_blocks[expected_pos] is block
    assert tree.conditional_blocks[expected_pos].condition is cond


def test_non_conditional_block_does_not_register() -> None:
    from tqec.compile.graph import TopologicalComputationGraph
    from tqec.compile.observables.fixed_bulk_builder import (
        FIXED_BULK_OBSERVABLE_BUILDER as DEFAULT_OBSERVABLE_BUILDER,
    )
    from tqec.computation.cube import ZXCube
    from tqec.utils.position import BlockPosition3D

    builder = FixedBulkCubeBuilder(IdentityPlaquetteCompiler)
    block = builder(CubeSpec(kind=ZXCube.XZX), LinearFunction(2, -1))
    graph = TopologicalComputationGraph(
        scalable_qubit_shape=block.scalable_shape,
        observable_builder=DEFAULT_OBSERVABLE_BUILDER,
    )
    graph.add_cube(BlockPosition3D(0, 0, 0), block)
    tree = graph.to_layer_tree()
    assert tree.conditional_blocks == {}
