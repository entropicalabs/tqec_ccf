"""Multi-conditional emission: ≤1 conditional cube per z-layer."""

from __future__ import annotations

import pytest

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D


_INIT_KIND = ConditionalLeafCubeKind.XZX_XZZ.value[0]  # XZX


def _placeholder_condition(pos: Position3D) -> CorrelationSurface:
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(pos, Basis.Z), ZXNode(pos, Basis.Z))])
    )


def test_two_conditional_cubes_distinct_z_emits_separate_ifblocks() -> None:
    g = BlockGraph("multi_cond_distinct_z")
    # Column A: z=0 init -> z=1 conditional (leaf).
    a0 = Position3D(0, 0, 0)
    a1 = Position3D(0, 0, 1)
    g.add_cube(a0, _INIT_KIND)
    g.add_cube(
        a1, ConditionalLeafCubeKind.XZX_XZZ, condition=_placeholder_condition(a0)
    )
    g.add_pipe(a0, a1)
    # Column B: z=0 init -> z=1 mid -> z=2 mid -> z=3 conditional (leaf).
    b0 = Position3D(2, 0, 0)
    b1 = Position3D(2, 0, 1)
    b2 = Position3D(2, 0, 2)
    b3 = Position3D(2, 0, 3)
    g.add_cube(b0, _INIT_KIND)
    g.add_cube(b1, _INIT_KIND)
    g.add_cube(b2, _INIT_KIND)
    g.add_cube(
        b3, ConditionalLeafCubeKind.XZX_XZZ, condition=_placeholder_condition(b2)
    )
    g.add_pipe(b0, b1)
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)

    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(k=1)
    # Each conditional cube produces multiple IfBlocks (per-CEO slot + detector).
    # Expect two distinct clusters in the output.
    assert text.count("IF(") >= 2
    # Confirm IfBlocks land in different z-layers (different SHIFT_COORDS frames).
    lines = text.splitlines()
    if_z_indices: list[int] = []
    z_index = 0
    for line in lines:
        if line.startswith("SHIFT_COORDS"):
            z_index += 1
        if line.startswith("IF("):
            if_z_indices.append(z_index)
    assert len(set(if_z_indices)) >= 2, (
        f"expected IfBlocks across >=2 z-layers, got {if_z_indices}"
    )


def test_two_conditional_cubes_same_z_raises() -> None:
    g = BlockGraph("multi_cond_same_z")
    a0 = Position3D(0, 0, 0)
    a1 = Position3D(0, 0, 1)
    b0 = Position3D(2, 0, 0)
    b1 = Position3D(2, 0, 1)
    g.add_cube(a0, _INIT_KIND)
    g.add_cube(
        a1, ConditionalLeafCubeKind.XZX_XZZ, condition=_placeholder_condition(a0)
    )
    g.add_pipe(a0, a1)
    g.add_cube(b0, _INIT_KIND)
    g.add_cube(
        b1, ConditionalLeafCubeKind.XZX_XZZ, condition=_placeholder_condition(b0)
    )
    g.add_pipe(b0, b1)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match="z=1"):
        cg.generate_conditional_stim_text(k=1)
