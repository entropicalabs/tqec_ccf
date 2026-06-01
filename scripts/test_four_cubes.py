from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root ``tools`` package importable when running this script
# directly (uv run python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import resolve_if_else
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _condition() -> CorrelationSurface:
    p = Position3D(0, 0, 1)
    return CorrelationSurface(span=frozenset([ZXEdge(ZXNode(p, Basis.Z), ZXNode(p, Basis.Z))]))


def main() -> None:
    """Four cubes in a square over two layers.

    Layer 0:
        XZX    XZX
        XZX    Conditional(XZX, XZZ)
    Layer 1:
        XZX    XZX
        XZX    XZX
    """
    g_conditional = BlockGraph("four_cubes_conditional")
    g_zero = BlockGraph("four_cubes_branch_zero")
    g_one = BlockGraph("four_cubes_branch_one")
    cubes = [
        (Position3D(0, 0, 0), "XZX"),
        (Position3D(0, 0, 1), "XZX"),
        (Position3D(1, 0, 0), "XZX"),
        (Position3D(1, 0, 1), "XZX"),
        (Position3D(0, 1, 0), "XZX"),
        (Position3D(0, 1, 1), "XZX"),
        (Position3D(1, 1, 1), "XZX"),
    ]
    pos_cond = Position3D(1, 1, 0)
    pipes = [(0, 1), (2, 3), (4, 5)]

    for g in [g_conditional, g_zero, g_one]:
        for cube, kind in cubes:
            g.add_cube(cube, kind)
        for u, v in pipes:
            g.add_pipe(cubes[u][0], cubes[v][0])

    # conditional block graph
    g_conditional.add_cube(pos_cond, ConditionalLeafCubeKind.XZX_XZZ, condition=_condition())
    g_conditional.add_pipe(pos_cond, cubes[6][0])

    cg = compile_block_graph(g_conditional, FIXED_BULK_CONVENTION, observables=None)
    conditional_stim = cg.generate_conditional_stim_text(k=1, condition_recs={pos: -1 for pos in cg._conditional_blocks})

    # resolve
    branch_zero = resolve_if_else(conditional_stim, conditions={-1: 0})
    branch_one = resolve_if_else(conditional_stim, conditions={-1: 1})

    # branch zero
    g_zero.add_cube(pos_cond, "XZX")
    g_zero.add_pipe(pos_cond, cubes[6][0])

    cg_zero = compile_block_graph(g_zero, FIXED_BULK_CONVENTION, observables=None)
    zero_stim = cg_zero.generate_stim_circuit(k=1)
    assert zero_stim == branch_zero, (
        "Branch zero should match the corresonding non conditional block graph stimulus"
    )

    # branch one
    g_one.add_cube(pos_cond, "XZZ")
    g_one.add_pipe(pos_cond, cubes[6][0])

    cg_one = compile_block_graph(g_one, FIXED_BULK_CONVENTION, observables=None)
    one_stim = cg_one.generate_stim_circuit(k=1)
    assert one_stim == branch_one, (
        "Branch one should match the corresonding non conditional block graph stimulus"
    )

    print(conditional_stim)


if __name__ == "__main__":
    main()
