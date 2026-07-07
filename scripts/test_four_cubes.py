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
    # TODO: this graph places the cond cube at z=0 (lowest slice). No real
    # causal layer exists below it, so this is a mock surface at z=-1 that
    # satisfies the structural causality check at Cube.__post_init__ and
    # passes through compile_correlation_surface_to_abstract_observable's
    # single-node fallback (uses block_graph.cubes[0] rather than the
    # surface position). The resolver yields no recs → placeholder rec[-1]
    # falls back, with a UserWarning. Restructure the graph (lift cond cube
    # to z>=1) for a real IF/ELSE condition.
    p = Position3D(0, 0, -1)
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
    conditional_stim = cg.generate_conditional_stim_text(k=1)

    # resolve (rec offset(s) come from emitted IF clause)
    import re as _re

    m = _re.search(r"IF\(([^)]+)\)", conditional_stim)
    assert m is not None
    rec_offsets = [int(s.strip().lstrip("rec[").rstrip("]")) for s in m.group(1).split("^")]
    rec = rec_offsets[0]
    branch_zero = resolve_if_else(conditional_stim, conditions={rec: 0})
    branch_one = resolve_if_else(conditional_stim, conditions={rec: 1})

    # branch zero
    g_zero.add_cube(pos_cond, "XZX")
    g_zero.add_pipe(pos_cond, cubes[6][0])

    cg_zero = compile_block_graph(g_zero, FIXED_BULK_CONVENTION, observables=None)
    zero_stim = cg_zero.generate_stim_circuit(k=1)
    _assert_equivalent_modulo_detector_order(branch_zero, zero_stim, "branch zero")

    # branch one
    g_one.add_cube(pos_cond, "XZZ")
    g_one.add_pipe(pos_cond, cubes[6][0])

    cg_one = compile_block_graph(g_one, FIXED_BULK_CONVENTION, observables=None)
    one_stim = cg_one.generate_stim_circuit(k=1)
    _assert_equivalent_modulo_detector_order(branch_one, one_stim, "branch one")

    print("OK: branches match reference compiles modulo DETECTOR order.")

    print(conditional_stim)


def _assert_equivalent_modulo_detector_order(actual, expected, label: str) -> None:
    """Single-pass conditional emission groups shared detectors first then
    divergent (commit 4a/4b); from-scratch compile interleaves them in radius-2
    lookback order. Both circuits define the same detector multiset.
    """

    def _split(circuit):
        non_det = []
        detector_blocks: list[frozenset[str]] = []
        current: set[str] = set()
        for inst in circuit:
            text = str(inst)
            if inst.name == "DETECTOR":
                current.add(text)
                continue
            if current:
                detector_blocks.append(frozenset(current))
                current = set()
            non_det.append(text)
        if current:
            detector_blocks.append(frozenset(current))
        return non_det, detector_blocks

    a_non, a_blocks = _split(actual)
    e_non, e_blocks = _split(expected)
    assert a_non == e_non, f"{label}: non-DETECTOR instructions differ"
    assert a_blocks == e_blocks, f"{label}: DETECTOR multisets differ per block"


if __name__ == "__main__":
    main()
