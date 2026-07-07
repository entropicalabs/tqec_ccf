"""Stack two T-gadgets along z with a connected 4-resolution observable.

Two T-gadgets are joined by a vertical spine, and a 4-resolution
ConditionalCorrelationSurface connects their observables (visualization only).

Run with: uv run scripts/stack_two_gadgets.py
"""

from __future__ import annotations

import itertools

from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import (
    ConditionalCorrelationSurface,
    CorrelationSurface,
    ZXEdge,
    ZXNode,
)
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D

GADGET1_DZ = 1
GADGET2_DZ = 4


def make_t_gadget(
    suffix: str = "",
) -> tuple[BlockGraph, CorrelationSurface, CorrelationSurface]:
    """T-gadget with proxy T and S cubes, extending in +x. Returns (graph, o_true, o_false)."""
    port_label = f"t_gadget_{suffix}"

    p = Position3D(0, 0, 0)
    b1 = Position3D(1, 0, 0)
    t = Position3D(1, 0, -1)
    r = Position3D(2, 0, 0)
    b2 = Position3D(2, 1, 0)
    c = Position3D(2, 1, 1)
    b3 = Position3D(3, 0, 0)
    y = Position3D(3, 0, -1)

    condition = CorrelationSurface(
        span=frozenset({ZXEdge(ZXNode(p, Basis.Z), ZXNode(b1, Basis.Z))})
    )

    g = BlockGraph(f"t_gadget_{suffix}")
    g.add_cube(p, "P", label=port_label)
    g.add_cube(b1, "XZX")
    g.add_cube(t, "XZX")  # T
    g.add_cube(r, "ZZX")
    g.add_cube(b2, "ZXX")
    g.add_cube(c, ConditionalLeafCubeKind.ZXX_ZXZ, condition=condition)
    g.add_cube(b3, "XZX")
    g.add_cube(y, "XZX")  # Y

    for a, b in [(p, b1), (t, b1), (b1, r), (r, b2), (b2, c), (r, b3), (b3, y)]:
        g.add_pipe(a, b)

    t_es = {
        Basis.X: {(p, b1), (t, b1), (b1, r), (r, b2), (b2, c)},
    }
    f_es = {
        Basis.X: {(p, b1), (t, b1), (b1, r), (r, b3), (y, b3)},
        Basis.Z: {(p, b1), (b1, r), (r, b2), (b2, c)},
    }

    def _obs_from_set(s: dict) -> CorrelationSurface:
        return CorrelationSurface(
            span=frozenset(
                {
                    ZXEdge(ZXNode(u, basis), ZXNode(v, basis))
                    for basis, edges in s.items()
                    for u, v in edges
                }
            )
        )

    return g, _obs_from_set(t_es), _obs_from_set(f_es)


def _spine_pos(z: int) -> Position3D:
    return Position3D(0, 0, z)


def stack_two_gadgets() -> BlockGraph:
    """Assemble the stacked block graph. See module docstring / plan for geometry."""
    stacked = BlockGraph("stacked_t_gadgets")

    # --- gadget bodies (skip their ports; the spine owns x=0,y=0) ---
    for dz in (GADGET1_DZ, GADGET2_DZ):
        gad = make_t_gadget(dz)[0].shift_by(dz=dz)
        for cube in gad.cubes:
            if cube.is_port:
                continue
            stacked.add_cube(cube.position, cube.kind, cube.label, cube.condition)
        for pipe in gad.pipes:
            # The port pipe (p->b1) references the skipped port; add it later,
            # once the spine cube at the port position exists.
            u, v = pipe.u.position, pipe.v.position
            if u not in stacked or v not in stacked:
                continue
            stacked.add_pipe(u, v, pipe.kind)

    # --- spine cubes ---
    stacked.add_cube(_spine_pos(0), "P", label="bottom")
    stacked.add_cube(_spine_pos(1), "XZX")  # filled gadget-1 port
    stacked.add_cube(_spine_pos(2), "XZX")
    stacked.add_cube(_spine_pos(3), "XZX")
    stacked.add_cube(_spine_pos(4), "XZX")  # filled gadget-2 port
    stacked.add_cube(_spine_pos(5), "P", label="top")

    # port -> b1 pipes (now the spine cube exists at the port position)
    stacked.add_pipe(_spine_pos(1), Position3D(1, 0, 1))
    stacked.add_pipe(_spine_pos(4), Position3D(1, 0, 4))

    # vertical spine pipes
    for z in range(5):
        stacked.add_pipe(_spine_pos(z), _spine_pos(z + 1))

    return stacked


SPINE_BOTTOM = 0  # input port
SPINE_TOP = 5  # output port


def _spine_edges(z_from: int, z_to: int, basis: Basis) -> frozenset[ZXEdge]:
    """Edges of the given basis along the spine between z_from and z_to (x=0,y=0)."""
    lo, hi = sorted((z_from, z_to))
    return frozenset(
        ZXEdge(ZXNode(_spine_pos(z), basis), ZXNode(_spine_pos(z + 1), basis))
        for z in range(lo, hi)
    )


def _spine_x_edges() -> frozenset[ZXEdge]:
    """X membrane spanning the full spine (both ports): z=0..5 at x=0,y=0."""
    return _spine_edges(SPINE_BOTTOM, SPINE_TOP, Basis.X)


def build_conditional_observable() -> ConditionalCorrelationSurface:
    """4-resolution observable: each key picks o_true/o_false per gadget, joined by the spine."""
    _, o_true_1, o_false_1 = make_t_gadget(GADGET1_DZ)
    _, o_true_2, o_false_2 = make_t_gadget(GADGET2_DZ)

    # Per gadget: (false-arm span, true-arm span). The X leg is carried by the
    # shared full-spine X membrane (below). o_false additionally has a Z leg,
    # routed down the spine to the input port at z=0.
    arms = {
        GADGET1_DZ: (
            o_false_1.shift_by(dz=GADGET1_DZ).span
            | _spine_edges(SPINE_BOTTOM, GADGET1_DZ, Basis.Z),
            o_true_1.shift_by(dz=GADGET1_DZ).span,
        ),
        GADGET2_DZ: (
            o_false_2.shift_by(dz=GADGET2_DZ).span
            | _spine_edges(SPINE_BOTTOM, GADGET2_DZ, Basis.Z),
            o_true_2.shift_by(dz=GADGET2_DZ).span,
        ),
    }
    spine = _spine_x_edges()

    # Combine by symmetric difference (Z2 composition): gadget-internal edges are
    # disjoint (behaves like union), while the two gadgets' Z-to-z0 spine segments
    # overlap on z0..z1 and cancel when both are false -- the only parity-valid way
    # to route both Z legs down the single spine column (avoids the 3-leg Z
    # junction at the filled port (0,0,1)).
    resolutions: dict[tuple[bool, ...], CorrelationSurface] = {}
    for b0, b1 in itertools.product((False, True), repeat=2):
        span = arms[GADGET1_DZ][b0] ^ arms[GADGET2_DZ][b1] ^ spine
        resolutions[(b0, b1)] = CorrelationSurface(span=frozenset(span))

    def _condition(dz: int) -> CorrelationSurface:
        edge = ZXEdge(
            ZXNode(_spine_pos(dz), Basis.Z), ZXNode(Position3D(1, 0, dz), Basis.Z)
        )
        return CorrelationSurface(span=frozenset({edge}))

    cond1, cond2 = _condition(GADGET1_DZ), _condition(GADGET2_DZ)
    return ConditionalCorrelationSurface(
        conditions=(cond1, cond2), resolutions=resolutions
    )


if __name__ == "__main__":
    g = stack_two_gadgets()
    print("cubes:", len(g.cubes), "pipes:", len(g.pipes), "ports:", g.ports)

    cco = build_conditional_observable()
    S = cco.resolutions[(False, False)]
    d0 = S ^ cco.resolutions[(True, False)]
    d1 = S ^ cco.resolutions[(False, True)]
    assert cco.resolutions[(True, True)].span == (S ^ d0 ^ d1).span, "flat-XOR broken"
    print("flat-XOR check: OK")

    for key, res in cco.resolutions.items():
        name = "".join("T" if b else "F" for b in key)
        path = f"notebooks/output/stack_{name}.html"
        g.view_as_html(write_html_filepath=path, show_correlation_surface=res)
        print("wrote", path)
