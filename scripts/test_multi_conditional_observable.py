"""Multi-conditional observable: one ConditionalCorrelationSurface gated by
two conditional cubes living on distinct z-layers.

Layout (two disconnected columns):

    Column A: a0 (z=0, ZXX) ----> a1 (z=1, conditional ZXX_ZXZ)
    Column B: b0 (z=0, ZXX) -> b1 (z=1) -> b2 (z=2) -> b3 (z=3, conditional ZXX_ZXZ)

The conditional observable has two branch surfaces. Both run X-paths from
each initialiser up to its conditional cube. branch_zero terminates with
extra X-edges that branch_one does not include, and vice-versa. Each
conditional cube therefore owns its own divergent slice; the emitted
Stim text contains two OBSERVABLE_INCLUDE(0) IfBlocks, one per cube,
sharing the same observable index.
"""

from tqec.compile.compile import compile_block_graph
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


_INIT_KIND = ConditionalLeafCubeKind.ZXX_ZXZ.value[0]  # ZXX

# Column A
a0 = Position3D(0, 0, 0)
a1 = Position3D(0, 0, 1)
# Column B
b0 = Position3D(2, 0, 0)
b1 = Position3D(2, 0, 1)
b2 = Position3D(2, 0, 2)
b3 = Position3D(2, 0, 3)


def build_graph() -> BlockGraph:
    g = BlockGraph("multi_conditional_observable")
    # Conditional cube placeholder conditions (mock recs; warnings are expected).
    g.add_cube(a0, _INIT_KIND)
    g.add_cube(
        a1,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset({ZXEdge(u=ZXNode(a0, Basis.Z), v=ZXNode(a0, Basis.Z))})
        ),
    )
    g.add_pipe(a0, a1)

    g.add_cube(b0, _INIT_KIND)
    g.add_cube(b1, _INIT_KIND)
    g.add_cube(b2, _INIT_KIND)
    g.add_cube(
        b3,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset({ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b2, Basis.Z))})
        ),
    )
    g.add_pipe(b0, b1)
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)
    return g


# branch_zero: takes the X-readout at the top of each conditional cube
# (a1 -> ZXX, kind.z=X; b3 -> ZXX, kind.z=X). Surface runs through both columns.
_shared_span = frozenset(
    {
        ZXEdge(u=ZXNode(a0, Basis.X), v=ZXNode(a1, Basis.X)),
        ZXEdge(u=ZXNode(b0, Basis.X), v=ZXNode(b1, Basis.X)),
        ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(b2, Basis.X)),
        ZXEdge(u=ZXNode(b2, Basis.X), v=ZXNode(b3, Basis.X)),
    }
)
# Same input surface for both branches. Divergence arises because the
# resolved conditional cube kinds differ (ZXX vs ZXZ -> z-face X vs Z), so
# `compile_correlation_surface_to_abstract_observable` lowers the surface
# differently per branch: top-readouts at a1/b3 only fire in branch_zero
# (where kind.z=X matches the X correlation).
branch_zero_surface = CorrelationSurface(span=_shared_span)
branch_one_surface = CorrelationSurface(span=_shared_span)

if __name__ == "__main__":
    g = build_graph()
    cond_a = next(c.condition for c in g.cubes if c.position == a1)
    cond_b = next(c.condition for c in g.cubes if c.position == b3)
    assert cond_a is not None and cond_b is not None
    conditional_observable = ConditionalCorrelationSurface(
        conditions=(cond_a, cond_b),
        resolutions={
            (False, False): branch_zero_surface,
            (False, True): branch_one_surface,
            (True, False): branch_one_surface,
            (True, True): branch_zero_surface,
        },
    )
    cg = compile_block_graph(g, observables=[conditional_observable])
    text = cg.generate_conditional_stim_text(k=1)
    print(text)

    import re

    if_blocks = re.findall(r"IF\(.*?\}(?:\s*ELSE\s*\{.*?\})?", text, re.DOTALL)
    obs_if_count = sum(1 for b in if_blocks if "OBSERVABLE_INCLUDE(0)" in b)
    total_if = text.count("IF(")
    print()
    print(f"# multi-cube observable IfBlocks: {obs_if_count}")
    print(f"# total IF(...) blocks (observable + circuit + detectors): {total_if}")
