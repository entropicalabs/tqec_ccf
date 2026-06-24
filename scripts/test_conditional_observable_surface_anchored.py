"""Surface-anchored conditional observable on a *plain* BlockGraph (no
``ConditionalLeafCubeKind`` cubes). The conditional observable is gated by a
condition surface that resolves directly to past pipe measurements; no per-
branch circuit shape change happens — only ``OBSERVABLE_INCLUDE`` is gated.

Minimal demo: a 3-cube vertical spine `b1 → b2 → b3`. The condition is the
pipe edge `b1↔b2` (X-basis). Two resolutions: with and without an extra
side hop into `c` (a spatial neighbour of `b3`).
"""

from __future__ import annotations

from tqec.compile.compile import compile_block_graph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import (
    ConditionalCorrelationSurface,
    CorrelationSurface,
    ZXEdge,
    ZXNode,
)
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


b1 = Position3D(0, 0, 0)
b2 = Position3D(0, 0, 1)
b3 = Position3D(0, 0, 2)
c = Position3D(1, 0, 2)


_cond_a = CorrelationSurface(
    span=frozenset({ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(b2, Basis.X))})
)


def build_graph() -> BlockGraph:
    g = BlockGraph("surface_anchored_demo")
    g.add_cube(b1, "ZXZ")
    g.add_cube(b2, "ZXZ")
    g.add_cube(b3, "ZXZ")
    g.add_cube(c, "ZXZ")
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)
    g.add_pipe(b3, c)
    return g


_obs_off = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(b2, Basis.Z)),
            ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b3, Basis.Z)),
        }
    )
)

# When condition fires (a=True), extend the observable to include the
# spatial hop b3 ↔ c (Z-basis), so the resolutions differ.
_obs_on = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(b2, Basis.Z)),
            ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b3, Basis.Z)),
            ZXEdge(u=ZXNode(b3, Basis.Z), v=ZXNode(c, Basis.Z)),
        }
    )
)


conditional_observable = ConditionalCorrelationSurface(
    conditions=(_cond_a,),
    resolutions={
        (False,): _obs_off,
        (True,): _obs_on,
    },
)


if __name__ == "__main__":
    g = build_graph()
    cg = compile_block_graph(g, observables=[conditional_observable])
    text = cg.generate_conditional_stim_text(k=1)
    print(text)

    import re

    if_then = re.findall(r"IF\(.*?\) \{\n(.*?)\n\}(?!\s*ELSE)", text, re.DOTALL)
    obs_ifs = [body for body in if_then if "OBSERVABLE_INCLUDE(0)" in body]
    print()
    print(f"# OBSERVABLE_INCLUDE(0) IF blocks (no ELSE): {len(obs_ifs)}")
    print(f"# total OBSERVABLE_INCLUDE(0) occurrences: {text.count('OBSERVABLE_INCLUDE(0)')}")
