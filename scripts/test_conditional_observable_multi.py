"""Multi-condition ConditionalCorrelationSurface — flat-XOR decomposable case.

Two parallel columns; each column ends in a conditional cube whose condition
is anchored at a measurement strictly in its past (a self-loop condition on
an upstream init cube). The observable is the union of both columns plus,
per condition's outcome, the corresponding column's top-readout.

Resolutions are XOR-decomposable across the two conditions:

    O(a, b) = shared ⊕ a · Δ_a ⊕ b · Δ_b

so emission produces a plain ``OBSERVABLE_INCLUDE`` for the shared baseline
and one ``IF(rec_i) { OBSERVABLE_INCLUDE Δ_i }`` per condition — no ELSE.

NOTE: the second example in ``notebooks/observable_multiple_conditions.ipynb``
violates the causality rule (conditional cubes whose conditions reference
their own z-layer) and does not compile under the current
``Cube.__post_init__`` check. This script supplies a structurally valid
graph that demonstrates the same flat-XOR canonical form.
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
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


a0 = Position3D(0, 0, 0)
a1 = Position3D(0, 0, 1)
b0 = Position3D(2, 0, 0)
b1 = Position3D(2, 0, 1)
b2 = Position3D(2, 0, 2)
b3 = Position3D(2, 0, 3)


_cond_a_surface = CorrelationSurface(
    span=frozenset({ZXEdge(u=ZXNode(a0, Basis.Z), v=ZXNode(a0, Basis.Z))})
)
_cond_b_surface = CorrelationSurface(
    span=frozenset({ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b2, Basis.Z))})
)


def build_graph() -> BlockGraph:
    init_kind = ConditionalLeafCubeKind.ZXX_ZXZ.value[0]  # ZXX
    g = BlockGraph("conditional_obs_multi")
    g.add_cube(a0, init_kind)
    g.add_cube(a1, ConditionalLeafCubeKind.ZXX_ZXZ, condition=_cond_a_surface)
    g.add_pipe(a0, a1)
    g.add_cube(b0, init_kind)
    g.add_cube(b1, init_kind)
    g.add_cube(b2, init_kind)
    g.add_cube(b3, ConditionalLeafCubeKind.ZXX_ZXZ, condition=_cond_b_surface)
    g.add_pipe(b0, b1)
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)
    return g


# Base observable spans both columns up to (but not through) each conditional
# cube. Per-condition flips toggle the top-readout edge into that column's
# conditional cube — independent, so flat-XOR decomposes cleanly.
_shared_a_edge = ZXEdge(u=ZXNode(a0, Basis.X), v=ZXNode(a1, Basis.X))
_shared_b_edges = {
    ZXEdge(u=ZXNode(b0, Basis.X), v=ZXNode(b1, Basis.X)),
    ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(b2, Basis.X)),
    ZXEdge(u=ZXNode(b2, Basis.X), v=ZXNode(b3, Basis.X)),
}
_base_span = frozenset({_shared_a_edge, *_shared_b_edges})

# Same surface across the truth table → divergence comes from per-branch
# AbstractObservable lowering (kind.z=X vs Z changes which top-readouts fire).
# Decomposability is checked at emission against the lowered qubit sets.
_resolution = CorrelationSurface(span=_base_span)


conditional_observable = ConditionalCorrelationSurface(
    conditions=(_cond_a_surface, _cond_b_surface),
    resolutions={
        (False, False): _resolution,
        (False, True): _resolution,
        (True, False): _resolution,
        (True, True): _resolution,
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
