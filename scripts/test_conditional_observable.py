"""One BlockGraph + one ConditionalCorrelationSurface with two branch shapes.

The conditional cube at ``t2`` resolves to either branch-zero (kind ZXX,
observable runs b1 -> c1 -> c2 -> t2) or branch-one (kind ZXZ, observable
runs b1 -> c1 -> t1). The generated Stim text carries per-branch
``OBSERVABLE_INCLUDE`` annotations inside the IF/ELSE.
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


b1 = Position3D(0, 0, 0)
c1 = Position3D(0, 0, 1)
c2 = Position3D(1, 0, 1)
t1 = Position3D(0, 0, 2)
t2 = Position3D(1, 0, 2)


def build_graph() -> BlockGraph:
    g = BlockGraph("conditional_observable_demo")
    g.add_cube(b1, "ZXX")
    g.add_cube(c1, "ZXZ")
    g.add_cube(c2, "ZXZ")
    g.add_cube(t1, "ZXX")
    g.add_cube(
        t2,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset(
                {ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(c1, Basis.Z))}
            )
        ),
    )
    g.add_pipe(b1, c1)
    g.add_pipe(c1, c2)
    g.add_pipe(c1, t1)
    g.add_pipe(c2, t2)
    return g


branch_zero_surface = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
            ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(c2, Basis.X)),
            ZXEdge(u=ZXNode(c2, Basis.X), v=ZXNode(t2, Basis.X)),
        }
    )
)

branch_one_surface = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
            ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(t1, Basis.X)),
        }
    )
)

conditional_observable = ConditionalCorrelationSurface(
    branch_zero=branch_zero_surface,
    branch_one=branch_one_surface,
    conditional_cube_positions=(t2,),
)

g = build_graph()
cg = compile_block_graph(g, observables=[conditional_observable])
text = cg.generate_conditional_stim_text(k=1)
print(text)
print()
print(f"# OBSERVABLE_INCLUDE in trunk:   {text.count(chr(10) + 'OBSERVABLE_INCLUDE')}")
print(f"# IF/ELSE blocks:                {text.count('IF(')}")
