"""Two conditional cubes on distinct z-layers: each gets its own IfBlock."""

from tqec.compile.compile import compile_block_graph
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


g = BlockGraph("two_conditional_cubes_distinct_z")

# Column A: init at z=0, conditional at z=1
a0 = Position3D(0, 0, 0)
a1 = Position3D(0, 0, 1)
# Column B: init at z=0, mid at z=1, mid at z=2, conditional at z=3
b0 = Position3D(2, 0, 0)
b1 = Position3D(2, 0, 1)
b2 = Position3D(2, 0, 2)
b3 = Position3D(2, 0, 3)

init_kind = ConditionalLeafCubeKind.XZX_XZZ.value[0]  # XZX

g.add_cube(a0, init_kind)
g.add_cube(
    a1,
    ConditionalLeafCubeKind.XZX_XZZ,
    condition=CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(a0, Basis.Z), ZXNode(a0, Basis.Z))])
    ),
)
g.add_pipe(a0, a1)

g.add_cube(b0, init_kind)
g.add_cube(b1, init_kind)
g.add_cube(b2, init_kind)
g.add_cube(
    b3,
    ConditionalLeafCubeKind.XZX_XZZ,
    condition=CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(b2, Basis.Z), ZXNode(b2, Basis.Z))])
    ),
)
g.add_pipe(b0, b1)
g.add_pipe(b1, b2)
g.add_pipe(b2, b3)


cg = compile_block_graph(g, observables=None)
text = cg.generate_conditional_stim_text(k=1)
print(text)
n_if = text.count("IF(")
print(f"\n# IfBlocks emitted: {n_if}")
