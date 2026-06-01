"""Stage 3d: LayerTree.generate_conditional_circuit assembles a single
ConditionalCircuit from per-leaf branch annotations.
"""

from __future__ import annotations

import stim

from tqec.compile.compile import compile_block_graph
from tqec.compile.conditional.circuit import ConditionalCircuit, IfBlock
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _two_cube_graph(pair_name: str = "XZX_XZZ") -> BlockGraph:
    p0, p1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    init_kind = ConditionalLeafCubeKind[pair_name].value[0]
    g = BlockGraph(f"e2e {pair_name}")
    g.add_cube(p0, init_kind)
    cond = CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p0, Basis.Z), ZXNode(p0, Basis.Z))])
    )
    g.add_cube(p1, ConditionalLeafCubeKind[pair_name], condition=cond)
    g.add_pipe(p0, p1)
    return g


def test_generate_conditional_circuit_returns_conditional_circuit_with_ifblocks() -> None:
    g = _two_cube_graph("XZX_XZZ")
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    tree = cg.to_layer_tree()
    cc = tree.generate_conditional_circuit(
        k=1, condition_rec=-1, do_not_use_database=True
    )
    assert isinstance(cc, ConditionalCircuit)
    if_blocks = [e for e in cc.entries if isinstance(e, IfBlock)]
    assert if_blocks, "expected at least one IfBlock at top level"
    assert all(ib.condition_rec == -1 for ib in if_blocks)

    # QUBIT_COORDS preamble present, only once per qubit.
    qubit_coord_indices: list[int] = []
    for entry in cc.entries:
        if isinstance(entry, stim.CircuitInstruction) and entry.name == "QUBIT_COORDS":
            qubit_coord_indices.append(entry.targets_copy()[0].qubit_value)
    assert len(qubit_coord_indices) == len(set(qubit_coord_indices)), (
        "QUBIT_COORDS duplicated in preamble"
    )


def test_generate_conditional_circuit_text_renders_ifblock_syntax() -> None:
    g = _two_cube_graph("XZX_XZZ")
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    tree = cg.to_layer_tree()
    cc = tree.generate_conditional_circuit(
        k=1, condition_rec=-1, do_not_use_database=True
    )
    text = cc.to_stim_text()
    assert "IF(rec[-1])" in text
    assert "ELSE" in text
