"""Stage 6 first cut: end-to-end IF/ELSE emission for single conditional cube."""

from __future__ import annotations

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _condition() -> CorrelationSurface:
    p = Position3D(0, 0, 0)
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p, Basis.Z), ZXNode(p, Basis.Z))])
    )


def _single_conditional_graph(pair_name: str) -> BlockGraph:
    # Conditional cubes are leaves; they require exactly one pipe.
    # Build init cube -> temporal pipe -> conditional cube.
    g = BlockGraph(f"Conditional {pair_name}")
    init_kind = ConditionalLeafCubeKind[pair_name].value[0]
    p0 = Position3D(0, 0, 0)
    p1 = Position3D(0, 0, 1)
    g.add_cube(p0, init_kind)
    g.add_cube(p1, ConditionalLeafCubeKind[pair_name], condition=_condition())
    g.add_pipe(p0, p1)
    return g


def test_single_conditional_cube_emits_if_else_block() -> None:
    g = _single_conditional_graph("XZX_XZZ")
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(
        k=1, condition_recs={pos: -1 for pos in cg._conditional_blocks}
    )
    assert "IF(rec[-1]) {" in text
    assert "} ELSE {" in text
    # The IF/ELSE wraps the conditional measurement layer, where one branch
    # emits a single combined MX run and the other emits the CEO-interleaved
    # MX / M sequence.
    assert "MX 1 4 5 6" in text or "MX 1 8 9 10" in text


def test_no_conditional_block_passthrough() -> None:
    # Non-conditional graph: generate_conditional_stim_text should return the
    # same text as the regular generate_stim_circuit path.
    from tqec.computation.cube import ZXCube

    g = BlockGraph("Memory")
    g.add_cube(Position3D(0, 0, 0), ZXCube.XZX)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text_cond = cg.generate_conditional_stim_text(k=1, condition_recs={})
    circuit = cg.generate_stim_circuit(k=1)
    assert text_cond == str(circuit)


def test_multi_conditional_raises_not_implemented() -> None:
    import pytest

    g = BlockGraph("MultiCondTest")
    init0 = Position3D(0, 0, 0)
    cond0 = Position3D(0, 0, 1)
    init1 = Position3D(2, 0, 0)
    cond1 = Position3D(2, 0, 1)
    init_kind = ConditionalLeafCubeKind.XZX_XZZ.value[0]
    g.add_cube(init0, init_kind)
    g.add_cube(cond0, ConditionalLeafCubeKind.XZX_XZZ, condition=_condition())
    g.add_pipe(init0, cond0)
    g.add_cube(init1, init_kind)
    g.add_cube(cond1, ConditionalLeafCubeKind.XZX_XZZ, condition=_condition())
    g.add_pipe(init1, cond1)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(NotImplementedError, match="[Mm]ulti-conditional"):
        cg.generate_conditional_stim_text(
            k=1, condition_recs={pos: -1 for pos in cg._conditional_blocks}
        )
