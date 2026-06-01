"""Stage 3c: AnnotateCircuitOnLayerNode propagates ConditionalCircuit when
``condition_rec`` is supplied and the leaf carries a non-empty
``conditional_layers`` map.
"""

from __future__ import annotations

from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.compile import compile_block_graph
from tqec.compile.conditional.circuit import ConditionalCircuit, IfBlock
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.compile.tree.annotators.circuit import AnnotateCircuitOnLayerNode
from tqec.compile.tree.node import LayerNode, NodeWalker
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _build_conditional_tree():
    p0, p1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    g = BlockGraph("annot-cond")
    init_kind = ConditionalLeafCubeKind.XZX_XZZ.value[0]
    g.add_cube(p0, init_kind)
    cond = CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p0, Basis.Z), ZXNode(p0, Basis.Z))])
    )
    g.add_cube(p1, ConditionalLeafCubeKind.XZX_XZZ, condition=cond)
    g.add_pipe(p0, p1)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    return cg.to_layer_tree()


class _CollectLayoutNodes(NodeWalker):
    def __init__(self) -> None:
        self.nodes: list[LayerNode] = []

    def visit_node(self, node: LayerNode) -> None:
        if isinstance(node._layer, LayoutLayer):
            self.nodes.append(node)


def test_annotator_skips_conditional_when_condition_rec_missing() -> None:
    tree = _build_conditional_tree()
    walker = AnnotateCircuitOnLayerNode(k=1, condition_rec=None)
    tree._root.walk(walker)
    collector = _CollectLayoutNodes()
    tree._root.walk(collector)
    for node in collector.nodes:
        annotations = node.get_annotations(1)
        assert annotations.circuit is not None  # branch-zero ScheduledCircuit always set
        assert annotations.conditional_circuit is None


def test_annotator_populates_conditional_circuit_when_condition_rec_given() -> None:
    tree = _build_conditional_tree()
    walker = AnnotateCircuitOnLayerNode(k=1, condition_rec=-7)
    tree._root.walk(walker)

    collector = _CollectLayoutNodes()
    tree._root.walk(collector)

    conditional_nodes = [
        n for n in collector.nodes if n._layer.conditional_layers
    ]
    non_conditional_nodes = [
        n for n in collector.nodes if not n._layer.conditional_layers
    ]
    assert conditional_nodes, "expected at least one LayoutLayer with conditional_layers"

    for node in conditional_nodes:
        annotations = node.get_annotations(1)
        assert annotations.circuit is not None
        assert isinstance(annotations.conditional_circuit, ConditionalCircuit)

    for node in non_conditional_nodes:
        annotations = node.get_annotations(1)
        assert annotations.conditional_circuit is None

    # At least one of the conditional leaves should surface an IfBlock (the
    # final-measurement slice).
    any_if = any(
        any(isinstance(e, IfBlock) for e in n.get_annotations(1).conditional_circuit.entries)
        for n in conditional_nodes
    )
    assert any_if, "expected at least one IfBlock in the annotated ConditionalCircuit"
