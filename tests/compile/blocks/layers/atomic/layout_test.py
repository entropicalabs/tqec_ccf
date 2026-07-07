from typing import Final

import pytest

from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.positioning import LayoutPosition2D
from tqec.plaquette.plaquette import Plaquettes
from tqec.plaquette.rpng.rpng import RPNGDescription
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.templates._testing import FixedTemplate
from tqec.templates.layout import LayoutTemplate
from tqec.templates.qubit import QubitTemplate
from tqec.utils.exceptions import TQECError
from tqec.utils.frozendefaultdict import FrozenDefaultDict
from tqec.utils.position import BlockPosition2D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

LOGICAL_QUBIT_SIDE: Final = LinearFunction(4, 5)
LOGICAL_QUBIT_SHAPE: Final = PhysicalQubitScalable2D(LOGICAL_QUBIT_SIDE, LOGICAL_QUBIT_SIDE)
_TRANSLATOR: Final = DefaultRPNGTranslator()
_EMPTY_PLAQUETTE = _TRANSLATOR.translate(RPNGDescription.empty())


@pytest.fixture(name="empty_plaquette_layer")
def empty_plaquette_layer_fixture() -> PlaquetteLayer:
    template = QubitTemplate()
    plaquettes = Plaquettes(FrozenDefaultDict({}, default_value=_EMPTY_PLAQUETTE))
    return PlaquetteLayer(template, plaquettes)


@pytest.fixture(name="plaquette_layer")
def plaquette_layer_fixture() -> PlaquetteLayer:
    template = FixedTemplate([[1]])
    plaquettes = Plaquettes(
        FrozenDefaultDict(
            {1: _TRANSLATOR.translate(RPNGDescription.from_string("-x1- -x2- -x3- -x4-"))},
            default_value=_EMPTY_PLAQUETTE,
        )
    )
    return PlaquetteLayer(template, plaquettes)


def test_creation(empty_plaquette_layer: PlaquetteLayer) -> None:
    with pytest.raises(TQECError, match=r".*should have at least one layer.$"):
        LayoutLayer({}, LOGICAL_QUBIT_SHAPE)
    pos = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    LayoutLayer({pos: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-93485, 12))
    LayoutLayer({pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    LayoutLayer({pos: empty_plaquette_layer, pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)


def test_bounds(empty_plaquette_layer: PlaquetteLayer) -> None:
    with pytest.raises(TQECError, match=r".*should have at least one layer.$"):
        LayoutLayer({}, LOGICAL_QUBIT_SHAPE)
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-93485, 12))
    layer = LayoutLayer({pos1: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    assert layer.bounds == (BlockPosition2D(0, 0), BlockPosition2D(0, 0))
    layer = LayoutLayer(
        {pos1: empty_plaquette_layer, pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE
    )
    assert layer.bounds == (BlockPosition2D(-93485, 0), BlockPosition2D(0, 12))


def test_scalable_qubit_bound() -> None:
    plaquettes = Plaquettes(FrozenDefaultDict({}, default_value=_EMPTY_PLAQUETTE))
    fixed_layer = PlaquetteLayer(FixedTemplate([[1]]), plaquettes)
    qubit_layer = PlaquetteLayer(QubitTemplate(), plaquettes)

    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-1, 12))
    # Important note for the hard-coded values here: plaquette origin are currently located in
    # the center of the plaquette. That means that the top-left qubit is in position (-1, -1). So
    # we should apply a shift to recover the correct bounds. This is why there are `` - (1, 1)``
    # to each PhysicalQubitScalable2D.
    fixed_layout_layer = LayoutLayer(
        {pos1: fixed_layer, pos2: fixed_layer}, fixed_layer.scalable_shape
    )
    assert fixed_layout_layer.qubit_bounds == (
        PhysicalQubitScalable2D(LinearFunction(0, -2), LinearFunction(0, 0)) - (1, 1),
        PhysicalQubitScalable2D(LinearFunction(0, 2), LinearFunction(0, 26)) - (1, 1),
    )

    qubit_layout_layer = LayoutLayer(
        {pos1: qubit_layer, pos2: qubit_layer}, qubit_layer.scalable_shape
    )
    # qubit_layer.scalable_shape is "4x + 5" (in qubit coordinates, NOT in plaquette coordinates).
    assert qubit_layout_layer.qubit_bounds == (
        PhysicalQubitScalable2D(LinearFunction(-4, -4), LinearFunction(0, 0)) - (1, 1),
        PhysicalQubitScalable2D(LinearFunction(4, 4), LinearFunction(52, 52)) - (1, 1),
    )


def test_scalable_timesteps(empty_plaquette_layer: PlaquetteLayer) -> None:
    with pytest.raises(TQECError, match=r".*should have at least one layer.$"):
        LayoutLayer({}, LOGICAL_QUBIT_SHAPE)
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    layer = LayoutLayer({pos1: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    assert layer.scalable_timesteps == LinearFunction(0, 1)


def test_scalable_shape(empty_plaquette_layer: PlaquetteLayer) -> None:
    with pytest.raises(TQECError, match=r".*should have at least one layer.$"):
        LayoutLayer({}, LOGICAL_QUBIT_SHAPE)
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-1, 12))
    layer = LayoutLayer({pos1: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    assert layer.scalable_shape == LOGICAL_QUBIT_SHAPE
    layer = LayoutLayer(
        {pos1: empty_plaquette_layer, pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE
    )
    assert layer.scalable_shape == PhysicalQubitScalable2D(
        2 * LOGICAL_QUBIT_SIDE - 1, 13 * (LOGICAL_QUBIT_SIDE - 1) + 1
    )


def test_to_template_and_plaquettes_single(empty_plaquette_layer: PlaquetteLayer) -> None:
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    layer = LayoutLayer({pos1: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    template, plaquettes = layer.to_template_and_plaquettes()

    assert isinstance(template, LayoutTemplate)
    assert len(template._layout) == 1
    pos_layout, template_layout = next(iter(template._layout.items()))
    assert pos_layout == BlockPosition2D(0, 0)
    assert type(template_layout) is type(empty_plaquette_layer.template)
    assert plaquettes == empty_plaquette_layer.plaquettes


def test_to_template_and_plaquettes_multiple(empty_plaquette_layer: PlaquetteLayer) -> None:
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-1, 12))
    layer = LayoutLayer(
        {pos1: empty_plaquette_layer, pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE
    )
    template, plaquettes = layer.to_template_and_plaquettes()

    assert isinstance(template, LayoutTemplate)
    assert len(template._layout) == 2
    assert set(template._layout.keys()) == {
        pos1.to_block_position(),
        pos2.to_block_position(),
    }
    for t in template._layout.values():
        assert type(t) is type(empty_plaquette_layer.template)
    assert plaquettes.collection == (
        FrozenDefaultDict(
            {
                **plaquettes.collection,
                **{
                    i + empty_plaquette_layer.template.expected_plaquettes_number: plaq
                    for i, plaq in plaquettes.collection.items()
                },
            },
            default_value=plaquettes.collection.default_value,
        )
    )


def test_to_conditional_circuit_requires_conditional_layers(
    empty_plaquette_layer: PlaquetteLayer,
) -> None:
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    layer = LayoutLayer({pos1: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE)
    with pytest.raises(TQECError):
        layer.to_conditional_circuit(k=1, condition_recs=[-1])


def test_to_conditional_circuit_weaves_ifblock_for_divergent_plaquette() -> None:
    """Exercise to_conditional_circuit on a LayoutLayer produced by the real
    compile pipeline for a single-cube conditional graph. Asserts at least one
    IfBlock surfaces with the requested condition_recs.
    """
    from tqec.compile.blocks.layers.atomic.layout import LayoutLayer as _LL
    from tqec.compile.compile import compile_block_graph
    from tqec.compile.convention import FIXED_BULK_CONVENTION
    from tqec.compile.conditional.circuit import ConditionalCircuit, IfBlock
    from tqec.compile.tree.node import LayerNode, NodeWalker
    from tqec.computation.block_graph import BlockGraph
    from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
    from tqec.computation.cube import ConditionalLeafCubeKind
    from tqec.utils.enums import Basis
    from tqec.utils.position import Position3D

    p0, p1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    init_kind = ConditionalLeafCubeKind.XZX_XZZ.value[0]
    g = BlockGraph("ll-cond")
    g.add_cube(p0, init_kind)
    cond = CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p0, Basis.Z), ZXNode(p0, Basis.Z))])
    )
    g.add_cube(p1, ConditionalLeafCubeKind.XZX_XZZ, condition=cond)
    g.add_pipe(p0, p1)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    tree = cg.to_layer_tree()

    found_layers: list[_LL] = []

    class _CollectConditionalLayoutLayers(NodeWalker):
        def visit_node(self, node: LayerNode) -> None:
            if isinstance(node._layer, _LL) and node._layer.conditional_layers:
                found_layers.append(node._layer)

    tree._root.walk(_CollectConditionalLayoutLayers())
    assert found_layers, "no LayoutLayer with conditional_layers in compiled tree"
    # Stabiliser-round LayoutLayers carry conditional_layers byte-equal to layers
    # (branches diverge only in the final measurement slice); search across all
    # conditional layers and assert at least one surfaces an IfBlock.
    all_if_blocks: list[IfBlock] = []
    for ll in found_layers:
        cc = ll.to_conditional_circuit(k=1, condition_recs=[-7])
        assert isinstance(cc, ConditionalCircuit)
        all_if_blocks.extend(e for e in cc.entries if isinstance(e, IfBlock))
    assert all_if_blocks, "expected at least one IfBlock for divergent plaquette content"
    assert all(ib.condition_recs == [-7] for ib in all_if_blocks)


def test_scalable_num_moments(
    empty_plaquette_layer: PlaquetteLayer, plaquette_layer: PlaquetteLayer
) -> None:
    pos1 = LayoutPosition2D.from_block_position(BlockPosition2D(0, 0))
    pos2 = LayoutPosition2D.from_block_position(BlockPosition2D(-1, 12))
    assert (
        LayoutLayer(
            {pos1: empty_plaquette_layer, pos2: empty_plaquette_layer}, LOGICAL_QUBIT_SHAPE
        ).scalable_num_moments
        == empty_plaquette_layer.scalable_num_moments
    )

    assert (
        LayoutLayer(
            {pos1: empty_plaquette_layer, pos2: plaquette_layer}, LOGICAL_QUBIT_SHAPE
        ).scalable_num_moments
        == plaquette_layer.scalable_num_moments
    )
