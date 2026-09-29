import dataclasses

import stim

from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.circuit.qubit import GridQubit
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.raw import FlowSpecLayer, RawCircuitLayer
from tqec.compile.blocks.positioning import LayoutCubePosition2D
from tqec.compile.conditional.circuit import IfBlock
from tqec.compile.conditional.condition_recs import leaf_measurement_spans, ordered_leaves
from tqec.compile.observables.abstract_observable import (
    AbstractObservable,
    ConditionalAbstractObservable,
)
from tqec.compile.observables.builder import (
    ObservableBuilder,
    ObservableComponent,
    get_observable_with_measurement_records,
)
from tqec.compile.tree.node import LayerNode
from tqec.templates.layout import LayoutTemplate
from tqec.utils.exceptions import TQECError


def _template_from_node(node: LayerNode) -> LayoutTemplate | None:
    """Return the plaquette template of a leaf.

    A leaf mixing raw Y-cap rounds with plaquette rounds is tolerated: the raw
    positions carry no template and are dropped. Returns ``None`` when the leaf
    is entirely raw (a lone Y cap).
    """
    layout = node._layer
    assert isinstance(layout, LayoutLayer)
    raw_positions = {
        pos for pos, layer in layout.layers.items() if isinstance(layer, RawCircuitLayer)
    }
    if not raw_positions:
        template, _ = layout.to_template_and_plaquettes()
        return template
    plaquette_layers = {
        pos: layer for pos, layer in layout.layers.items() if pos not in raw_positions
    }
    if not plaquette_layers:
        return None
    sub = LayoutLayer(plaquette_layers, layout.element_shape)
    template, _ = sub.to_template_and_plaquettes()
    return template


def _annotate_y_cube_readouts(
    leaves: list[LayerNode],
    obs_slice: AbstractObservable,
    k: int,
    observable_index: int,
) -> None:
    """Include a Y-basis measurement cap's logical-Y readout in the observable.

    A Y cube's logical readout is *not* the final-round data measurement (the
    normal top-readout path); it is the set of transition-round records that
    reconstruct the logical Y operator, published by that round as its
    ``observable_spec``. The representative used there is the one the observable
    builder expects for an arriving correlation surface --- the patch's middle
    lines --- rather than Gidney's corner-anchored one; see
    ``_tqec_logical_y_observable_spec``.

    This walks the z-slice's leaves for the round carrying such a spec, shifts
    its qubit coordinates (given in the local element frame) to the cube's
    position, and XORs the matching measurements into the observable. Only the Y
    cubes the surface actually reaches are read: another Y cube sharing the
    slice belongs to another surface, and its readout is random for this one.
    """
    y_positions = {
        (c.cube.position.x, c.cube.position.y)
        for c in obs_slice.top_readout_cubes
        if c.cube.is_y_cube
    }
    if not y_positions:
        return
    for leaf in leaves:
        layout = leaf._layer
        if not isinstance(layout, LayoutLayer):
            continue
        for pos, layer in layout.layers.items():
            if not isinstance(layer, FlowSpecLayer):
                continue
            if not isinstance(pos, LayoutCubePosition2D):
                raise TQECError("A RawCircuitLayer is only supported at a cube position.")
            block = pos.to_block_position()
            if (block.x, block.y) not in y_positions:
                continue
            spec = layer.observable_spec(k)
            if not spec:
                continue
            eshape = layout.element_shape.to_shape_2d(k)
            bp = block
            # The spec is in patch-local coordinates while ``records`` below is
            # keyed by the circuit's qubit coordinates, which are absolute: a
            # block at ``bp`` starts at ``bp * (eshape - 1)``. Offsetting by the
            # position *relative* to ``layout.bounds`` instead agrees only when
            # this layer's minimum block position is zero, and silently selects
            # the wrong measurements otherwise.
            dx = bp.x * (eshape.x - 1)
            dy = bp.y * (eshape.y - 1)
            qubits = {GridQubit(c[0] + dx, c[1] + dy) for c in spec}
            circuit = leaf.get_annotations(k).circuit
            assert circuit is not None
            records = MeasurementRecordsMap.from_scheduled_circuit(circuit)
            leaf.get_annotations(k).observables.append(
                get_observable_with_measurement_records(qubits, records, observable_index)
            )


def _get_ordered_leaves(root: LayerNode) -> list[LayerNode]:
    """Return the leaves of the tree in time order."""
    if root.is_leaf:
        return [root]
    return [n for child in root.children for n in _get_ordered_leaves(child)]


def annotate_observable(
    root: LayerNode,
    k: int,
    observable: AbstractObservable,
    observable_index: int,
    observable_builder: ObservableBuilder,
) -> None:
    """Annotates the observables on the tree.

    Args:
        root: root node of the tree.
        k: distance parameter.
        observable: observable to annotate.
        observable_index: index of the observable in the circuit.
        observable_builder: builder that computes and constructs qubits whose
            measurements will be included in the logical observable.

    """
    for z, subtree_root in enumerate(root.children):
        leaves = _get_ordered_leaves(subtree_root)
        obs_slice = observable.slice_at_z(z)
        # Annotate the observable at the bottom of the blocks
        _annotate_observable_at_node(
            leaves[0],
            obs_slice,
            k,
            observable_index,
            observable_builder,
            ObservableComponent.BOTTOM_STABILIZERS,
        )
        readout_layer = leaves[-1]
        if obs_slice.temporal_hadamard_pipes:
            readout_layer = leaves[-2]
            # Annotate the observable at the realignment layer in temporal hadamard pipes
            _annotate_observable_at_node(
                leaves[-1],
                obs_slice,
                k,
                observable_index,
                observable_builder,
                ObservableComponent.REALIGNMENT,
            )
        # Annotate the observable at the top of the blocks
        _annotate_observable_at_node(
            readout_layer,
            obs_slice,
            k,
            observable_index,
            observable_builder,
            ObservableComponent.TOP_READOUTS,
        )
        # A Y-basis measurement cap contributes its transition-round logical-Y
        # readout, not a final-round data readout, so it is handled separately
        # (and excluded from the normal top-readout path above).
        _annotate_y_cube_readouts(leaves, obs_slice, k, observable_index)


def annotate_conditional_observable(
    root: LayerNode,
    k: int,
    cond_observable: ConditionalAbstractObservable,
    observable_index: int,
    observable_builder: ObservableBuilder,
    min_z: int,
) -> None:
    """Annotate a truth-table-indexed logical observable on the tree.

    At every leaf where the observable reads measurements, all ``2 ** N`` branch
    resolutions are independently lowered to qubit sets. The N-condition
    flat-XOR decomposition

        O(key) = S ⊕ XOR_{i: key[i]} Δ_i

    is computed and validated; non-decomposable inputs (AND structure) are
    rejected with a descriptive error. ``S`` emits as a plain
    ``OBSERVABLE_INCLUDE`` at each leaf it reads; each ``Δ_i`` is wrapped in a
    single ``IF(condition_i) { OBSERVABLE_INCLUDE Δ_i }`` (no ELSE), placed on
    the latest leaf among the one its condition is anchored to and the ones
    ``Δ_i`` reads, so that every measurement it names already happened. Stim
    XORs all contributions into one logical observable.

    Args:
        root: root node of the tree.
        k: distance parameter.
        cond_observable: truth-table-indexed observable whose per-branch
            resolutions and per-bit ``condition_recs`` drive the emission.
        observable_index: index of the observable in the circuit.
        observable_builder: builder that computes and constructs qubits whose
            measurements will be included in the logical observable.
        min_z: smallest z-layer index spanned by the tree, used to convert the
            bindings' absolute anchor z into ``root.children`` offsets.

    Raises:
        TQECError: if ``cond_observable`` has no condition, does not cover every
            truth-table key, anchors a condition outside the tree, has not had
            its conditions resolved, is not XOR-decomposable, or would have to
            place an ``IF`` block inside a repeated layer.

    """
    bindings = cond_observable.condition_bindings
    n_bits = len(bindings)
    if n_bits == 0:
        raise TQECError("ConditionalAbstractObservable has no condition_bindings.")
    branches = cond_observable.branches
    expected_keys = {tuple(bool((i >> j) & 1) for j in range(n_bits)) for i in range(2**n_bits)}
    if set(branches.keys()) != expected_keys:
        raise TQECError(
            f"ConditionalAbstractObservable.branches must cover all 2^{n_bits} truth-table keys."
        )

    # anchor_z_by_bit: z-index (relative to min_z) where the i-th condition's
    # IfBlock physically sits. Cube-anchored bits use cube.z; surface-anchored
    # use anchor_z = max(condition span z) + 1. anchor_idx may equal
    # len(root.children) when the condition lives at the final z-layer (the
    # IfBlock then attaches to the last leaf of that layer — see anchor leaf
    # selection below).
    anchor_idx_by_bit = [b.anchor_z - min_z for b in bindings]
    n_layers = len(root.children)
    for bit, idx in enumerate(anchor_idx_by_bit):
        if idx < 0 or idx > n_layers:
            raise TQECError(
                f"ConditionalCorrelationSurface bit {bit} anchors at "
                f"z={min_z + idx} which is outside the tree's z range "
                f"[{min_z}, {min_z + n_layers}]."
            )

    # Per-bit rec lists from the resolver pipeline (graph.py populates
    # ``cond_observable.condition_recs`` before calling into the annotator).
    if cond_observable.condition_recs is None or len(cond_observable.condition_recs) != n_bits:
        raise TQECError(
            "ConditionalAbstractObservable.condition_recs has not been "
            "populated; call resolve_condition_recs (and the surface-anchored "
            "resolver) before annotation."
        )
    recs_per_bit: list[list[int]] = [list(r) for r in cond_observable.condition_recs]

    spans = leaf_measurement_spans(root, k)
    subtree_leaves = [ordered_leaves(subtree) for subtree in root.children]
    # Absolute indices of the measurements each Δ_i reads, and the leaves it
    # reads them in (the IF block has to come after the last of those).
    delta_per_bit: list[list[int]] = [[] for _ in range(n_bits)]
    delta_leaves_per_bit: list[list[LayerNode]] = [[] for _ in range(n_bits)]

    def _anchor_actions(
        leaves: list[LayerNode],
        slices: list[AbstractObservable],
    ) -> list[tuple[LayerNode, ObservableComponent]]:
        actions: list[tuple[LayerNode, ObservableComponent]] = [
            (leaves[0], ObservableComponent.BOTTOM_STABILIZERS)
        ]
        readout_leaf = leaves[-1]
        if any(sl.temporal_hadamard_pipes for sl in slices):
            readout_leaf = leaves[-2]
            actions.append((leaves[-1], ObservableComponent.REALIGNMENT))
        actions.append((readout_leaf, ObservableComponent.TOP_READOUTS))
        return actions

    zero_key = tuple(False for _ in range(n_bits))
    flip_keys = [tuple(j == i for j in range(n_bits)) for i in range(n_bits)]

    for z_idx, leaves in enumerate(subtree_leaves):
        slice_by_key = {key: obs.slice_at_z(z_idx) for key, obs in branches.items()}
        for anchor_leaf, component in _anchor_actions(leaves, list(slice_by_key.values())):
            assert isinstance(anchor_leaf._layer, LayoutLayer)
            template, _ = anchor_leaf._layer.to_template_and_plaquettes()
            qubits_by_key = {
                key: observable_builder.build(k, template, sl, component)
                for key, sl in slice_by_key.items()
            }
            shared = qubits_by_key[zero_key]
            deltas: list[frozenset] = [
                frozenset(qubits_by_key[flip_keys[i]] ^ shared) for i in range(n_bits)
            ]
            # Validate flat-XOR decomposability for every key.
            for key, qubits in qubits_by_key.items():
                predicted = shared
                for i, bit in enumerate(key):
                    if bit:
                        predicted = predicted ^ deltas[i]
                if predicted != qubits:
                    raise TQECError(
                        "ConditionalCorrelationSurface is not XOR-decomposable "
                        f"across its {n_bits} conditions at z={min_z + z_idx}, "
                        f"component={component.name}: O({key}) does not equal "
                        "S ⊕ XOR_{i: key[i]} Δ_i. Only flat-XOR-separable "
                        "observables are currently supported."
                    )
            if shared:
                circuit = anchor_leaf.get_annotations(k).circuit
                assert circuit is not None
                meas = MeasurementRecordsMap.from_scheduled_circuit(circuit)
                anchor_leaf.get_annotations(k).observables.append(
                    get_observable_with_measurement_records(shared, meas, observable_index)
                )
            span = spans[id(anchor_leaf)]
            for bit, delta in enumerate(deltas):
                indices = [i for q in delta if (i := span.absolute_index(q)) is not None]
                if indices:
                    delta_per_bit[bit].extend(indices)
                    delta_leaves_per_bit[bit].append(anchor_leaf)

    for bit in range(n_bits):
        delta_indices = delta_per_bit[bit]
        if not delta_indices:
            continue
        # The leaf the condition is anchored to: the last leaf of its anchor
        # z-layer, clamped when the anchor sits past the last layer.
        leaf_z = min(anchor_idx_by_bit[bit], len(subtree_leaves) - 1)
        candidates = [subtree_leaves[leaf_z][-1], *delta_leaves_per_bit[bit]]
        host_leaf = max(candidates, key=lambda leaf: spans[id(leaf)].end)
        host = spans[id(host_leaf)]
        if host.repeated:
            raise TQECError(
                f"The IF block for condition {bit} of a ConditionalCorrelationSurface "
                "would sit inside a repeated layer, which conditional emission does "
                "not support."
            )
        then_body: list = [
            stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(i - host.end) for i in sorted(delta_indices)],
                [observable_index],
            )
        ]
        annotations = host_leaf.get_annotations(k)
        if annotations.conditional_observables is None:
            annotations.conditional_observables = []
        annotations.conditional_observables.append(
            IfBlock(
                condition_recs=list(recs_per_bit[bit]),
                then_body=then_body,
                else_body=None,
            )
        )


def _annotate_observable_at_node(
    node: LayerNode,
    obs_slice: AbstractObservable,
    k: int,
    observable_index: int,
    observable_builder: ObservableBuilder,
    component: ObservableComponent,
) -> None:
    circuit = node.get_annotations(k).circuit
    assert circuit is not None
    measurement_record = MeasurementRecordsMap.from_scheduled_circuit(circuit)
    assert isinstance(node._layer, LayoutLayer)
    template = _template_from_node(node)
    if template is None:
        return
    # Y cubes contribute via their transition-round logical readout, handled by
    # _annotate_y_cube_readouts; drop them from the template-driven builder so it
    # does not pick up the final-round data measurements at their position.
    obs_slice = dataclasses.replace(
        obs_slice,
        top_readout_cubes=frozenset(c for c in obs_slice.top_readout_cubes if not c.cube.is_y_cube),
    )
    obs_qubits = observable_builder.build(k, template, obs_slice, component)
    if obs_qubits:
        obs_annotation = get_observable_with_measurement_records(
            obs_qubits, measurement_record, observable_index
        )
        node.get_annotations(k).observables.append(obs_annotation)
