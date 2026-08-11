import dataclasses

from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.circuit.qubit import GridQubit
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.raw import FlowSpecLayer, RawCircuitLayer
from tqec.compile.blocks.positioning import LayoutCubePosition2D
from tqec.compile.conditional.circuit import IfBlock
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
    position, and XORs the matching measurements into the observable.
    """
    if not any(c.cube.is_y_cube for c in obs_slice.top_readout_cubes):
        return
    for leaf in leaves:
        layout = leaf._layer
        if not isinstance(layout, LayoutLayer):
            continue
        for pos, layer in layout.layers.items():
            if not isinstance(layer, FlowSpecLayer):
                continue
            spec = layer.observable_spec(k)
            if not spec:
                continue
            if not isinstance(pos, LayoutCubePosition2D):
                raise TQECError("A RawCircuitLayer is only supported at a cube position.")
            eshape = layout.element_shape.to_shape_2d(k)
            mincube, _ = layout.bounds
            bp = pos.to_block_position()
            dx = (bp.x - mincube.x) * (eshape.x - 1)
            dy = (bp.y - mincube.y) * (eshape.y - 1)
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
    condition_recs_by_z: dict[int, list[int]],  # noqa: ARG001 — back-compat; per-bit recs now read off cond_observable.condition_recs
    min_z: int,
) -> None:
    """Annotate a truth-table-indexed logical observable on the tree.

    Conditional cubes in ``cond_observable.conditional_cube_positions`` each
    own the leaves at z-layers ``(prev_cube_z, this_cube_z]`` (after sorting by
    z). At every anchor leaf, all ``2 ** N`` branch resolutions are independently
    lowered to qubit sets. The N-condition flat-XOR decomposition

        O(key) = S ⊕ XOR_{i: key[i]} Δ_i

    is computed and validated; non-decomposable inputs (AND structure) are
    rejected with a descriptive error. ``S`` emits as a plain
    ``OBSERVABLE_INCLUDE`` on the trunk; each ``Δ_i`` is wrapped in a single
    ``IF(condition_recs_i) { OBSERVABLE_INCLUDE Δ_i }`` (no ELSE) at its
    owning cube's anchor leaf. Stim XORs all contributions into one logical
    observable.

    Args:
        root: root node of the tree.
        k: distance parameter.
        cond_observable: truth-table-indexed observable whose per-branch
            resolutions and per-bit ``condition_recs`` drive the emission.
        observable_index: index of the observable in the circuit.
        observable_builder: builder that computes and constructs qubits whose
            measurements will be included in the logical observable.
        condition_recs_by_z: retained for backward compatibility only and no
            longer read; per-bit measurement records are taken from
            ``cond_observable.condition_recs``.
        min_z: smallest z-layer index spanned by the tree, used to convert the
            bindings' absolute anchor z into ``root.children`` offsets.

    """
    import stim  # local: avoid module-level dep when unused

    from tqec.compile.conditional.condition_recs import (  # noqa: PLC0415
        _collect_pre_cond_entries,
        _compute_tail_shifts,
    )

    bindings = cond_observable.condition_bindings
    n_bits = len(bindings)
    if n_bits == 0:
        raise TQECError(
            "ConditionalAbstractObservable has no condition_bindings."
        )
    branches = cond_observable.branches
    expected_keys = {
        tuple(bool((i >> j) & 1) for j in range(n_bits)) for i in range(2**n_bits)
    }
    if set(branches.keys()) != expected_keys:
        raise TQECError(
            "ConditionalAbstractObservable.branches must cover all "
            f"2^{n_bits} truth-table keys."
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

    bits_in_z_order = sorted(range(n_bits), key=lambda b: anchor_idx_by_bit[b])
    sorted_cube_z_indices = [anchor_idx_by_bit[b] for b in bits_in_z_order]

    # Per-bit rec lists from the resolver pipeline (graph.py populates
    # ``cond_observable.condition_recs`` before calling into the annotator).
    if cond_observable.condition_recs is None or len(cond_observable.condition_recs) != n_bits:
        raise TQECError(
            "ConditionalAbstractObservable.condition_recs has not been "
            "populated; call resolve_condition_recs (and the surface-anchored "
            "resolver) before annotation."
        )
    recs_per_bit: list[list[int]] = [list(r) for r in cond_observable.condition_recs]

    max_idx = sorted_cube_z_indices[-1]
    # Cap max_idx so we don't try to collect entries beyond the tree.
    max_walk_z = min(max_idx, n_layers - 1)
    entries, subtree_leaves = _collect_pre_cond_entries(root, k, max_idx + 1)
    tail_shifts_max = _compute_tail_shifts(entries)
    entry_by_leaf_id = {id(e.leaf): (e, i) for i, e in enumerate(entries)}

    # For each z-sorted anchor, find the index of its last entry and the
    # count of measurements after that entry up to the end of the walk.
    # Subtracting tail_beyond from tail_shifts_max maps rec offsets from the
    # max-anchor frame into that anchor's IfBlock frame.
    end_entry_idx_per_sorted: list[int] = []
    j = 0
    for anchor_idx in sorted_cube_z_indices:
        while j < len(entries) and entries[j].z <= anchor_idx:
            j += 1
        end_entry_idx_per_sorted.append(j - 1)
    running_total = sum(e.num_measurements for e in entries)
    cumulative_through: list[int] = []
    cum = 0
    for e in entries:
        cum += e.num_measurements
        cumulative_through.append(cum)
    tail_beyond_per_bit: list[int] = [0] * n_bits
    for s, end_idx in enumerate(end_entry_idx_per_sorted):
        through = cumulative_through[end_idx] if end_idx >= 0 else 0
        tail_beyond_per_bit[bits_in_z_order[s]] = running_total - through

    delta_per_bit: list[list[int]] = [[] for _ in range(n_bits)]

    def _collect_recs(leaf: LayerNode, qubits: set, target: list[int], bit: int) -> None:
        entry, idx = entry_by_leaf_id[id(leaf)]
        shift = tail_shifts_max[idx] - tail_beyond_per_bit[bit]
        for q in qubits:
            if q not in entry.records:
                continue
            local = entry.records[q][-1]
            target.append(local - shift)

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
            deltas: list[frozenset] = []
            for i in range(n_bits):
                deltas.append(frozenset(qubits_by_key[flip_keys[i]] ^ shared))
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
                    get_observable_with_measurement_records(
                        shared, meas, observable_index
                    )
                )
            for bit, delta in enumerate(deltas):
                _collect_recs(anchor_leaf, delta, delta_per_bit[bit], bit)

    for bit in range(n_bits):
        delta_recs = delta_per_bit[bit]
        if not delta_recs:
            continue
        # Anchor leaf for the IfBlock: last leaf at the binding's anchor z
        # (clamped if the anchor sits beyond the walked range).
        leaf_z = min(anchor_idx_by_bit[bit], len(subtree_leaves) - 1)
        cube_leaf = subtree_leaves[leaf_z][-1]
        then_body: list = [
            stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(o) for o in sorted(delta_recs)],
                [observable_index],
            )
        ]
        annotations = cube_leaf.get_annotations(k)
        if annotations.conditional_observables is None:
            annotations.conditional_observables = []
        annotations.conditional_observables.append(
            IfBlock(
                condition_recs=recs_per_bit[bit],
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
