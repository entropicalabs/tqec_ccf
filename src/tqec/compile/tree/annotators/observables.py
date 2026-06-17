from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
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
from tqec.utils.exceptions import TQECError


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


def annotate_conditional_observable(
    root: LayerNode,
    k: int,
    cond_observable: ConditionalAbstractObservable,
    observable_index: int,
    observable_builder: ObservableBuilder,
    condition_recs_by_z: dict[int, list[int]],
    min_z: int,
) -> None:
    """Annotate a branch-aware logical observable on the tree.

    For each leaf at ``z <= cube_z``, both branch surfaces are lowered to qubit
    sets via :class:`ObservableBuilder`. Shared qubits emit a plain
    ``OBSERVABLE_INCLUDE`` on that leaf's trunk. Divergent qubits' measurement
    records are shifted into the IfBlock emission frame (the conditional
    cube's last leaf at z=cube_z) and bundled into a single per-branch
    ``OBSERVABLE_INCLUDE`` instruction, then wrapped in an
    :class:`IfBlock` gated by the cube's ``condition_recs`` and emitted on
    the cube's last leaf.

    Currently only a single conditional cube per observable is supported;
    multi-cube ConditionalCorrelationSurface is reserved for a follow-up.
    """
    import stim  # local: avoid module-level dep when unused

    from tqec.compile.conditional.condition_recs import (  # noqa: PLC0415
        _collect_pre_cond_entries,
        _compute_tail_shifts,
    )

    if len(cond_observable.conditional_cube_positions) != 1:
        raise NotImplementedError(
            "ConditionalCorrelationSurface with multiple "
            "conditional_cube_positions is not yet supported. Use exactly one "
            "cube position; multi-cube emission is a follow-up."
        )
    cube_pos = cond_observable.conditional_cube_positions[0]
    cube_z = cube_pos.z
    cube_z_idx = cube_z - min_z
    if cube_z_idx < 0 or cube_z_idx >= len(root.children):
        raise TQECError(
            f"ConditionalCorrelationSurface names a conditional cube at "
            f"z={cube_z} that does not correspond to any z-layer in the "
            f"tree (min_z={min_z}, layers={len(root.children)})."
        )
    recs = condition_recs_by_z.get(cube_z)
    if recs is None:
        raise TQECError(
            f"ConditionalCorrelationSurface references a conditional cube at "
            f"z={cube_z} but no resolved condition_recs were produced for "
            "that z-layer. Ensure the named cube is actually conditional."
        )

    # Walk leaves at z <= cube_z. _collect_pre_cond_entries takes a strict
    # upper bound, so pass cube_z_idx + 1.
    entries, subtree_leaves = _collect_pre_cond_entries(
        root, k, cube_z_idx + 1
    )
    tail_shifts = _compute_tail_shifts(entries)
    entry_by_leaf_id = {id(e.leaf): (e, i) for i, e in enumerate(entries)}

    div_zero_recs: list[int] = []
    div_one_recs: list[int] = []

    def _collect_recs(leaf: LayerNode, qubits: set, target: list[int]) -> None:
        entry, idx = entry_by_leaf_id[id(leaf)]
        shift = tail_shifts[idx]
        for q in qubits:
            if q not in entry.records:
                continue
            local = entry.records[q][-1]
            target.append(local - shift)

    def _anchor_actions(leaves: list[LayerNode], slice_zero: AbstractObservable,
                        slice_one: AbstractObservable) -> list[tuple[LayerNode, ObservableComponent]]:
        actions: list[tuple[LayerNode, ObservableComponent]] = [
            (leaves[0], ObservableComponent.BOTTOM_STABILIZERS)
        ]
        readout_leaf = leaves[-1]
        if slice_zero.temporal_hadamard_pipes or slice_one.temporal_hadamard_pipes:
            readout_leaf = leaves[-2]
            actions.append((leaves[-1], ObservableComponent.REALIGNMENT))
        actions.append((readout_leaf, ObservableComponent.TOP_READOUTS))
        return actions

    for z_idx, leaves in enumerate(subtree_leaves):
        slice_zero = cond_observable.branch_zero.slice_at_z(z_idx)
        slice_one = cond_observable.branch_one.slice_at_z(z_idx)
        for anchor_leaf, component in _anchor_actions(leaves, slice_zero, slice_one):
            assert isinstance(anchor_leaf._layer, LayoutLayer)
            template, _ = anchor_leaf._layer.to_template_and_plaquettes()
            qubits_zero = observable_builder.build(k, template, slice_zero, component)
            qubits_one = observable_builder.build(k, template, slice_one, component)
            shared = qubits_zero & qubits_one
            div_zero = qubits_zero - shared
            div_one = qubits_one - shared
            if shared:
                circuit = anchor_leaf.get_annotations(k).circuit
                assert circuit is not None
                meas = MeasurementRecordsMap.from_scheduled_circuit(circuit)
                anchor_leaf.get_annotations(k).observables.append(
                    get_observable_with_measurement_records(
                        shared, meas, observable_index
                    )
                )
            _collect_recs(anchor_leaf, div_zero, div_zero_recs)
            _collect_recs(anchor_leaf, div_one, div_one_recs)

    if not (div_zero_recs or div_one_recs):
        return

    cube_leaf = subtree_leaves[cube_z_idx][-1]
    then_body: list = []
    else_body: list = []
    if div_one_recs:
        then_body.append(
            stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(o) for o in sorted(div_one_recs)],
                [observable_index],
            )
        )
    if div_zero_recs:
        else_body.append(
            stim.CircuitInstruction(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(o) for o in sorted(div_zero_recs)],
                [observable_index],
            )
        )
    annotations = cube_leaf.get_annotations(k)
    if annotations.conditional_observables is None:
        annotations.conditional_observables = []
    annotations.conditional_observables.append(
        IfBlock(
            condition_recs=list(recs),
            then_body=then_body,
            else_body=else_body if else_body else None,
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
    template, _ = node._layer.to_template_and_plaquettes()
    obs_qubits = observable_builder.build(k, template, obs_slice, component)
    if obs_qubits:
        obs_annotation = get_observable_with_measurement_records(
            obs_qubits, measurement_record, observable_index
        )
        node.get_annotations(k).observables.append(obs_annotation)
