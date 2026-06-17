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

    Each conditional cube in ``conditional_cube_positions`` owns the leaves
    at z-layers in ``(prev_cube_z, this_cube_z]`` (after sorting by z).
    Shared per-leaf qubits emit as plain ``OBSERVABLE_INCLUDE`` on the trunk.
    Divergent qubits' rec offsets are shifted into the owning cube's IfBlock
    frame and bundled into one per-branch ``OBSERVABLE_INCLUDE`` instruction
    inside an :class:`IfBlock` gated by that cube's ``condition_recs``.
    All IfBlocks reuse the same ``observable_index`` — Stim XORs them into
    a single logical observable.
    """
    import stim  # local: avoid module-level dep when unused

    from tqec.compile.conditional.condition_recs import (  # noqa: PLC0415
        _collect_pre_cond_entries,
        _compute_tail_shifts,
    )

    if not cond_observable.conditional_cube_positions:
        raise TQECError(
            "ConditionalAbstractObservable has no conditional_cube_positions."
        )
    cube_z_indices = sorted(p.z - min_z for p in cond_observable.conditional_cube_positions)
    n_layers = len(root.children)
    for z_idx in cube_z_indices:
        if z_idx < 0 or z_idx >= n_layers:
            raise TQECError(
                f"ConditionalCorrelationSurface names a conditional cube at "
                f"z={min_z + z_idx} that does not correspond to any z-layer "
                f"in the tree (min_z={min_z}, layers={n_layers})."
            )
    recs_per_cube: list[list[int]] = []
    for z_idx in cube_z_indices:
        z_actual = min_z + z_idx
        recs = condition_recs_by_z.get(z_actual)
        if recs is None:
            raise TQECError(
                f"ConditionalCorrelationSurface references a conditional cube at "
                f"z={z_actual} but no resolved condition_recs were produced for "
                "that z-layer. Ensure the named cube is actually conditional."
            )
        recs_per_cube.append(list(recs))

    max_idx = cube_z_indices[-1]
    entries, subtree_leaves = _collect_pre_cond_entries(root, k, max_idx + 1)
    tail_shifts_max = _compute_tail_shifts(entries)
    entry_by_leaf_id = {id(e.leaf): (e, i) for i, e in enumerate(entries)}

    # For each cube, find the index of its last entry (in `entries`) and the
    # count of measurements that happen AFTER that point up to the end of
    # the walk. Subtracting that from tail_shifts_max translates rec offsets
    # from the max-cube frame to this cube's IfBlock frame.
    end_entry_idx_per_cube: list[int] = []
    j = 0
    for cube_z_idx in cube_z_indices:
        while j < len(entries) and entries[j].z <= cube_z_idx:
            j += 1
        end_entry_idx_per_cube.append(j - 1)
    tail_beyond_per_cube: list[int] = []
    total_after = 0
    running_total = sum(e.num_measurements for e in entries)
    cumulative_through: list[int] = []
    cum = 0
    for e in entries:
        cum += e.num_measurements
        cumulative_through.append(cum)
    for end_idx in end_entry_idx_per_cube:
        through = cumulative_through[end_idx] if end_idx >= 0 else 0
        tail_beyond_per_cube.append(running_total - through)
    del total_after  # not used

    # owner_for_z_idx[z_idx] = cube index that owns this z-layer
    owner_for_z_idx: list[int] = [0] * (max_idx + 1)
    c = 0
    for z_idx in range(max_idx + 1):
        while cube_z_indices[c] < z_idx:
            c += 1
        owner_for_z_idx[z_idx] = c

    div_zero_per_cube: list[list[int]] = [[] for _ in cube_z_indices]
    div_one_per_cube: list[list[int]] = [[] for _ in cube_z_indices]

    def _collect_recs(leaf: LayerNode, qubits: set, target: list[int], cube_idx: int) -> None:
        entry, idx = entry_by_leaf_id[id(leaf)]
        shift = tail_shifts_max[idx] - tail_beyond_per_cube[cube_idx]
        for q in qubits:
            if q not in entry.records:
                continue
            local = entry.records[q][-1]
            target.append(local - shift)

    def _anchor_actions(
        leaves: list[LayerNode],
        slice_zero: AbstractObservable,
        slice_one: AbstractObservable,
    ) -> list[tuple[LayerNode, ObservableComponent]]:
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
        cube_owner = owner_for_z_idx[z_idx]
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
            _collect_recs(anchor_leaf, div_zero, div_zero_per_cube[cube_owner], cube_owner)
            _collect_recs(anchor_leaf, div_one, div_one_per_cube[cube_owner], cube_owner)

    for c, cube_z_idx in enumerate(cube_z_indices):
        d0 = div_zero_per_cube[c]
        d1 = div_one_per_cube[c]
        if not (d0 or d1):
            continue
        cube_leaf = subtree_leaves[cube_z_idx][-1]
        then_body: list = []
        else_body: list = []
        if d1:
            then_body.append(
                stim.CircuitInstruction(
                    "OBSERVABLE_INCLUDE",
                    [stim.target_rec(o) for o in sorted(d1)],
                    [observable_index],
                )
            )
        if d0:
            else_body.append(
                stim.CircuitInstruction(
                    "OBSERVABLE_INCLUDE",
                    [stim.target_rec(o) for o in sorted(d0)],
                    [observable_index],
                )
            )
        annotations = cube_leaf.get_annotations(k)
        if annotations.conditional_observables is None:
            annotations.conditional_observables = []
        annotations.conditional_observables.append(
            IfBlock(
                condition_recs=recs_per_cube[c],
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
