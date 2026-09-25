"""Resolve correlation surfaces into IF/ELSE ``condition_recs`` lists.

A conditional cube's IF/ELSE branch is selected by the XOR of a set of
measurement records. The set is defined by the cube's attached
:class:`CorrelationSurface` (``Cube.condition``), which by invariant
lives **strictly below** the cube's z-layer (enforced at
:meth:`Cube.__post_init__`).

:func:`resolve_condition_recs` walks the :class:`LayerTree` and, for
each ``(LayoutPosition3D, AbstractObservable)`` pair (the
:class:`AbstractObservable` is pre-compiled by
:func:`compile_block_graph` from the cube's surface, when the
:class:`BlockGraph` context is in scope), returns the list of negative
``rec`` offsets in the frame of the IfBlock emission point (start of
the conditional cube's first leaf moment).

Per-z-slice contributing qubits come from
:meth:`ObservableBuilder.build` for each
:class:`ObservableComponent` (bottom stabilisers, top readouts,
realignment); per-leaf
:meth:`MeasurementRecordsMap.from_scheduled_circuit` provides local
rec offsets, then a "tail shift" (count of measurements after the
contributing leaf, up to the IfBlock point) translates each local
offset into the IfBlock frame.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.observables.abstract_observable import AbstractObservable
from tqec.compile.observables.builder import ObservableBuilder, ObservableComponent
from tqec.utils.exceptions import TQECError

if TYPE_CHECKING:
    from tqec.circuit.qubit import GridQubit
    from tqec.compile.blocks.positioning import LayoutPosition3D
    from tqec.compile.tree.node import LayerNode
    from tqec.compile.tree.tree import LayerTree


@dataclass
class _LeafEntry:
    """Per-leaf bookkeeping for tail-shift accounting."""

    leaf: "LayerNode"
    z: int
    position_in_subtree: int
    leaves_in_subtree: int
    records: MeasurementRecordsMap
    num_measurements: int


def _get_ordered_leaves(root: "LayerNode") -> list["LayerNode"]:
    """Return the leaves of the subtree in time order."""
    if root.is_leaf:
        return [root]
    return [n for child in root.children for n in _get_ordered_leaves(child)]


def _collect_pre_cond_entries(
    tree_root: "LayerNode", k: int, cond_z: int
) -> tuple[list[_LeafEntry], list[list["LayerNode"]]]:
    """Walk subtrees at ``z < cond_z`` in time order, returning per-leaf
    entries plus the per-z list of ordered leaves (needed by the
    component dispatch)."""
    entries: list[_LeafEntry] = []
    subtree_leaves: list[list["LayerNode"]] = []
    for z, subtree in enumerate(tree_root.children):
        if z >= cond_z:
            break
        leaves = _get_ordered_leaves(subtree)
        subtree_leaves.append(leaves)
        for pos_in_subtree, leaf in enumerate(leaves):
            circuit = leaf.get_annotations(k).circuit
            if circuit is None:
                raise TQECError(
                    "resolve_condition_recs: leaf at z={z}, position {pos_in_subtree} "
                    "has no circuit annotation. Run circuit annotation before "
                    "condition resolution."
                )
            records = MeasurementRecordsMap.from_scheduled_circuit(circuit)
            num = sum(len(o) for o in records.mapping.values())
            entries.append(
                _LeafEntry(
                    leaf=leaf,
                    z=z,
                    position_in_subtree=pos_in_subtree,
                    leaves_in_subtree=len(leaves),
                    records=records,
                    num_measurements=num,
                )
            )
    return entries, subtree_leaves


def _compute_tail_shifts(entries: list[_LeafEntry]) -> list[int]:
    """Return ``shifts`` where ``shifts[i]`` is the count of measurements
    that happen *after* ``entries[i]`` and up to the IfBlock emission
    point (which is the moment immediately after the last entry). A
    qubit measured at ``entries[i]`` with local offset ``-r`` translates
    to IfBlock-frame offset ``-r - shifts[i]``."""
    shifts: list[int] = [0] * len(entries)
    running = 0
    for i in range(len(entries) - 1, -1, -1):
        shifts[i] = running
        running += entries[i].num_measurements
    return shifts


def _qubits_for_component(
    k: int,
    leaf: "LayerNode",
    obs_slice: AbstractObservable,
    component: ObservableComponent,
    observable_builder: ObservableBuilder,
) -> set["GridQubit"]:
    assert isinstance(leaf._layer, LayoutLayer)
    template, _ = leaf._layer.to_template_and_plaquettes()
    return observable_builder.build(k, template, obs_slice, component)


def _resolve_one(
    cond_pos: "LayoutPosition3D",
    obs: AbstractObservable,
    entries: list[_LeafEntry],
    subtree_leaves: list[list["LayerNode"]],
    tail_shifts: list[int],
    k: int,
    observable_builder: ObservableBuilder,
) -> list[int]:
    entry_by_leaf_id: dict[int, tuple[_LeafEntry, int]] = {
        id(e.leaf): (e, i) for i, e in enumerate(entries)
    }
    recs: list[int] = []

    def collect(leaf: "LayerNode", qubits: set["GridQubit"]) -> None:
        entry, idx = entry_by_leaf_id[id(leaf)]
        shift = tail_shifts[idx]
        for q in qubits:
            if q not in entry.records:
                # qubit named by builder but not measured in this leaf
                # (builder may emit stretched-stabiliser placeholders);
                # skip exactly like get_observable_with_measurement_records.
                continue
            local = entry.records[q][-1]
            recs.append(local - shift)

    for z, leaves in enumerate(subtree_leaves):
        obs_slice = obs.slice_at_z(z)
        # bottom stabilisers land on the first leaf of the z-slice
        bot_qubits = _qubits_for_component(
            k, leaves[0], obs_slice, ObservableComponent.BOTTOM_STABILIZERS, observable_builder
        )
        if bot_qubits:
            collect(leaves[0], bot_qubits)

        # top readouts land on the last leaf, except when a temporal
        # hadamard pipe inserts a realignment layer (then second-to-last)
        readout_leaf = leaves[-1]
        if obs_slice.temporal_hadamard_pipes:
            readout_leaf = leaves[-2]
            real_qubits = _qubits_for_component(
                k, leaves[-1], obs_slice, ObservableComponent.REALIGNMENT, observable_builder
            )
            if real_qubits:
                collect(leaves[-1], real_qubits)
        top_qubits = _qubits_for_component(
            k, readout_leaf, obs_slice, ObservableComponent.TOP_READOUTS, observable_builder
        )
        if top_qubits:
            collect(readout_leaf, top_qubits)

    return sorted(recs)


def resolve_surface_condition_recs(
    tree: "LayerTree",
    k: int,
    obs: AbstractObservable,
    anchor_z: int,
    observable_builder: ObservableBuilder,
    *,
    debug_label: str = "<surface-anchored condition>",
) -> list[int]:
    """Resolve a surface-anchored condition (no associated conditional cube)
    to a list of ``rec`` offsets, computed at an IfBlock that lives on the
    leaf at z = ``anchor_z`` (exclusive — so the condition's measurements
    must live at z < anchor_z).

    Mirrors :func:`resolve_condition_recs` but for a single condition that
    is not tied to a cube position. Uses the same machinery
    (``_collect_pre_cond_entries`` + ``_compute_tail_shifts`` +
    ``_resolve_one``) keyed purely on ``anchor_z``.
    """
    entries, subtree_leaves = _collect_pre_cond_entries(tree._root, k, anchor_z)
    tail_shifts = _compute_tail_shifts(entries)
    recs = _resolve_one(
        # cond_pos is only used for diagnostics inside _resolve_one (none here)
        None,  # type: ignore[arg-type]
        obs,
        entries,
        subtree_leaves,
        tail_shifts,
        k,
        observable_builder,
    )
    if not recs:
        warnings.warn(
            f"resolve_surface_condition_recs: {debug_label} resolved to no "
            f"measurement records in any leaf at z<{anchor_z}. Falling back "
            "to placeholder rec[-1]; the IF branch selection is meaningless.",
            stacklevel=2,
        )
        recs = [-1]
    return recs


def resolve_condition_recs(
    tree: "LayerTree",
    k: int,
    conditional_observables: dict["LayoutPosition3D", AbstractObservable],
    observable_builder: ObservableBuilder,
) -> dict["LayoutPosition3D", list[int]]:
    """Resolve each pre-compiled :class:`AbstractObservable` to a list of
    ``rec`` offsets in the IfBlock emission frame.

    For each ``(cond_pos, obs)`` pair:

    1. Walk the leaves of subtrees at ``z < cond_pos.z`` to collect
       per-leaf :class:`MeasurementRecordsMap` and per-leaf measurement
       counts.
    2. For each z-slice and each observable component (bottom
       stabilisers, top readouts, realignment), invoke
       :meth:`ObservableBuilder.build` on the appropriate leaf to derive
       the contributing qubits, then look up each qubit's local rec
       offset and shift it into the IfBlock emission frame.

    Causality (every surface node lives at ``z < cube.z``) is enforced
    upstream at :meth:`Cube.__post_init__`; no defensive re-check here.

    Returns:
        ``dict`` mapping each ``cond_pos`` to a sorted list of negative
        ``rec`` offsets (the IF/ELSE condition is their XOR).
    """
    result: dict["LayoutPosition3D", list[int]] = {}
    for cond_pos, obs in conditional_observables.items():
        entries, subtree_leaves = _collect_pre_cond_entries(
            tree._root, k, cond_pos.z
        )
        tail_shifts = _compute_tail_shifts(entries)
        recs = _resolve_one(
            cond_pos, obs, entries, subtree_leaves, tail_shifts, k, observable_builder
        )
        if not recs:
            warnings.warn(
                f"resolve_condition_recs: surface for conditional cube at "
                f"{cond_pos} resolved to no measurement records in any "
                "pre-cond leaf. This likely means the surface picks "
                "data-qubit readouts that are absorbed by a temporal pipe, "
                "or otherwise targets a measurement that does not exist in "
                "z<cond_z. Falling back to placeholder rec[-1] so emission "
                "produces syntactically valid Stim text; the resulting "
                "IF/ELSE branch selection is meaningless.",
                stacklevel=2,
            )
            recs = [-1]
        result[cond_pos] = recs
    return result
