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
:class:`BlockGraph` context is in scope), returns the **absolute** indices
of the measurements whose XOR is the condition, counted from the first
measurement of the whole circuit.

Absolute indices, rather than ``rec`` offsets, because one condition is shared
by every ``IF`` a conditional cube emits (gates, detectors, observables), and
each of those sits after a different number of measurements. The
:class:`~tqec.compile.conditional.circuit.IfBlock` keeps them absolute and
:meth:`~tqec.compile.conditional.circuit.ConditionalCircuit.to_stim_text`
converts them to ``rec`` offsets at the position of each block.

Per-z-slice contributing qubits come from :meth:`ObservableBuilder.build` for
each :class:`ObservableComponent` (bottom stabilisers, top readouts,
realignment). :func:`leaf_measurement_spans` gives each leaf's position in the
circuit's measurement sequence, counting every repetition of a
:class:`~tqec.compile.blocks.layers.composed.repeated.RepeatedLayer`.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class LeafMeasurements:
    """Where a leaf's measurements sit in the measurement sequence of the circuit.

    Attributes:
        start: absolute index of the leaf's first measurement. For a leaf that
            is repeated, this is its *last* repetition, the one whose records
            later rounds refer to.
        count: number of measurements the leaf performs, per repetition.
        records: the leaf's own measurement records.
        repeated: whether the leaf sits inside a repeated layer.

    """

    start: int
    count: int
    records: MeasurementRecordsMap
    repeated: bool

    def absolute_index(self, qubit: GridQubit) -> int | None:
        """Return the absolute index of the last measurement of ``qubit``, if any."""
        if qubit not in self.records:
            return None
        return self.start + self.count + self.records[qubit][-1]

    @property
    def end(self) -> int:
        """Absolute index just past the leaf's last measurement."""
        return self.start + self.count


def leaf_measurement_spans(root: LayerNode, k: int) -> dict[int, LeafMeasurements]:
    """Locate every leaf of the tree in the circuit's measurement sequence.

    Walks the tree in time order, counting a
    :class:`~tqec.compile.blocks.layers.composed.repeated.RepeatedLayer` body
    once per repetition at ``k``, which is how the final circuit performs it.

    Args:
        root: root of the tree. Its first measurement has index 0.
        k: scaling factor.

    Returns:
        a mapping from ``id(leaf)`` to that leaf's :class:`LeafMeasurements`.

    Raises:
        TQECError: if a leaf has not been annotated with its circuit yet.

    """
    spans: dict[int, LeafMeasurements] = {}

    def walk(node: LayerNode, start: int, repeated: bool) -> int:
        if node.is_leaf:
            circuit = node.get_annotations(k).circuit
            if circuit is None:
                raise TQECError(
                    "Cannot locate measurements before the leaves have been "
                    "annotated with their circuits."
                )
            records = MeasurementRecordsMap.from_scheduled_circuit(circuit)
            count = circuit.get_circuit().num_measurements
            spans[id(node)] = LeafMeasurements(start, count, records, repeated)
            return count
        repetitions = node.repetitions
        if repetitions is not None:
            (body,) = node.children
            reps = repetitions.integer_eval(k)
            # Measure the body once, then place it at its last repetition.
            per_repetition = walk(body, start, True)
            walk(body, start + (reps - 1) * per_repetition, True)
            return reps * per_repetition
        total = 0
        for child in node.children:
            total += walk(child, start + total, repeated)
        return total

    walk(root, 0, False)
    return spans


def ordered_leaves(root: LayerNode) -> list[LayerNode]:
    """Return the leaves of the subtree in time order."""
    if root.is_leaf:
        return [root]
    return [n for child in root.children for n in ordered_leaves(child)]


def _qubits_for_component(
    k: int,
    leaf: LayerNode,
    obs_slice: AbstractObservable,
    component: ObservableComponent,
    observable_builder: ObservableBuilder,
) -> set[GridQubit]:
    assert isinstance(leaf._layer, LayoutLayer)
    template, _ = leaf._layer.to_template_and_plaquettes()
    return observable_builder.build(k, template, obs_slice, component)


def _resolve_one(
    obs: AbstractObservable,
    subtree_leaves: list[list[LayerNode]],
    spans: dict[int, LeafMeasurements],
    k: int,
    observable_builder: ObservableBuilder,
) -> list[int]:
    """Return the sorted absolute indices of the measurements ``obs`` reads."""
    indices: list[int] = []

    def collect(leaf: LayerNode, qubits: set[GridQubit]) -> None:
        span = spans[id(leaf)]
        for q in qubits:
            # The builder may name a qubit this leaf does not measure (a
            # stretched-stabiliser placeholder); skip it, exactly like
            # get_observable_with_measurement_records.
            index = span.absolute_index(q)
            if index is not None:
                indices.append(index)

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

    return sorted(indices)


def _leaves_below(tree: LayerTree, z_index: int) -> list[list[LayerNode]]:
    """Return the time-ordered leaves of each z-subtree strictly below ``z_index``."""
    return [ordered_leaves(subtree) for subtree in tree._root.children[:z_index]]


def resolve_surface_condition_recs(
    tree: LayerTree,
    k: int,
    obs: AbstractObservable,
    anchor_z: int,
    observable_builder: ObservableBuilder,
    min_z: int = 0,
    *,
    debug_label: str = "<surface-anchored condition>",
) -> list[int]:
    """Resolve a surface-anchored condition to absolute measurement indices.

    A surface-anchored condition has no conditional cube of its own; it reads
    the measurements of every z-layer strictly below ``anchor_z``. Uses the same
    machinery as :func:`resolve_condition_recs`.

    Args:
        tree: the layer tree, with its leaves annotated with their circuits.
        k: scaling factor.
        obs: the condition, compiled to an abstract observable.
        anchor_z: absolute z of the first layer the condition may not read.
        observable_builder: builder lowering ``obs`` to measured qubits.
        min_z: smallest z of the computation, the z of the tree's first layer.
        debug_label: how to name the condition in an error message.

    Returns:
        the sorted absolute indices (0 for the circuit's first measurement) of
        the measurements whose XOR is the condition.

    Raises:
        TQECError: if the condition reads no measurement at all.

    """
    spans = leaf_measurement_spans(tree._root, k)
    indices = _resolve_one(obs, _leaves_below(tree, anchor_z - min_z), spans, k, observable_builder)
    if not indices:
        raise TQECError(
            f"{debug_label} reads no measurement in any layer below z={anchor_z}, "
            "so no IF can be conditioned on it. This happens when the surface only "
            "crosses data qubits that a temporal pipe carries on instead of "
            "measuring."
        )
    return indices


def resolve_condition_recs(
    tree: LayerTree,
    k: int,
    conditional_observables: dict[LayoutPosition3D, AbstractObservable],
    observable_builder: ObservableBuilder,
    min_z: int = 0,
) -> dict[LayoutPosition3D, list[int]]:
    """Resolve each conditional cube's condition to absolute measurement indices.

    For each ``(cond_pos, obs)`` pair, walks the leaves of the layers strictly
    below ``cond_pos.z``, lowers each z-slice and observable component (bottom
    stabilisers, top readouts, realignment) to measured qubits with
    :meth:`ObservableBuilder.build`, and locates each measurement in the
    circuit with :func:`leaf_measurement_spans`.

    Causality (every surface node lives at ``z < cube.z``) is enforced
    upstream at :meth:`Cube.__post_init__`; no defensive re-check here.

    Args:
        tree: the layer tree, with its leaves annotated with their circuits.
        k: scaling factor.
        conditional_observables: each conditional cube's condition, compiled to
            an abstract observable.
        observable_builder: builder lowering the conditions to measured qubits.
        min_z: smallest z of the computation, the z of the tree's first layer.

    Returns:
        ``dict`` mapping each ``cond_pos`` to the sorted absolute indices (0
        for the circuit's first measurement) of the measurements whose XOR
        selects the branch.

    Raises:
        TQECError: if a condition reads no measurement at all.

    """
    spans = leaf_measurement_spans(tree._root, k)
    result: dict[LayoutPosition3D, list[int]] = {}
    for cond_pos, obs in conditional_observables.items():
        indices = _resolve_one(
            obs, _leaves_below(tree, cond_pos.z - min_z), spans, k, observable_builder
        )
        if not indices:
            raise TQECError(
                f"The condition of the conditional cube at {cond_pos} reads no "
                "measurement in any layer below it, so its IF/ELSE branch cannot "
                "be selected. This happens when the surface only crosses data "
                "qubits that a temporal pipe carries on instead of measuring."
            )
        result[cond_pos] = indices
    return result
