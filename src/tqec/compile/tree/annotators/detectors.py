from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from tqec.circuit.measurement_map import MeasurementRecordsMap
from tqec.circuit.qubit import GridQubit
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.raw import RawCircuitLayer
from tqec.utils.coordinates import StimCoordinates
from tqec.compile.conditional.circuit import IfBlock
from tqec.compile.detectors.compute import compute_detectors_for_fixed_radius
from tqec.compile.detectors.database import DetectorDatabase
from tqec.compile.tree.annotations import DetectorAnnotation
from tqec.compile.tree.node import LayerNode, NodeWalker
from tqec.plaquette.plaquette import Plaquettes
from tqec.templates.base import Template
from tqec.utils.exceptions import TQECError


@dataclass(frozen=True)
class LookbackInformation:
    """Stores data on one QEC round.

    This data-structure is used to store information that might be useful to
    compute detectors. It only represents one QEC round.

    Attributes:
        template: template representing the QEC round.
        plaquettes: plaquettes that can be used in conjunction with
            ``self.template`` to generate the quantum circuit representing the
            QEC round.
        measurement_records: all the measurement records of the QEC round
            represented by ``self``. This could in theory be computed from
            ``self.template`` and ``self.plaquettes`` by generating the quantum
            circuit and then extracting the measurement records from it, but it
            turns out that we already have access to these records when creating
            such a structure, so we store them to avoid re-computing.

    """

    template: Template
    plaquettes: Plaquettes
    measurement_records: MeasurementRecordsMap
    plaquettes_branch_one: Plaquettes | None = None
    """Branch-one plaquettes for the round, when this round belongs to a
    conditional cube. ``None`` for rounds where both branches share the same
    plaquettes (every round outside a conditional cube, plus the conditional
    cube's stabiliser-round slices that happen to be byte-equal across
    branches)."""


@dataclass
class LookbackInformationList:
    """A sequence of :class:`LookbackInformation` instances."""

    infos: list[LookbackInformation] = field(default_factory=list)

    def append(
        self,
        template: Template,
        plaquettes: Plaquettes,
        measurement_records: MeasurementRecordsMap,
        plaquettes_branch_one: Plaquettes | None = None,
    ) -> None:
        """Add the provided parameters to the lookback window.

        This method might remove older items that should not be considered anymore from the lookback
        stack.
        """
        self.infos.append(
            LookbackInformation(template, plaquettes, measurement_records, plaquettes_branch_one)
        )

    def extend(self, other: LookbackInformationList, repetitions: int = 1) -> None:
        """Add the provided lookback information to self, potentially repeating it several times.

        This method can be used when exiting a REPEAT block to update the lookback information by
        taking into account that it might be repeated several times.

        """
        self.infos.extend(other.infos * repetitions)

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, index: int | slice) -> LookbackInformation | list[LookbackInformation]:
        return self.infos[index]  # pragma: no cover


class LookbackStack:
    def __init__(self) -> None:
        """Initialise the lookback stack.

        The lookback stack can be used to query the current state for detector computation.

        This data-structure keeps information about the past QEC rounds in order to be able to query
        them and help in detector computation by only considering the ``N`` last rounds.

        In particular, this data-structure is useful to keep track of previous rounds in the
        presence of ``REPEAT`` blocks.

        """
        self._stack: list[LookbackInformationList] = [LookbackInformationList()]

    def enter_repeat_block(self) -> None:
        """Append a new entry to the stack."""
        self._stack.append(LookbackInformationList())

    def close_repeat_block(self, repetitions: int) -> None:
        """Remove the last entry on the stack, repeating it as needed into the new last entry."""
        if len(self._stack) < 2:
            raise TQECError(
                f"Only got {len(self._stack)} < 2 entries in the stack. That "
                "means that we are not in a REPEAT block. Cannot call "
                "close_repeat_block()."
            )
        self._stack[-2].extend(self._stack[-1], repetitions)
        self._stack.pop(-1)

    def append(
        self,
        template: Template,
        plaquettes: Plaquettes,
        measurement_records: MeasurementRecordsMap,
        plaquettes_branch_one: Plaquettes | None = None,
    ) -> None:
        """Append a new QEC round in the data-structure."""
        self._stack[-1].append(
            template, plaquettes, measurement_records, plaquettes_branch_one
        )

    def _get_last_n(
        self, n: int
    ) -> tuple[
        list[Template],
        list[Plaquettes],
        list[MeasurementRecordsMap],
        list[Plaquettes | None],
    ]:
        if n < 0:
            raise TQECError(
                f"Cannot look back a negative number of rounds. Got a lookback value of {n}."
            )
        if n == 0:
            return [], [], [], []
        templates: list[Template] = []
        plaquettes: list[Plaquettes] = []
        measurement_records: list[MeasurementRecordsMap] = []
        plaquettes_one: list[Plaquettes | None] = []
        # Filling the lists in reverse order (i.e., from earlier time to oldest
        # time) and correcting when returning.
        for element in reversed(self._stack):
            for info in reversed(element.infos):
                templates.append(info.template)
                plaquettes.append(info.plaquettes)
                measurement_records.append(info.measurement_records)
                plaquettes_one.append(info.plaquettes_branch_one)
                if len(templates) == n:
                    return (
                        templates[::-1],
                        plaquettes[::-1],
                        measurement_records[::-1],
                        plaquettes_one[::-1],
                    )

        return (
            templates[::-1],
            plaquettes[::-1],
            measurement_records[::-1],
            plaquettes_one[::-1],
        )

    def lookback(
        self,
        n: int,
    ) -> tuple[list[Template], list[Plaquettes], MeasurementRecordsMap]:
        """Get the last ``self._lookback`` QEC rounds."""
        templates, plaquettes, measurement_records, _ = self._get_last_n(n)
        measurement_record = MeasurementRecordsMap()
        for mrec in measurement_records:
            measurement_record = measurement_record.with_added_measurements(mrec)
        return templates, plaquettes, measurement_record

    def lookback_per_branch(
        self,
        n: int,
    ) -> tuple[
        list[Template],
        list[Plaquettes],
        list[Plaquettes],
        MeasurementRecordsMap,
    ]:
        """Get the last ``n`` QEC rounds with parallel branch-zero and branch-one
        plaquette lists. Rounds with no branch-one alternate fall back to the
        branch-zero entry (both branches share that round's content)."""
        templates, plaquettes_zero, measurement_records, plaquettes_one = (
            self._get_last_n(n)
        )
        plaquettes_one_filled: list[Plaquettes] = [
            (po if po is not None else pz)
            for po, pz in zip(plaquettes_one, plaquettes_zero)
        ]
        measurement_record = MeasurementRecordsMap()
        for mrec in measurement_records:
            measurement_record = measurement_record.with_added_measurements(mrec)
        return templates, plaquettes_zero, plaquettes_one_filled, measurement_record

    def __len__(self) -> int:
        if len(self._stack) > 1:
            raise TQECError(
                "Cannot get a meaningful stack length when a REPEAT block is in construction."
            )
        return len(self._stack[0])


class AnnotateDetectorsOnLayerNode(NodeWalker):
    def __init__(
        self,
        k: int,
        manhattan_radius: int = 2,
        detector_database: DetectorDatabase | None = None,
        only_use_database: bool = False,
        lookback: int = 2,
        parallel_process_count: int = 1,
        condition_recs: dict[int, list[int]] | None = None,
        min_z: int = 0,
    ):
        """Walker computing and annotating detectors on leaf nodes.

        This class keeps track of the ``lookback`` previous leaf nodes seen and
        uses them to automatically compute the detectors at all the leaf nodes
        it encounters.

        Args:
            k: scaling factor.
            manhattan_radius: Parameter for the automatic computation of detectors.
                Should be large enough so that flows cancelling each other to
                form a detector are strictly contained in plaquettes that are at
                most at a distance of ``manhattan_radius`` from the central
                plaquette. Detector computation runtime grows with this parameter,
                so you should try to keep it to its minimum. A value too low might
                produce invalid detectors.
            detector_database: existing database of detectors that is used to
                avoid computing detectors if the database already contains them.
                Default to `None` which result in not using any kind of database
                and unconditionally performing the detector computation.
            only_use_database: if ``True``, only detectors from the database will be
                used. An error will be raised if a situation that is not registered
                in the database is encountered. Default to ``False``.
            lookback: number of QEC rounds to consider to try to find detectors. Including more
                rounds increases computation time.
            parallel_process_count: number of processes to use for parallel processing.
                1 for sequential processing, >1 for parallel processing using
                ``parallel_process_count`` processes, and -1 for using all available
                CPU cores. Default to 1.

        """
        if lookback < 1:
            raise TQECError(
                "Cannot compute detectors without any layer. The `lookback` "
                f"parameter should be >= 1 but got {lookback}."
            )
        self._k = k
        self._manhattan_radius = manhattan_radius
        self._database = detector_database if detector_database is not None else DetectorDatabase()
        self._only_use_database = only_use_database
        self._lookback_size = lookback
        self._lookback_stack = LookbackStack()
        self._parallel_process_count = parallel_process_count
        self._condition_recs = condition_recs
        self._min_z = min_z
        self._depth = 0
        self._z_index = -1
        # Explicit multi-qubit end_spec handed forward by the transition round
        # (Y cap), consumed by the next raw round: for each (shifted) stabilizer
        # ancilla coordinate, the (shifted) qubit coordinates the transition
        # measured to prepare that stabilizer.
        self._pending_end_spec: dict[tuple[int, int], list[tuple[int, int]]] | None = None

    @override
    def visit_node(self, node: LayerNode) -> None:
        if not isinstance(node._layer, LayoutLayer):
            return
        annotations = node.get_annotations(self._k)
        if annotations.circuit is None:
            raise TQECError("Cannot compute detectors without the circuit annotation.")

        raw_by_pos = {
            pos: layer
            for pos, layer in node._layer.layers.items()
            if isinstance(layer, RawCircuitLayer)
        }
        if raw_by_pos:
            if len(raw_by_pos) == len(node._layer.layers):
                self._annotate_raw_slice(node, annotations, list(raw_by_pos.values()))
            else:
                self._annotate_mixed_slice(node, annotations, raw_by_pos)
            return

        template_zero, plaquettes_zero = node._layer.to_template_and_plaquettes()
        plaquettes_one_for_round: Plaquettes | None = None
        active_condition_recs: list[int] | None = None
        if self._condition_recs is not None and node._layer.conditional_layers:
            active_condition_recs = self._condition_recs.get(
                self._min_z + self._z_index
            )
        leaf_is_conditional = active_condition_recs is not None
        if leaf_is_conditional:
            template_one, plaquettes_one_for_round = (
                node._layer._compute_template_and_plaquettes(
                    node._layer._branch_one_layers()
                )
            )
            if template_one != template_zero:
                raise TQECError(
                    "AnnotateDetectorsOnLayerNode: per-branch templates differ; "
                    "EMC + CEO violated."
                )
        self._lookback_stack.append(
            template_zero,
            plaquettes_zero,
            MeasurementRecordsMap.from_scheduled_circuit(annotations.circuit),
            plaquettes_branch_one=plaquettes_one_for_round,
        )

        templates, plaquettes, measurement_records = self._lookback_stack.lookback(
            self._lookback_size
        )
        detectors_zero = compute_detectors_for_fixed_radius(
            templates,
            self._k,
            plaquettes,
            self._manhattan_radius,
            self._database,
            self._only_use_database,
            self._parallel_process_count,
        )

        if not leaf_is_conditional:
            for detector in detectors_zero:
                annotations.detectors.append(
                    DetectorAnnotation.from_detector(detector, measurement_records)
                )
            return

        # Per-branch detector computation. The lookback gives parallel
        # branch-zero / branch-one plaquette lists; everything else (templates,
        # measurement records) is shared by EMC + CEO.
        templates_pb, plaquettes_zero_lb, plaquettes_one_lb, measurement_records_pb = (
            self._lookback_stack.lookback_per_branch(self._lookback_size)
        )
        detectors_one = compute_detectors_for_fixed_radius(
            templates_pb,
            self._k,
            plaquettes_one_lb,
            self._manhattan_radius,
            self._database,
            self._only_use_database,
            self._parallel_process_count,
        )
        zero_set = set(detectors_zero)
        one_set = set(detectors_one)
        shared = zero_set & one_set
        zero_only = [d for d in detectors_zero if d not in shared]
        one_only = [d for d in detectors_one if d not in shared]
        # Common detectors stay on annotations.detectors so non-conditional
        # callers continue to see them. The divergent slice is recorded
        # separately on conditional_detectors.
        for detector in detectors_zero:
            if detector in shared:
                annotations.detectors.append(
                    DetectorAnnotation.from_detector(detector, measurement_records_pb)
                )
        # Coarse single-IfBlock-per-leaf: zero-only -> else, one-only -> then.
        if zero_only or one_only:
            assert active_condition_recs is not None
            then_body = [
                DetectorAnnotation.from_detector(d, measurement_records_pb).to_instruction()
                for d in one_only
            ]
            else_body = [
                DetectorAnnotation.from_detector(d, measurement_records_pb).to_instruction()
                for d in zero_only
            ]
            if annotations.conditional_detectors is None:
                annotations.conditional_detectors = []
            annotations.conditional_detectors.append(
                IfBlock(
                    condition_recs=list(active_condition_recs),
                    then_body=then_body,
                    else_body=else_body if else_body else None,
                )
            )

    def _annotate_raw_slice(
        self, node: LayerNode, annotations: object, raw_layers: list[RawCircuitLayer]
    ) -> None:
        """Handle a leaf whose layer carries a :class:`RawCircuitLayer`.

        The Y-basis measurement cap is sliced into per-round raw layers
        (transition, boundary0, repeated boundary, final). A single round carries
        no cross-round detectors: those *seam* / *bulk* detectors are formed here
        from each round's flow specs, which close a round's ``start_spec`` against
        the previous round's measurements. The previous round is either

        * a *standard* round (the below memory round, a boundary round, or the
          last boundary before the final round): its stabilizers are measured by
          a single ancilla, recovered from the previous round's measurement
          records keyed by ancilla coordinate (via the lookback stack); or
        * the *transition* round, whose ``end_spec`` prepares each degenerate
          stabilizer with multiple records not recoverable from a
          coordinate-keyed record map, so it is handed forward explicitly via
          ``self._pending_end_spec``.

        Any detectors internal to a single round (the final round's stabilizer
        reconstruction) stay inside the round's own circuit.
        """
        if len(raw_layers) != 1:
            raise TQECError("Only a single RawCircuitLayer per layer is supported.")
        raw_layer = raw_layers[0]
        raw_records = MeasurementRecordsMap.from_scheduled_circuit(annotations.circuit)
        raw_total = annotations.circuit.get_circuit().num_measurements

        # A lone Y cap occupies block position (0, 0) of its layer, so no shift.
        self._emit_raw_seam(annotations, raw_layer, raw_records, raw_total, (0, 0))

        # Push this slice's own records so later rounds can look back through it.
        self._lookback_stack.append(None, None, raw_records)  # type: ignore[arg-type]

    def _raw_shift(self, node: LayerNode, pos: object) -> tuple[int, int]:
        """Qubit-coordinate offset applied to a raw layer at ``pos`` within its
        enclosing :class:`LayoutLayer` (mirrors ``LayoutLayer._mixed_to_circuit``)."""
        layout = node._layer
        eshape = layout.element_shape.to_shape_2d(self._k)  # type: ignore[attr-defined]
        mincube, _ = layout.bounds  # type: ignore[attr-defined]
        bp = pos.to_block_position()  # type: ignore[attr-defined]
        return (bp.x - mincube.x) * (eshape.x - 1), (bp.y - mincube.y) * (eshape.y - 1)

    def _emit_raw_seam(
        self,
        annotations: object,
        raw_layer: RawCircuitLayer,
        raw_records: MeasurementRecordsMap,
        raw_total: int,
        shift: tuple[int, int],
    ) -> None:
        """Emit the cross-round seam / bulk / reconstruction detectors for one
        raw round, whose flow-spec qubit coordinates are offset by ``shift`` into
        the layer's qubit frame.

        All spec values are qubit coordinates; a measurement is located by
        looking the (shifted) coordinate up in the relevant round's
        measurement-record map (``raw_records`` for this round, the lookback for
        the previous round). This is robust to a coexisting memory cube whose
        measurements interleave with the cap's in the merged round circuit.

        Offsets are expressed relative to the end of *this* round's circuit (as
        :class:`DetectorAnnotation` requires): this round's records already are;
        the previous round's records (relative to that round's end) are rebased
        by subtracting ``raw_total`` (this round's measurement count).
        """
        dx, dy = shift

        def sh(c: tuple[int, int]) -> tuple[int, int]:
            return (c[0] + dx, c[1] + dy)

        def this_offsets(coords: list[tuple[int, int]]) -> list[int]:
            return [raw_records[GridQubit(*sh(c))][-1] for c in coords]

        start_getter = getattr(raw_layer, "start_spec", None)
        start_spec = start_getter(self._k) if start_getter is not None else {}

        if start_spec:
            _, _, prev_records = self._lookback_stack.lookback(1)
            pending = self._pending_end_spec
            for anc_coord, cur_coords in start_spec.items():
                sc = sh(anc_coord)
                cur_offsets = this_offsets(cur_coords)
                if pending is not None:
                    # Previous round is the transition round: XOR the qubits it
                    # used to prepare this stabilizer (resolved in the previous
                    # round's record map) with this round's ancilla.
                    if sc not in pending:
                        continue
                    prev_offsets = [
                        prev_records[GridQubit(*pc)][-1] - raw_total for pc in pending[sc]
                    ]
                else:
                    # Previous round is a standard round: this stabilizer's single
                    # ancilla measurement, recovered by coordinate.
                    gq = GridQubit(*sc)
                    if gq not in prev_records:
                        raise TQECError(
                            f"Y-cap seam detector references ancilla {sc} that was "
                            "not measured in the previous round; the below round "
                            "does not match the Y patch."
                        )
                    prev_offsets = [prev_records[gq][-1] - raw_total]
                annotations.detectors.append(
                    DetectorAnnotation(
                        StimCoordinates(float(sc[0]), float(sc[1]), 0.0),
                        sorted(prev_offsets + cur_offsets),
                    )
                )

        # Reconstruction detectors internal to this round (final round only).
        reconstruction_getter = getattr(raw_layer, "reconstruction_spec", None)
        reconstruction = (
            reconstruction_getter(self._k) if reconstruction_getter is not None else None
        )
        if reconstruction:
            for anc_coord, coords in reconstruction.items():
                sc = sh(anc_coord)
                annotations.detectors.append(
                    DetectorAnnotation(
                        StimCoordinates(float(sc[0]), float(sc[1]), 2.0),
                        sorted(this_offsets(coords)),
                    )
                )

        # Hand this round's explicit end_spec forward, if any (only the
        # transition round has one). Cleared once consumed by the next round.
        end_getter = getattr(raw_layer, "end_spec", None)
        end_spec = end_getter(self._k) if end_getter is not None else None
        if end_spec is not None:
            self._pending_end_spec = {sh(c): [sh(q) for q in v] for c, v in end_spec.items()}
        else:
            self._pending_end_spec = None

    def _annotate_mixed_slice(
        self, node: LayerNode, annotations: object, raw_by_pos: dict[object, RawCircuitLayer]
    ) -> None:
        """Handle a leaf whose :class:`LayoutLayer` mixes raw Y-cap rounds with
        coexisting plaquette (memory) rounds at distinct cube positions.

        Memory-position detectors are computed with the standard template path
        (restricted to the plaquette positions); each raw position contributes
        its shift-aware seam detectors. Both read measurement offsets from the
        combined round circuit, so the two sets never collide (they reference
        disjoint qubit coordinates).
        """
        layout = node._layer
        full_records = MeasurementRecordsMap.from_scheduled_circuit(annotations.circuit)
        raw_total = annotations.circuit.get_circuit().num_measurements

        # Raw (Y-cap) seam detectors first, while the lookback's most recent
        # entry is still the *previous* round (this round is pushed below).
        for pos, raw_layer in raw_by_pos.items():
            self._emit_raw_seam(
                annotations, raw_layer, full_records, raw_total, self._raw_shift(node, pos)
            )

        # Plaquette (memory) detectors: build a sub-layer over just those
        # positions, push this round, and reuse the fixed-radius computation
        # against the lookback (which now includes this round).
        plaquette_layers = {
            pos: layer for pos, layer in layout.layers.items() if pos not in raw_by_pos
        }
        sub_layer = LayoutLayer(plaquette_layers, layout.element_shape)
        template, plaquettes = sub_layer.to_template_and_plaquettes()
        self._lookback_stack.append(template, plaquettes, full_records)
        templates, plaqs, measurement_records = self._lookback_stack.lookback(
            self._lookback_size
        )
        detectors = compute_detectors_for_fixed_radius(
            templates,
            self._k,
            plaqs,
            self._manhattan_radius,
            self._database,
            self._only_use_database,
            self._parallel_process_count,
        )
        for detector in detectors:
            annotations.detectors.append(
                DetectorAnnotation.from_detector(detector, measurement_records)
            )

    @override
    def enter_node(self, node: LayerNode) -> None:
        self._depth += 1
        if self._depth == 2:
            self._z_index += 1
        if node.is_repeated:
            self._lookback_stack.enter_repeat_block()

    @override
    def exit_node(self, node: LayerNode) -> None:
        self._depth -= 1
        if not node.is_repeated:
            return
        # Note: this is the place to perform checks. In particular, checking that
        # detectors computed at the first repetition of the REPEAT block are also
        # valid at any repetitions. This is a requirement for the REPEAT block to
        # make sense, but that would be nice to include a check to avoid
        # misleadingly include detectors that are incorrect sometimes.
        repetitions = node.repetitions
        assert repetitions is not None
        self._lookback_stack.close_repeat_block(repetitions.integer_eval(self._k))
