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
        # Explicit multi-record end_spec handed forward by the transition round
        # (Y cap), consumed by the next raw round. ``(end_spec, prev_total)``.
        self._pending_end_spec: tuple[dict[tuple[int, int], list[int]], int] | None = None

    @override
    def visit_node(self, node: LayerNode) -> None:
        if not isinstance(node._layer, LayoutLayer):
            return
        annotations = node.get_annotations(self._k)
        if annotations.circuit is None:
            raise TQECError("Cannot compute detectors without the circuit annotation.")

        raw_layers = [
            layer
            for layer in node._layer.layers.values()
            if isinstance(layer, RawCircuitLayer)
        ]
        if raw_layers:
            self._annotate_raw_slice(node, annotations, raw_layers)
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

        start_getter = getattr(raw_layer, "start_spec", None)
        start_spec = start_getter(self._k) if start_getter is not None else {}

        if start_spec:
            if self._pending_end_spec is not None:
                # Previous round is the transition round: match against its
                # explicit multi-record end_spec.
                prev_end, prev_total = self._pending_end_spec
                for anc_coord, cur_idxs in start_spec.items():
                    if anc_coord not in prev_end:
                        continue
                    prev_offsets = [i - prev_total - raw_total for i in prev_end[anc_coord]]
                    cur_offsets = [i - raw_total for i in cur_idxs]
                    annotations.detectors.append(
                        DetectorAnnotation(
                            StimCoordinates(float(anc_coord[0]), float(anc_coord[1]), 0.0),
                            sorted(prev_offsets + cur_offsets),
                        )
                    )
            else:
                # Previous round is a standard round: recover each stabilizer's
                # ancilla measurement from the previous round's record map.
                _, _, prev_records = self._lookback_stack.lookback(1)
                for anc_coord, cur_idxs in start_spec.items():
                    gq = GridQubit(anc_coord[0], anc_coord[1])
                    if gq not in prev_records:
                        raise TQECError(
                            f"Y-cap seam detector references ancilla {anc_coord} "
                            "that was not measured in the previous round; the "
                            "below round does not match the Y patch."
                        )
                    prev_offset = prev_records[gq][-1]
                    offsets = [prev_offset - raw_total] + [i - raw_total for i in cur_idxs]
                    annotations.detectors.append(
                        DetectorAnnotation(
                            StimCoordinates(float(anc_coord[0]), float(anc_coord[1]), 0.0),
                            sorted(offsets),
                        )
                    )

        # Hand this round's explicit end_spec forward, if any (only the
        # transition round has one). Cleared once consumed by the next round.
        end_getter = getattr(raw_layer, "end_spec", None)
        end_spec = end_getter(self._k) if end_getter is not None else None
        self._pending_end_spec = (end_spec, raw_total) if end_spec is not None else None

        # Push this slice's own records so later rounds can look back through it.
        self._lookback_stack.append(None, None, raw_records)  # type: ignore[arg-type]

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
