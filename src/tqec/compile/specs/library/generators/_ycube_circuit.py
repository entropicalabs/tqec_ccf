"""Dependency-free circuit construction for the native Y-half cube.

Builds the surface-code rounds (memory, transition, boundary, final) that make
up a Y-basis measurement, directly as a ``stim.Circuit`` in tqec integer
coordinates, on top of the patch geometry in
:mod:`tqec.compile.specs.library.generators.ycube`. No dependency on Gidney's
``gen`` framework at runtime; correctness is checked against a vendored copy of
``gen`` used purely as a test oracle.

The construction mirrors Gidney's ``build_surface_code_round_circuit`` (gate
schedule) and his per-chunk ``Flow`` bookkeeping (which measurements form a
detector), but reimplemented over the local geometry. Detectors are formed by
matching a round's *end* flows to the next round's *start* flows (same
stabilizer, adjacent rounds).

This module is built up incrementally; this first layer provides the standard
surface-code round and the memory experiment used to validate the machinery.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import stim
from typing_extensions import override

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.enums import SpatialBlockBorder, TemporalBlockBorder
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.raw import RawCircuitLayer
from tqec.compile.blocks.layers.composed.base import BaseComposedLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.specs.library.generators.ycube import (
    PatchGeometry,
    Stabilizer,
    gidney_to_tqec,
    tqec_to_gidney,
    xtop_qubit_patch,
    ztop_yboundary_patch,
)
from tqec.templates.base import RectangularTemplate
from tqec.utils.enums import Basis
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

Coord = tuple[int, int]

# The Y-cap raw slice occupies the same physical footprint as a memory cube
# (element shape 4k+5) and 6k+15 moments (transition + d//2 boundary rounds +
# final, each round ticked; d = 2k+1).
_YCAP_ELEMENT_SHAPE = PhysicalQubitScalable2D(LinearFunction(4, 5), LinearFunction(4, 5))
_YCAP_NUM_MOMENTS = LinearFunction(6, 15)

# Gidney's diagonal directions in complex coordinates (DR, DL, UL, UR).
_DR = 0.5 + 0.5j
_DL = -0.5 + 0.5j
_UL = -0.5 - 0.5j
_UR = 0.5 - 0.5j


@dataclass
class _Builder:
    """Accumulates a ``stim.Circuit`` over integer-coordinate qubits, tracking
    measurement record indices keyed by ``(round_tag, qubit_coord)``.

    A stabilizer ancilla is measured once per round, so ``(round_tag, coord)``
    uniquely identifies a measurement and lets later rounds form detectors that
    reference earlier rounds by coordinate.
    """

    circuit: stim.Circuit = field(default_factory=stim.Circuit)
    q2i: dict[Coord, int] = field(default_factory=dict)
    num_measurements: int = 0
    # (round_tag, coord) -> absolute measurement index (0-based, in emission order).
    records: dict[tuple[str, Coord], int] = field(default_factory=dict)

    def allocate(self, coords: set[Coord]) -> None:
        """Assign qubit indices (sorted by (y, x)) and emit QUBIT_COORDS."""
        for c in sorted(coords, key=lambda p: (p[1], p[0])):
            if c not in self.q2i:
                idx = len(self.q2i)
                self.q2i[c] = idx
                self.circuit.append("QUBIT_COORDS", [idx], [float(c[0]), float(c[1])])

    def _idx(self, coords: list[Coord]) -> list[int]:
        return [self.q2i[c] for c in coords]

    def gate1(self, name: str, coords: list[Coord]) -> None:
        if coords:
            self.circuit.append(name, self._idx(coords))

    def gate2(self, name: str, pairs: list[tuple[Coord, Coord]]) -> None:
        if pairs:
            flat: list[int] = []
            for a, b in pairs:
                flat.extend((self.q2i[a], self.q2i[b]))
            self.circuit.append(name, flat)

    def tick(self) -> None:
        self.circuit.append("TICK")

    def measure(self, name: str, coords: list[Coord], round_tag: str) -> None:
        """Emit a measurement instruction; record each qubit's absolute index."""
        if not coords:
            return
        self.circuit.append(name, self._idx(coords))
        for c in coords:
            self.records[(round_tag, c)] = self.num_measurements
            self.num_measurements += 1

    def rec(self, round_tag: str, coord: Coord) -> int:
        """Absolute record index of the measurement of ``coord`` in ``round_tag``."""
        return self.records[(round_tag, coord)]

    def detector(self, abs_indices: list[int], coord: tuple[float, ...]) -> None:
        """Append a DETECTOR referencing the given absolute measurement indices."""
        targets = [stim.target_rec(i - self.num_measurements) for i in abs_indices]
        self.circuit.append("DETECTOR", targets, list(coord))


def _split_by_basis(stabs: tuple[Stabilizer, ...]) -> tuple[list[Stabilizer], list[Stabilizer]]:
    xs = [s for s in stabs if s.basis == Basis.X]
    zs = [s for s in stabs if s.basis == Basis.Z]
    return xs, zs


def standard_round(
    b: _Builder,
    patch: PatchGeometry,
    round_tag: str,
    *,
    init_data_basis: dict[Coord, Basis] | None = None,
    measure_data_basis: dict[Coord, Basis] | None = None,
) -> None:
    """Emit one standard surface-code round on ``patch`` into ``b``.

    Port of Gidney's ``build_surface_code_round_circuit``: reset X-ancillas in
    X and Z-ancillas in Z (plus any data-qubit inits), run the four CX layers
    (ancilla->data for X stabilizers, data->ancilla for Z), then measure
    X-ancillas in X and Z-ancillas in Z (plus any data-qubit measurements).
    """
    init_data_basis = init_data_basis or {}
    measure_data_basis = measure_data_basis or {}
    xs, zs = _split_by_basis(patch.stabilizers)
    x_anc = [s.ancilla for s in xs]
    z_anc = [s.ancilla for s in zs]

    # Resets.
    b.gate1("RX", x_anc)
    for basis in (Basis.X, Basis.Z):
        qs = [q for q, bb in init_data_basis.items() if bb == basis]
        b.gate1(f"R{basis.value}", qs)
    b.gate1("R", z_anc)
    b.tick()

    # Four CX layers (ordered_data has length 4).
    for k in range(4):
        pairs: list[tuple[Coord, Coord]] = []
        for s in patch.stabilizers:
            data = s.ordered_data[k]
            if data is None:
                continue
            # X-basis: control = ancilla, target = data. Z-basis: reversed.
            pairs.append((s.ancilla, data) if s.basis == Basis.X else (data, s.ancilla))
        b.gate2("CX", pairs)
        b.tick()

    # Measurements: X-ancillas, then data (by basis), then Z-ancillas.
    b.measure("MX", x_anc, round_tag)
    for basis in (Basis.X, Basis.Z):
        qs = [q for q, bb in measure_data_basis.items() if bb == basis]
        b.measure(f"M{basis.value}", qs, round_tag)
    b.measure("M", z_anc, round_tag)
    # Trailing TICK so the next round's resets land in a fresh moment (each
    # moment holds at most one operation per qubit, as tqec's Moment requires).
    b.tick()


def _bulk_detectors(
    b: _Builder, patch: PatchGeometry, prev_tag: str, cur_tag: str
) -> None:
    """Consecutive-round detectors: each ancilla this round XOR the same ancilla
    last round (both measure the same stabilizer)."""
    for s in patch.stabilizers:
        b.detector(
            [b.rec(prev_tag, s.ancilla), b.rec(cur_tag, s.ancilla)],
            (s.ancilla[0], s.ancilla[1], 0),
        )


def _first_round_detectors(
    b: _Builder, patch: PatchGeometry, tag: str, init_data_basis: dict[Coord, Basis]
) -> None:
    """First round after a data reset: a stabilizer all of whose data qubits are
    reset in that stabilizer's own basis is deterministic on its own."""
    for s in patch.stabilizers:
        data = [d for d in s.ordered_data if d is not None]
        if all(init_data_basis.get(d) == s.basis for d in data):
            b.detector([b.rec(tag, s.ancilla)], (s.ancilla[0], s.ancilla[1], 0))


def _split_dl_md_ur(
    ps: set[complex],
) -> tuple[set[complex], set[complex], set[complex]]:
    """Port of Gidney's ``_split_dl_md_ur``: partition measure qubits into
    below-diagonal / on-diagonal / above-diagonal groups."""
    dl: set[complex] = set()
    md: set[complex] = set()
    ur: set[complex] = set()
    for m in ps:
        if m.real > m.imag + 1:
            ur.add(m)
        elif m.real == m.imag or m.real == m.imag + 1:
            md.add(m)
        else:
            dl.add(m)
    return dl, md, ur


@dataclass
class TransitionFlows:
    """The measurement records (as tqec coords, in the transition round) that
    form each stabilizer's detector, matched to adjacent rounds by ancilla.

    Attributes:
        start: ``{xtop ancilla (complex) -> [tqec coords measured this round]}``.
            Matched against the preceding memory round's measurement of the
            same ancilla to form the seam detector.
        end: ``{ztop ancilla (complex) -> [tqec coords measured this round]}``.
            Matched against the following boundary round's measurement of the
            same ancilla.
        observable: tqec coords measured this round that flow into the logical-Y
            observable, in Gidney's representative (corner Y plus the X-basis
            ancillas). The compiler emits a different representative, matched to
            the observable builder's middle-line convention --- see
            :func:`_tqec_logical_y_observable_spec`.
    """

    start: dict[complex, list[Coord]]
    end: dict[complex, list[Coord]]
    observable: list[Coord]


def transition_round(
    b: _Builder, distance: int, round_tag: str, transposed: bool = False
) -> TransitionFlows:
    """Emit the Y-basis transition round (xtop patch -> degenerate ztop patch).

    Direct port of Gidney's ``make_y_transition_round_nesw_xzxz_to_xzzx``: the
    corner data qubit is measured in the Y basis and the patch is folded onto
    the degenerate Y-boundary patch. Gate positions are computed in Gidney's
    complex-plane convention and mapped to tqec integer coordinates on emission.
    """
    d = distance
    start = xtop_qubit_patch(d, transposed)
    end = ztop_yboundary_patch(d, transposed)
    # `used`: every qubit either patch touches, in Gidney complex coords.
    used: set[complex] = set()
    for patch in (start, end):
        for s in patch.stabilizers:
            # reconstruct gidney data qubits from the stabilizer's gidney ancilla
            used.add(s.gidney_ancilla)
        for dq in patch.data_qubits:
            used.add(tqec_to_gidney(dq, transposed))

    def mbasis(q: complex) -> str | None:
        if q.real % 1 == 0:
            return None
        return "X" if int(q.real + q.imag) & 1 == 0 else "Z"

    xs = {q for q in used if mbasis(q) == "X"}
    zs = {q for q in used if mbasis(q) == "Z"}
    top_row = {q for q in used if q.imag == -0.5}
    right_col = {q for q in used if q.real == d - 0.5}

    def toward(qs: set[complex], delta: complex, sign: int) -> list[tuple[complex, complex]]:
        result = []
        for q in qs:
            if q + delta in used:
                pair = (q, q + delta)
                result.append(pair if sign == 1 else pair[::-1])
        return result

    xs_dl, xs_md, xs_ur = _split_dl_md_ur(xs)
    zs_dl, zs_md, zs_ur = _split_dl_md_ur(zs)

    def g(q: complex) -> Coord:
        return gidney_to_tqec(q, transposed)

    def e1(name: str, qs: set[complex]) -> None:
        b.gate1(name, [g(q) for q in qs])

    def e2(name: str, pairs: list[tuple[complex, complex]]) -> None:
        b.gate2(name, [(g(a), g(bb)) for a, bb in pairs])

    e1("RX", (xs - right_col) | top_row)
    e1("R", (zs - top_row) | right_col)
    b.tick()
    e2("CX", toward(xs - right_col, _DL, +1))
    e2("CX", toward(zs - top_row, _DL, -1))
    b.tick()
    e2("CX", toward(xs - right_col, _DR, +1))
    e2("CX", toward(zs - top_row, _UL, -1))
    b.tick()
    e2("CX", toward(xs_ur | xs_md, _UL, -1))
    e2("CX", toward(zs_ur, _DR, +1))
    e2("XCY", toward(zs_md, _DR, +1))
    e2("CX", toward(xs_dl, _UL, +1))
    e2("CX", toward(zs_dl, _DR, -1))
    b.tick()
    e2("CX", toward(xs_ur, _DL, -1))
    e2("CX", toward(zs_ur, _DL, +1))
    e2("CX", toward(xs_dl, _UR, +1))
    e2("CX", toward(zs_dl, _UR, -1))
    b.tick()
    e2("XCY", toward(xs_md - top_row, _DL, -1))
    b.tick()
    e1("H", {q for q in used if q.real > q.imag})
    e1("SQRT_X", {q for q in used if q.real == q.imag and q.real % 1 == 0.5})
    b.tick()

    # Measurements: X-basis (xms), then the corner data qubit in Y, then Z-basis.
    xms = (xs - top_row) | right_col
    zms = (zs - right_col) | top_row
    b.measure("MX", [g(q) for q in sorted(xms, key=lambda q: (q.imag, q.real))], round_tag)
    b.measure("MY", [g(0j)], round_tag)
    b.measure("M", [g(q) for q in sorted(zms, key=lambda q: (q.imag, q.real))], round_tag)
    # Trailing TICK so the next round starts in a fresh moment.
    b.tick()

    # Start flows: input stabilizers (xtop tiles) measured this round.
    start_flows: dict[complex, list[Coord]] = {}
    for s in start.stabilizers:
        m = s.gidney_ancilla
        if m.real == m.imag:
            meas = [m, m + 1]
        elif m.real == d - 0.5:
            meas = [m]
        elif m.imag == -0.5:
            meas = [m]
        elif m.real > m.imag and s.basis == Basis.X:
            meas = [m - 1j]
        elif m.real > m.imag and s.basis == Basis.Z:
            meas = [m + 1]
        elif m.real < m.imag:
            meas = [m]
        else:
            raise NotImplementedError(f"unexpected start ancilla {m!r}")
        start_flows[m] = [g(q) for q in meas]

    # End flows: output stabilizers (ztop tiles) prepared this round.
    end_flows: dict[complex, list[Coord]] = {}
    for s in end.stabilizers:
        m = s.gidney_ancilla
        if m == 0.5 + 0.5j:
            meas = [m, m + 1, m - 1j, m + _UL]
        elif m == d - 0.5 + 0.5j:
            meas = [m]
        elif m == d - 1.5 - 0.5j:
            meas = [m]
        elif m.real == d - 0.5:
            meas = [m, m - 1j]
        elif m.imag == -0.5:
            meas = [m, m + 1]
        elif m.real == m.imag:
            meas = [m, m + 1, m - 1j]
        else:
            meas = [m]
        end_flows[m] = [g(q) for q in meas]

    # Observable flow: logical Y = corner Y + xms records.
    obs = [g(0j)] + [g(q) for q in xms]

    return TransitionFlows(start=start_flows, end=end_flows, observable=obs)


def _final_round(
    b: _Builder,
    patch: PatchGeometry,
    prev_tag: str,
    tag: str,
    distance: int,
    transposed: bool = False,
) -> None:
    """Emit the transversal final data measurement on the degenerate patch and
    the stabilizer-reconstruction detectors it enables.

    Data qubit basis follows Gidney's anti-diagonal split: ``Z`` if
    ``x + y < d`` else ``X`` (in Gidney coords). A stabilizer whose every data
    qubit is measured in the stabilizer's own basis is reconstructed and paired
    with its last ancilla measurement in ``prev_tag``.
    """
    measure_basis: dict[Coord, Basis] = {}
    for dq in patch.data_qubits:
        g = tqec_to_gidney(dq, transposed)
        measure_basis[dq] = Basis.Z if g.real + g.imag < distance else Basis.X
    standard_round(b, patch, tag, measure_data_basis=measure_basis)
    # Bulk detectors: this round's ancilla measurement vs the previous round's
    # (the final round still measures every stabilizer via its ancilla).
    _bulk_detectors(b, patch, prev_tag, tag)
    # Reconstruction detectors: a stabilizer whose data are all measured in its
    # own basis is closed by the transversal data measurement.
    for s in patch.stabilizers:
        data = [d for d in s.ordered_data if d is not None]
        if all(measure_basis.get(d) == s.basis for d in data):
            b.detector(
                [b.rec(tag, s.ancilla)] + [b.rec(tag, d) for d in data],
                (s.ancilla[0], s.ancilla[1], 2),
            )


def y_cap_segment_circuit(
    distance: int, mem_rounds: int, init_basis: Basis
) -> stim.Circuit:
    """Full standalone Y-cap experiment for validation.

    Initialise the xtop patch data in ``init_basis``, run ``mem_rounds`` memory
    rounds, then the Y-cap segment (transition, ``d//2`` boundary rounds, final
    transversal measurement), wiring every detector including the transition
    seam detectors. No logical observable is included (a lone Y cap has no
    deterministic logical on a non-Y-eigenstate input); this validates that the
    detector structure across the transition is deterministic.
    """
    d = distance
    xtop = xtop_qubit_patch(d)
    ztop = ztop_yboundary_patch(d)
    b = _Builder()
    all_coords = (
        set(xtop.data_qubits)
        | {s.ancilla for s in xtop.stabilizers}
        | set(ztop.data_qubits)
        | {s.ancilla for s in ztop.stabilizers}
    )
    b.allocate(all_coords)

    init = {dq: init_basis for dq in xtop.data_qubits}
    mem_tags = [f"m{i}" for i in range(mem_rounds)]
    standard_round(b, xtop, mem_tags[0], init_data_basis=init)
    _first_round_detectors(b, xtop, mem_tags[0], init)
    for i in range(1, mem_rounds):
        standard_round(b, xtop, mem_tags[i])
        _bulk_detectors(b, xtop, mem_tags[i - 1], mem_tags[i])

    # Transition round + seam detectors against the last memory round.
    t_tag = "T"
    flows = transition_round(b, d, t_tag)
    for s in xtop.stabilizers:
        m = s.gidney_ancilla
        b.detector(
            [b.rec(mem_tags[-1], s.ancilla)] + [b.rec(t_tag, c) for c in flows.start[m]],
            (s.ancilla[0], s.ancilla[1], 0),
        )

    # Boundary (padding) rounds on the degenerate patch.
    pad = d // 2
    b_tags = [f"b{i}" for i in range(pad)]
    prev = t_tag
    for i in range(pad):
        standard_round(b, ztop, b_tags[i])
        if i == 0:
            # First boundary round's start flows match the transition end flows.
            for s in ztop.stabilizers:
                m = s.gidney_ancilla
                b.detector(
                    [b.rec(t_tag, c) for c in flows.end[m]] + [b.rec(b_tags[0], s.ancilla)],
                    (s.ancilla[0], s.ancilla[1], 0),
                )
        else:
            _bulk_detectors(b, ztop, b_tags[i - 1], b_tags[i])
        prev = b_tags[i]

    # Final transversal data measurement.
    _final_round(b, ztop, prev, "F", d)
    return b.circuit


def y_cap_raw_circuit(distance: int) -> tuple[stim.Circuit, dict[Coord, list[int]]]:
    """The raw Y-cap slice ``[transition, boundary x d//2, final]`` for a
    ``RawCircuitLayer``, in tqec integer coordinates.

    The data qubits are NOT reset here -- they carry the logical state in from
    the below cube's memory column, which tqec emits natively. All detectors
    that are internal to the slice are emitted here (transition end-flows vs the
    first boundary round, boundary bulk, and the final round's bulk +
    reconstruction), together with nothing that references a prior slice.

    The transition round's *start* flows can only be closed against the below
    cube's last memory round, which lives in a different slice and whose record
    layout is not visible to a ``circuit_factory``. Those seam detectors are
    therefore left for the detector annotator: this function returns a
    ``seam_spec`` mapping each below-cube ancilla coordinate to the list of
    *raw-slice-relative* measurement record indices (0-based within the returned
    circuit) that, XORed with that ancilla's measurement in the previous round,
    form the seam detector.

    Returns:
        ``(circuit, seam_spec)``.
    """
    d = distance
    xtop = xtop_qubit_patch(d)
    ztop = ztop_yboundary_patch(d)
    b = _Builder()
    b.allocate(
        set(xtop.data_qubits)
        | {s.ancilla for s in xtop.stabilizers}
        | set(ztop.data_qubits)
        | {s.ancilla for s in ztop.stabilizers}
    )

    t_tag = "T"
    flows = transition_round(b, d, t_tag)

    # Seam spec: below-cube ancilla coord -> raw-relative record indices of the
    # transition measurements that close its stabilizer. gidney_to_tqec(m) is the
    # coordinate the below round measured that same stabilizer's ancilla at.
    seam_spec: dict[Coord, list[int]] = {}
    for s in xtop.stabilizers:
        m = s.gidney_ancilla
        seam_spec[s.ancilla] = [b.rec(t_tag, c) for c in flows.start[m]]

    pad = d // 2
    b_tags = [f"b{i}" for i in range(pad)]
    prev = t_tag
    for i in range(pad):
        standard_round(b, ztop, b_tags[i])
        if i == 0:
            for s in ztop.stabilizers:
                m = s.gidney_ancilla
                b.detector(
                    [b.rec(t_tag, c) for c in flows.end[m]] + [b.rec(b_tags[0], s.ancilla)],
                    (s.ancilla[0], s.ancilla[1], 0),
                )
        else:
            _bulk_detectors(b, ztop, b_tags[i - 1], b_tags[i])
        prev = b_tags[i]

    _final_round(b, ztop, prev, "F", d)
    return b.circuit, seam_spec


# --- Per-round Y-cap slices (milestone 2) --------------------------------
#
# The monolithic ``[transition, boundary x k, final]`` slice is sliced into its
# constituent single rounds, each emitted as its own ``RawCircuitLayer`` so the
# compile tree treats the cap like any other cube: ``transition (1)`` +
# ``boundary0 (1)`` + ``RepeatedLayer(boundary, k-1)`` + ``final (1)`` = ``k+2``
# rounds for ``d = 2k+1`` (``k = d//2`` boundary rounds in total).
#
# A single round carries no cross-round detectors: those are formed by the
# detector annotator across consecutive raw layers from the *flow specs* below.
# The only detectors internal to a round are the final round's stabilizer
# reconstruction detectors (they reference only the final round's own records).
#
# Flow specs (all keyed by tqec ancilla coordinate; values are lists of *qubit
# coordinates* measured this round, in the round's local element frame):
#   start_spec : the qubits this round measures to detect a stabilizer, matched
#                against the *previous* round's measurement of the same ancilla.
#   end_spec   : the qubits this round measures preparing a stabilizer, to be
#                matched by the *next* round's start_spec. Only the transition
#                round has a non-trivial (multi-qubit) end_spec; standard rounds
#                measure each stabilizer with a single ancilla, so their
#                successors recover the match from the measurement records keyed
#                by coordinate.
#   reconstruction_spec : (final round only) the ancilla + transversally-measured
#                data qubits that reconstruct a stabilizer within the final round.
#
# Values are *coordinates*, not record indices, because when a Y cap coexists
# with a memory cube the two rounds' measurements are interleaved by
# ``merge_scheduled_circuits`` -- so a local record index no longer identifies
# the right measurement, but a qubit coordinate always does (each qubit is
# measured at most once per round). The detector annotator resolves each
# coordinate against the merged round's measurement-record map.

_TRANSITION_NUM_MOMENTS = LinearFunction(0, 8)
_STANDARD_NUM_MOMENTS = LinearFunction(0, 6)


def _strip_trailing_tick(circuit: stim.Circuit) -> stim.Circuit:
    """Drop a single trailing ``TICK`` so the round matches the leaf-circuit
    convention (the compile tree inserts the inter-round ``TICK`` itself)."""
    n = len(circuit)
    if n and circuit[n - 1].name == "TICK":
        stripped = stim.Circuit()
        for i in range(n - 1):
            stripped.append(circuit[i])
        return stripped
    return circuit


def _measurement_coordinates(circuit: stim.Circuit) -> list[Coord]:
    """The tqec coordinate measured by each measurement of ``circuit``, in order."""
    qubit_coords = circuit.get_final_qubit_coordinates()
    coords: list[Coord] = []
    for instruction in circuit.flattened():
        if instruction.name not in ("M", "MX", "MY", "MZ"):
            continue
        for target in instruction.targets_copy():
            c = qubit_coords[target.qubit_value]
            coords.append((int(c[0]), int(c[1])))
    return coords


@functools.cache
def _tqec_logical_y_observable_spec(
    distance: int, transposed: bool = False
) -> tuple[Coord, ...]:
    """The transition-round records measuring the logical Y operator *in tqec's
    representative*.

    :func:`transition_round` reports a logical-Y flow of its own
    (``TransitionFlows.observable``, Gidney's corner-Y plus X-ancilla
    representative). That is a valid logical Y, but it is not the representative
    the rest of the compiler uses: an incoming correlation surface is lowered by
    the observable builder onto the patch's *middle* lines --- the X sheet onto
    the data column ``x = d`` and the Z sheet onto the data row ``y = d`` (see
    ``build_regular_cube_top_readout_qubits``). The two representatives differ by
    a product of input-patch stabilizers, so combining Gidney's readout with the
    builder's host measurements leaves the observable non-deterministic.

    Rather than hard-code that stabilizer correction, ask ``stim`` for the
    records implementing the flow ``X(column x=d) . Z(row y=d) -> I`` through the
    transition round. Any solution is equally valid --- solutions differ only by
    sets that are deterministic within the round --- and it is exact and correct
    at every distance by construction.
    """
    d = distance
    circuit, _ = _build_transition_round(d, transposed)
    index_of = {
        (int(c[0]), int(c[1])): q for q, c in circuit.get_final_qubit_coordinates().items()
    }
    target = stim.PauliString(circuit.num_qubits)
    # Expressed on the Gidney lattice and mapped through ``gidney_to_tqec`` so
    # that both middle lines follow the patch's orientation rather than being
    # pinned to a fixed tqec row/column.
    middle = (d - 1) // 2
    for i in range(d):
        target[index_of[gidney_to_tqec(complex(middle, i), transposed)]] = "X"
    for i in range(d):
        qubit = index_of[gidney_to_tqec(complex(i, middle), transposed)]
        # The centre qubit carries both sheets, i.e. X . Z = Y.
        target[qubit] = "Y" if target[qubit] == 1 else "Z"
    (solution,) = stim.Circuit.solve_flow_measurements(
        circuit,
        [stim.Flow(input=target, output=stim.PauliString(circuit.num_qubits))],
    )
    coords = _measurement_coordinates(circuit)
    return tuple(sorted(coords[i] for i in solution))


def _build_transition_round(
    distance: int, transposed: bool = False
) -> tuple[stim.Circuit, TransitionFlows]:
    """The transition round on its own (no trailing ``TICK``) and its flows."""
    d = distance
    xtop = xtop_qubit_patch(d, transposed)
    ztop = ztop_yboundary_patch(d, transposed)
    b = _Builder()
    b.allocate(
        set(xtop.data_qubits)
        | {s.ancilla for s in xtop.stabilizers}
        | set(ztop.data_qubits)
        | {s.ancilla for s in ztop.stabilizers}
    )
    flows = transition_round(b, d, "T", transposed)
    return _strip_trailing_tick(b.circuit), flows


def transition_raw_slice(
    distance: int, transposed: bool = False
) -> tuple[stim.Circuit, dict[Coord, list[int]], dict[Coord, list[int]], list[int]]:
    """The single transition round as a standalone circuit plus its flow specs.

    Returns ``(circuit, start_spec, end_spec, observable_spec)`` where
    ``start_spec`` closes the transition against the below memory round (the
    seam), ``end_spec`` prepares the degenerate patch stabilizers for the first
    boundary round, and ``observable_spec`` are the records reconstructing the
    logical-Y operator in the representative the observable builder uses (see
    :func:`_tqec_logical_y_observable_spec`).
    """
    d = distance
    circuit, flows = _build_transition_round(d, transposed)
    xtop = xtop_qubit_patch(d, transposed)
    ztop = ztop_yboundary_patch(d, transposed)
    start_spec = {s.ancilla: list(flows.start[s.gidney_ancilla]) for s in xtop.stabilizers}
    end_spec = {s.ancilla: list(flows.end[s.gidney_ancilla]) for s in ztop.stabilizers}
    return (
        circuit,
        start_spec,
        end_spec,
        list(_tqec_logical_y_observable_spec(d, transposed)),
    )


def boundary_raw_slice(
    distance: int, transposed: bool = False
) -> tuple[stim.Circuit, dict[Coord, list[Coord]]]:
    """A single boundary (padding) round on the degenerate patch, plus its
    ``start_spec`` (each stabilizer measured by its ancilla once)."""
    d = distance
    ztop = ztop_yboundary_patch(d, transposed)
    b = _Builder()
    b.allocate(set(ztop.data_qubits) | {s.ancilla for s in ztop.stabilizers})
    standard_round(b, ztop, "B")
    start_spec = {s.ancilla: [s.ancilla] for s in ztop.stabilizers}
    return _strip_trailing_tick(b.circuit), start_spec


def final_raw_slice(
    distance: int, transposed: bool = False
) -> tuple[stim.Circuit, dict[Coord, list[Coord]], dict[Coord, list[Coord]]]:
    """The transversal final data measurement round, plus its ``start_spec`` and
    ``reconstruction_spec``.

    The bulk detector against the last boundary round is left to the annotator
    (via ``start_spec`` matched by ancilla coordinate). The stabilizer-
    reconstruction detectors are also emitted by the annotator (via
    ``reconstruction_spec``) rather than inline, because when the cap coexists
    with a memory cube the merged round interleaves both patches' measurements
    and a local record index would no longer be valid.
    """
    d = distance
    ztop = ztop_yboundary_patch(d, transposed)
    b = _Builder()
    b.allocate(set(ztop.data_qubits) | {s.ancilla for s in ztop.stabilizers})
    measure_basis: dict[Coord, Basis] = {}
    for dq in ztop.data_qubits:
        g = tqec_to_gidney(dq, transposed)
        measure_basis[dq] = Basis.Z if g.real + g.imag < d else Basis.X
    standard_round(b, ztop, "F", measure_data_basis=measure_basis)
    start_spec = {s.ancilla: [s.ancilla] for s in ztop.stabilizers}
    reconstruction_spec: dict[Coord, list[Coord]] = {}
    for s in ztop.stabilizers:
        data = [dd for dd in s.ordered_data if dd is not None]
        if all(measure_basis.get(dd) == s.basis for dd in data):
            reconstruction_spec[s.ancilla] = [s.ancilla, *data]
    return _strip_trailing_tick(b.circuit), start_spec, reconstruction_spec


class _YRoundRawLayer(RawCircuitLayer):
    """One round of the Y-basis measurement cap, as a :class:`RawCircuitLayer`.

    Exposes the round's flow specs (``start_spec`` / ``end_spec`` /
    ``observable_spec`` / ``reconstruction_spec``) so the detector annotator can
    form the cross-round detectors the sliced structure no longer carries inline.
    All spec values are qubit coordinates in the local element frame.
    """

    def __init__(
        self,
        circuit_factory: Callable[[int], stim.Circuit],
        num_moments: LinearFunction,
    ) -> None:
        self._make_circuit = circuit_factory
        super().__init__(self._make_scheduled_circuit, _YCAP_ELEMENT_SHAPE, num_moments)

    def _make_scheduled_circuit(self, k: int) -> ScheduledCircuit:
        return ScheduledCircuit.from_circuit(self._make_circuit(2 * k + 1))

    def start_spec(self, k: int) -> dict[Coord, list[Coord]]:
        """Qubits matched against the previous round's measurement of the same
        ancilla (keyed by tqec ancilla coordinate)."""
        return {}

    def end_spec(self, k: int) -> dict[Coord, list[Coord]] | None:
        """Multi-qubit preparation spec handed to the next round, or ``None``
        when the successor recovers the match from measurement records by
        coordinate (every standard round)."""
        return None

    def observable_spec(self, k: int) -> list[Coord] | None:
        """Qubits reconstructing the logical-Y operator, or ``None``."""
        return None

    def reconstruction_spec(self, k: int) -> dict[Coord, list[Coord]] | None:
        """Stabilizer-reconstruction spec internal to this round, or ``None``."""
        return None


class _TransitionRawLayer(_YRoundRawLayer):
    def __init__(self, transposed: bool = False) -> None:
        self._transposed = transposed
        super().__init__(
            lambda d: transition_raw_slice(d, transposed)[0], _TRANSITION_NUM_MOMENTS
        )

    def start_spec(self, k: int) -> dict[Coord, list[Coord]]:
        return transition_raw_slice(2 * k + 1, self._transposed)[1]

    def end_spec(self, k: int) -> dict[Coord, list[Coord]] | None:
        return transition_raw_slice(2 * k + 1, self._transposed)[2]

    def observable_spec(self, k: int) -> list[Coord] | None:
        return transition_raw_slice(2 * k + 1, self._transposed)[3]


class _BoundaryRawLayer(_YRoundRawLayer):
    def __init__(self, transposed: bool = False) -> None:
        self._transposed = transposed
        super().__init__(
            lambda d: boundary_raw_slice(d, transposed)[0], _STANDARD_NUM_MOMENTS
        )

    def start_spec(self, k: int) -> dict[Coord, list[Coord]]:
        return boundary_raw_slice(2 * k + 1, self._transposed)[1]


class _FinalRawLayer(_YRoundRawLayer):
    def __init__(self, transposed: bool = False) -> None:
        self._transposed = transposed
        super().__init__(
            lambda d: final_raw_slice(d, transposed)[0], _STANDARD_NUM_MOMENTS
        )

    def start_spec(self, k: int) -> dict[Coord, list[Coord]]:
        return final_raw_slice(2 * k + 1, self._transposed)[1]

    def reconstruction_spec(self, k: int) -> dict[Coord, list[Coord]] | None:
        return final_raw_slice(2 * k + 1, self._transposed)[2]


def make_y_cap_layers(transposed: bool = False) -> list[BaseLayer | BaseComposedLayer]:
    """Build the sliced Y-cap layer sequence ``[transition, boundary0,
    RepeatedLayer(boundary, k-1), final]`` (``k`` boundary rounds in total)."""
    return [
        _TransitionRawLayer(transposed),
        _BoundaryRawLayer(transposed),
        RepeatedLayer(_BoundaryRawLayer(transposed), LinearFunction(1, -1)),
        _FinalRawLayer(transposed),
    ]


class YHalfCubeBlock(Block):
    """A Y-basis measurement cap, which *gains* its junction round rather than
    having its first round overwritten.

    For an ordinary cube, a temporal pipe below replaces the block's
    ``Z_NEGATIVE`` border --- ``layer_sequence[0]`` --- with the pipe's junction
    layer. A Y cap has no round to spare there: its first layer is the transition
    round, and letting the substitution proceed would overwrite it with a memory
    round. Nothing in the substitution machinery objects (``RawCircuitLayer`` is
    a ``BaseLayer``, so ``get_atomic_temporal_border`` happily returns it); the
    loss only surfaces further downstream, as a seam-detector mismatch.

    Earlier revisions dodged this by prefixing the cap with a hard-coded memory
    "adapter" round that existed only to absorb the substitution. That forced the
    adapter's orientation to be guessed (it was pinned to ``HORIZONTAL``, i.e. a
    ``ZX*`` cube below). Prepending instead lets the junction round come from the
    pipe itself, whose kind is derived from the cube below, so no orientation is
    assumed here. The resulting layer sequence is identical to the one the
    adapter produced.

    ``template`` is the block's spatial footprint, needed to build the
    :class:`~tqec.compile.specs.base.PipeSpec` of the temporal pipe below (raw
    layers carry no template of their own). It describes shape only --- the
    orientation lives in the plaquettes, which the pipe supplies --- so carrying
    it here reintroduces no orientation assumption.
    """

    def __init__(
        self,
        layer_sequence: Sequence[BaseLayer | BaseComposedLayer],
        trimmed_spatial_borders: frozenset[SpatialBlockBorder] = frozenset(),
        template: RectangularTemplate | None = None,
    ) -> None:
        super().__init__(layer_sequence, trimmed_spatial_borders)
        self.template = template

    @property
    @override
    def releases_its_qubits(self) -> bool:
        # The cap's final round measures every data qubit of the degenerate
        # patch transversally, so nothing of it survives into later rounds.
        return True

    @override
    def with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> YHalfCubeBlock | None:
        # Keep the subclass: the merged slice reads ``releases_its_qubits`` off
        # the block *after* the pipe below has been substituted in.
        if not border_replacements:
            return self
        layers = self._layers_with_temporal_borders_replaced(border_replacements)
        if not layers:
            return None
        return YHalfCubeBlock(layers, self.trimmed_spatial_borders, self.template)

    @override
    def with_spatial_borders_trimmed(
        self, borders: Iterable[SpatialBlockBorder]
    ) -> YHalfCubeBlock:
        # Keep the subclass so a later temporal substitution still prepends.
        return YHalfCubeBlock(
            self._layers_with_spatial_borders_trimmed(borders),
            self.trimmed_spatial_borders | frozenset(borders),
            self.template,
        )

    @override
    def _layers_with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> list[BaseLayer | BaseComposedLayer]:
        below = border_replacements.get(TemporalBlockBorder.Z_NEGATIVE)
        remaining = {
            border: layer
            for border, layer in border_replacements.items()
            if border is not TemporalBlockBorder.Z_NEGATIVE
        }
        layers = super()._layers_with_temporal_borders_replaced(remaining)
        if below is not None:
            layers.insert(0, below)
        return layers


def memory_experiment_circuit(distance: int, rounds: int, basis: Basis) -> stim.Circuit:
    """A self-contained surface-code memory experiment on the xtop patch.

    Used to validate the standard-round + detector machinery independently of
    the Y transition: initialise all data qubits in ``basis``, run ``rounds``
    standard rounds, then measure all data qubits in ``basis``. The resulting
    circuit's detectors must all be deterministic (its DEM builds cleanly).
    """
    patch = xtop_qubit_patch(distance)
    b = _Builder()
    b.allocate(set(patch.data_qubits) | {s.ancilla for s in patch.stabilizers})
    init = {d: basis for d in patch.data_qubits}
    tags = [f"r{i}" for i in range(rounds)]

    standard_round(b, patch, tags[0], init_data_basis=init)
    _first_round_detectors(b, patch, tags[0], init)
    for i in range(1, rounds):
        standard_round(b, patch, tags[i])
        _bulk_detectors(b, patch, tags[i - 1], tags[i])

    # Final data measurement + stabilizer reconstruction.
    final_tag = "final"
    b.measure(f"M{basis.value}", sorted(patch.data_qubits, key=lambda p: (p[1], p[0])), final_tag)
    for s in patch.stabilizers:
        if s.basis != basis:
            continue
        data = [d for d in s.ordered_data if d is not None]
        b.detector(
            [b.rec(tags[-1], s.ancilla)] + [b.rec(final_tag, d) for d in data],
            (s.ancilla[0], s.ancilla[1], 1),
        )
    return b.circuit
