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

from dataclasses import dataclass, field

import stim

from tqec.compile.specs.library.generators.ycube import (
    PatchGeometry,
    Stabilizer,
    gidney_to_tqec,
    xtop_qubit_patch,
    ztop_yboundary_patch,
)
from tqec.utils.enums import Basis

Coord = tuple[int, int]

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
            observable.
    """

    start: dict[complex, list[Coord]]
    end: dict[complex, list[Coord]]
    observable: list[Coord]


def transition_round(b: _Builder, distance: int, round_tag: str) -> TransitionFlows:
    """Emit the Y-basis transition round (xtop patch -> degenerate ztop patch).

    Direct port of Gidney's ``make_y_transition_round_nesw_xzxz_to_xzzx``: the
    corner data qubit is measured in the Y basis and the patch is folded onto
    the degenerate Y-boundary patch. Gate positions are computed in Gidney's
    complex-plane convention and mapped to tqec integer coordinates on emission.
    """
    d = distance
    start = xtop_qubit_patch(d)
    end = ztop_yboundary_patch(d)
    # `used`: every qubit either patch touches, in Gidney complex coords.
    used: set[complex] = set()
    for patch in (start, end):
        for s in patch.stabilizers:
            # reconstruct gidney data qubits from the stabilizer's gidney ancilla
            used.add(s.gidney_ancilla)
        for dq in patch.data_qubits:
            used.add(complex((dq[0] - 1) / 2, (dq[1] - 1) / 2))

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
        return gidney_to_tqec(q)

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
    b: _Builder, patch: PatchGeometry, prev_tag: str, tag: str, distance: int
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
        gx, gy = (dq[0] - 1) / 2, (dq[1] - 1) / 2
        measure_basis[dq] = Basis.Z if gx + gy < distance else Basis.X
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
