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
    xtop_qubit_patch,
)
from tqec.utils.enums import Basis

Coord = tuple[int, int]


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
