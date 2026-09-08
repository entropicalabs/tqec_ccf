"""Non-fault-tolerant state-injection encoder for the injection cube.

Builds the encoding circuit that prepares an arbitrary single-qubit state on the
logical qubit of a rotated surface-code patch, directly as a ``stim.Circuit`` in
tqec integer coordinates. The circuit performs **no measurements**: it resets the
patch's data qubits, applies one single-qubit gate to the centre data qubit, and
entangles the patch with a fixed schedule of ``CX`` layers, leaving every
stabilizer of the patch in its ``+1`` eigenstate.

Which state is prepared is named by a string; the reset basis and the gate come
from :data:`~tqec.utils.injection_state.INJECTION_STATES`. Two of the eight
states are not stabilizer states and are compiled as a tagged Clifford stand-in,
because stim has no gate for either --- see that module for why substituting the
real gate afterwards cannot change any detector.

The construction is the one used by the sibling Entropica project ``noncliff``
(``noncliff/injection.py``), vendored as a test oracle in
``tests/_vendor/noncliff``. A ``d = 3`` patch is encoded directly, and a larger
patch is reached by wrapping the ``d - 2`` patch in a *shell*: a ring of freshly
reset boundary qubits entangled with the interior by two ``CX`` layers.

Injection is **not** fault tolerant. A single fault during the encoder can
corrupt the injected state, so the encoded state carries an error rate of order
``p`` rather than ``p**d``. That is inherent to state injection, and the reason
the encoder is a distinct cube kind rather than an initialisation basis.

## Coordinate frame

``noncliff`` and tqec already agree on the qubit-coordinate frame --- data qubits
on odd-odd grid points, stabilizer ancillas on even-even points, with the same
checkerboard parity --- so no coordinate transform is involved. The patch this
encoder prepares is exactly
:func:`~tqec.compile.specs.library.generators.ycube.xtop_qubit_patch`, whose
left and right walls are ``Z`` (a ``ZX*`` cube above the injection). An ``XZ*``
cube above has those walls in ``X``, and its patch is the reflection across the
main diagonal, obtained with ``transposed=True``.

The centre data qubit, which carries the injected state, sits at ``(d, d)`` and
is therefore fixed by that reflection.
"""

from __future__ import annotations

import stim

from tqec.compile.specs.library.generators.ycube import xtop_qubit_patch
from tqec.utils.exceptions import TQECError
from tqec.utils.injection_state import (
    DEFAULT_INJECTION_STATE,
    injection_state_operations,
)
from tqec.utils.scale import LinearFunction

_Coord = tuple[int, int]
_Pair = tuple[_Coord, _Coord]

INJECTION_ENCODER_MOMENTS = LinearFunction(3, 4)
"""Number of moments of the encoder for ``d = 2k + 1``.

Seven moments encode the ``d = 3`` patch, and every shell that grows the patch by
two adds three more (resets, then the shell's two ``CX`` layers).
"""


def _shell_layer_a(distance: int, ring: int) -> list[_Pair]:
    """First ``CX`` layer of the shell that grows a patch from ``d - 2`` to ``d``.

    ``ring`` is counted inwards from the boundary of the ``distance x distance``
    patch: ring ``0`` is the outermost shell, and ring ``r`` is the shell that
    grew the patch to ``distance - 2 * r``.

    Every pair is ``(control, target)``.
    """
    lo, hi = 2 * ring + 1, 2 * (distance - ring) - 1
    pairs: list[_Pair] = [((hi, lo + 2), (hi - 2, lo + 2))]
    pairs += [((x, lo), (x - 2, lo)) for x in range(hi - 2, lo + 3, -4)]
    pairs.append(((lo + 2, lo + 2), (lo + 2, lo)))
    pairs += [((lo, y), (lo, y - 2)) for y in range(lo + 4, hi - 3, 4)]
    pairs.append(((lo, hi - 2), (lo + 2, hi - 2)))
    pairs += [((x, hi), (x + 2, hi)) for x in range(lo + 2, hi - 3, 4)]
    pairs.append(((hi - 2, hi - 2), (hi - 2, hi)))
    pairs += [((hi, y), (hi, y + 2)) for y in range(hi - 4, lo + 3, -4)]
    return pairs


def _shell_layer_b(distance: int, ring: int) -> list[_Pair]:
    """Second ``CX`` layer of the same shell. See :func:`_shell_layer_a`."""
    lo, hi = 2 * ring + 1, 2 * (distance - ring) - 1
    pairs: list[_Pair] = [((x, lo + 2), (x, lo)) for x in range(hi, lo + 3, -2)]
    pairs += [((lo, y), (lo + 2, y)) for y in range(lo, hi - 3, 2)]
    pairs += [((x, hi - 2), (x, hi)) for x in range(lo, hi - 3, 2)]
    pairs += [((hi, y), (hi - 2, y)) for y in range(lo + 4, hi + 1, 2)]
    return pairs


def _shell_reset_bases(distance: int, ring: int) -> dict[_Coord, str]:
    """Reset basis of each qubit the shell at ``ring`` adds to the patch.

    A new boundary qubit is prepared in the basis it is first used in: ``|+>``
    (``RX``) if the first two-qubit gate touching it uses it as a control,
    ``|0>`` (``R``) if as a target.
    """
    lo, hi = 2 * ring + 1, 2 * (distance - ring) - 1
    perimeter = {
        (x, y)
        for x in range(lo, hi + 1, 2)
        for y in range(lo, hi + 1, 2)
        if x in {lo, hi} or y in {lo, hi}
    }
    bases: dict[_Coord, str] = {}
    for control, target in _shell_layer_a(distance, ring) + _shell_layer_b(distance, ring):
        if control in perimeter:
            bases.setdefault(control, "RX")
        if target in perimeter:
            bases.setdefault(target, "R")
    return bases


# The ``d = 3`` encoder, in the frame of a patch whose lowest data qubit is at
# (1, 1). Resets first, then the four CX layers; every pair is (control, target).
_BASE_RESETS: dict[str, tuple[_Coord, ...]] = {
    "R": ((1, 1), (1, 5), (5, 1), (5, 5)),
    "RX": ((1, 3), (3, 1), (3, 5), (5, 3)),
}
_BASE_CX_LAYERS: tuple[tuple[_Pair, ...], ...] = (
    (((3, 1), (1, 1)), ((1, 3), (3, 3)), ((3, 5), (5, 5))),
    (((1, 3), (1, 5)), ((3, 3), (3, 1)), ((5, 3), (5, 1))),
    (((1, 3), (1, 1)), ((5, 1), (3, 1)), ((5, 3), (3, 3))),
    (((5, 3), (5, 5)), ((3, 3), (3, 5))),
)


class _Encoder:
    """Accumulates the encoder circuit over coordinate-addressed qubits."""

    def __init__(self, qubits: set[_Coord], transposed: bool) -> None:
        self._transposed = transposed
        self._circuit = stim.Circuit()
        self._started = False
        self._q2i: dict[_Coord, int] = {}
        for coord in sorted(qubits, key=lambda c: (c[1], c[0])):
            self._q2i[coord] = len(self._q2i)
            self._circuit.append(
                "QUBIT_COORDS", [self._q2i[coord]], [float(coord[0]), float(coord[1])]
            )

    @property
    def circuit(self) -> stim.Circuit:
        """Get the accumulated circuit."""
        return self._circuit

    def _index(self, coord: _Coord) -> int:
        x, y = coord
        return self._q2i[(y, x) if self._transposed else (x, y)]

    def _open_moment(self) -> None:
        """Separate this moment from the previous one, if there was one."""
        if self._started:
            self._circuit.append(stim.CircuitInstruction("TICK", []))
        self._started = True

    def moment_1q(self, gates: dict[str, list[_Coord]], tag: str = "") -> None:
        """Append one moment of single-qubit gates, keyed by instruction name.

        ``tag`` is attached to every instruction emitted in this moment. It marks
        a gate that stands in for one stim cannot represent; see
        :mod:`tqec.utils.injection_state`.
        """
        self._open_moment()
        for name, coords in gates.items():
            if coords:
                self._circuit.append(
                    stim.CircuitInstruction(name, [self._index(c) for c in coords], tag=tag)
                )

    def moment_cx(self, pairs: list[_Pair]) -> None:
        """Append one moment of ``CX`` gates, given as ``(control, target)`` pairs."""
        self._open_moment()
        if pairs:
            targets: list[int] = []
            for control, target in pairs:
                targets += [self._index(control), self._index(target)]
            self._circuit.append(stim.CircuitInstruction("CX", targets))


def injection_encoder_circuit(
    distance: int, *, transposed: bool = False, state: str = DEFAULT_INJECTION_STATE
) -> stim.Circuit:
    """Build the state-injection encoder for a ``distance x distance`` patch.

    Args:
        distance: the code distance ``d``, odd and at least 3.
        transposed: reflect the patch across its main diagonal. The encoder is
            written for a patch whose left and right walls are ``Z`` (a ``ZX*``
            cube above the injection); an ``XZ*`` cube above needs the
            reflection.
        state: which single-qubit state to prepare on the logical qubit. One of
            :data:`~tqec.utils.injection_state.INJECTION_STATES`, matched
            exactly. ``"T"`` and ``"T_DAG"`` are compiled as a tagged Clifford
            stand-in, since stim has no gate for either.

    Returns:
        the encoder circuit, in tqec integer coordinates, declaring
        ``QUBIT_COORDS`` for every qubit of the patch (data qubits and
        stabilizer ancillas alike, so the footprint matches a memory round) and
        acting only on the data qubits. It contains no measurement.

    Raises:
        TQECError: if ``distance`` is even or smaller than 3, or if ``state`` is
            not one of :data:`~tqec.utils.injection_state.INJECTION_STATES`.

    """
    if distance < 3 or distance % 2 == 0:
        raise TQECError(f"The code distance must be odd and at least 3, got {distance}.")
    reset, gate, stands_in_for = injection_state_operations(state)

    patch = xtop_qubit_patch(distance)
    encoder = _Encoder(
        set(patch.data_qubits) | {stabilizer.ancilla for stabilizer in patch.stabilizers},
        transposed,
    )

    # The d = 3 encoder acts on the innermost patch, whose lowest data qubit is
    # at (1 + shift, 1 + shift); the centre data qubit lands on (d, d).
    shift = distance - 3

    def shifted(coord: _Coord) -> _Coord:
        return (coord[0] + shift, coord[1] + shift)

    # The gate moment is always emitted, as ``I`` where the state needs none, so
    # the encoder's moment count stays ``3k + 4`` whatever is injected.
    centre = (distance, distance)
    encoder.moment_1q({reset: [centre]})
    encoder.moment_1q({gate: [centre]}, tag=stands_in_for)
    encoder.moment_1q(
        {basis: [shifted(c) for c in coords] for basis, coords in _BASE_RESETS.items()}
    )
    for layer in _BASE_CX_LAYERS:
        encoder.moment_cx([(shifted(c), shifted(t)) for c, t in layer])

    # Then grow the patch outwards, one shell per ring, innermost ring first.
    for ring in range((distance - 5) // 2, -1, -1):
        bases = _shell_reset_bases(distance, ring)
        encoder.moment_1q(
            {basis: sorted(c for c, b in bases.items() if b == basis) for basis in ("R", "RX")}
        )
        encoder.moment_cx(_shell_layer_a(distance, ring))
        encoder.moment_cx(_shell_layer_b(distance, ring))

    return encoder.circuit
