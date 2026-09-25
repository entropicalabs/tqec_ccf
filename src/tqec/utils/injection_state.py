"""The single-qubit states a state-injection cube can prepare.

An injection cube encodes an arbitrary single-qubit state onto the logical qubit
of a surface-code patch. Which state is named by a short string, and each name
maps to a reset basis and one gate applied to the patch's centre data qubit
before the encoder entangles it into the code.

Two of the eight states are not stabilizer states, and stim has no gate for
either: ``stim.Circuit("T 0")`` raises. Those two are compiled as the Clifford
gate of the same rotation axis and half the angle --- a *stand-in* --- carrying a
stim tag naming the gate it stands in for. The compile therefore runs on an
ordinary :class:`stim.Circuit` throughout, and only the text emitter substitutes
the real gate back in. See
:func:`~tqec.compile.graph.TopologicalComputationGraph.generate_stim_text`.

That substitution is sound because **the patch's stabilizer group does not depend
on the injected state**. ``"0"``/``"1"`` are orthogonal and both leave every
stabilizer at ``+1``, as are ``"+"``/``"-"``; two orthogonal pairs in
complementary bases span the single-qubit state space, so every stabilizer pulls
back through the encoder's ``CX`` network to an operator acting as identity on
the centre data qubit. No stabilizer, and hence no detector anywhere downstream,
can tell which state was injected --- including a non-stabilizer one. So the
detectors computed from the stand-in circuit are exactly the detectors of the
circuit the emitted text describes.

This module deliberately depends on nothing: both :mod:`tqec.computation.cube`
and the encoder in :mod:`tqec.compile.specs.library.generators.injection` import
it, and it needs no stim.
"""

from __future__ import annotations

from typing import Final

from tqec.utils.exceptions import TQECError

INJECTION_STATES: Final[dict[str, tuple[str, str, str]]] = {
    # state: (centre reset, gate compiled onto the centre, gate it stands in for)
    "0": ("R", "I", ""),
    "1": ("R", "X", ""),
    "+": ("RX", "I", ""),
    "-": ("RX", "Z", ""),
    "i": ("RX", "S", ""),
    "-i": ("RX", "S_DAG", ""),
    "T": ("RX", "S", "T"),
    "T_DAG": ("RX", "S_DAG", "T_DAG"),
}
"""Each injectable state, as ``(reset, gate, stands_in_for)``.

``stands_in_for`` is empty for a state stim can represent directly. Where it is
not, it is both the stim tag the compiled gate carries and the name the text
emitter substitutes, so the emitter and the encoder read one source of truth.
"""

DEFAULT_INJECTION_STATE: Final[str] = "i"
"""The state an injection cube prepares unless told otherwise."""

NON_CLIFFORD_INJECTION_TAGS: Final[frozenset[str]] = frozenset(
    stands_in_for for _, _, stands_in_for in INJECTION_STATES.values() if stands_in_for
)
"""Every tag naming a gate stim cannot represent."""


def validate_injection_state(state: str) -> None:
    """Check that ``state`` names an injectable state.

    Matching is exact and deliberately case-sensitive: ``"i"`` is a state while
    ``"I"`` is the string representation of the ``INJECTION`` cube *kind*, so
    normalising case would quietly give ``"I"`` a meaning it should not have.

    Raises:
        TQECError: if ``state`` is not one of :data:`INJECTION_STATES`.

    """
    if state not in INJECTION_STATES:
        raise TQECError(
            f"Unknown injected state {state!r}. Expected one of "
            f"{sorted(INJECTION_STATES)} (matched exactly, including case)."
        )


def injection_state_operations(state: str) -> tuple[str, str, str]:
    """Return ``(reset, gate, stands_in_for)`` for ``state``.

    Raises:
        TQECError: if ``state`` is not one of :data:`INJECTION_STATES`.

    """
    validate_injection_state(state)
    return INJECTION_STATES[state]


def is_clifford_injection_state(state: str) -> bool:
    """Return whether ``state`` is one stim can represent directly.

    Raises:
        TQECError: if ``state`` is not one of :data:`INJECTION_STATES`.

    """
    return not injection_state_operations(state)[2]
