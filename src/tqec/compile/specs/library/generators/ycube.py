"""Native Y-half-cube (S-gate cap) circuit generation.

This module builds the surface-code Y-basis-measurement segment natively, in
tqec's own qubit-coordinate frame, so it can be emitted by a
:class:`~tqec.compile.blocks.layers.atomic.raw.RawCircuitLayer` inside tqec's
own layer tree (no external splice/stitch machinery).

The physics is Craig Gidney's in-place Y-basis measurement ("Inplace Access to
the Surface Code Y Basis", 2024): a memory round, a transition round that folds
the ``xtop`` qubit patch onto a degenerate ``ztop`` Y-boundary patch (measuring
the corner data qubit in the Y basis), padding rounds, and a transversal final
data measurement. Rather than depend on Gidney's ``gen`` framework at runtime,
the constructions here are reimplemented directly and verified against a
vendored copy of ``gen`` used purely as a test oracle
(``tests/_vendor/midout``); see ``tests/compile/specs/library/generators``.

This first layer provides the **geometry** only: the surface-code patch
(data qubits, stabilizers) reproduced in tqec integer coordinates. The circuit
(gate schedules, detectors, observable) is built on top of it.

## Coordinate frame

Gidney places data qubits at integer points of the complex plane and stabilizer
ancillas at half-integer points. tqec places data qubits at odd integer grid
points and ancillas at even integer points. The two are related by a single
uniform transform, verified empirically against tqec's own memory patch::

    tqec_coord(q) = (2 * q.real + 1, 2 * q.imag + 1)

(a Gidney data qubit ``col + row*1j`` -> tqec ``(2*col+1, 2*row+1)``; a Gidney
ancilla at a half-integer coordinate maps to an even tqec grid point). This is
the local, single-cube element frame; positioning of the patch within a larger
computation is handled by the enclosing ``LayoutLayer``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqec.utils.enums import Basis

# The four diagonal directions from a stabilizer ancilla to its data qubits, in
# Gidney's half-step complex convention: DR, DL, UL, UR (= (0.5+0.5j)*1j**d).
_DIRS: tuple[complex, ...] = tuple((0.5 + 0.5j) * 1j**d for d in range(4))
_DR, _DL, _UL, _UR = _DIRS


def gidney_to_tqec(q: complex) -> tuple[int, int]:
    """Map a Gidney complex-plane qubit coordinate to a tqec integer grid point.

    Args:
        q: a qubit coordinate in Gidney's convention (data qubits at integer
            points, ancillas at half-integer points).

    Returns:
        the ``(x, y)`` tqec grid coordinate. Data qubits land on odd-odd points,
        ancillas on even-even points.

    """
    x = 2 * q.real + 1
    y = 2 * q.imag + 1
    return (int(round(x)), int(round(y)))


def _checkerboard_basis(m: complex) -> Basis:
    """Gidney's ``checkerboard_basis``: ``X`` where ``Re + Im`` is even, else ``Z``."""
    return Basis.X if int(m.real + m.imag) % 2 == 0 else Basis.Z


@dataclass(frozen=True)
class Stabilizer:
    """One stabilizer of a surface-code patch, in tqec integer coordinates.

    Attributes:
        ancilla: the ``(x, y)`` tqec grid coordinate of the syndrome qubit.
        basis: the Pauli basis (``X`` or ``Z``) this stabilizer measures.
        ordered_data: the up-to-four data-qubit ``(x, y)`` coordinates in the
            gate-schedule order (``None`` for a missing neighbour on a boundary
            stabilizer). The order follows Gidney's ``order_func`` so the
            two-qubit gate schedule reproduces the vendored construction.
        gidney_ancilla: the original Gidney complex coordinate of the ancilla,
            retained so callers can reproduce Gidney's position-arithmetic flow
            rules (which are stated in Gidney coordinates).
    """

    ancilla: tuple[int, int]
    basis: Basis
    ordered_data: tuple[tuple[int, int] | None, ...]
    gidney_ancilla: complex


@dataclass(frozen=True)
class PatchGeometry:
    """A surface-code patch in tqec integer coordinates.

    Attributes:
        distance: the code distance ``d``.
        data_qubits: every data-qubit ``(x, y)`` coordinate in the patch.
        stabilizers: every stabilizer of the patch.
    """

    distance: int
    data_qubits: frozenset[tuple[int, int]]
    stabilizers: tuple[Stabilizer, ...]


def _rectangular_patch(
    *,
    distance: int,
    top_basis: Basis,
    bot_basis: Basis,
    left_basis: Basis,
    right_basis: Basis,
    order_func,
) -> PatchGeometry:
    """Dependency-free port of Gidney's ``rectangular_surface_code_patch``.

    Reproduces the exact data-qubit / measure-qubit / tile construction of
    ``tests/_vendor/midout/circuits/steps/_patches.py`` (verified against it),
    then maps every coordinate into tqec's grid via :func:`gidney_to_tqec`.
    """
    width = height = distance
    possible_data = {complex(x, y) for x in range(width) for y in range(height)}

    def is_boundary(m: complex, b: Basis) -> bool:
        if top_basis == b and m.imag == -0.5:
            return True
        if left_basis == b and m.real == -0.5:
            return True
        if bot_basis == b and m.imag == height - 0.5:
            return True
        if right_basis == b and m.real == width - 0.5:
            return True
        return False

    possible_measure = {q + d for q in possible_data for d in _DIRS}
    measure_qubits = {
        m
        for m in possible_measure
        if sum((m + d) in possible_data for d in _DIRS) > 1
        if is_boundary(m, Basis.X) <= (_checkerboard_basis(m) == Basis.X)
        if is_boundary(m, Basis.Z) <= (_checkerboard_basis(m) == Basis.Z)
    }
    data_qubits = {
        q for q in possible_data if sum((q + d) in measure_qubits for d in _DIRS) > 1
    }

    stabilizers: list[Stabilizer] = []
    for m in sorted(measure_qubits, key=lambda q: (q.imag, q.real)):
        basis = _checkerboard_basis(m)
        ordered = tuple(
            gidney_to_tqec(m + d) if (d is not None and (m + d) in data_qubits) else None
            for d in order_func(m)
        )
        stabilizers.append(
            Stabilizer(
                ancilla=gidney_to_tqec(m),
                basis=basis,
                ordered_data=ordered,
                gidney_ancilla=m,
            )
        )
    return PatchGeometry(
        distance=distance,
        data_qubits=frozenset(gidney_to_tqec(q) for q in data_qubits),
        stabilizers=tuple(stabilizers),
    )


def _xtop_order(m: complex) -> list[complex]:
    order_s = [_UR, _UL, _DR, _DL]
    order_n = [_UR, _DR, _UL, _DL]
    return order_s if _checkerboard_basis(m) == Basis.X else order_n


def _ztop_order(m: complex) -> list[complex]:
    order_s = [_UR, _UL, _DR, _DL]
    order_n = [_UR, _DR, _UL, _DL]
    return order_s if _checkerboard_basis(m) == Basis.X else order_n


def xtop_qubit_patch(distance: int) -> PatchGeometry:
    """The ``xtop`` qubit patch used during memory rounds (N=X, E=Z, S=X, W=Z).

    Dependency-free port of ``make_xtop_qubit_patch`` in tqec coordinates.
    """
    return _rectangular_patch(
        distance=distance,
        top_basis=Basis.X,
        right_basis=Basis.Z,
        bot_basis=Basis.X,
        left_basis=Basis.Z,
        order_func=_xtop_order,
    )


def ztop_yboundary_patch(distance: int) -> PatchGeometry:
    """The degenerate ``ztop`` Y-boundary patch after the transition (N=Z, E=X, S=X, W=Z).

    Dependency-free port of ``make_ztop_yboundary_patch`` in tqec coordinates.
    """
    return _rectangular_patch(
        distance=distance,
        top_basis=Basis.Z,
        right_basis=Basis.X,
        bot_basis=Basis.X,
        left_basis=Basis.Z,
        order_func=_ztop_order,
    )
