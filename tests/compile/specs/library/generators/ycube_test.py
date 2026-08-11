"""Oracle-parity tests for the native Y-cube patch geometry.

Verifies that the dependency-free patch reconstruction in
:mod:`tqec.compile.specs.library.generators.ycube` exactly reproduces Craig
Gidney's vendored ``gen`` patches (``tests/_vendor/midout``), which are used
here purely as a ground-truth oracle.
"""

from __future__ import annotations

import pytest

from tests._vendor.midout.circuits.steps._patches import (
    make_xtop_qubit_patch,
    make_ztop_yboundary_patch,
)
from tqec.compile.specs.library.generators.ycube import (
    gidney_to_tqec,
    xtop_qubit_patch,
    ztop_yboundary_patch,
)
from tqec.utils.enums import Basis


def _oracle_summary(patch):
    """Return ``(data set, {ancilla_tqec: (basis, ordered_data_tqec)})`` from a patch."""
    data = {gidney_to_tqec(q) for q in patch.data_set}
    stabs = {}
    for tile in patch.tiles:
        m = tile.measurement_qubit
        ordered = tuple(
            gidney_to_tqec(d) if d is not None else None for d in tile.ordered_data_qubits
        )
        stabs[gidney_to_tqec(m)] = (tile.basis, ordered)
    return data, stabs


def _native_summary(geom):
    data = set(geom.data_qubits)
    stabs = {s.ancilla: (s.basis.value, s.ordered_data) for s in geom.stabilizers}
    return data, stabs


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_xtop_patch_matches_oracle(distance: int) -> None:
    o_data, o_stabs = _oracle_summary(make_xtop_qubit_patch(distance=distance))
    n_data, n_stabs = _native_summary(xtop_qubit_patch(distance))
    assert n_data == o_data
    assert n_stabs == o_stabs


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_ztop_yboundary_patch_matches_oracle(distance: int) -> None:
    o_data, o_stabs = _oracle_summary(make_ztop_yboundary_patch(distance=distance))
    n_data, n_stabs = _native_summary(ztop_yboundary_patch(distance))
    assert n_data == o_data
    assert n_stabs == o_stabs


def test_transform_lands_data_on_odd_and_ancilla_on_even() -> None:
    geom = xtop_qubit_patch(3)
    for x, y in geom.data_qubits:
        assert x % 2 == 1 and y % 2 == 1
    for s in geom.stabilizers:
        assert s.ancilla[0] % 2 == 0 and s.ancilla[1] % 2 == 0
        assert s.basis in (Basis.X, Basis.Z)
