"""Unit tests for ``tools.measurements``."""

from __future__ import annotations

import pytest
import stim

from tools import (
    MeasurementInfo,
    build_measurement_map,
    locate_record,
    observable_records,
    resolve_if_else,
    resolve_if_else_by_order,
)


def test_build_map_tracks_coords_and_ticks() -> None:
    """Coords and ticks are tracked per measurement record."""
    circuit = stim.Circuit(
        "\n".join(
            [
                "QUBIT_COORDS(1, 2) 0",
                "QUBIT_COORDS(3, 4) 1",
                "MX 0",
                "TICK",
                "M 1 0",
            ]
        )
    )
    m = build_measurement_map(circuit)
    assert m == [
        MeasurementInfo((1.0, 2.0), 0),  # MX 0 before any TICK
        MeasurementInfo((3.0, 4.0), 1),  # M 1 after one TICK
        MeasurementInfo((1.0, 2.0), 1),  # M 0 after one TICK
    ]


def test_missing_coords_are_none() -> None:
    """Measurements without QUBIT_COORDS get a None coord."""
    m = build_measurement_map(stim.Circuit("M 7"))
    assert m == [MeasurementInfo(None, 0)]


def test_mpp_product_is_one_record_first_qubit_coord() -> None:
    """An MPP product is one record, attributed to its first qubit."""
    circuit = stim.Circuit(
        "\n".join(
            [
                "QUBIT_COORDS(0, 0) 0",
                "QUBIT_COORDS(1, 1) 1",
                "QUBIT_COORDS(2, 2) 2",
                "MPP X0*Z1 X2",
            ]
        )
    )
    m = build_measurement_map(circuit)
    # one record per product; first product attributed to qubit 0
    assert m == [MeasurementInfo((0.0, 0.0), 0), MeasurementInfo((2.0, 2.0), 0)]


def test_locate_record_resolves_relative_offset() -> None:
    """rec[-k] resolves against the measurement frame."""
    m = build_measurement_map(stim.Circuit("QUBIT_COORDS(5, 6) 0\nM 0 0 0"))
    assert locate_record(m, 3, -1) == m[2]
    assert locate_record(m, 3, -3) == m[0]


def test_observable_records_xor_cancels_repeats() -> None:
    """A record included twice cancels (XOR) from the observable."""
    circuit = stim.Circuit(
        "\n".join(
            [
                "QUBIT_COORDS(0, 0) 0",
                "QUBIT_COORDS(1, 1) 1",
                "M 0 1",
                "OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]",
                "OBSERVABLE_INCLUDE(0) rec[-2]",  # cancels the first rec[-2]
            ]
        )
    )
    assert observable_records(circuit, 0) == [MeasurementInfo((1.0, 1.0), 0)]


def test_observable_index_is_respected() -> None:
    """Only the requested observable index is extracted."""
    circuit = stim.Circuit(
        "\n".join(
            [
                "QUBIT_COORDS(0, 0) 0",
                "QUBIT_COORDS(1, 1) 1",
                "M 0 1",
                "OBSERVABLE_INCLUDE(0) rec[-2]",
                "OBSERVABLE_INCLUDE(1) rec[-1]",
            ]
        )
    )
    assert observable_records(circuit, 0) == [MeasurementInfo((0.0, 0.0), 0)]
    assert observable_records(circuit, 1) == [MeasurementInfo((1.0, 1.0), 0)]


def test_composes_with_multirec_resolve_if_else() -> None:
    """Multi-rec IF resolves then feeds observable_records."""
    text = "\n".join(
        [
            "QUBIT_COORDS(0, 0) 0",
            "QUBIT_COORDS(1, 0) 1",
            "M 0 1",
            "IF(rec[-2]^rec[-1]) {",  # multi-rec XOR header (keyed by first rec -2)
            "  OBSERVABLE_INCLUDE(0) rec[-1]",
            "}",
        ]
    )
    fired = resolve_if_else(text, {-2: 1})
    assert observable_records(fired, 0) == [MeasurementInfo((1.0, 0.0), 0)]
    dropped = resolve_if_else(text, {-2: 0})
    assert observable_records(dropped, 0) == []


def test_resolve_by_order_drives_same_rec_blocks_independently() -> None:
    """resolve_if_else_by_order selects per position, even with equal rec offsets."""
    text = "\n".join(
        [
            "M 0",
            "IF(rec[-1]) {",
            "  X 0",
            "} ELSE {",
            "  Y 0",
            "}",
            "IF(rec[-1]) {",  # same rec offset as the first block
            "  Z 0",
            "}",
        ]
    )
    c = resolve_if_else_by_order(text, [1, 0])  # first -> IF(X), second -> drop
    assert [inst.name for inst in c] == ["M", "X"]
    c2 = resolve_if_else_by_order(text, [0, 1])  # first -> ELSE(Y), second -> Z
    assert [inst.name for inst in c2] == ["M", "Y", "Z"]


def test_resolve_by_order_too_few_outcomes_raises() -> None:
    """A short outcomes list raises rather than silently dropping blocks."""
    text = "M 0\nIF(rec[-1]) {\n  X 0\n}"
    with pytest.raises(ValueError, match="outcomes has"):
        resolve_if_else_by_order(text, [])
