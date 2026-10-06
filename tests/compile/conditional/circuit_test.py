"""Unit tests for ``ConditionalCircuit`` / ``IfBlock``."""

from __future__ import annotations

import pytest
import stim

from tqec.compile.conditional.circuit import (
    ConditionalCircuit,
    IfBlock,
)


def _circuit(text: str) -> stim.Circuit:
    return stim.Circuit(text.strip())


def test_to_stim_text_simple_passthrough() -> None:
    c = ConditionalCircuit()
    c.append("H", [0, 1])
    c.append("M", [0, 1])
    text = c.to_stim_text()
    assert "H 0 1" in text
    assert "M 0 1" in text


def test_to_stim_text_with_if_else() -> None:
    inner_then = stim.CircuitInstruction("M", [stim.GateTarget(7)])
    inner_else = stim.CircuitInstruction("MX", [stim.GateTarget(7)])
    block = IfBlock(condition_recs=[-3], then_body=[inner_then], else_body=[inner_else])
    c = ConditionalCircuit()
    c.append("CX", [1, 2])
    c.append_if(block)
    c.append("CX", [3, 4])
    text = c.to_stim_text()
    expected = [
        "CX 1 2",
        "IF(rec[-3]) {",
        "  M 7",
        "} ELSE {",
        "  MX 7",
        "}",
        "CX 3 4",
    ]
    assert text.splitlines() == expected


def test_to_stim_text_multi_rec_xor() -> None:
    inner = stim.CircuitInstruction("X", [stim.GateTarget(0)])
    block = IfBlock(condition_recs=[-5, -2, -1], then_body=[inner])
    c = ConditionalCircuit()
    c.append_if(block)
    text = c.to_stim_text()
    assert "IF(rec[-5]^rec[-2]^rec[-1]) {" in text


def test_ifblock_rejects_empty_condition_recs() -> None:
    with pytest.raises(ValueError, match="at least one condition rec"):
        IfBlock(condition_recs=[])


def test_to_stim_circuit_strict_rejects_if_block() -> None:
    c = ConditionalCircuit()
    c.append_if(IfBlock(condition_recs=[-1]))
    with pytest.raises(ValueError, match="IfBlock is still present"):
        c.to_stim_circuit_strict()


def test_to_stim_circuit_strict_roundtrip_when_no_branches() -> None:
    c = ConditionalCircuit()
    c.append("H", [0])
    c.append("M", [0])
    sc = c.to_stim_circuit_strict()
    assert len(sc) == 2


def test_absolute_condition_is_rendered_relative_to_each_block() -> None:
    """One absolute condition, shared by two IF blocks, reads the same measurement."""
    c = ConditionalCircuit()
    c.append("M", [0, 1])
    c.append_if(IfBlock(condition_recs=[1], then_body=[stim.CircuitInstruction("X", [2])]))
    c.append("M", [3, 4, 5])
    c.append_if(IfBlock(condition_recs=[1], then_body=[stim.CircuitInstruction("X", [2])]))
    ifs = [line for line in c.to_stim_text().splitlines() if line.startswith("IF")]
    assert ifs == ["IF(rec[-1]) {", "IF(rec[-4]) {"]


def test_absolute_condition_counts_measurements_inside_if_blocks() -> None:
    c = ConditionalCircuit()
    c.append("M", [0])
    c.append_if(
        IfBlock(
            condition_recs=[0],
            then_body=[stim.CircuitInstruction("M", [1])],
            else_body=[stim.CircuitInstruction("MX", [1])],
        )
    )
    c.append_if(IfBlock(condition_recs=[0], then_body=[stim.CircuitInstruction("X", [2])]))
    ifs = [line for line in c.to_stim_text().splitlines() if line.startswith("IF")]
    assert ifs == ["IF(rec[-1]) {", "IF(rec[-2]) {"]


def test_absolute_condition_on_a_future_measurement_is_rejected() -> None:
    c = ConditionalCircuit()
    c.append("M", [0])
    c.append_if(IfBlock(condition_recs=[1]))
    with pytest.raises(ValueError, match="only 1 measurements precede"):
        c.to_stim_text()


def test_ifblock_rejects_mixed_relative_and_absolute_condition() -> None:
    with pytest.raises(ValueError, match="mixes relative"):
        IfBlock(condition_recs=[-1, 3])


@pytest.mark.parametrize(
    "else_body",
    [
        [stim.CircuitInstruction("R", [0])],
        [stim.CircuitInstruction("M", [1])],
    ],
    ids=["different-count", "different-qubit"],
)
def test_branches_measuring_differently_are_rejected(
    else_body: list[stim.CircuitInstruction],
) -> None:
    c = ConditionalCircuit()
    c.append_if(
        IfBlock(
            condition_recs=[-1],
            then_body=[stim.CircuitInstruction("M", [0])],
            else_body=list(else_body),
        )
    )
    with pytest.raises(ValueError, match="do not measure the same qubits"):
        c.to_stim_text()


def _two_condition_circuit() -> ConditionalCircuit:
    """Two blocks on measurements {0, 1}, one in relative form, and one on {2}."""
    c = ConditionalCircuit()
    c.append("M", [0, 1])
    c.append_if(
        IfBlock(
            condition_recs=[0, 1],
            then_body=[stim.CircuitInstruction("M", [2])],
            else_body=[stim.CircuitInstruction("MX", [2])],
        )
    )
    c.append_if(
        IfBlock(
            condition_recs=[2],
            then_body=[stim.CircuitInstruction("X", [3])],
        )
    )
    c.append_if(
        IfBlock(
            condition_recs=[-3, -2],
            then_body=[stim.CircuitInstruction("Z", [4])],
            else_body=[stim.CircuitInstruction("Y", [4])],
        )
    )
    return c


def test_conditions_are_absolute_and_distinct_in_order_of_first_use() -> None:
    assert _two_condition_circuit().conditions == [(0, 1), (2,)]


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ({(0, 1): 1, (2,): 1}, "M 0 1 2\nX 3\nZ 4"),
        ({(0, 1): 0, (2,): 1}, "M 0 1\nMX 2\nX 3\nY 4"),
        ({(0, 1): 1, (2,): 0}, "M 0 1 2\nZ 4"),
    ],
)
def test_resolve_keeps_the_chosen_arm_of_every_block(
    outcomes: dict[tuple[int, ...], int], expected: str
) -> None:
    assert _two_condition_circuit().resolve(outcomes) == _circuit(expected)


def test_resolve_rejects_a_missing_or_invalid_outcome() -> None:
    with pytest.raises(ValueError, match=r"No outcome given .* \[2\]"):
        _two_condition_circuit().resolve({(0, 1): 1})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _two_condition_circuit().resolve({(0, 1): 2, (2,): 0})
