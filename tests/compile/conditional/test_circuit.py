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
    block = IfBlock(condition_rec=-3, then_body=[inner_then], else_body=[inner_else])
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


def test_to_stim_circuit_strict_rejects_if_block() -> None:
    c = ConditionalCircuit()
    c.append_if(IfBlock(condition_rec=-1))
    with pytest.raises(ValueError, match="IfBlock is still present"):
        c.to_stim_circuit_strict()


def test_to_stim_circuit_strict_roundtrip_when_no_branches() -> None:
    c = ConditionalCircuit()
    c.append("H", [0])
    c.append("M", [0])
    sc = c.to_stim_circuit_strict()
    assert len(sc) == 2

