"""Unit tests for ``ConditionalCircuit`` / ``IfBlock`` / ``branch_diff``."""

from __future__ import annotations

import pytest
import stim

from tqec.compile.conditional.circuit import (
    ConditionalCircuit,
    IfBlock,
    branch_diff,
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


def test_branch_diff_all_identical() -> None:
    z = _circuit("H 0\nM 0")
    o = _circuit("H 0\nM 0")
    entries = branch_diff(z, o, condition_rec=-1)
    assert len(entries) == 2
    assert all(isinstance(e, stim.CircuitInstruction) for e in entries)


def test_branch_diff_one_differing_instruction() -> None:
    z = _circuit("H 0\nMX 0\nCX 1 2")
    o = _circuit("H 0\nM 0\nCX 1 2")
    entries = branch_diff(z, o, condition_rec=-5)
    # H (shared), IfBlock (MX/M), CX (shared)
    assert len(entries) == 3
    assert isinstance(entries[0], stim.CircuitInstruction)
    assert isinstance(entries[1], IfBlock)
    assert isinstance(entries[2], stim.CircuitInstruction)
    assert entries[1].condition_rec == -5
    assert len(entries[1].then_body) == 1
    assert entries[1].then_body[0].name == "M"
    assert len(entries[1].else_body) == 1
    assert entries[1].else_body[0].name == "MX"


def test_branch_diff_collapses_consecutive_differences() -> None:
    # stim auto-merges adjacent same-name instructions when parsing text, so
    # build by append() to keep them separate at the instruction level.
    z = stim.Circuit()
    z.append("H", [0])
    z.append("MX", [0])
    z.append("MY", [1])
    z.append("CX", [1, 2])
    o = stim.Circuit()
    o.append("H", [0])
    o.append("M", [0])
    o.append("MR", [1])
    o.append("CX", [1, 2])
    entries = branch_diff(z, o, condition_rec=-1)
    # H (shared), IfBlock (two differing instrs), CX (shared)
    assert len(entries) == 3
    block = entries[1]
    assert isinstance(block, IfBlock)
    assert [i.name for i in block.then_body] == ["M", "MR"]
    assert [i.name for i in block.else_body] == ["MX", "MY"]


def test_branch_diff_length_mismatch_raises() -> None:
    z = _circuit("H 0\nM 0")
    o = _circuit("H 0\nM 0\nX 1")
    with pytest.raises(ValueError, match="instruction count mismatch"):
        branch_diff(z, o, condition_rec=-1)
