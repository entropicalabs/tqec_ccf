"""Tests that stim instruction tags survive tqec's circuit manipulation.

stim lets an instruction carry a tag, rendered as ``S[T] 0`` and preserved
losslessly through a text round trip. tqec used to rebuild instructions from
name, targets and args alone, dropping the tag, which matters because the
state-injection encoder marks a non-Clifford gate by tagging the Clifford
stand-in it compiles as: lose the tag and ``S[T]`` silently becomes an ordinary
``S``, which is a different state rather than an error.

These tests use hand-built circuits so they stay independent of the injection
feature.
"""

from __future__ import annotations

import pytest
import stim

from tqec.circuit.moment import Moment
from tqec.circuit.non_clifford import (
    render_with_non_clifford_gates,
    rewrite_non_clifford_tags,
)
from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.circuit.schedule.manipulation import (
    _instruction_signature,
    merge_instructions,
    merge_scheduled_circuits,
    remove_duplicate_instructions,
)
from tqec.compile.conditional.circuit import IfBlock, remap_entry_qubit_indices
from tqec.utils.exceptions import TQECWarning
from tqec.utils.noise_model import NoiseModel

_TAG = "T"


def _tagged(name: str, targets: list[int], tag: str = _TAG) -> stim.CircuitInstruction:
    return stim.CircuitInstruction(name, targets, tag=tag)


def _tags(circuit: stim.Circuit) -> dict[str, str]:
    """Map each instruction's rendered form to its tag."""
    return {str(instruction): instruction.tag for instruction in circuit.flattened()}


def _scheduled(*instructions: stim.CircuitInstruction) -> ScheduledCircuit:
    circuit = stim.Circuit()
    for index, instruction in enumerate(instructions):
        if index:
            circuit.append("TICK")
        circuit.append(instruction)
    return ScheduledCircuit.from_circuit(circuit)


def test_stim_supports_tags_at_all() -> None:
    # Pins the assumption the rest of this file rests on.
    circuit = stim.Circuit()
    circuit.append(_tagged("S", [0]))
    assert str(circuit).strip() == "S[T] 0"
    assert stim.Circuit(str(circuit)) == circuit
    assert circuit[0].tag == _TAG


def test_moment_preserves_a_tag_when_remapping_qubits() -> None:
    circuit = stim.Circuit()
    circuit.append(_tagged("S", [0]))
    circuit.append("H", [1])
    remapped = Moment(circuit, _avoid_checks=True).with_mapped_qubit_indices({0: 5, 1: 6})
    assert _tags(remapped.circuit) == {"S[T] 5": _TAG, "H 6": ""}


def test_moment_preserves_a_tag_when_filtering_qubits() -> None:
    circuit = stim.Circuit()
    circuit.append(_tagged("S", [0]))
    circuit.append("H", [1])
    filtered = Moment(circuit, _avoid_checks=True).filter_by_qubits([0])
    assert _tags(filtered.circuit) == {"S[T] 0": _TAG}


def test_scheduled_circuit_preserves_a_tag_when_remapping_qubits() -> None:
    scheduled = _scheduled(_tagged("S", [0]), stim.CircuitInstruction("M", [0]))
    remapped = scheduled.map_qubit_indices({0: 7})
    assert _tags(remapped.get_circuit(include_qubit_coords=False))["S[T] 7"] == _TAG


def test_merge_instructions_does_not_fuse_differently_tagged_instructions() -> None:
    # Regression: this used to return a single untagged `S 0 1`, silently
    # turning a stand-in gate into the gate it stands in for.
    merged = merge_instructions([_tagged("S", [0]), stim.CircuitInstruction("S", [1])])
    assert len(merged) == 2
    assert {str(instruction): instruction.tag for instruction in merged} == {
        "S[T] 0": _TAG,
        "S 1": "",
    }


def test_merge_instructions_still_fuses_identically_tagged_instructions() -> None:
    merged = merge_instructions([_tagged("S", [0]), _tagged("S", [1])])
    assert len(merged) == 1
    assert str(merged[0]) == "S[T] 0 1"


def test_remove_duplicate_instructions_keeps_tags_apart() -> None:
    # Two instructions that differ only by tag are not duplicates. Keeping them
    # apart leaves two gates on one qubit, which is not a valid moment -- and the
    # warning saying so is the honest signal. Merging them silently, as this used
    # to, would have turned the stand-in gate into the gate it stands in for.
    with pytest.warns(TQECWarning, match="do not form a valid moment"):
        kept = remove_duplicate_instructions(
            [_tagged("S", [0]), stim.CircuitInstruction("S", [0])],
            mergeable_instruction_names=frozenset({"S"}),
        )
    assert sorted(str(instruction) for instruction in kept) == ["S 0", "S[T] 0"]


def test_instruction_signature_distinguishes_tags() -> None:
    assert _instruction_signature(_tagged("S", [0])) != _instruction_signature(
        stim.CircuitInstruction("S", [0])
    )


def test_merge_scheduled_circuits_preserves_a_tag() -> None:
    qubit_map = QubitMap({0: GridQubit(0, 0), 1: GridQubit(2, 2)})
    merged = merge_scheduled_circuits(
        [_scheduled(_tagged("S", [0])), _scheduled(stim.CircuitInstruction("H", [1]))],
        qubit_map,
    )
    assert _tags(merged.get_circuit(include_qubit_coords=False))["S[T] 0"] == _TAG


def test_noise_model_preserves_a_tag() -> None:
    circuit = stim.Circuit()
    circuit.append("RX", [0])
    circuit.append("TICK")
    circuit.append(_tagged("S", [0]))
    noisy = NoiseModel.uniform_depolarizing(1e-3).noisy_circuit(circuit)
    tags = _tags(noisy)
    assert tags["S[T] 0"] == _TAG
    # The gate is still noised; the tag does not make it immune.
    assert any(name.startswith("DEPOLARIZE1") for name in tags)


def test_noise_model_preserves_a_tag_in_an_exempt_moment() -> None:
    circuit = stim.Circuit()
    circuit.append(_tagged("S", [0]))
    noisy = NoiseModel.uniform_depolarizing(1e-3).noisy_circuit(
        circuit, noiseless_moments=frozenset({0})
    )
    assert _tags(noisy) == {"S[T] 0": _TAG}


def test_remap_entry_qubit_indices_preserves_a_tag() -> None:
    remapped = remap_entry_qubit_indices(_tagged("S", [0]), {0: 4})
    assert isinstance(remapped, stim.CircuitInstruction)
    assert str(remapped) == "S[T] 4"


def test_remap_entry_qubit_indices_preserves_a_tag_inside_an_if_block() -> None:
    block = IfBlock(condition_recs=[-1], then_body=[_tagged("S", [0])])
    remapped = remap_entry_qubit_indices(block, {0: 4})
    assert isinstance(remapped, IfBlock)
    assert str(remapped.then_body[0]) == "S[T] 4"


def test_render_with_non_clifford_gates_is_transparent_when_untagged() -> None:
    circuit = stim.Circuit("H 0\nTICK\nM 0")
    body = stim.Circuit("M 0\nTICK")
    circuit += body * 3
    assert render_with_non_clifford_gates(circuit) == str(circuit)


def test_render_with_non_clifford_gates_substitutes_the_real_gate() -> None:
    circuit = stim.Circuit()
    circuit.append("RX", [0])
    circuit.append("TICK")
    circuit.append(_tagged("S_DAG", [12], "T_DAG"))
    rendered = render_with_non_clifford_gates(circuit)
    assert "T_DAG 12" in rendered
    assert "S_DAG" not in rendered


def test_render_with_non_clifford_gates_leaves_an_unrelated_tag_alone() -> None:
    circuit = stim.Circuit()
    circuit.append(_tagged("H", [0], "something-else"))
    assert render_with_non_clifford_gates(circuit) == str(circuit)


def test_rewrite_non_clifford_tags_handles_indented_text() -> None:
    # `ConditionalCircuit.to_stim_text` indents an IF/ELSE body.
    text = "IF(rec[-1]) {\n  S[T] 4\n}\nSHIFT_COORDS(0, 0, 1)"
    assert rewrite_non_clifford_tags(text) == ("IF(rec[-1]) {\n  T 4\n}\nSHIFT_COORDS(0, 0, 1)")


def test_rewrite_non_clifford_tags_leaves_an_unrelated_tag_alone() -> None:
    assert rewrite_non_clifford_tags("H[whatever] 0") == "H[whatever] 0"
