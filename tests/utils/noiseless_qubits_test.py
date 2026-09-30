"""Tests for ``NoiseModel``'s per-moment, per-qubit noise exemption.

Kept out of ``noise_model_test.py``, which is recovered third-party code under a
different licence.
"""

from __future__ import annotations

import pytest
import stim

from tqec.utils.noise_model import NoiseModel


def _noise(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    return [
        i
        for i in circuit.flattened()
        if stim.gate_data(i.name).is_noisy_gate and not stim.gate_data(i.name).produces_measurements
    ]


def _noisy_qubits(circuit: stim.Circuit) -> set[int]:
    return {t.value for i in _noise(circuit) for t in i.targets_copy()}


def test_exempt_qubits_get_no_noise_in_their_moment() -> None:
    circuit = stim.Circuit("R 0 1 2 3\nTICK\nCX 0 1 2 3\nTICK\nM 0 1 2 3")
    model = NoiseModel.uniform_depolarizing(0.01)
    # Exempt qubits 0 and 1 in moment 1, the CX: the CX on 2 3 keeps its noise.
    exempt = model.noisy_circuit(circuit, noiseless_qubits={1: {0, 1}})
    moment_1 = str(exempt).split("TICK")[1]
    assert "DEPOLARIZE2(0.01) 2 3" in moment_1
    assert "0 1" not in moment_1.split("DEPOLARIZE2")[1]
    # Other moments are untouched by the exemption.
    assert _noisy_qubits(exempt) == {0, 1, 2, 3}
    assert "CX 0 1" in str(exempt)


def test_a_neighbour_sharing_the_moment_keeps_its_idling_noise() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nH 0\nTICK\nM 0 1")
    model = NoiseModel.uniform_depolarizing(0.01)
    exempt = model.noisy_circuit(circuit, noiseless_qubits={1: {0}})
    moment_1 = str(exempt).split("TICK")[1]
    assert "H 0" in moment_1
    assert "DEPOLARIZE1(0.01) 1" in moment_1  # qubit 1 idles, and idling is noisy
    assert "DEPOLARIZE1(0.01) 0" not in moment_1


def test_exempting_every_qubit_of_every_moment_yields_the_original_circuit() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1")
    model = NoiseModel.uniform_depolarizing(0.01)
    exempt = model.noisy_circuit(circuit, noiseless_qubits={m: {0, 1} for m in range(3)})
    assert not _noise(exempt)
    assert [i.name for i in exempt.flattened()] == [i.name for i in circuit.flattened()]


def test_no_noiseless_qubits_matches_the_default() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1")
    model = NoiseModel.uniform_depolarizing(0.01)
    assert model.noisy_circuit(circuit, noiseless_qubits={}) == model.noisy_circuit(circuit)


def test_a_repeat_block_cannot_be_exempted() -> None:
    circuit = stim.Circuit("R 0\nTICK\nREPEAT 2 {\n    H 0\n    TICK\n}\nM 0")
    model = NoiseModel.uniform_depolarizing(0.01)
    with pytest.raises(ValueError, match="REPEAT block"):
        model.noisy_circuit(circuit, noiseless_qubits={1: {0}})
