"""Tests for ``NoiseModel``'s per-moment noise exemption.

Kept out of ``noise_model_test.py``, which is recovered third-party code under a
different licence.
"""

from __future__ import annotations

import stim

from tqec.utils.noise_model import NoiseModel


def test_noiseless_moments_are_left_untouched() -> None:
    circuit = stim.Circuit("""
        R 0 1
        TICK
        CX 0 1
        TICK
        M 0 1
    """)
    model = NoiseModel.uniform_depolarizing(0.01)

    everywhere = model.noisy_circuit(circuit)
    # Moment 1 is the CX; exempting it drops its DEPOLARIZE2 and nothing else.
    exempt = model.noisy_circuit(circuit, noiseless_moments=frozenset({1}))

    def noise_names(c: stim.Circuit) -> list[str]:
        return [i.name for i in c.flattened() if i.name.startswith("DEPOLARIZE")]

    assert "DEPOLARIZE2" in noise_names(everywhere)
    assert "DEPOLARIZE2" not in noise_names(exempt)
    assert len(noise_names(exempt)) < len(noise_names(everywhere))
    # The exempted moment keeps its own operation.
    assert "CX" in [i.name for i in exempt.flattened()]


def test_exempting_every_moment_yields_the_original_circuit() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1")
    model = NoiseModel.uniform_depolarizing(0.01)
    exempt = model.noisy_circuit(circuit, noiseless_moments=frozenset(range(3)))
    assert [i.name for i in exempt.flattened()] == [i.name for i in circuit.flattened()]


def test_no_noiseless_moments_matches_the_default() -> None:
    circuit = stim.Circuit("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1")
    model = NoiseModel.uniform_depolarizing(0.01)
    assert model.noisy_circuit(circuit, noiseless_moments=frozenset()) == model.noisy_circuit(
        circuit
    )
