"""End-to-end compilation of an injection cube capping a memory cube.

The graph is the smallest one an injection cube can appear in: the cube at the
origin, a temporal pipe, and a memory cube above it that reads the patch out
transversally.

What these tests pin down is the detector structure, because that is where the
injection cube differs from every other way of starting a column. The encoder
leaves *all* ``d**2 - 1`` stabilizers in their ``+1`` eigenstate, whereas a
transversal reset only fixes those of the reset basis, so the first syndrome
round carries a single-record detector for every stabilizer rather than for half
of them.
"""

from __future__ import annotations

import pytest
import stim

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION
from tqec.compile.specs.library.generators._injection_layer import (
    InjectionCubeBlock,
    InjectionRawLayer,
)
from tqec.computation.block_graph import BlockGraph
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D

_ORIGIN = Position3D(0, 0, 0)
_ABOVE = Position3D(0, 0, 1)
_KS = (1, 2)
_SLOW_K = 3


def _injection_column(above: str = "ZXX") -> BlockGraph:
    graph = BlockGraph(f"injection into {above}")
    graph.add_cube(_ORIGIN, "I")
    graph.add_cube(_ABOVE, above)
    graph.add_pipe(_ORIGIN, _ABOVE)
    graph.validate()
    return graph


def _without_observables(circuit: stim.Circuit) -> stim.Circuit:
    """Drop the observable, which the injected state makes non-deterministic.

    The injected state sits on the XY equator, so reading the logical operator
    out is a coin flip -- that randomness is the point of the experiment. stim
    refuses to build a detector error model for a circuit whose observable is
    non-deterministic, so the detectors are checked on their own.
    """
    return stim.Circuit(
        "\n".join(
            line
            for line in str(circuit).splitlines()
            if not line.strip().startswith("OBSERVABLE_INCLUDE")
        )
    )


def _detectors_by_round(circuit: stim.Circuit) -> dict[float, list[int]]:
    """Map each round's detector coordinate to the record counts of its detectors."""
    rounds: dict[float, list[int]] = {}
    for instruction in circuit.flattened():
        if instruction.name != "DETECTOR":
            continue
        coordinates = instruction.gate_args_copy()
        rounds.setdefault(coordinates[2], []).append(len(instruction.targets_copy()))
    return rounds


@pytest.mark.parametrize("k", _KS)
@pytest.mark.parametrize("above", ["ZXX", "ZXZ", "XZX", "XZZ"])
def test_every_detector_is_deterministic(k: int, above: str) -> None:
    circuit = compile_block_graph(_injection_column(above), FIXED_BULK_CONVENTION)
    # Raises if any detector is non-deterministic in the noiseless circuit.
    _without_observables(circuit.generate_stim_circuit(k)).detector_error_model()


@pytest.mark.parametrize("k", _KS)
@pytest.mark.parametrize("above", ["ZXX", "XZX"])
def test_first_round_detects_every_stabilizer(k: int, above: str) -> None:
    distance = 2 * k + 1
    circuit = compile_block_graph(_injection_column(above), FIXED_BULK_CONVENTION)
    rounds = _detectors_by_round(circuit.generate_stim_circuit(k))
    first = rounds[min(rounds)]
    # One single-record detector per stabilizer -- twice what a transversal
    # reset would give, and the reason the annotator needs the encoder's
    # preparation spec.
    assert len(first) == distance**2 - 1
    assert set(first) == {1}


@pytest.mark.parametrize("k", _KS)
def test_bulk_rounds_are_ordinary(k: int) -> None:
    distance = 2 * k + 1
    circuit = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    rounds = _detectors_by_round(circuit.generate_stim_circuit(k))
    ordered = [rounds[z] for z in sorted(rounds)]
    # Every round between the first and the readout is a plain consecutive-round
    # comparison: one two-record detector per stabilizer.
    for counts in ordered[1:-1]:
        assert len(counts) == distance**2 - 1
        assert set(counts) == {2}


@pytest.mark.parametrize("k", _KS)
def test_encoder_contributes_no_measurement(k: int) -> None:
    distance = 2 * k + 1
    circuit = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    stim_circuit = circuit.generate_stim_circuit(k)
    rounds = _detectors_by_round(stim_circuit)
    # Syndrome rounds measure every ancilla; the readout measures every data
    # qubit. The encoder adds nothing, so this accounts for every measurement.
    expected = len(rounds) * (distance**2 - 1) + distance**2
    assert stim_circuit.num_measurements == expected


@pytest.mark.parametrize("k", _KS)
def test_encoder_occupies_a_single_round(k: int) -> None:
    circuit = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    rounds = _detectors_by_round(circuit.generate_stim_circuit(k))
    # The injection block holds the encoder plus the pipe round it gains, and
    # the memory cube above contributes 2k + 1 layers, of which the pipe took
    # the first. One of those layers is the encoder, which is not a syndrome
    # round -- hence 2k + 2 rounds rather than 2k + 3.
    assert len(rounds) == 2 * k + 2


@pytest.mark.parametrize("k", _KS)
def test_injected_state_is_read_out_at_random(k: int) -> None:
    circuit = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    stim_circuit = circuit.generate_stim_circuit(k)
    assert stim_circuit.num_observables == 1
    _, observables = stim_circuit.compile_detector_sampler().sample(2000, separate_observables=True)
    # S_DAG |+> is a Y eigenstate, so an X or Z readout of it is unbiased. A
    # deterministic outcome here would mean the encoder prepared a stabilizer
    # state instead.
    assert 0.4 < observables.mean() < 0.6


@pytest.mark.slow
@pytest.mark.parametrize("above", ["ZXX", "XZX"])
def test_detector_structure_holds_at_distance_seven(above: str) -> None:
    distance = 2 * _SLOW_K + 1
    circuit = compile_block_graph(_injection_column(above), FIXED_BULK_CONVENTION)
    stim_circuit = circuit.generate_stim_circuit(_SLOW_K)
    _without_observables(stim_circuit).detector_error_model()
    rounds = _detectors_by_round(stim_circuit)
    first = rounds[min(rounds)]
    assert len(first) == distance**2 - 1
    assert set(first) == {1}
    assert len(rounds) == 2 * _SLOW_K + 2


def test_block_is_a_cube_with_a_constant_temporal_extent() -> None:
    block = InjectionCubeBlock([InjectionRawLayer()])
    assert block.is_cube
    assert not block.is_pipe
    assert block.scalable_timesteps.is_constant()


def test_fixed_boundary_convention_is_not_supported() -> None:
    graph = _injection_column()
    with pytest.raises(NotImplementedError, match="only implemented for the fixed-bulk"):
        compile_block_graph(graph, FIXED_BOUNDARY_CONVENTION)


@pytest.mark.parametrize("k", _KS)
def test_crumble_url_with_polygons(k: int) -> None:
    # The encoder is a raw round with no plaquettes, so it contributes no
    # stabilizer outlines. Regression: this used to raise ``NotImplementedError``
    # out of ``LayoutLayer.to_template_and_plaquettes``.
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    url = compiled.generate_crumble_url(k=k, add_polygons=True)
    assert url.startswith("https://algassert.com/crumble#circuit=")
    # Every syndrome round is outlined; the encoder round is not.
    assert sum(1 for layer in url.split(";") if "POLYGON" in layer) > 0


@pytest.mark.parametrize("k", _KS)
def test_crumble_url_without_polygons(k: int) -> None:
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    assert compiled.generate_crumble_url(k=k, add_polygons=False).startswith(
        "https://algassert.com/crumble#circuit="
    )


def test_proxy_false_reaches_the_generator_through_the_compile() -> None:
    graph = BlockGraph("proxy false")
    graph.add_cube(_ORIGIN, "I", proxy=False)
    graph.add_cube(_ABOVE, "ZXX")
    graph.add_pipe(_ORIGIN, _ABOVE)
    graph.validate()
    # Regression: the flag used to be reset by the graph shift inside
    # ``compile_block_graph``, silently producing the proxy=True circuit.
    with pytest.raises(NotImplementedError, match="stim does not support"):
        compile_block_graph(graph, FIXED_BULK_CONVENTION).generate_stim_circuit(1)


def _proxy_stripped(circuit: stim.Circuit) -> stim.Circuit:
    """Drop the injected proxy gate, leaving a logical ``|+>`` memory.

    The distance of an injection column can only be measured this way: stim needs
    a deterministic observable, and the injected state's readout is a coin flip.
    What is left is the encoder acting as an ordinary Clifford preparation, which
    is exactly what the oracle's ``clifft_sim`` measures.
    """
    return stim.Circuit(
        "\n".join(
            line for line in str(circuit).splitlines() if not line.strip().startswith("S_DAG")
        )
    )


def _noise_instruction_count(circuit: stim.Circuit) -> int:
    return sum(
        1
        for instruction in circuit.flattened()
        if instruction.name.startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR"))
    )


@pytest.mark.parametrize("k", _KS)
def test_noiseless_injection_removes_noise(k: int) -> None:
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    noisy = compiled.generate_stim_circuit(k, noise_model=noise)
    exempt = compiled.generate_stim_circuit(k, noise_model=noise, noiseless_injection=True)
    assert _noise_instruction_count(exempt) < _noise_instruction_count(noisy)


@pytest.mark.parametrize("k", _KS)
def test_a_noisy_encoder_makes_the_column_distance_one(k: int) -> None:
    # State injection is not fault tolerant: one fault in the encoder corrupts
    # the state outright. That is the point of injection, not a defect.
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    circuit = _proxy_stripped(
        compiled.generate_stim_circuit(k, noise_model=NoiseModel.uniform_depolarizing(1e-3))
    )
    assert len(circuit.shortest_graphlike_error(ignore_ungraphlike_errors=False)) == 1


@pytest.mark.parametrize("k", _KS)
def test_a_noiseless_encoder_restores_the_code_distance(k: int) -> None:
    # With the encoder idealised, everything downstream is a plain memory and
    # protects the logical qubit to the full code distance. This reproduces the
    # oracle's own assertion, which noises the circuit with `skip_idx` set.
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    circuit = _proxy_stripped(
        compiled.generate_stim_circuit(
            k, noise_model=NoiseModel.uniform_depolarizing(1e-3), noiseless_injection=True
        )
    )
    assert len(circuit.shortest_graphlike_error(ignore_ungraphlike_errors=False)) == 2 * k + 1


def test_noiseless_injection_needs_a_noise_model_to_do_anything() -> None:
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    assert compiled.generate_stim_circuit(1, noiseless_injection=True) == (
        compiled.generate_stim_circuit(1)
    )


def test_noiseless_injection_is_harmless_without_an_injection_cube() -> None:
    graph = BlockGraph("plain memory")
    graph.add_cube(_ORIGIN, "ZXZ")
    graph.add_cube(_ABOVE, "ZXZ")
    graph.add_pipe(_ORIGIN, _ABOVE)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    assert compiled.generate_stim_circuit(
        1, noise_model=noise, noiseless_injection=True
    ) == compiled.generate_stim_circuit(1, noise_model=noise)
