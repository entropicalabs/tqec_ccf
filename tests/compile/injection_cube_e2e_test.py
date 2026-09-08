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
