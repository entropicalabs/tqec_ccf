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
from tqec.utils.exceptions import TQECError
from tqec.utils.injection_state import INJECTION_STATES
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


@pytest.mark.parametrize("state", tuple(INJECTION_STATES))
def test_the_state_reaches_the_generator_through_the_compile(state: str) -> None:
    # Regression: the attribute used to be reset by the graph shift inside
    # ``compile_block_graph``, silently producing the default state's circuit.
    graph = BlockGraph(f"injecting {state}")
    graph.add_cube(_ORIGIN, "I", state=state)
    graph.add_cube(_ABOVE, "ZXX")
    graph.add_pipe(_ORIGIN, _ABOVE)
    graph.validate()
    # The unguarded builder, so this can inspect a non-Clifford state's tagged
    # stand-in -- which the public generate_stim_circuit rightly refuses to hand
    # out.
    circuit = compile_block_graph(graph, FIXED_BULK_CONVENTION)._build_stim_circuit(1)
    reset, gate, stands_in_for = INJECTION_STATES[state]
    names = [i.name for i in circuit.flattened() if i.name != "QUBIT_COORDS"]
    # The encoder opens by preparing the centre qubit alone, before any TICK.
    assert names[: names.index("TICK")] == [reset]
    # A non-Clifford state compiles as a tagged Clifford stand-in.
    tags = [i.tag for i in circuit.flattened() if i.tag]
    assert tags == ([stands_in_for] if stands_in_for else [])
    assert gate in names


@pytest.mark.parametrize("state", tuple(INJECTION_STATES))
def test_every_state_yields_the_same_detectors(state: str) -> None:
    # The encoder is an isometry onto the codespace whatever it injects, so the
    # patch's stabilizers -- and every detector downstream -- are unchanged. This
    # is what licenses computing detectors from a Clifford stand-in and shipping
    # them with the real gate substituted back in.
    graph = BlockGraph(f"injecting {state}")
    graph.add_cube(_ORIGIN, "I", state=state)
    graph.add_cube(_ABOVE, "ZXX")
    graph.add_pipe(_ORIGIN, _ABOVE)
    compiled = compile_block_graph(graph, FIXED_BULK_CONVENTION)
    # Counted from the text, since a non-Clifford state has no stim.Circuit.
    text = compiled.generate_stim_text(1)
    detectors = sum(1 for line in text.splitlines() if line.startswith("DETECTOR"))
    assert detectors == 3**2 - 1 + 3 * (3**2 - 1) + 4
    if INJECTION_STATES[state][2]:
        return
    # Determinism can only be checked where stim can hold the circuit.
    _without_observables(compiled.generate_stim_circuit(1)).detector_error_model()


def _without_injected_gate(circuit: stim.Circuit, state: str = "i") -> stim.Circuit:
    """Drop the injected gate, leaving a logical ``|+>`` memory.

    The distance of an injection column can only be measured this way: stim needs
    a deterministic observable, and the injected state's readout is a coin flip.
    What is left is the encoder acting as an ordinary Clifford preparation, which
    is exactly what the oracle's ``clifft_sim`` measures.

    Matched on the exact first token rather than a prefix --- the default state
    emits ``S``, and ``startswith("S")`` would also swallow ``SHIFT_COORDS``.
    """
    gate = INJECTION_STATES[state][1]
    return stim.Circuit(
        "\n".join(line for line in str(circuit).splitlines() if line.split()[:1] != [gate])
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
    circuit = _without_injected_gate(
        compiled.generate_stim_circuit(k, noise_model=NoiseModel.uniform_depolarizing(1e-3))
    )
    assert len(circuit.shortest_graphlike_error(ignore_ungraphlike_errors=False)) == 1


@pytest.mark.parametrize("k", _KS)
def test_a_noiseless_encoder_restores_the_code_distance(k: int) -> None:
    # With the encoder idealised, everything downstream is a plain memory and
    # protects the logical qubit to the full code distance. This reproduces the
    # oracle's own assertion, which noises the circuit with `skip_idx` set.
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    circuit = _without_injected_gate(
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


def _t_column() -> BlockGraph:
    graph = BlockGraph("injecting T")
    graph.add_cube(_ORIGIN, "I", state="T")
    graph.add_cube(_ABOVE, "ZXX")
    graph.add_pipe(_ORIGIN, _ABOVE)
    graph.validate()
    return graph


@pytest.mark.parametrize("k", _KS)
def test_stim_text_matches_the_circuit_for_a_clifford_graph(k: int) -> None:
    # Keeps the two entry points from drifting: with nothing stim cannot hold,
    # the text is exactly the circuit's own rendering.
    compiled = compile_block_graph(_injection_column(), FIXED_BULK_CONVENTION)
    assert compiled.generate_stim_text(k) == str(compiled.generate_stim_circuit(k))


def test_a_non_clifford_state_is_refused_by_the_circuit_entry_points() -> None:
    compiled = compile_block_graph(_t_column(), FIXED_BULK_CONVENTION)
    for call in (
        lambda: compiled.generate_stim_circuit(1),
        lambda: compiled.generate_crumble_url(k=1),
    ):
        with pytest.raises(TQECError, match="stim has no T gate"):
            call()


def test_stim_text_emits_a_real_non_clifford_gate() -> None:
    text = compile_block_graph(_t_column(), FIXED_BULK_CONVENTION).generate_stim_text(1)
    # One real T, on the centre data qubit of the patch.
    assert [line.split() for line in text.splitlines() if line.split()[:1] == ["T"]] == [
        ["T", "12"]
    ]
    # And the text is deliberately not a stim circuit -- that is the whole point.
    with pytest.raises(ValueError, match="Gate not found: 'T'"):
        stim.Circuit(text)


def test_stim_text_keeps_the_non_clifford_gate_under_noise() -> None:
    # The noisy path is the actual use case: the tag has to survive the noise
    # model as well as the compile.
    text = compile_block_graph(_t_column(), FIXED_BULK_CONVENTION).generate_stim_text(
        1, noise_model=NoiseModel.uniform_depolarizing(1e-3)
    )
    assert any(line.split()[:1] == ["T"] for line in text.splitlines())
    assert any(line.startswith("DEPOLARIZE") for line in text.splitlines())


def test_noiseless_injection_works_for_a_non_clifford_state() -> None:
    compiled = compile_block_graph(_t_column(), FIXED_BULK_CONVENTION)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    noisy = compiled.generate_stim_text(1, noise_model=noise)
    exempt = compiled.generate_stim_text(1, noise_model=noise, noiseless_injection=True)
    assert sum(1 for line in exempt.splitlines() if line.startswith("DEPOLARIZE")) < sum(
        1 for line in noisy.splitlines() if line.startswith("DEPOLARIZE")
    )
    assert any(line.split()[:1] == ["T"] for line in exempt.splitlines())


_NEIGHBOUR = Position3D(1, 0, 0)
_NEIGHBOUR_ABOVE = Position3D(1, 0, 1)


def _injection_beside_a_column() -> BlockGraph:
    """Join an injection column at ``z = 1`` to an ordinary three-cube column.

    This is the smallest graph in which an injection cube shares a z-slice with a
    block of a different temporal height: at ``z = 0`` the injection block has
    two rounds while the memory cube has ``2k+1``.
    """
    graph = BlockGraph("injection beside a column")
    graph.add_cube(Position3D(0, 0, 0), "XZX")
    graph.add_cube(Position3D(0, 0, 1), "XZX")
    graph.add_cube(Position3D(0, 0, 2), "XZX")
    graph.add_cube(_NEIGHBOUR, "I")
    graph.add_cube(_NEIGHBOUR_ABOVE, "XZX")
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    graph.add_pipe(Position3D(0, 0, 1), Position3D(0, 0, 2))
    graph.add_pipe(Position3D(0, 0, 1), _NEIGHBOUR_ABOVE)
    graph.add_pipe(_NEIGHBOUR, _NEIGHBOUR_ABOVE)
    graph.validate()
    return graph


def test_the_block_acquires_its_qubits() -> None:
    # The encoder resets every data qubit of the patch, so nothing of the block
    # exists before its own first round -- which is what licenses end-aligning it
    # in a merged slice rather than padding it with memory rounds.
    block = InjectionCubeBlock([InjectionRawLayer()])
    assert block.acquires_its_qubits
    assert not block.releases_its_qubits


@pytest.mark.parametrize("k", _KS)
def test_injection_compiles_beside_a_taller_column(k: int) -> None:
    # Regression: raised ``NotImplementedError`` out of ``_block_pad_body``,
    # which required a ``RepeatedLayer`` to pad a short block with. An injection
    # block has none, being deliberately non-scalable in time.
    compiled = compile_block_graph(_injection_beside_a_column(), FIXED_BULK_CONVENTION)
    circuit = compiled.generate_stim_circuit(k)
    # Raises if any detector is non-deterministic in the noiseless circuit.
    _without_observables(circuit).detector_error_model()


@pytest.mark.parametrize("k", _KS)
def test_the_encoder_runs_at_the_end_of_the_slice(k: int) -> None:
    # End-aligned, so the encoder is the second-to-last round of the z-slice
    # rather than its first: the injected state is not fault-tolerantly encoded,
    # and every round it is held for is exposure.
    compiled = compile_block_graph(_injection_beside_a_column(), FIXED_BULK_CONVENTION)
    circuit = compiled.generate_stim_circuit(k)
    moments = str(circuit).split("TICK")
    injected = [
        index
        for index, moment in enumerate(moments)
        if any(line.strip().split()[:1] == ["S"] for line in moment.splitlines())
    ]
    assert len(injected) == 1
    # The neighbour's rounds of the same slice come first, so the encoder cannot
    # be at the very beginning.
    assert injected[0] > 0


@pytest.mark.parametrize("k", _KS)
def test_the_injection_patch_is_absent_from_the_leading_rounds(k: int) -> None:
    # The measure of end-alignment: the injection patch is measured in exactly one
    # round of its own z-slice, not in all ``2k+1`` of them. Compare against the
    # same graph with the injection cube replaced by an ordinary memory cube,
    # which does occupy every round.
    beside = compile_block_graph(_injection_beside_a_column(), FIXED_BULK_CONVENTION)
    ordinary = _injection_beside_a_column().to_dict()
    for cube in ordinary["cubes"]:
        if cube["kind"] == "I":
            cube["kind"] = "XZX"
            cube.pop("state", None)
    full_height = compile_block_graph(BlockGraph.from_dict(ordinary), FIXED_BULK_CONVENTION)
    saved = (
        full_height.generate_stim_circuit(k).num_measurements
        - beside.generate_stim_circuit(k).num_measurements
    )
    distance = 2 * k + 1
    # Exactly ``2k`` rounds skipped, each of which would have measured every
    # ancilla of the patch. Nothing else moves: the ordinary cube's own final
    # round is absorbed by the temporal pipe above it, just as the injection
    # block's is.
    assert saved == 2 * k * (distance**2 - 1)


@pytest.mark.parametrize("k", _KS)
def test_noiseless_injection_still_finds_a_late_encoder(k: int) -> None:
    # The encoder is no longer the first round, which used to raise: the moment
    # indices are computed by walking the tree, so preceding rounds are fine as
    # long as none of them is a ``REPEAT`` block.
    compiled = compile_block_graph(_injection_beside_a_column(), FIXED_BULK_CONVENTION)
    noise = NoiseModel.uniform_depolarizing(1e-3)
    noisy = compiled.generate_stim_circuit(k, noise_model=noise)
    exempt = compiled.generate_stim_circuit(k, noise_model=noise, noiseless_injection=True)
    assert _noise_instruction_count(exempt) < _noise_instruction_count(noisy)
    # And the exempted moments really are the encoder's: the moment holding the
    # injected gate carries no noise at all.
    for moment in str(exempt).split("TICK"):
        lines = moment.splitlines()
        if any(line.strip().split()[:1] == ["S"] for line in lines):
            assert not any(
                line.strip().startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR")) for line in lines
            )
            break
    else:  # pragma: no cover
        pytest.fail("no moment holding the injected gate")


@pytest.mark.parametrize("add_polygons", [False, True])
def test_crumble_url_beside_a_taller_column(add_polygons: bool) -> None:
    # Two regressions at once: ``generate_crumble_url`` used to call
    # ``to_layer_tree()`` without ``k``, which a mismatched-schedule slice cannot
    # be merged without; and the polygon annotator used to refuse a slice mixing
    # the encoder's raw round with a neighbour's plaquette round.
    compiled = compile_block_graph(_injection_beside_a_column(), FIXED_BULK_CONVENTION)
    url = compiled.generate_crumble_url(k=1, add_polygons=add_polygons)
    assert url.startswith("https://algassert.com/crumble#circuit=")
