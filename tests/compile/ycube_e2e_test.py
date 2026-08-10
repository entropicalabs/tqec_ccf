"""End-to-end compilation tests for the native Y-half cube (S-gate cap).

A single Y-capped memory column must compile all the way through
``compile_block_graph`` to a valid ``stim.Circuit`` whose detectors are all
deterministic, with no fictitious ``MPP`` elements (the physical-seam
property).
"""

from __future__ import annotations

import pytest

from tqec import BlockGraph, compile_block_graph
from tqec.computation.cube import LeafCubeKind, ZXCube
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D


def _y_capped_column() -> BlockGraph:
    g = BlockGraph("y_capped_column")
    g.add_cube(Position3D(0, 0, 0), ZXCube.from_str("ZXZ"))
    g.add_cube(Position3D(0, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    return g


@pytest.mark.parametrize("k", [1, 2])
def test_y_capped_column_compiles(k: int) -> None:
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=k)
    assert circuit.num_detectors > 0
    assert circuit.num_qubits > 0
    # A lone Y column has no closed correlation surface, so no observable.
    assert circuit.num_observables == 0


@pytest.mark.parametrize("k", [1, 2])
def test_y_capped_column_detectors_are_deterministic(k: int) -> None:
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=k)
    # Raises if any detector is non-deterministic -- exercises the seam detectors
    # the annotator emits across the transition round.
    circuit.detector_error_model(decompose_errors=False)


def test_y_capped_column_has_no_mpp() -> None:
    """The native Y cap uses no fictitious multi-qubit Pauli measurements."""
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=1)
    assert not any(inst.name == "MPP" for inst in circuit.flattened())


@pytest.mark.parametrize("k", [1, 2])
def test_temporal_pipe_does_not_overwrite_the_transition_round(k: int) -> None:
    """The temporal pipe below a Y cap must *prepend* its junction round, not
    replace the cap's first layer.

    For an ordinary cube the pipe substitutes the ``Z_NEGATIVE`` border, i.e.
    ``layer_sequence[0]``. A Y cap's first layer is the transition round, and
    since ``RawCircuitLayer`` is a ``BaseLayer`` the substitution machinery
    accepts it without complaint, dropping the Y measurement. The transition
    round is the only source of ``MY`` in the circuit, so counting them pins the
    behaviour: exactly one per Y cube, at every ``k``.

    (Substituting rather than prepending currently also trips the seam-detector
    check before this assertion is reached. That check is incidental --- it
    depends on the junction round's plaquettes disagreeing with the transition's
    expected ancillas --- so the ``MY`` count is asserted as the direct
    invariant.)
    """
    for graph, expected_y_cubes in (
        (_y_capped_column(), 1),
        (_two_y_caps_with_main_column(), 2),
    ):
        circuit = compile_block_graph(graph, observables="auto").generate_stim_circuit(k=k)
        my_targets = sum(
            len(inst.targets_copy()) for inst in circuit.flattened() if inst.name == "MY"
        )
        assert my_targets == expected_y_cubes


def _two_y_caps_with_main_column() -> BlockGraph:
    """The notebook cells 7-8 graph: a 5-cube main column (x=0) with two Y caps
    on side branches (x=1) at z=2 and z=4, each coexisting in a z-slice with a
    continuing main-column memory cube (mismatched temporal schedules)."""
    g = BlockGraph("two_y_caps")
    b = [Position3D(0, 0, i) for i in range(5)]
    c1, c3 = Position3D(1, 0, 1), Position3D(1, 0, 3)
    y2, y4 = Position3D(1, 0, 2), Position3D(1, 0, 4)
    for p in b:
        g.add_cube(p, ZXCube.from_str("ZXZ"))
    g.add_cube(c1, ZXCube.from_str("ZXZ"))
    g.add_cube(c3, ZXCube.from_str("ZXZ"))
    g.add_cube(y2, LeafCubeKind.Y_HALF_CUBE)
    g.add_cube(y4, LeafCubeKind.Y_HALF_CUBE)
    for i in range(4):
        g.add_pipe(b[i], b[i + 1])
    g.add_pipe(b[1], c1)
    g.add_pipe(b[3], c3)
    g.add_pipe(c1, y2)
    g.add_pipe(c3, y4)
    return g


@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_coexistence_compiles_dem_clean(k: int) -> None:
    """Two Y caps coexisting with a continuing memory column (mismatched
    temporal schedules) compile to a circuit whose detectors are all
    deterministic, with no MPP."""
    circuit = compile_block_graph(
        _two_y_caps_with_main_column(), observables=[]
    ).generate_stim_circuit(k=k)
    assert circuit.num_detectors > 0
    circuit.detector_error_model(decompose_errors=False)  # raises if non-deterministic
    assert not any(inst.name == "MPP" for inst in circuit.flattened())


def _zzyy_surface() -> tuple[BlockGraph, list]:
    graph = _two_y_caps_with_main_column()
    surfaces = graph.find_correlation_surfaces()
    assert [cs.external_stabilizer_on_graph(graph) for cs in surfaces] == ["ZZYY"]
    return graph, surfaces


@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_observable_is_deterministic(k: int) -> None:
    """The closed ``Z_in . Z_out . Y_cap1 . Y_cap2`` surface (the S^2 = Z algebra)
    lowers to a single observable that takes the same value on every shot."""
    graph, surfaces = _zzyy_surface()
    circuit = compile_block_graph(graph, observables=surfaces).generate_stim_circuit(k=k)
    assert circuit.num_observables == 1
    circuit.detector_error_model(decompose_errors=False)
    _, observables = circuit.compile_detector_sampler().sample(
        1000, separate_observables=True
    )
    assert len({bool(v) for v in observables.reshape(-1)}) == 1


@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_observable_preserves_distance(k: int) -> None:
    """The Y seam does not collapse the code distance of the closed surface."""
    graph, surfaces = _zzyy_surface()
    circuit = compile_block_graph(graph, observables=surfaces).generate_stim_circuit(k=k)
    noisy = NoiseModel.uniform_depolarizing(0.001).noisy_circuit(circuit)
    error = noisy.shortest_graphlike_error(
        ignore_ungraphlike_errors=False, canonicalize_circuit_errors=True
    )
    assert len(error) == 2 * k + 1
