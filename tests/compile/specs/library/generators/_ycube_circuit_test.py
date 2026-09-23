"""Validation tests for the native Y-cube circuit construction.

These exercise the standard surface-code round + detector machinery
independently of the Y transition, by building a plain memory experiment and
checking it is a well-formed, full-distance surface code.
"""

from __future__ import annotations

import pytest
import stim

from tests._vendor.midout import gen
from tests._vendor.midout.circuits.steps._measure_y_transition_round import (
    make_y_transition_round_nesw_xzxz_to_xzzx,
)
from tests._vendor.midout.circuits.steps._patches import (
    make_xtop_qubit_patch,
    make_ztop_yboundary_patch,
)
from tqec import BlockGraph, compile_block_graph
from tqec.compile.specs.library.generators._ycube_circuit import (
    _build_transition_round,
    _Builder,
    _bulk_detectors,
    _final_round,
    _first_round_detectors,
    _middle_line_correction_plaquettes,
    _tqec_logical_y_observable_spec,
    memory_experiment_circuit,
    reverse_round_circuit,
    standard_round,
    transition_round,
    y_cap_magic_experiment_circuit,
    y_cap_raw_circuit,
    y_cap_segment_circuit,
    y_init_magic_experiment_circuit,
)
from tqec.compile.specs.library.generators.ycube import (
    gidney_to_tqec,
    xtop_qubit_patch,
    ztop_yboundary_patch,
)
from tqec.computation.cube import LeafCubeKind, ZXCube
from tqec.utils.enums import Basis
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_memory_experiment_detectors_are_deterministic(distance: int, basis: Basis) -> None:
    """A noiseless memory experiment has only deterministic detectors.

    Equivalently, its detector error model builds without error.
    """
    circuit = memory_experiment_circuit(distance, rounds=distance, basis=basis)
    # Raises ValueError if any detector/observable is non-deterministic.
    circuit.detector_error_model(decompose_errors=False)


def _memory_experiment_with_logical(distance: int, rounds: int, basis: Basis) -> stim.Circuit:
    patch = xtop_qubit_patch(distance)
    b = _Builder()
    b.allocate(set(patch.data_qubits) | {s.ancilla for s in patch.stabilizers})
    init = {d: basis for d in patch.data_qubits}
    tags = [f"r{i}" for i in range(rounds)]
    standard_round(b, patch, tags[0], init_data_basis=init)
    _first_round_detectors(b, patch, tags[0], init)
    for i in range(1, rounds):
        standard_round(b, patch, tags[i])
        _bulk_detectors(b, patch, tags[i - 1], tags[i])
    final_tag = "final"
    data = sorted(patch.data_qubits, key=lambda p: (p[1], p[0]))
    b.measure(f"M{basis.value}", data, final_tag)
    for s in patch.stabilizers:
        if s.basis != basis:
            continue
        dd = [x for x in s.ordered_data if x is not None]
        b.detector(
            [b.rec(tags[-1], s.ancilla)] + [b.rec(final_tag, x) for x in dd],
            (s.ancilla[0], s.ancilla[1], 1),
        )
    # xtop has X boundaries top/bottom -> X logical is vertical (column x==1);
    # Z boundaries left/right -> Z logical is horizontal (row y==1).
    logical = [q for q in data if q[0] == 1] if basis == Basis.X else [q for q in data if q[1] == 1]
    b.circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(b.rec(final_tag, q) - b.num_measurements) for q in logical],
        0,
    )
    return b.circuit


def _instructions(circuit: stim.Circuit) -> list[stim.CircuitInstruction]:
    """Return a circuit's instructions, which none of these circuits nest."""
    instructions: list[stim.CircuitInstruction] = []
    for instruction in circuit:
        assert isinstance(instruction, stim.CircuitInstruction), "unexpected REPEAT block"
        instructions.append(instruction)
    return instructions


def _with_depolarizing_noise(circuit: stim.Circuit, p: float = 0.001) -> stim.Circuit:
    all_q = sorted(
        {
            t.value
            for inst in _instructions(circuit)
            for t in inst.targets_copy()
            if t.is_qubit_target
        }
    )
    out = stim.Circuit()
    for inst in _instructions(circuit):
        out.append(inst)
        if inst.name == "TICK":
            out.append("DEPOLARIZE1", all_q, p)
    return out


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("basis", [Basis.X, Basis.Z])
def test_memory_experiment_has_full_code_distance(distance: int, basis: Basis) -> None:
    """The memory experiment is a genuine distance-``d`` code.

    That is, its shortest graphlike logical error has weight ``d``.
    """
    circuit = _with_depolarizing_noise(_memory_experiment_with_logical(distance, distance, basis))
    circuit.detector_error_model(decompose_errors=False)
    assert len(circuit.shortest_graphlike_error()) == distance


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("init_basis", [Basis.X, Basis.Z])
def test_y_cap_segment_detectors_are_deterministic(distance: int, init_basis: Basis) -> None:
    """The full Y-cap segment has only deterministic detectors.

    The segment is memory + transition + boundary + final, and the transition
    seam detectors are included in the check.
    """
    circuit = y_cap_segment_circuit(distance, mem_rounds=distance, init_basis=init_basis)
    circuit.detector_error_model(decompose_errors=False)


def _cap_only_circuit(distance: int):
    """Build ``[transition, boundary x d//2, final]`` with data qubits unreset.

    Used for observable-flow checks. Returns ``(builder, transition flows)``.
    """
    xtop = xtop_qubit_patch(distance)
    ztop = ztop_yboundary_patch(distance)
    b = _Builder()
    b.allocate(
        set(xtop.data_qubits)
        | {s.ancilla for s in xtop.stabilizers}
        | set(ztop.data_qubits)
        | {s.ancilla for s in ztop.stabilizers}
    )
    flows = transition_round(b, distance, "T")
    prev = "T"
    for i in range(distance // 2):
        standard_round(b, ztop, f"b{i}")
        prev = f"b{i}"
    _final_round(b, ztop, prev, "F", distance)
    return b, flows


@pytest.mark.parametrize("distance", [3, 5])
def test_transition_measures_logical_y(distance: int) -> None:
    """The transition round's observable records reconstruct the logical Y.

    The incoming operator is Y at the corner, Z along the x-axis boundary and X
    along the y-axis boundary; the reconstruction is verified with stim's
    ``has_flow``.
    """
    b, flows = _cap_only_circuit(distance)
    nq = max(b.q2i.values()) + 1
    arr = ["I"] * nq
    arr[b.q2i[gidney_to_tqec(0j)]] = "Y"
    for q in range(1, distance):
        arr[b.q2i[gidney_to_tqec(complex(q, 0))]] = "Z"
        arr[b.q2i[gidney_to_tqec(complex(0, q))]] = "X"
    inp = stim.PauliString("".join(arr))
    obs_recs = [b.rec("T", c) for c in flows.observable]
    flow = stim.Flow(input=inp, output=stim.PauliString(nq), measurements=obs_recs)
    assert b.circuit.has_flow(flow)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_observable_spec_reads_out_the_middle_cross(distance: int, transposed: bool) -> None:
    """The emitted readout measures the logical Y on the patch's middle lines.

    That is the representative the observable builder lowers an arriving
    correlation surface onto, and the one the geometric plaquette-strip
    correction in ``_tqec_logical_y_observable_spec`` targets --- not Gidney's
    corner-anchored operator, which
    :func:`test_transition_measures_logical_y` covers.

    The flow holds up to sign: ``+Y`` at the crossing of the two strings is a
    convention, and the readout implements its negation. A constant frame offset
    leaves the observable deterministic, which is all the compiler needs.
    """
    circuit, _ = _build_transition_round(distance, transposed)
    coords = {(int(c[0]), int(c[1])): q for q, c in circuit.get_final_qubit_coordinates().items()}
    middle = (distance - 1) // 2

    target = stim.PauliString(circuit.num_qubits)
    for i in range(distance):
        target[coords[gidney_to_tqec(complex(middle, i), transposed)]] = "X"
    for i in range(distance):
        qubit = coords[gidney_to_tqec(complex(i, middle), transposed)]
        target[qubit] = "Y" if target[qubit] == 1 else "Z"

    spec = set(_tqec_logical_y_observable_spec(distance, transposed))
    measured = _measured_coordinates(circuit)
    records = [i for i, coord in enumerate(measured) if coord in spec]
    assert len(records) == len(spec)

    flow = stim.Flow(
        input=target, output=stim.PauliString(circuit.num_qubits), measurements=records
    )
    assert circuit.has_flow(flow, unsigned=True)


def _measured_coordinates(circuit: stim.Circuit) -> list[tuple[int, int]]:
    """Return the coordinate measured by each measurement of ``circuit``, in order."""
    qubit_coords = circuit.get_final_qubit_coordinates()
    out: list[tuple[int, int]] = []
    for inst in _instructions(circuit):
        if inst.name not in ("M", "MX", "MY", "MZ"):
            continue
        for target in inst.targets_copy():
            qubit = target.qubit_value
            assert qubit is not None
            coord = qubit_coords[qubit]
            out.append((int(coord[0]), int(coord[1])))
    return out


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_middle_line_correction_is_the_plaquette_strip(distance: int, transposed: bool) -> None:
    """The two logical-Y representatives differ by exactly two plaquette strips.

    Moving the ``X`` string from the boundary column to the middle one multiplies
    it by every ``X`` plaquette in between, and likewise for the ``Z`` string and
    the rows. Pinning the strips pins the whole construction: a wrong strip
    silently reads out a different operator.
    """
    middle = (distance - 1) // 2
    ancillas = _middle_line_correction_plaquettes(distance, transposed)
    assert len(ancillas) == len(set(ancillas)) == 2 * middle * (middle + 1)

    patch = xtop_qubit_patch(distance, transposed)
    basis_of = {s.gidney_ancilla: s.basis for s in patch.stabilizers}
    x_strip = [a for a in ancillas if basis_of[a] == Basis.X]
    z_strip = [a for a in ancillas if basis_of[a] == Basis.Z]
    assert len(x_strip) == len(z_strip)
    assert all(0 < a.real < middle for a in x_strip)
    assert all(0 < a.imag < middle for a in z_strip)
    # And nothing eligible was left out.
    assert {a for a, b in basis_of.items() if b == Basis.X and 0 < a.real < middle} == set(x_strip)
    assert {a for a, b in basis_of.items() if b == Basis.Z and 0 < a.imag < middle} == set(z_strip)


@pytest.mark.parametrize(
    ("transposed", "expected"),
    [
        (False, ((1, 1), (2, 0), (2, 2), (4, 6), (6, 4))),
        (True, ((0, 2), (1, 1), (2, 2), (4, 6), (6, 4))),
    ],
)
def test_observable_spec_at_distance_three_is_stable(
    transposed: bool, expected: tuple[tuple[int, int], ...]
) -> None:
    """Golden readout at ``d=3``, so a change of representative cannot slip in.

    These coordinates were cross-checked against
    ``stim.Circuit.solve_flow_measurements`` for ``d`` up to 11 in both
    orientations when the geometric construction replaced it.
    """
    assert _tqec_logical_y_observable_spec(3, transposed) == expected


def _oracle_y_cap_segment(distance: int, mem_rounds: int):
    """Build the vendored-gen equivalent of ``y_cap_segment_circuit``.

    That is Z-init memory + transition + boundary + final, with the observable
    flow stripped, used as a detector-parity oracle.
    """
    d = distance
    xtop = make_xtop_qubit_patch(distance=d)
    ztop = make_ztop_yboundary_patch(distance=d)
    trans = make_y_transition_round_nesw_xzxz_to_xzzx(distance=d)
    trans = gen.Chunk(
        circuit=trans.circuit,
        q2i=trans.q2i,
        flows=[f for f in trans.flows if f.obs_index is None],
    )
    chunks = [gen.standard_surface_code_chunk(xtop, init_data_basis="Z")]
    chunks += [gen.standard_surface_code_chunk(xtop) for _ in range(mem_rounds - 1)]
    chunks.append(trans)
    chunks += [gen.standard_surface_code_chunk(ztop) for _ in range(d // 2)]
    chunks.append(
        gen.standard_surface_code_chunk(
            ztop,
            measure_data_basis={q: "Z" if q.real + q.imag < d else "X" for q in ztop.data_set},
        )
    )
    return gen.compile_chunks_into_circuit(chunks, include_detectors=True).flattened()


@pytest.mark.parametrize("distance", [3, 5])
def test_y_cap_segment_detector_count_matches_oracle(distance: int) -> None:
    """The native Y-cap segment emits as many detectors as the oracle.

    Both are built from the same recipe: Z-init memory + Y cap.
    """
    native = y_cap_segment_circuit(distance, mem_rounds=distance, init_basis=Basis.Z)
    oracle = _oracle_y_cap_segment(distance, mem_rounds=distance)
    assert native.num_detectors == oracle.num_detectors
    assert native.num_qubits == oracle.num_qubits


def _compose_below_and_raw(distance: int, mem_rounds: int, init_basis: Basis) -> stim.Circuit:
    """Emulate what the detector annotator does.

    A native below memory column, the raw Y-cap slice appended, and the seam
    detectors formed from the raw slice's ``seam_spec`` against the below
    column's last round.
    """
    xtop = xtop_qubit_patch(distance)
    b = _Builder()
    b.allocate(set(xtop.data_qubits) | {s.ancilla for s in xtop.stabilizers})
    init = {dq: init_basis for dq in xtop.data_qubits}
    tags = [f"m{i}" for i in range(mem_rounds)]
    standard_round(b, xtop, tags[0], init_data_basis=init)
    _first_round_detectors(b, xtop, tags[0], init)
    for i in range(1, mem_rounds):
        standard_round(b, xtop, tags[i])
        _bulk_detectors(b, xtop, tags[i - 1], tags[i])
    last = tags[-1]

    raw, seam = y_cap_raw_circuit(distance)
    idx2coord = {
        inst.targets_copy()[0].value: (int(inst.gate_args_copy()[0]), int(inst.gate_args_copy()[1]))
        for inst in _instructions(raw)
        if inst.name == "QUBIT_COORDS"
    }
    b.allocate(set(idx2coord.values()))
    raw_base = b.num_measurements
    running = 0
    for inst in _instructions(raw):
        if inst.name == "QUBIT_COORDS":
            continue
        if inst.name == "DETECTOR":
            newt = [
                stim.target_rec((raw_base + running + t.value) - (raw_base + running))
                for t in inst.targets_copy()
            ]
            b.circuit.append("DETECTOR", newt, inst.gate_args_copy())
            continue
        targs = [b.q2i[idx2coord[t.value]] if t.is_qubit_target else t for t in inst.targets_copy()]
        b.circuit.append(inst.name, targs, inst.gate_args_copy())
        if inst.name in ("M", "MX", "MY", "MZ"):
            running += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
    b.num_measurements = raw_base + running
    for anc_coord, raw_idxs in seam.items():
        recs = [b.rec(last, anc_coord)] + [raw_base + i for i in raw_idxs]
        b.circuit.append(
            "DETECTOR",
            [stim.target_rec(r - b.num_measurements) for r in recs],
            (anc_coord[0], anc_coord[1], 0),
        )
    return b.circuit


@pytest.mark.parametrize("distance", [3, 5])
def test_raw_slice_plus_seam_matches_segment(distance: int) -> None:
    """The raw Y-cap slice composes onto a native memory column.

    With annotator-style seam detectors it must be deterministic and carry as
    many detectors as the monolithic ``y_cap_segment_circuit``.
    """
    composed = _compose_below_and_raw(distance, distance, Basis.Z)
    composed.detector_error_model(decompose_errors=False)
    segment = y_cap_segment_circuit(distance, mem_rounds=distance, init_basis=Basis.Z)
    assert composed.num_detectors == segment.num_detectors


def _detector_signature(circuit: stim.Circuit, coord_map: dict[int, tuple]) -> set[frozenset]:
    """Return a frame-independent detector fingerprint.

    Each detector becomes the frozenset of the
    ``(qubit_coord, k-th-measurement-of-that-qubit)`` labels it references.
    """
    times: dict[tuple, int] = {}
    rec_id: dict[int, tuple] = {}
    n = 0
    for inst in _instructions(circuit):
        if inst.name in ("M", "MX", "MY", "MZ"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    c = coord_map[t.value]
                    k = times.get(c, 0)
                    times[c] = k + 1
                    rec_id[n] = (c, k)
                    n += 1
    dets: set[frozenset] = set()
    running = 0
    for inst in _instructions(circuit):
        if inst.name in ("M", "MX", "MY", "MZ"):
            running += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
        elif inst.name == "DETECTOR":
            dets.add(frozenset(rec_id[running + t.value] for t in inst.targets_copy()))
    return dets


def _coord_map(circuit: stim.Circuit, transform) -> dict[int, tuple]:
    return {
        inst.targets_copy()[0].value: transform(*inst.gate_args_copy())
        for inst in _instructions(circuit)
        if inst.name == "QUBIT_COORDS"
    }


@pytest.mark.parametrize("distance", [3, 5])
def test_raw_slice_detectors_match_oracle_exactly(distance: int) -> None:
    """The native Y-cap slice has the same detector set as the oracle.

    The slice is memory column + raw slice + seam. Not merely the same count:
    the same detectors, each referencing the same
    ``(qubit, k-th-measurement)`` labels once both are expressed in a common
    coordinate frame.
    """
    native = _compose_below_and_raw(distance, distance, Basis.Z)
    oracle = _oracle_y_cap_segment(distance, mem_rounds=distance)
    native_sig = _detector_signature(native, _coord_map(native, lambda x, y, *r: (int(x), int(y))))
    oracle_sig = _detector_signature(
        oracle, _coord_map(oracle, lambda x, y, *r: gidney_to_tqec(complex(x, y)))
    )
    assert native_sig == oracle_sig


def _compiled_y_capped_column(k: int) -> stim.Circuit:
    """Compile a ``ZXZ`` memory column capped by a Y-half cube.

    This goes through the production path. :func:`_compose_below_and_raw`
    emulates the annotator around the monolithic :func:`y_cap_raw_circuit`;
    this instead exercises the per-round :class:`RawCircuitLayer` slices and
    the real ``AnnotateDetectorsOnLayerNode`` seam emission.
    """
    graph = BlockGraph("y_capped_column")
    graph.add_cube(Position3D(0, 0, 0), ZXCube.from_str("ZXZ"))
    graph.add_cube(Position3D(0, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    return compile_block_graph(graph, observables=None).generate_stim_circuit(k).flattened()


@pytest.mark.parametrize("k", [1, 2])
def test_compiled_y_capped_column_detectors_match_oracle_exactly(k: int) -> None:
    """The *compiled* Y-capped column has the same detector set as the oracle.

    :func:`test_raw_slice_detectors_match_oracle_exactly` pins the monolithic
    slice with a hand-rolled seam; this pins what tqec actually emits: per-round
    raw layers, the template path for the memory column below, and the seam
    detectors built by the detector annotator from the flow specs.

    The compiled column runs ``2k + 2`` memory rounds below the cap, so the
    oracle is built with that many.
    """
    compiled = _compiled_y_capped_column(k)
    oracle = _oracle_y_cap_segment(2 * k + 1, mem_rounds=2 * k + 2)
    compiled_sig = _detector_signature(
        compiled, _coord_map(compiled, lambda x, y, *rest: (int(x), int(y)))
    )
    oracle_sig = _detector_signature(
        oracle, _coord_map(oracle, lambda x, y, *rest: gidney_to_tqec(complex(x, y)))
    )
    assert compiled_sig == oracle_sig


_NOISE_INSTRUCTIONS = frozenset(
    {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "E",
        "ELSE_CORRELATED_ERROR",
    }
)


def _noisy_except_mpp(circuit: stim.Circuit, p: float = 0.001) -> stim.Circuit:
    """Layer uniform depolarizing noise, leaving the magic ``MPP`` moments alone.

    The magic head/tail is a simulator-only readout with no physical counterpart,
    so it must stay noiseless -- including the idling noise its moments would
    otherwise charge every other qubit. tqec's ``NoiseModel`` also cannot
    classify a multi-product ``MPP``: ``_measure_basis`` assumes a single Pauli
    product. Both are handled by keeping those moments out of the model entirely.
    """
    moments: list[list[stim.CircuitInstruction]] = [[]]
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            moments.append([])
        elif instruction.name != "QUBIT_COORDS":
            moments[-1].append(instruction)
    magic = [any(i.name == "MPP" for i in moment) for moment in moments]

    body = stim.Circuit()
    for index, coords in circuit.get_final_qubit_coordinates().items():
        body.append("QUBIT_COORDS", [index], list(coords))
    for position, moment in enumerate(moments):
        if magic[position]:
            continue
        for instruction in moment:
            body.append(instruction)
        body.append("TICK")
    noisy_body = NoiseModel.uniform_depolarizing(p).noisy_circuit(
        body, system_qubits=set(range(circuit.num_qubits))
    )

    noised: list[list[stim.CircuitInstruction]] = [[]]
    for instruction in noisy_body.flattened():
        if instruction.name == "TICK":
            noised.append([])
        elif instruction.name != "QUBIT_COORDS":
            noised[-1].append(instruction)

    out = stim.Circuit()
    for index, coords in circuit.get_final_qubit_coordinates().items():
        out.append("QUBIT_COORDS", [index], list(coords))
    cursor = 0
    for position, moment in enumerate(moments):
        source = moment if magic[position] else noised[cursor]
        if not magic[position]:
            cursor += 1
        for instruction in source:
            out.append(instruction)
        if position != len(moments) - 1:
            out.append("TICK")
    assert out.num_measurements == circuit.num_measurements
    assert out.num_detectors == circuit.num_detectors
    return out


def _circuit_distance(circuit: stim.Circuit) -> int:
    noisy = _noisy_except_mpp(circuit)
    return len(
        noisy.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=4,
            dont_explore_edges_increasing_symptom_degree=False,
        )
    )


def test_reverse_round_circuit_inverts_the_transition_round() -> None:
    """Reversing the transition round turns it into a patch *preparation*.

    The forward round folds the ``xtop`` patch onto the degenerate Y-boundary
    patch and measures its stabilizers out; the reverse must prepare every
    ``xtop`` stabilizer, which is what ``solve_flow_measurements`` is asked here.
    """
    d = 3
    forward, _ = _build_transition_round(d)
    reversed_circuit = reverse_round_circuit(forward)
    assert {i.name for i in reversed_circuit.flattened()} >= {"RY", "SQRT_X_DAG"}

    index = {
        (int(v[0]), int(v[1])): i for i, v in reversed_circuit.get_final_qubit_coordinates().items()
    }
    n = reversed_circuit.num_qubits
    for stabilizer in xtop_qubit_patch(d).stabilizers:
        paulis = ["_"] * n
        for data in stabilizer.ordered_data:
            if data is not None:
                paulis[index[data]] = stabilizer.basis.value
        flow = stim.Flow(input=stim.PauliString(n), output=stim.PauliString("".join(paulis)))
        assert reversed_circuit.solve_flow_measurements([flow])[0] is not None, stabilizer.ancilla


@pytest.mark.parametrize("d", [3, 5])
def test_y_basis_initialisation_matches_the_measurement_cap(d: int) -> None:
    """The time-reversed cap is a full-distance Y-basis initialisation.

    Both experiments are bracketed by Gidney-style noiseless magic ``MPP``
    operations: a preparation head for the measurement cap, a readout tail for
    the initialisation. Each must be deterministic and carry the code's full
    distance, and the reversal must preserve the detector count exactly.
    """
    cap = y_cap_magic_experiment_circuit(d, mem_rounds=d)
    init = y_init_magic_experiment_circuit(d, mem_rounds=d)

    for name, circuit in (("cap", cap), ("init", init)):
        circuit.detector_error_model(decompose_errors=False)  # raises if non-deterministic
        _, observables = circuit.compile_detector_sampler().sample(200, separate_observables=True)
        assert len({bool(v) for v in observables.reshape(-1)}) == 1, name
        assert _circuit_distance(circuit) == d, name

    assert init.num_detectors == cap.num_detectors
    assert init.num_observables == cap.num_observables
    # Reversing turns the cap's transversal data measurement into a reset, so the
    # initialisation makes strictly fewer measurements.
    assert init.num_measurements < cap.num_measurements
