"""End-to-end compilation tests for the native Y-half cube (S-gate cap).

A single Y-capped memory column must compile all the way through
``compile_block_graph`` to a valid ``stim.Circuit`` whose detectors are all
deterministic, with no fictitious ``MPP`` elements (the physical-seam
property).
"""

from __future__ import annotations

import pytest
import stim

from tqec import BlockGraph, compile_block_graph
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.positioning import LayoutPosition2D, LayoutPosition3D
from tqec.compile.specs.base import CubeSpec
from tqec.compile.specs.library.generators.fixed_bulk import FixedBulkConventionGenerator
from tqec.compile.tree.node import LayerNode
from tqec.compile.tree.tree import LayerTree
from tqec.computation.cube import LeafCubeKind, ZXCube
from tqec.plaquette.compilation.base import IdentityPlaquetteCompiler
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.utils.enums import Orientation
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import BlockPosition3D, Direction3D, Position3D


def _y_capped_column(kind: str = "ZXZ") -> BlockGraph:
    g = BlockGraph("y_capped_column")
    g.add_cube(Position3D(0, 0, 0), ZXCube.from_str(kind))
    g.add_cube(Position3D(0, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    return g


# Both spatial orientations of the cube below the cap. ``ZX*`` is the one
# Gidney's construction is written for; ``XZ*`` needs the reflected patch.
_BELOW_KINDS = ["ZXZ", "ZXX", "XZX", "XZZ"]


@pytest.mark.parametrize("kind", _BELOW_KINDS)
def test_y_cap_compiles_over_either_patch_orientation(kind: str) -> None:
    """The Y cap follows the orientation of the cube below it.

    A cap over an ``XZ*`` cube runs on the diagonal reflection of the patch
    Gidney's construction assumes; ``CubeSpec.y_cap_transposed`` selects it. All
    four below-cube kinds must give a fully deterministic circuit with no MPP.
    """
    circuit = compile_block_graph(_y_capped_column(kind), observables="auto").generate_stim_circuit(
        k=1
    )
    circuit.detector_error_model(decompose_errors=False)
    assert not any(inst.name == "MPP" for inst in circuit.flattened())


def test_y_cap_transposed_is_derived_from_the_cube_below() -> None:
    """``y_cap_transposed`` follows the below cube's ``x`` boundary basis.

    It is never set for the cube below itself.
    """
    for kind, expected in (("ZXZ", False), ("ZXX", False), ("XZX", True), ("XZZ", True)):
        graph = _y_capped_column(kind)
        specs = {cube.position: CubeSpec.from_cube(cube, graph) for cube in graph.cubes}
        assert specs[Position3D(0, 0, 1)].y_cap_transposed is expected, kind
        assert specs[Position3D(0, 0, 0)].y_cap_transposed is False, kind


def _y_init_column(kind: str = "ZXZ") -> BlockGraph:
    """Build a Y-basis initialisation feeding a memory cube above it."""
    g = BlockGraph("y_init_column")
    g.add_cube(Position3D(0, 0, 0), LeafCubeKind.Y_HALF_CUBE)
    g.add_cube(Position3D(0, 0, 1), ZXCube.from_str(kind))
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    return g


def test_a_y_cube_attached_only_to_a_port_is_rejected_with_a_clear_error() -> None:
    """A Y cube needs a regular cube on one temporal side or the other.

    With a cube below it is a measurement cap, with one above it is a Y-basis
    initialisation; with neither there is no patch to run on, and the failure is
    much clearer here than the seam-detector mismatch it would otherwise become.
    """
    graph = BlockGraph("y_on_port")
    graph.add_cube(Position3D(0, 0, 0), LeafCubeKind.PORT, label="p")
    graph.add_cube(Position3D(0, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1), "ZXO")
    # Asserted at the spec, which is where the rule lives: a whole-graph compile
    # of this never reaches it, because open ports are refused earlier.
    with pytest.raises(NotImplementedError, match="no regular cube directly below or above"):
        CubeSpec.from_cube(graph[Position3D(0, 0, 1)], graph)


@pytest.mark.parametrize("kind", _BELOW_KINDS)
@pytest.mark.parametrize("k", [1, 2])
def test_y_basis_initialisation_compiles_deterministically(k: int, kind: str) -> None:
    """A Y cube *below* a regular cube initialises it in the Y basis.

    The time reverse of the measurement cap: deterministic throughout and free of
    fictitious ``MPP``. It runs one round *longer* than a cap, because it needs
    two rounds in the cap's own interaction order next to its fold where a cap
    needs one, and the temporal pipe supplies only one of them --- see
    :func:`handoff_raw_slice`.
    """
    init = compile_block_graph(_y_init_column(kind), observables="auto").generate_stim_circuit(k=k)
    cap = compile_block_graph(_y_capped_column(kind), observables="auto").generate_stim_circuit(k=k)
    init.detector_error_model(decompose_errors=False)  # raises if non-deterministic
    assert not any(inst.name == "MPP" for inst in init.flattened())
    assert init.num_detectors > cap.num_detectors
    assert init.num_measurements > cap.num_measurements


def test_y_cube_direction_is_derived_from_its_neighbour() -> None:
    """``y_cube_initialises`` follows which side the regular cube is on."""
    for graph, position, expected in (
        (_y_capped_column(), Position3D(0, 0, 1), False),
        (_y_init_column(), Position3D(0, 0, 0), True),
    ):
        spec = CubeSpec.from_cube(graph[position], graph)
        assert spec.y_cube_initialises is expected


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
    """The temporal pipe below a Y cap prepends its junction round.

    It must not replace the cap's first layer. For an ordinary cube the pipe
    substitutes the ``Z_NEGATIVE`` border, i.e. ``layer_sequence[0]``. A Y cap's
    first layer is the transition round, and
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
        my_targets = 0
        for inst in circuit.flattened():
            assert isinstance(inst, stim.CircuitInstruction), "flattened yields instructions"
            if inst.name == "MY":
                my_targets += len(inst.targets_copy())
        assert my_targets == expected_y_cubes


def _branch_axis(kind: str) -> Direction3D:
    """Return the axis a side branch of ``kind`` cubes may run along.

    A pipe's two walls must carry different bases. A pipe along ``X`` has its
    walls in ``Y`` and ``Z``, one along ``Y`` has them in ``X`` and ``Z``; so a
    ``ZXZ``/``XZX`` column can branch sideways in ``x`` and a ``ZXX``/``XZZ`` one
    in ``y``. Neither can do both, which is why the two-cap graph exists in two
    mirror-image shapes rather than one.
    """
    if kind[1] != kind[2]:
        return Direction3D.X
    assert kind[0] != kind[2], f"{kind} cannot branch along either spatial axis"
    return Direction3D.Y


def _two_y_caps_with_main_column(kind: str = "ZXZ") -> BlockGraph:
    """Build the notebook cells 7-8 graph.

    A 5-cube main column at the origin with two Y caps on side branches at z=2
    and z=4, each coexisting in a z-slice with a continuing main-column memory
    cube (mismatched temporal schedules). The branch runs along whichever spatial
    axis ``kind`` permits -- see :func:`_branch_axis`.
    """
    g = BlockGraph("two_y_caps")
    b = [Position3D(0, 0, i) for i in range(5)]

    def beside(z: int) -> Position3D:
        if _branch_axis(kind) is Direction3D.X:
            return Position3D(1, 0, z)
        return Position3D(0, 1, z)

    c1, c3 = beside(1), beside(3)
    y2, y4 = beside(2), beside(4)
    for p in b:
        g.add_cube(p, ZXCube.from_str(kind))
    g.add_cube(c1, ZXCube.from_str(kind))
    g.add_cube(c3, ZXCube.from_str(kind))
    g.add_cube(y2, LeafCubeKind.Y_HALF_CUBE)
    g.add_cube(y4, LeafCubeKind.Y_HALF_CUBE)
    for i in range(4):
        g.add_pipe(b[i], b[i + 1])
    g.add_pipe(b[1], c1)
    g.add_pipe(b[3], c3)
    g.add_pipe(c1, y2)
    g.add_pipe(c3, y4)
    return g


@pytest.mark.parametrize("kind", _BELOW_KINDS)
@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_coexistence_compiles_dem_clean(k: int, kind: str) -> None:
    """Two Y caps coexisting with a continuing memory column compile cleanly.

    Their temporal schedules are mismatched; the resulting circuit must have
    only deterministic detectors, and no MPP.
    """
    circuit = compile_block_graph(
        _two_y_caps_with_main_column(kind), observables=[]
    ).generate_stim_circuit(k=k)
    assert circuit.num_detectors > 0
    circuit.detector_error_model(decompose_errors=False)  # raises if non-deterministic
    assert not any(inst.name == "MPP" for inst in circuit.flattened())


# The closed surface is the S^2 = Z algebra of the two gadgets. Its basis on the
# main column follows that column's top face.
_SURFACE_BY_KIND = {"ZXZ": "ZZYY", "ZXX": "XXYY", "XZX": "XXYY", "XZZ": "ZZYY"}


def _closed_surface(kind: str = "ZXZ") -> tuple[BlockGraph, list]:
    graph = _two_y_caps_with_main_column(kind)
    surfaces = graph.find_correlation_surfaces()
    assert [cs.external_stabilizer_on_graph(graph) for cs in surfaces] == [_SURFACE_BY_KIND[kind]]
    return graph, surfaces


@pytest.mark.parametrize("kind", _BELOW_KINDS)
@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_observable_is_deterministic(k: int, kind: str) -> None:
    """The closed ``in . out . Y_cap1 . Y_cap2`` surface lowers deterministically.

    That surface is the S^2 = Z algebra of the two gadgets; it must lower to a
    single observable that takes the same value on every shot.
    """
    graph, surfaces = _closed_surface(kind)
    circuit = compile_block_graph(graph, observables=surfaces).generate_stim_circuit(k=k)
    assert circuit.num_observables == 1
    circuit.detector_error_model(decompose_errors=False)
    _, observables = circuit.compile_detector_sampler().sample(1000, separate_observables=True)
    assert len({bool(v) for v in observables.reshape(-1)}) == 1


def _bulk_z_rpngs(tree: LayerTree, k: int) -> list[tuple[int, LayoutPosition2D, set[str]]]:
    """Every ``(leaf index, position, RPNG strings)`` of the plaquette sub-layers."""

    def leaves(node: LayerNode) -> list[LayerNode]:
        return [node] if node.is_leaf else [n for c in node.children for n in leaves(c)]

    found = []
    for index, leaf in enumerate(leaves(tree._root)):
        layer = leaf._layer
        if not isinstance(layer, LayoutLayer):
            continue
        for position, sublayer in layer.layers.items():
            if not isinstance(sublayer, PlaquetteLayer):
                continue
            found.append(
                (
                    index,
                    position,
                    {
                        str(plaquette.debug_information.rpng)
                        for plaquette in sublayer.plaquettes.collection.values()
                        if plaquette.debug_information.rpng is not None
                    },
                )
            )
    return found


def test_the_cap_interaction_order_reaches_exactly_the_junction_rounds() -> None:
    """Only the round directly below a Y cap is re-timed.

    The temporal pipe below a cap supplies two memory rounds. Its ``Z_POSITIVE``
    border becomes the cap's junction round and must carry the cap's interaction
    order; its ``Z_NEGATIVE`` border is the last round of the cube *below*, which
    can share a time slice with a spatial pipe still on the fixed-bulk schedule.
    Re-timing that one instead makes the two collide on their shared data qubits,
    so the scope of the change is asserted rather than left to chance.
    """
    generator = FixedBulkConventionGenerator(DefaultRPNGTranslator(), IdentityPlaquetteCompiler)
    retimed = {
        str(plaquette.debug_information.rpng)
        for plaquette in generator.get_y_cap_junction_plaquettes(
            Orientation.HORIZONTAL, False
        ).collection.values()
        if plaquette.debug_information.rpng is not None
    }
    plain = {
        str(plaquette.debug_information.rpng)
        for plaquette in generator.get_memory_qubit_plaquettes(
            Orientation.HORIZONTAL, None, None
        ).collection.values()
        if plaquette.debug_information.rpng is not None
    }
    marker = next(iter(retimed - plain))

    graph = _two_y_caps_with_main_column("ZXZ")
    tree = compile_block_graph(graph, observables=[]).to_layer_tree(k=1)
    carrying = [
        (index, position) for index, position, rpngs in _bulk_z_rpngs(tree, 1) if marker in rpngs
    ]
    # One junction round per cap, and both at the capped column's position.
    assert len(carrying) == 2
    assert len({position for _, position in carrying}) == 1
    # Consecutive z-slices, so the two are genuinely the two caps' first rounds.
    assert carrying[0][0] < carrying[1][0]

    # A graph with no Y cap is untouched.
    memory = BlockGraph("memory_column")
    for z in range(3):
        memory.add_cube(Position3D(0, 0, z), ZXCube.from_str("ZXZ"))
    for z in range(2):
        memory.add_pipe(Position3D(0, 0, z), Position3D(0, 0, z + 1))
    plain_tree = compile_block_graph(memory, observables=[]).to_layer_tree(k=1)
    assert not [entry for entry in _bulk_z_rpngs(plain_tree, 1) if marker in entry[2]], (
        "the fixed-bulk schedule must be untouched wherever no Y cap is involved"
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_y_cap_junction_plaquettes_change_only_the_interaction_order(transposed: bool) -> None:
    """Re-timing the junction round preserves everything except the times.

    The round must keep measuring exactly the same stabilizers on exactly the same
    qubits -- only the order in which their data qubits are touched may change.
    """
    orientation = Orientation.VERTICAL if transposed else Orientation.HORIZONTAL
    generator = FixedBulkConventionGenerator(DefaultRPNGTranslator(), IdentityPlaquetteCompiler)
    plain = generator.get_memory_qubit_plaquettes(orientation, None, None)
    retimed = generator.get_y_cap_junction_plaquettes(orientation, transposed)

    assert set(plain.collection.keys()) == set(retimed.collection.keys())
    changed = 0
    for index, before in plain.collection.items():
        after = retimed.collection[index]
        b, a = before.debug_information.rpng, after.debug_information.rpng
        assert b is not None and a is not None
        assert a.ancilla == b.ancilla
        # Same corners occupied, same reset/measure/basis on each.
        assert [(c.r, c.p, c.g) for c in a.corners] == [(c.r, c.p, c.g) for c in b.corners]
        if [c.n for c in a.corners] != [c.n for c in b.corners]:
            changed += 1
    assert changed > 0, "the re-timed round must actually differ from the fixed-bulk one"


def _two_y_inits_with_main_column(kind: str = "ZXZ") -> BlockGraph:
    """The two-cap graph run backwards: Y cubes below, feeding up into the branch."""
    g = BlockGraph("two_y_inits")
    b = [Position3D(0, 0, i) for i in range(5)]

    def beside(z: int) -> Position3D:
        if _branch_axis(kind) is Direction3D.X:
            return Position3D(1, 0, z)
        return Position3D(0, 1, z)

    c1, c3 = beside(1), beside(3)
    y0, y2 = beside(0), beside(2)
    for p in [*b, c1, c3]:
        g.add_cube(p, ZXCube.from_str(kind))
    g.add_cube(y0, LeafCubeKind.Y_HALF_CUBE)
    g.add_cube(y2, LeafCubeKind.Y_HALF_CUBE)
    for i in range(4):
        g.add_pipe(b[i], b[i + 1])
    g.add_pipe(b[1], c1)
    g.add_pipe(b[3], c3)
    g.add_pipe(y0, c1)
    g.add_pipe(y2, c3)
    return g


@pytest.mark.parametrize("kind", _BELOW_KINDS)
@pytest.mark.parametrize("k", [1, 2])
def test_two_y_inits_observable_preserves_distance(k: int, kind: str) -> None:
    """Two Y-basis initialisations close the same surface at full distance.

    The mirror of :func:`test_two_y_caps_observable_preserves_distance`. An
    initialisation needs *two* rounds in the cap's interaction order after its
    fold where a cap needs one before it; with only the pipe's junction round the
    distance is 4 instead of 5 at ``k = 2``, and the block's own handoff round is
    what restores it.

    That defect is invisible to :meth:`~stim.Circuit.shortest_graphlike_error` in
    *both* its modes: two of the four faults flip four detectors each, so the
    error is a hyperedge no matching graph can represent and the graphlike search
    reports the full ``2k + 1`` regardless. Only
    ``search_for_undetectable_logical_errors`` sees it.
    """
    graph = _two_y_inits_with_main_column(kind)
    surfaces = graph.find_correlation_surfaces()
    assert [cs.external_stabilizer_on_graph(graph) for cs in surfaces] == [_SURFACE_BY_KIND[kind]]
    circuit = compile_block_graph(graph, observables=surfaces).generate_stim_circuit(k=k)
    circuit.detector_error_model(decompose_errors=False)
    _, observables = circuit.compile_detector_sampler().sample(300, separate_observables=True)
    assert len({bool(v) for v in observables.reshape(-1)}) == 1

    noisy = NoiseModel.uniform_depolarizing(0.001).noisy_circuit(circuit)
    error = noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    assert len(error) == 2 * k + 1


@pytest.mark.parametrize("kind", _BELOW_KINDS)
@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_observable_preserves_distance(k: int, kind: str) -> None:
    """The Y seam does not collapse the code distance of the closed surface.

    This is a regression test for a hook-error mismatch at the junction between
    the memory round the temporal pipe supplies and the cap's transition round.
    While the junction round ran the fixed-bulk interaction order, two faults
    straddling that seam produced the same syndrome but different logical
    effects, giving an undetectable weight-2 logical error and a circuit distance
    of ``k + 1`` instead of ``2k + 1``. See
    :meth:`.FixedBulkConventionGenerator.get_y_cap_junction_plaquettes`.

    Uses :meth:`stim.Circuit.search_for_undetectable_logical_errors` rather than
    :meth:`~stim.Circuit.shortest_graphlike_error`. The latter, called with
    ``ignore_ungraphlike_errors=False``, decomposes the error model before
    searching; that decomposition split one of the two parallel
    ``D_a D_b`` / ``D_a D_b L0`` edges into boundary components and so reported
    the full ``2k + 1`` throughout the period when the circuit really had
    distance ``k + 1``. It measures the decomposed matching graph, not the
    circuit.

    The defect only appears under *correlated* two-qubit gate noise: each of the
    two faults was a single ``DEPOLARIZE2`` carrying both an X and a Z component.
    Splitting that noise into independent single-qubit errors hides it, so
    ``uniform_depolarizing`` (which emits ``DEPOLARIZE2`` after two-qubit gates)
    is required here.
    """
    graph, surfaces = _closed_surface(kind)
    circuit = compile_block_graph(graph, observables=surfaces).generate_stim_circuit(k=k)
    noisy = NoiseModel.uniform_depolarizing(0.001).noisy_circuit(circuit)
    error = noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False,
    )
    assert len(error) == 2 * k + 1


def test_y_cap_releases_its_qubits_but_a_memory_cube_does_not() -> None:
    """Only a block that measures out its data qubits may leave a merged slice.

    Any other block must stay present in the slice's trailing layers.
    """
    graph = _two_y_caps_with_main_column()
    compiled = compile_block_graph(graph, observables=[])
    y_block = compiled._blocks[LayoutPosition3D.from_block_position(BlockPosition3D(1, 0, 2))]
    memory_block = compiled._blocks[LayoutPosition3D.from_block_position(BlockPosition3D(0, 0, 2))]
    assert y_block.releases_its_qubits is True
    assert memory_block.releases_its_qubits is False


def test_finished_y_cap_is_absent_from_the_trailing_merged_layers() -> None:
    """A Y cap shorter than its slice drops out instead of idling.

    At k=3 the cap (k+3 = 6 rounds) is shorter than the memory column it shares
    a z-slice with (2k+1 = 7), so it must be gone from the slice's last layer
    rather than idle through it on padded boundary rounds.
    """
    k = 3
    tree = compile_block_graph(_two_y_caps_with_main_column(), observables=[]).to_layer_tree(k=k)

    def leaves(node: LayerNode) -> list[LayerNode]:
        return [node] if node.is_leaf else [n for c in node.children for n in leaves(c)]

    slice_leaves = leaves(tree._root.children[2])  # z = 2
    positions = []
    for leaf in slice_leaves:
        assert isinstance(leaf._layer, LayoutLayer)
        positions.append(set(leaf._layer.layers))
    assert len(positions) == 7
    # The cap contributes to the first six layers and is gone from the last.
    assert len(positions[0]) == 2
    assert len(positions[-1]) == 1
    assert positions[-1] < positions[0]
