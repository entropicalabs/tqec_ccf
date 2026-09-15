"""End-to-end compilation tests for the native Y-half cube (S-gate cap).

A single Y-capped memory column must compile all the way through
``compile_block_graph`` to a valid ``stim.Circuit`` whose detectors are all
deterministic, with no fictitious ``MPP`` elements (the physical-seam
property).
"""

from __future__ import annotations

from itertools import pairwise

import pytest
import stim

from tqec import BlockGraph, compile_block_graph
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.positioning import LayoutPosition3D
from tqec.compile.specs.base import CubeSpec
from tqec.compile.tree.node import LayerNode
from tqec.computation.cube import LeafCubeKind, ZXCube
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import BlockPosition3D, Position3D


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


def test_y_basis_initialisation_is_rejected_with_a_clear_error() -> None:
    """A Y cube with no cube below it is refused where it is diagnosable.

    Only the measurement half of the construction is lowered. Without this
    check, a Y-basis initialisation is built as a measurement cap and fails much
    later with a seam-detector mismatch.
    """
    graph = BlockGraph("y_init")
    graph.add_cube(Position3D(0, 0, 0), LeafCubeKind.Y_HALF_CUBE)
    graph.add_cube(Position3D(0, 0, 1), ZXCube.from_str("ZXZ"))
    graph.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    with pytest.raises(NotImplementedError, match="not a Y-basis measurement cap"):
        compile_block_graph(graph, observables=[]).generate_stim_circuit(k=1)


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


def _two_y_caps_with_main_column(kind: str = "ZXZ") -> BlockGraph:
    """Build the notebook cells 7-8 graph.

    A 5-cube main column (x=0) with two Y caps on side branches (x=1) at z=2 and
    z=4, each coexisting in a z-slice with a continuing main-column memory cube
    (mismatched temporal schedules).
    """
    g = BlockGraph("two_y_caps")
    b = [Position3D(0, 0, i) for i in range(5)]
    c1, c3 = Position3D(1, 0, 1), Position3D(1, 0, 3)
    y2, y4 = Position3D(1, 0, 2), Position3D(1, 0, 4)
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


@pytest.mark.parametrize("kind", ["ZXZ", "XZX"])
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
# main column follows that column's top face: ``ZZYY`` for ``ZXZ``, ``XXYY`` for
# the transposed ``XZX``.
_SURFACE_BY_KIND = {"ZXZ": "ZZYY", "XZX": "XXYY"}


def _closed_surface(kind: str = "ZXZ") -> tuple[BlockGraph, list]:
    graph = _two_y_caps_with_main_column(kind)
    surfaces = graph.find_correlation_surfaces()
    assert [cs.external_stabilizer_on_graph(graph) for cs in surfaces] == [_SURFACE_BY_KIND[kind]]
    return graph, surfaces


@pytest.mark.parametrize("kind", ["ZXZ", "XZX"])
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: the Y cap collapses the circuit distance of the closed "
        "surface to k + 1 (2 at k=1, 3 at k=2) instead of 2k + 1. The shortest "
        "undetectable logical error is a timelike string of ancilla measurement "
        "flips running from the transition round through the cap's boundary and "
        "final rounds. Reproduced for both ZXZ and XZX; the same block graph with "
        "ordinary caps instead of Y caps keeps full distance, and Gidney's own "
        "Y-basis memory experiment (tests/_vendor/midout) keeps full distance, so "
        "the loss is specific to tqec's Y cap."
    ),
)
@pytest.mark.parametrize("kind", ["ZXZ", "XZX"])
@pytest.mark.parametrize("k", [1, 2])
def test_two_y_caps_observable_preserves_distance(k: int, kind: str) -> None:
    """The Y seam does not collapse the code distance of the closed surface.

    Uses :meth:`stim.Circuit.search_for_undetectable_logical_errors` rather than
    :meth:`~stim.Circuit.shortest_graphlike_error`. The latter, called with
    ``ignore_ungraphlike_errors=False``, decomposes the error model before
    searching; that decomposition splits one of the two parallel
    ``D_a D_b`` / ``D_a D_b L0`` edges into boundary components and so hides the
    weight-2 logical error. It is the distance of the decomposed matching graph,
    not of the circuit.
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


@pytest.mark.parametrize("add_polygons", [False, True])
def test_crumble_url_for_a_mismatched_schedule_slice(add_polygons: bool) -> None:
    """Regression: ``generate_crumble_url`` used to drop ``k``.

    A z-slice whose blocks have mismatched temporal schedules --- a Y cap beside
    a continuing memory cube --- can only be merged by flattening it at a
    concrete scaling factor, so ``to_layer_tree()`` with no argument raised. The
    circuit itself compiled fine, which is what kept this hidden.
    """
    compiled = compile_block_graph(_two_y_caps_with_main_column())
    url = compiled.generate_crumble_url(k=1, add_polygons=add_polygons)
    assert url.startswith("https://algassert.com/crumble#circuit=")


def _capped_column_at(x: int, kind: str = "ZXZ") -> BlockGraph:
    """Build a Y-capped column whose blocks sit at block position ``x``."""
    g = BlockGraph(f"capped column at x={x}")
    g.add_cube(Position3D(x, 0, 0), ZXCube.from_str(kind))
    g.add_cube(Position3D(x, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    g.add_pipe(Position3D(x, 0, 0), Position3D(x, 0, 1))
    return g


def _moved_then_capped(x: int = 0, kind: str = "ZXZ") -> BlockGraph:
    """Move the logical qubit one block sideways, then cap it.

    The cap ends up **alone** in its z-slice, at a block position that is not the
    origin -- the case the seam-detector coordinate frame used to get wrong.
    """
    g = BlockGraph(f"y-move at x={x}")
    cubes = [
        (Position3D(x, 0, 0), ZXCube.from_str(kind)),
        (Position3D(x, 0, 1), ZXCube.from_str(kind)),
        (Position3D(x + 1, 0, 1), ZXCube.from_str(kind)),
        (Position3D(x + 1, 0, 2), LeafCubeKind.Y_HALF_CUBE),
    ]
    for pos, k in cubes:
        g.add_cube(pos, k)
    for (p1, _), (p2, _) in pairwise(cubes):
        g.add_pipe(p1, p2)
    return g


@pytest.mark.parametrize("x", [0, 1, 3])
def test_y_cap_column_away_from_the_origin(x: int) -> None:
    """A Y-capped column compiles wherever it sits in the block grid.

    Regression: the cap's seam detectors were resolved against a coordinate frame
    computed relative to the z-slice's own bounding box, while the measurement
    records are keyed by absolute qubit coordinates. The two agree only when the
    slice contains a block at the computation's minimum x/y, so a column at
    ``x >= 1`` raised ``KeyError`` on an ancilla one block pitch away.
    """
    circuit = compile_block_graph(_capped_column_at(x)).generate_stim_circuit(k=1)
    circuit.detector_error_model(decompose_errors=False)


@pytest.mark.parametrize("kind", ["ZXZ", "XZX"])
@pytest.mark.parametrize("k", [1, 2])
def test_moving_a_logical_qubit_before_capping_it(k: int, kind: str) -> None:
    """A qubit that moves sideways and is then Y-capped compiles cleanly.

    The cap is the only block in its z-slice and is not at the origin, so its
    slice minimum equals its own position and the old slice-relative offset
    collapsed to zero. ``build_yy``-style graphs never caught this: there the cap
    always shares its slice with the main column at x=0, which makes the
    slice-relative and absolute frames coincide.
    """
    circuit = compile_block_graph(_moved_then_capped(kind=kind)).generate_stim_circuit(k=k)
    circuit.detector_error_model(decompose_errors=False)


@pytest.mark.parametrize("x", [0, 1, 2])
def test_moved_and_capped_column_away_from_the_origin(x: int) -> None:
    """The moved-then-capped column is position-independent too."""
    circuit = compile_block_graph(_moved_then_capped(x=x)).generate_stim_circuit(k=1)
    circuit.detector_error_model(decompose_errors=False)


def _cap_beside_a_column_at(x: int, kind: str = "ZXZ") -> BlockGraph:
    """Build a Y cap sharing its z-slice with a continuing column, at block ``x``."""
    g = BlockGraph(f"cap beside column at x={x}")
    a, b = x, x + 1
    for z in range(3):
        g.add_cube(Position3D(a, 0, z), ZXCube.from_str(kind))
    g.add_cube(Position3D(b, 0, 1), ZXCube.from_str(kind))
    g.add_cube(Position3D(b, 0, 2), LeafCubeKind.Y_HALF_CUBE)
    g.add_pipe(Position3D(a, 0, 0), Position3D(a, 0, 1))
    g.add_pipe(Position3D(a, 0, 1), Position3D(a, 0, 2))
    g.add_pipe(Position3D(a, 0, 1), Position3D(b, 0, 1))
    g.add_pipe(Position3D(b, 0, 1), Position3D(b, 0, 2))
    return g


@pytest.mark.parametrize("x", [0, 1, 2])
def test_mixed_slice_away_from_the_origin(x: int) -> None:
    """A Y cap coexisting with a memory cube compiles wherever the pair sits.

    Regression: ``LayoutLayer._mixed_to_circuit`` placed its plaquette sublayers
    in the absolute qubit frame but its raw sublayers relative to the layer's own
    bounds. With a layer minimum of one, the plaquette cube at ``bp = 1`` and the
    raw cube at ``bp = 2`` were emitted onto the same qubits
    (``MultipleOperationsOnSameQubitError``); with a minimum of two the cap landed
    a block away from where its seam detectors looked for it.
    """
    circuit = compile_block_graph(_cap_beside_a_column_at(x)).generate_stim_circuit(k=1)
    circuit.detector_error_model(decompose_errors=False)
