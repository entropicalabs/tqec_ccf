"""Defines :func:`~.compile.compile_block_graph`."""

from typing import Final, Literal

from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.atomic.raw import RawCircuitLayer
from tqec.compile.blocks.layers.composed.base import BaseComposedLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.convention import FIXED_BULK_CONVENTION, Convention
from tqec.compile.graph import TopologicalComputationGraph
from tqec.compile.observables.abstract_observable import (
    AbstractObservable,
    ConditionalAbstractObservable,
    _ConditionBinding,
    compile_correlation_surface_to_abstract_observable,
)
from tqec.compile.specs.base import CubeSpec, PipeSpec
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import (
    ConditionalCorrelationSurface,
    CorrelationSurface,
)
from tqec.computation.cube import Cube
from tqec.templates.base import RectangularTemplate
from tqec.utils.exceptions import TQECError
from tqec.utils.position import BlockPosition3D, Direction3D, Position3D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

_DEFAULT_SCALABLE_QUBIT_SHAPE: Final = PhysicalQubitScalable2D(
    LinearFunction(4, 5), LinearFunction(4, 5)
)

_DEFAULT_BLOCK_REPETITIONS: LinearFunction = LinearFunction(2, -1)


def _resolve_conditional_cubes(
    bg: BlockGraph, branch_assignment: "dict[Position3D, int] | int"
) -> BlockGraph:
    """Return a BlockGraph copy in which every conditional cube is replaced by
    one of its branch kinds.

    ``branch_assignment`` may be:

    - an ``int`` (legacy single-branch path): every conditional cube takes the
      same branch index.
    - a ``dict[Position3D, int]`` mapping each conditional cube's position to
      its selected branch index (0 or 1). Cubes not present in the dict are
      left unchanged — caller must include every conditional cube it cares
      about.

    The result has only ZXCube-kinded cubes, so the existing observable-
    compilation helper (which asserts ZXCube) can consume it directly.
    """
    new_bg = BlockGraph(bg.name)
    is_legacy_int = isinstance(branch_assignment, int)
    for cube in bg.cubes:
        kind = cube.kind
        if cube.is_conditional:
            if is_legacy_int:
                idx = branch_assignment  # type: ignore[assignment]
            else:
                idx = branch_assignment[cube.position]  # type: ignore[index]
            kind = kind.value[idx]
        new_bg.add_cube(cube.position, kind, label=cube.label)
    for pipe in bg.pipes:
        new_bg.add_pipe(pipe.u.position, pipe.v.position, pipe.kind)
    return new_bg


def _classify_conditions(
    bg: BlockGraph, surface: "ConditionalCorrelationSurface"
) -> tuple[_ConditionBinding, ...]:
    """Bind each entry of ``surface.conditions`` either to an existing
    conditional cube (when ``Cube.condition == surface.conditions[i]``) or to
    a surface-anchored position derived from the condition's max measurement
    z.

    Returns one :class:`_ConditionBinding` per surface condition, ordered the
    same as ``surface.conditions`` — that ordering defines the bit order of
    every key in ``surface.resolutions``.

    Raises:
        TQECError: if a condition matches more than one conditional cube
            (ambiguous binding), or if a surface-anchored condition spans an
            empty position range.
    """
    bindings: list[_ConditionBinding] = []
    for i, cond in enumerate(surface.conditions):
        matches = [
            c.position
            for c in bg.cubes
            if c.is_conditional and c.condition == cond
        ]
        if len(matches) > 1:
            raise TQECError(
                f"ConditionalCorrelationSurface.conditions[{i}] matches "
                f"multiple conditional cubes at {matches}; conditions must be "
                "uniquely identifying."
            )
        if len(matches) == 1:
            cube_pos = matches[0]
            bindings.append(
                _ConditionBinding(
                    surface_index=i,
                    anchor_z=cube_pos.z,
                    cube_position=cube_pos,
                )
            )
        else:
            # Surface-anchored: condition gates an observable without a
            # corresponding conditional cube. anchor_z = max-z reached by
            # any node in the condition's surface, +1 so the rec offset is
            # strictly in the past at the emission leaf.
            zs = [
                node.position.z
                for edge in cond.span
                for node in (edge.u, edge.v)
            ]
            if not zs:
                raise TQECError(
                    f"ConditionalCorrelationSurface.conditions[{i}] has an "
                    "empty span; surface-anchored conditions must reference "
                    "at least one measurement."
                )
            bindings.append(
                _ConditionBinding(
                    surface_index=i,
                    anchor_z=max(zs) + 1,
                    cube_position=None,
                )
            )
    return tuple(bindings)


def _get_template_from_layer(
    root: BaseLayer | BaseComposedLayer,
) -> RectangularTemplate | None:
    """Get a unique template from any given layer.

    This helper function try its best to recover the template a given layer uses.

    Raises:
        TQECError: if an instance of :class:`.BaseLayer` is something else than an instance of
            :class:`.PlaquetteLayer`, because :class:`.PlaquetteLayer` is the only class from which
            we can recover a template instance.
        TQECError: if an instance of :class:`.SequencedLayers` contains sub-layers with
            different templates.
        NotImplementedError: if an unknown layer is found.

    Returns:
        the template used by the provided ``root`` layer.

    """
    if isinstance(root, BaseLayer):
        if isinstance(root, RawCircuitLayer):
            # A raw-circuit layer (e.g. one round of the Y-basis measurement cap)
            # carries no Template. It is skipped when recovering a block's
            # template; a block that mixes raw layers with plaquette layers (a Y
            # cap once its junction round has been prepended) yields the
            # plaquette layer's template for the temporal-pipe junction.
            return None
        if not isinstance(root, PlaquetteLayer):
            raise TQECError(
                f"Trying to get the Template from a {type(root).__name__} "
                "instance that does not have any Template."
            )
        return root.template
    elif isinstance(root, SequencedLayers):
        # A block built only from raw layers (the Y-basis measurement cap) has no
        # plaquette layer to recover a template from, so it declares its spatial
        # footprint directly. Checked before walking the layers, which would
        # otherwise find nothing and raise.
        if isinstance(root, Block) and root.declared_template is not None:
            return root.declared_template
        possible_templates = {
            template
            for layer in root.layer_sequence
            if (template := _get_template_from_layer(layer)) is not None
        }
        if len(possible_templates) > 1:
            raise TQECError(
                "Multiple possible Template found:\n  -"
                + "\n  -".join(type(t).__name__ for t in possible_templates)
                + "\nWhich is not supported at the moment."
            )
        if not possible_templates:
            raise TQECError("No Template found among the layers of a block.")
        return next(iter(possible_templates))
    elif isinstance(root, RepeatedLayer):
        return _get_template_from_layer(root.internal_layer)
    else:
        raise NotImplementedError("Unknown layer type encountered:", type(root).__name__)


def compile_block_graph(
    block_graph: BlockGraph,
    convention: Convention = FIXED_BULK_CONVENTION,
    observables: (
        list[CorrelationSurface | ConditionalCorrelationSurface]
        | Literal["auto"]
        | None
    ) = "auto",
    block_temporal_height: LinearFunction = _DEFAULT_BLOCK_REPETITIONS,
) -> TopologicalComputationGraph:
    """Compile a block graph.

    Args:
        block_graph: The block graph to compile.
        convention: convention used to generate the quantum circuits.
        observables: correlation surfaces that should be compiled into
            observables and included in the compiled circuit.
            If set to ``"auto"``, the correlation surfaces will be automatically
            determined from the block graph. If a list of correlation surfaces
            is provided, only those surfaces will be compiled into observables
            and included in the compiled circuit. If set to ``None``, no
            observables will be included in the compiled circuit.
        block_temporal_height: the number of rounds of stabilizer measurements
            (ignoring one layer for initialization and another for final measurement).
            Defaults to `2k-1`.

    Returns:
        A :class:`TopologicalComputationGraph` object that can be used to generate a
        ``stim.Circuit`` and scale easily.

    """
    # All the ports should be filled before compiling the block graph.
    if block_graph.num_ports != 0:
        raise TQECError(
            "Can not compile a block graph with open ports into circuits. "
            "You might want to call `fill_ports` or `fill_ports_for_minimal_simulation` "
            "on the block graph before compiling it."
        )
    # Validate the graph can represent a valid computation.
    block_graph.validate()

    # Fix the shadowed faces of the cubes to avoid using spatial cubes
    # when a non-spatial cube can be used at the same position.
    # For example, when three XXZ cubes are connected in a row along the x-axis,
    # the middle one can be replaced by a ZXX cube because the faces along the
    # x-axis are shadowed by the connected pipes.
    block_graph = block_graph.fix_shadowed_faces()

    # Set the minimum z of block graph to 0.(time starts from zero)
    minz = min(cube.position.z for cube in block_graph.cubes)
    if minz != 0:
        block_graph = block_graph.shift_by(dz=-minz)

    # We need to know exactly which spatial pipes will be placed on a time slice where extended
    # plaquettes will be used, in order to adapt the schedule of the measurement layer.
    def has_pipes_in_both_spatial_dimensions(cube: Cube) -> bool:
        return frozenset(
            pipe.direction for pipe in block_graph.pipes_at(cube.position) if pipe.kind.is_spatial
        ) == frozenset([Direction3D.X, Direction3D.Y])

    extended_stabilizers_pipe_slices: frozenset[int] = frozenset(
        pipe.u.position.z
        for pipe in block_graph.pipes
        if (
            pipe.direction == Direction3D.Y
            and (
                has_pipes_in_both_spatial_dimensions(pipe.u)
                ^ has_pipes_in_both_spatial_dimensions(pipe.v)
            )
        )
    )
    cube_specs = {
        cube: CubeSpec.from_cube(cube, block_graph, extended_stabilizers_pipe_slices)
        for cube in block_graph.cubes
    }

    # 0. Get the abstract observables to be included in the compiled circuit.
    obs_included: list[AbstractObservable] = []
    cond_obs_included: list[ConditionalAbstractObservable] = []
    if observables is not None:
        if observables == "auto":
            observables = block_graph.find_correlation_surfaces()
        else:
            observables = [cs.shift_by(dz=-minz) for cs in observables]
        include_temporal_hadamard_pipes = convention.name == "fixed_bulk"
        for surface in observables:
            if isinstance(surface, ConditionalCorrelationSurface):
                import warnings as _warnings

                _warnings.warn(
                    "ConditionalCorrelationSurface: skipping per-branch surface "
                    "validation against substituted BlockGraph. TODO: replace the "
                    "conditional cube with each branch's ZXCube kind and run "
                    "_check_correlation_surface_validity on each branch.",
                    stacklevel=2,
                )
                # Bind each surface condition either to an existing
                # conditional cube (Cube.condition match) or to a past
                # measurement string (surface-anchored). Ordering of bindings
                # follows surface.conditions; it defines the bit order of
                # resolution keys.
                bindings = _classify_conditions(block_graph, surface)
                cube_bits = [
                    i for i, b in enumerate(bindings) if b.cube_position is not None
                ]
                # Compile every truth-table resolution against a graph that
                # has cube-anchored bits resolved per the key (surface-anchored
                # bits don't affect the BlockGraph topology).
                branches: dict[tuple[bool, ...], AbstractObservable] = {}
                for key, resolution in surface.resolutions.items():
                    if cube_bits:
                        assignment = {
                            bindings[i].cube_position: int(key[i])  # type: ignore[misc]
                            for i in cube_bits
                        }
                        bg_for_compile = _resolve_conditional_cubes(
                            block_graph, assignment
                        )
                    else:
                        bg_for_compile = block_graph
                    branches[key] = compile_correlation_surface_to_abstract_observable(
                        bg_for_compile,
                        resolution,
                        include_temporal_hadamard_pipes,
                        _skip_validation=True,
                    )
                # Pre-compile each condition surface into an AbstractObservable
                # so the resolver can derive rec offsets at emission time.
                resolved_conditions = tuple(
                    compile_correlation_surface_to_abstract_observable(
                        block_graph,
                        surface.conditions[b.surface_index],
                        include_temporal_hadamard_pipes,
                        _skip_validation=True,
                    )
                    for b in bindings
                )
                cond_obs_included.append(
                    ConditionalAbstractObservable(
                        branches=branches,
                        condition_bindings=bindings,
                        resolved_conditions=resolved_conditions,
                    )
                )
            else:
                obs_included.append(
                    compile_correlation_surface_to_abstract_observable(
                        block_graph, surface, include_temporal_hadamard_pipes
                    )
                )

    # 0.5 Pre-compile each conditional cube's condition surface into an
    # AbstractObservable. Doing it here (where the BlockGraph is in scope)
    # lets the resolver run at emission time without needing the BlockGraph.
    from tqec.compile.blocks.positioning import LayoutPosition3D

    conditional_observables = {
        LayoutPosition3D.from_block_position(
            BlockPosition3D(cube.position.x, cube.position.y, cube.position.z)
        ): compile_correlation_surface_to_abstract_observable(
            block_graph,
            cube.condition,
            include_temporal_hadamard_pipes=(convention.name == "fixed_bulk"),
        )
        for cube in block_graph.cubes
        if cube.is_conditional and cube.condition is not None
    }

    # 1. Create topological computation graph
    graph = TopologicalComputationGraph(
        _DEFAULT_SCALABLE_QUBIT_SHAPE,
        observables=obs_included,
        observable_builder=convention.triplet.observable_builder,
        conditional_observables=conditional_observables,
        conditional_abstract_observables=cond_obs_included,
    )

    # 2. Add cubes to the graph
    for cube in block_graph.cubes:
        spec = cube_specs[cube]
        position = BlockPosition3D(cube.position.x, cube.position.y, cube.position.z)
        graph.add_cube(position, convention.triplet.cube_builder(spec, block_temporal_height))

    # 3. Add pipes to the graph
    # Note that the order of the pipes to add is important.
    # To keep the time-direction pipes from removing the extra resets
    # added by the space-direction pipes, we first add the time-direction pipes
    pipes = block_graph.pipes
    time_pipes = [pipe for pipe in pipes if pipe.direction == Direction3D.Z]
    temporal_hadamard_z_positions: set[int] = {
        pipe.u.position.z for pipe in time_pipes if pipe.kind.has_hadamard
    }
    space_pipes = [pipe for pipe in pipes if pipe.direction != Direction3D.Z]
    for pipe in time_pipes + space_pipes:
        pos1, pos2 = pipe.u.position, pipe.v.position
        pos1 = BlockPosition3D(pos1.x, pos1.y, pos1.z)
        pos2 = BlockPosition3D(pos2.x, pos2.y, pos2.z)
        template1 = _get_template_from_layer(graph.get_cube(pos1))
        template2 = _get_template_from_layer(graph.get_cube(pos2))
        key = PipeSpec(
            (cube_specs[pipe.u], cube_specs[pipe.v]),
            (template1, template2),
            pipe.kind,
            has_spatial_up_or_down_pipe_in_timeslice=(
                pos1.z == pos2.z and pos1.z in extended_stabilizers_pipe_slices
            ),
            at_temporal_hadamard_layer=(
                pipe.kind.is_temporal and pos1.z in temporal_hadamard_z_positions
            ),
        )
        graph.add_pipe(pos1, pos2, convention.triplet.pipe_builder(key, block_temporal_height))

    return graph
