"""Defines :func:`~.compile.compile_block_graph`."""

import warnings
from collections.abc import Iterable
from dataclasses import replace
from typing import Final, Literal

from tqec.compile.blocks.block import Block
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.atomic.raw import RawCircuitLayer
from tqec.compile.blocks.layers.composed.base import BaseComposedLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.blocks.positioning import LayoutPosition3D
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
    ZXEdge,
    find_correlation_surfaces,
)
from tqec.computation.cube import ConditionalLeafCubeKind, Cube, LeafCubeKind
from tqec.templates.base import RectangularTemplate
from tqec.utils.exceptions import TQECError
from tqec.utils.position import BlockPosition3D, Direction3D, Position3D
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

_DEFAULT_SCALABLE_QUBIT_SHAPE: Final = PhysicalQubitScalable2D(
    LinearFunction(4, 5), LinearFunction(4, 5)
)

_DEFAULT_BLOCK_REPETITIONS: LinearFunction = LinearFunction(2, -1)


def _branch_independent_correlation_surfaces(
    block_graph: BlockGraph,
) -> list[CorrelationSurface]:
    """Return a generating set of the correlation surfaces that touch no conditional cube.

    A surface is valid when it is valid at every cube it spans, and its validity
    at a cube depends on that cube's kind alone. So a surface of the graph with
    every conditional cube fixed to its first branch that touches none of them
    is a surface of the graph whatever branch each cube takes. One that touches
    a conditional cube reads it in a basis that depends on the branch, and can
    only be a :class:`ConditionalCorrelationSurface`.

    The surfaces found on the first-branch graph generate all of its surfaces,
    but the ones among them that avoid the conditional cubes need not generate
    all of those that do: two that both cross a cube can XOR to one that does
    not. Gaussian elimination over GF(2), on the edges incident to the
    conditional cubes, recovers a generating set of the surfaces that avoid them.

    Args:
        block_graph: a graph with conditional cubes and no open port.

    Returns:
        a generating set of the surfaces of ``block_graph`` that touch no
        conditional cube.

    """
    conditional = {cube.position for cube in block_graph.cubes if cube.is_conditional}

    def crossing(surface: CorrelationSurface) -> frozenset[ZXEdge]:
        return frozenset(
            edge
            for edge in surface.span
            if edge.u.position in conditional or edge.v.position in conditional
        )

    def lead(edges: frozenset[ZXEdge]) -> ZXEdge:
        return max(edges)

    pivots: dict[ZXEdge, tuple[frozenset[ZXEdge], CorrelationSurface]] = {}
    avoiding: list[CorrelationSurface] = []
    branch_zero = _resolve_conditional_cubes(block_graph, 0)
    for found in find_correlation_surfaces(branch_zero.to_zx_graph()):
        surface, edges = found, crossing(found)
        while edges and lead(edges) in pivots:
            pivot_edges, pivot_surface = pivots[lead(edges)]
            edges, surface = edges ^ pivot_edges, surface ^ pivot_surface
        if edges:
            pivots[lead(edges)] = (edges, surface)
        elif surface.span:
            avoiding.append(surface)
    if pivots:
        warnings.warn(
            f'observables="auto" leaves out {len(pivots)} independent correlation '
            "surface(s) that cross a conditional cube: they depend on its branch. "
            "Give them explicitly, as ConditionalCorrelationSurface.",
            stacklevel=3,
        )
    return avoiding


def _resolve_conditional_cubes(
    bg: BlockGraph, branch_assignment: "dict[Position3D, int] | int"
) -> BlockGraph:
    """Return a copy of the graph with every conditional cube fixed to one branch.

    Return a BlockGraph copy in which every conditional cube is replaced by
    one of its branch kinds.

    ``branch_assignment`` may be:

    - an ``int`` (legacy single-branch path): every conditional cube takes the
      same branch index.
    - a ``dict[Position3D, int]`` mapping each conditional cube's position to
      its selected branch index (0 or 1). A conditional cube not present in the
      dict takes branch 0: the caller only lists the cubes it cares about, and
      must not read the others (a surface that does is rejected upstream, in
      :func:`compile_block_graph`).

    The result has only ZXCube-kinded cubes, so the existing observable-
    compilation helper (which asserts ZXCube) can consume it directly.
    """
    new_bg = BlockGraph(bg.name)
    is_legacy_int = isinstance(branch_assignment, int)
    for cube in bg.cubes:
        kind = cube.kind
        # ``isinstance`` rather than ``cube.is_conditional``: it narrows ``kind``
        # to the pair of ZXCube kinds indexed just below.
        if isinstance(kind, ConditionalLeafCubeKind):
            if is_legacy_int:
                idx = branch_assignment  # type: ignore[assignment]
            else:
                idx = branch_assignment.get(cube.position, 0)  # type: ignore[union-attr]
            # The chosen branch is an unconditional ZXCube, so the condition that
            # selected it is spent and must not travel with the cube.
            new_bg.insert_cube(replace(cube, kind=kind.value[idx], condition=None))
            continue
        new_bg.insert_cube(cube)
    for pipe in bg.pipes:
        new_bg.add_pipe(pipe.u.position, pipe.v.position, pipe.kind)
    return new_bg


def _classify_conditions(
    bg: BlockGraph, surface: "ConditionalCorrelationSurface"
) -> tuple[_ConditionBinding, ...]:
    """Bind each condition of a surface to a conditional cube or an anchor position.

    Bind each entry of ``surface.conditions`` either to an existing
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
        matches = [c.position for c in bg.cubes if c.is_conditional and c.condition == cond]
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
            zs = [node.position.z for edge in cond.span for node in (edge.u, edge.v)]
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


def _positions(positions: Iterable[Position3D]) -> str:
    """Format positions for an error message, sorted, as ``(x,y,z)``."""
    return ", ".join(str(p) for p in sorted(positions))


def _reject_conditional_cubes_beside_raw_cubes(bg: BlockGraph) -> None:
    """Reject a conditional cube in the same z-slice as a Y cube or an injection cube.

    Either cube gives the slice a temporal schedule of its own, and a slice of
    mismatched schedules is flattened at the concrete ``k``, which the
    conditional emission does not support. Checked here, rather than only when
    the slice is merged, so that :func:`compile_block_graph` fails instead of
    the first circuit generation, and with the cubes' own positions.

    Raises:
        NotImplementedError: if a conditional cube shares its z-slice with a Y
            cube or an injection cube.

    """
    raw_by_z: dict[int, list[Position3D]] = {}
    for cube in bg.cubes:
        if cube.is_y_cube or cube.kind is LeafCubeKind.INJECTION:
            raw_by_z.setdefault(cube.position.z, []).append(cube.position)
    for cube in bg.cubes:
        beside = raw_by_z.get(cube.position.z) if cube.is_conditional else None
        if beside:
            raise NotImplementedError(
                f"The conditional cube at {cube.position} shares its z-slice with the "
                f"Y or injection cube(s) at {_positions(beside)}, whose temporal schedule "
                "differs from it. A conditional cube cannot sit in a z-slice of "
                "mismatched schedules: move it to a z-slice of its own."
            )


def _reject_y_readout(observable: AbstractObservable, what: str, minz: int) -> None:
    """Reject a conditional surface that reads a Y-basis measurement.

    A Y cube's logical readout is its transition-round records, which only the
    plain observable annotator emits; the condition resolver and the
    conditional observable annotator would ask the observable builder for a
    data readout on a cube that has none.

    Args:
        observable: the compiled surface, in the graph's shifted frame.
        what: how to name the surface in the error message.
        minz: the shift applied to the graph, added back to the positions reported.

    Raises:
        NotImplementedError: if the surface ends on a Y-basis measurement.

    """
    y_cubes = _positions(
        c.cube.position.shift_by(dz=minz) for c in observable.top_readout_cubes if c.cube.is_y_cube
    )
    if y_cubes:
        raise NotImplementedError(
            f"{what} ends on the Y-basis measurement(s) at {y_cubes}. Reading a "
            "Y-basis measurement is only supported in a plain correlation surface, "
            "not in a condition or a ConditionalCorrelationSurface."
        )


def compile_block_graph(
    block_graph: BlockGraph,
    convention: Convention = FIXED_BULK_CONVENTION,
    observables: (
        list[CorrelationSurface | ConditionalCorrelationSurface] | Literal["auto"] | None
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
            determined from the block graph. With conditional cubes, those are
            the surfaces that touch no conditional cube, which are observables
            whichever branch each cube takes; one that crosses a conditional
            cube depends on its branch and must be given explicitly, as a
            :class:`~tqec.computation.correlation.ConditionalCorrelationSurface`.
            If a list of correlation surfaces is provided, only those surfaces
            will be compiled into observables
            and included in the compiled circuit. If set to ``None``, no
            observables will be included in the compiled circuit. Entries may be
            :class:`~tqec.computation.correlation.ConditionalCorrelationSurface`;
            those are only emitted by
            :meth:`~tqec.compile.graph.TopologicalComputationGraph.generate_conditional_stim_text`.
        block_temporal_height: the number of rounds of stabilizer measurements
            (ignoring one layer for initialization and another for final measurement).
            Defaults to `2k-1`.

    Returns:
        A :class:`TopologicalComputationGraph` object that can be used to generate a
        ``stim.Circuit`` and scale easily.

    Raises:
        TQECError: if the graph has open ports or is not valid, or if a resolution
            of a :class:`~tqec.computation.correlation.ConditionalCorrelationSurface`
            reaches a conditional cube none of its conditions selects.
        NotImplementedError: if a conditional cube shares its z-slice with a Y cube
            or an injection cube, or if a condition or a
            :class:`~tqec.computation.correlation.ConditionalCorrelationSurface`
            reads a Y-basis measurement.

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
    _reject_conditional_cubes_beside_raw_cubes(block_graph)

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
    # Surfaces "auto" finds are valid by construction; one found beside a
    # conditional cube cannot be re-checked on the graph, which PyZX cannot convert.
    skip_validation = False
    if observables is not None:
        if observables == "auto":
            # Deliberately not ``block_graph.find_correlation_surfaces()``: that
            # raises when the graph has no deterministic observable, which is the
            # right answer for a user asking for one explicitly but not here.
            # ``"auto"`` means "include whatever deterministic observables exist",
            # and a computation may legitimately have none -- a Y-basis
            # measurement cap reads out at random by construction.
            if any(cube.is_conditional for cube in block_graph.cubes):
                observables = list(_branch_independent_correlation_surfaces(block_graph))
                skip_validation = True
            else:
                observables = find_correlation_surfaces(block_graph.to_zx_graph())
        else:
            observables = [cs.shift_by(dz=-minz) for cs in observables]
        include_temporal_hadamard_pipes = convention.name == "fixed_bulk"
        for surface in observables:
            if isinstance(surface, ConditionalCorrelationSurface):
                warnings.warn(
                    "ConditionalCorrelationSurface: the per-branch resolutions "
                    "are not checked to be valid correlation surfaces of the "
                    "graph with each branch substituted in.",
                    stacklevel=2,
                )
                # Bind each surface condition either to an existing
                # conditional cube (Cube.condition match) or to a past
                # measurement string (surface-anchored). Ordering of bindings
                # follows surface.conditions; it defines the bit order of
                # resolution keys.
                bindings = _classify_conditions(block_graph, surface)
                cube_bits = [i for i, b in enumerate(bindings) if b.cube_position is not None]
                # Every other conditional cube is fixed to an arbitrary branch
                # below, which is only sound if no resolution reaches it.
                bound = {bindings[i].cube_position for i in cube_bits}
                unbound = {
                    cube.position
                    for cube in block_graph.cubes
                    if cube.is_conditional and cube.position not in bound
                }
                for key, resolution in surface.resolutions.items():
                    reached = unbound & resolution.positions
                    if reached:
                        positions = _positions(p.shift_by(dz=minz) for p in reached)
                        raise TQECError(
                            f"The resolution {key} of a ConditionalCorrelationSurface "
                            f"reaches the conditional cube(s) at {positions}, whose "
                            "conditions are not among the surface's conditions. Add "
                            "them to the surface's conditions."
                        )
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
                        bg_for_compile = _resolve_conditional_cubes(block_graph, assignment)
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
                for key, branch in branches.items():
                    _reject_y_readout(
                        branch, f"The resolution {key} of a ConditionalCorrelationSurface", minz
                    )
                for i, condition in enumerate(resolved_conditions):
                    _reject_y_readout(
                        condition, f"Condition {i} of a ConditionalCorrelationSurface", minz
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
                        block_graph,
                        surface,
                        include_temporal_hadamard_pipes,
                        _skip_validation=skip_validation,
                    )
                )

    # 0.5 Pre-compile each conditional cube's condition surface into an
    # AbstractObservable. Doing it here (where the BlockGraph is in scope)
    # lets the resolver run at emission time without needing the BlockGraph.
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
    for cube in block_graph.cubes:
        if cube.is_conditional and cube.condition is not None:
            layout_position = LayoutPosition3D.from_block_position(
                BlockPosition3D(cube.position.x, cube.position.y, cube.position.z)
            )
            _reject_y_readout(
                conditional_observables[layout_position],
                f"The condition of the conditional cube at {cube.position.shift_by(dz=minz)}",
                minz,
            )

    # 1. Create topological computation graph
    graph = TopologicalComputationGraph(
        _DEFAULT_SCALABLE_QUBIT_SHAPE,
        observables=obs_included,
        observable_builder=convention.triplet.observable_builder,
        conditional_observables=conditional_observables,
        conditional_abstract_observables=cond_obs_included,
        z_offset=minz,
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
