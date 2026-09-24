from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from tqec.compile.blocks.block import Block
from tqec.compile.specs.enums import SpatialArms
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import Cube, CubeKind, ZXCube
from tqec.computation.pipe import PipeKind
from tqec.templates.base import RectangularTemplate
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.injection_state import DEFAULT_INJECTION_STATE
from tqec.utils.position import Direction3D, Position3D
from tqec.utils.scale import LinearFunction

if TYPE_CHECKING:
    from tqec.computation.correlation import CorrelationSurface


def _y_cube_neighbour(cube: Cube, graph: BlockGraph) -> tuple[ZXCube, bool]:
    """Return the regular cube a ``Y_HALF_CUBE`` attaches to, and its direction.

    A Y cube is either a measurement *cap*, sitting on top of the cube it reads
    out, or a Y-basis *initialisation*, sitting underneath the cube it feeds. It
    runs on that cube's patch either way, so the neighbour decides the cap's
    orientation; the direction decides which way round the construction runs.

    Returns:
        the neighbouring cube's kind, and ``True`` when the Y cube is an
        initialisation (the neighbour is *above* it).

    Raises:
        NotImplementedError: if the Y cube has no regular cube on either
            temporal side --- a Y cube attached only to a ``Port``, which is not
            lowered.

    """
    position = cube.position
    below = Position3D(position.x, position.y, position.z - 1)
    if graph.has_pipe_between(below, position):
        below_kind = graph[below].kind
        if isinstance(below_kind, ZXCube):
            return below_kind, False
    above = Position3D(position.x, position.y, position.z + 1)
    if graph.has_pipe_between(position, above):
        above_kind = graph[above].kind
        if isinstance(above_kind, ZXCube):
            return above_kind, True
    raise NotImplementedError(
        f"The Y cube at {position} has no regular cube directly below or above "
        "it, so it is neither a Y-basis measurement cap nor a Y-basis "
        "initialisation. A Y cube connected only to a Port is not implemented."
    )


def _y_capped_cube_kind(cube: Cube, graph: BlockGraph) -> ZXCube:
    """Return the kind of the regular cube a ``Y_HALF_CUBE`` attaches to."""
    return _y_cube_neighbour(cube, graph)[0]


def _y_cube_is_initialisation(cube: Cube, graph: BlockGraph) -> bool:
    """Whether a ``Y_HALF_CUBE`` initialises rather than measures.

    ``False`` for any cube that is not a Y cube.
    """
    if not cube.is_y_cube:
        return False
    return _y_cube_neighbour(cube, graph)[1]


def _y_cap_is_transposed(cube: Cube, graph: BlockGraph) -> bool:
    """Whether a ``Y_HALF_CUBE``'s patch must be reflected across its main diagonal.

    A Y cap continues the patch of the cube below it, so it inherits that cube's
    spatial orientation. Gidney's construction is written for a ``ZX*`` cube
    (spatial boundaries normal to ``x`` in ``Z``); an ``XZ*`` cube needs the
    reflection. Returns ``False`` for any cube that is not a Y cube.
    """
    if not cube.is_y_cube:
        return False
    return _y_capped_cube_kind(cube, graph).x == Basis.X


def _injection_capped_cube_kind(cube: Cube, graph: BlockGraph) -> ZXCube:
    """Return the kind of the cube an ``INJECTION`` cube hands its state to.

    Raises:
        NotImplementedError: if the injection cube does not sit directly below a
            regular cube. ``BlockGraph._validate_locally_at_cube`` already rejects
            every such graph --- no upward temporal pipe, or one leading somewhere
            other than a ``ZXCube`` --- so reaching this means a spec was built
            from a graph that was never validated.

    """
    above = Position3D(cube.position.x, cube.position.y, cube.position.z + 1)
    above_kind = graph[above].kind if graph.has_pipe_between(cube.position, above) else None
    if not isinstance(above_kind, ZXCube):
        raise NotImplementedError(
            f"The injection cube at {cube.position} does not hand its state to a "
            "regular cube. Only injection into a ZX cube directly above is "
            "implemented; an injection cube connected to a Port is not."
        )
    return above_kind


def _injection_is_transposed(cube: Cube, graph: BlockGraph) -> bool:
    """Whether an ``INJECTION`` cube's patch must be reflected across its main diagonal.

    An injection cube prepares the patch of the cube above it, so it inherits
    that cube's spatial orientation. The encoder is written for a ``ZX*`` cube
    (spatial boundaries normal to ``x`` in ``Z``); an ``XZ*`` cube needs the
    reflection. Returns ``False`` for any cube that is not an injection cube.
    """
    if not cube.is_injection_cube:
        return False
    return _injection_capped_cube_kind(cube, graph).x == Basis.X


@dataclass(frozen=True)
class CubeSpec:
    """Specification of a cube in a block graph.

    The template of the `CompiledBlock` will be determined based on the specification.
    This class can be used as a key to look up the corresponding `CompiledBlock` before
    applying the substitution rules.

    Attributes:
        cube_kind: The kind of the cube.
        spatial_arms: Flag indicating the spatial directions the cube connects to the
            adjacent cubes. This is useful for spatial cubes (XXZ and ZZX) where
            the arms can determine the template used to implement the cube.
        has_spatial_up_or_down_pipe_in_timeslice: a flag indicating if a spatial
            pipe at the top or bottom of a spatial cube is executed on the same
            timeslice as this cube. This information is needed for the fixed
            boundary convention.
        condition: The correlation surface carried over from a conditional ``Cube``;
            ``None`` for non-conditional specs.
        y_cap_transposed: For a ``Y_HALF_CUBE`` only: whether the cap's patch is
            reflected across its main diagonal. The Y cap runs on the patch of
            the cube below it, and Gidney's construction is written for a ``ZX*``
            cube (left/right boundaries in ``Z``). An ``XZ*`` cube below has those
            boundaries in ``X`` and needs the reflected patch. ``False`` for every
            other cube kind.
        y_cube_initialises: For a ``Y_HALF_CUBE`` only: whether it is a Y-basis
            *initialisation* (the regular cube it attaches to sits above it)
            rather than a measurement cap (the regular cube sits below).
            ``False`` for every other cube kind.
        injection_transposed: For an ``INJECTION`` cube only: whether the encoder's
            patch is reflected across its main diagonal, for the same reason as
            ``y_cap_transposed``, read off the cube *above* instead of below.
            ``False`` for every other cube kind.
        state: For an ``INJECTION`` cube only: which single-qubit state to inject.
            See :py:attr:`~tqec.computation.cube.Cube.state`.

    """

    kind: CubeKind
    spatial_arms: SpatialArms = SpatialArms.NONE
    has_spatial_up_or_down_pipe_in_timeslice: bool = False
    condition: CorrelationSurface | None = None
    y_cap_transposed: bool = False
    y_cube_initialises: bool = False
    injection_transposed: bool = False
    state: str = DEFAULT_INJECTION_STATE

    def __post_init__(self) -> None:
        if self.spatial_arms != SpatialArms.NONE:
            if not self.is_spatial:
                raise TQECError(
                    "The `spatial_arms` attribute should be `SpatialArms.NONE` "
                    "for non-spatial cubes."
                )

    @property
    def is_spatial(self) -> bool:
        """Return ``True`` if ``self`` represents a spatial cube."""
        return isinstance(self.kind, ZXCube) and self.kind.is_spatial

    @staticmethod
    def from_cube(
        cube: Cube,
        graph: BlockGraph,
        spatial_up_or_down_pipes_slices: frozenset[int] = frozenset(),
    ) -> CubeSpec:
        """Return the cube spec from a cube in a block graph."""
        has_spatial_up_or_down_pipe_in_timeslice = (
            cube.position.z in spatial_up_or_down_pipes_slices
        )
        if not cube.is_spatial:
            return CubeSpec(
                cube.kind,
                has_spatial_up_or_down_pipe_in_timeslice=has_spatial_up_or_down_pipe_in_timeslice,
                condition=cube.condition,
                y_cap_transposed=_y_cap_is_transposed(cube, graph),
                y_cube_initialises=_y_cube_is_initialisation(cube, graph),
                injection_transposed=_injection_is_transposed(cube, graph),
                state=cube.state,
            )
        spatial_arms = SpatialArms.from_cube_in_graph(cube, graph)
        return CubeSpec(
            cube.kind,
            spatial_arms,
            has_spatial_up_or_down_pipe_in_timeslice,
            condition=cube.condition,
        )

    @property
    def pipe_dimensions(self) -> frozenset[Literal[Direction3D.X, Direction3D.Y]]:
        """Return the dimension(s) in which ``self`` has at least one pipe."""
        dimensions: list[Literal[Direction3D.X, Direction3D.Y]] = []
        if SpatialArms.LEFT in self.spatial_arms or SpatialArms.RIGHT in self.spatial_arms:
            dimensions.append(Direction3D.X)
        if SpatialArms.UP in self.spatial_arms or SpatialArms.DOWN in self.spatial_arms:
            dimensions.append(Direction3D.Y)
        return frozenset(dimensions)

    @property
    def has_spatial_pipe_in_both_dimensions(self) -> bool:
        """Return ``True`` if the provided spec has a pipe in each of the two spatial dimensions."""
        return self.spatial_arms.has_spatial_arm_in_both_dimensions


class CubeBuilder(Protocol):
    """Protocol for building a `Block` based on a `CubeSpec`."""

    def __call__(self, spec: CubeSpec, block_temporal_height: LinearFunction) -> Block:
        """Build a ``Block`` instance from a ``CubeSpec``.

        Args:
            spec: Specification of the cube in the block graph.
            block_temporal_height: the number of rounds of stabilizer measurements
            (ignoring one layer for initialization and another for final measurement).

        Returns:
            a ``Block`` based on the provided ``CubeSpec``.

        """
        ...


class PipeBuilder(Protocol):
    """Protocol for building a `Block` based on a `PipeSpec`."""

    def __call__(self, spec: PipeSpec, block_temporal_height: LinearFunction) -> Block:
        """Build a `CompiledBlock` instance from a `PipeSpec`.

        Args:
            spec: Specification of the cube in the block graph.
            block_temporal_height: the number of rounds of stabilizer measurements
            (ignoring one layer for initialization and another for final measurement).

        Returns:
            a `CompiledBlock` based on the provided `PipeSpec`.

        """
        ...


@dataclass(frozen=True)
class PipeSpec:
    """Specification of a pipe in a block graph.

    The `PipeSpec` is used to determine the substitution rules between the two
    `CompiledBlock`s connected by the pipe. The substitution rules are used to
    update the layers of the `CompiledBlock`s based on the plaquettes in the
    `Substitution`.

    Attributes:
        cube_specs: the ordered cube specifications. By convention, the cube
            corresponding to ``cube_specs[0]`` should have a smaller position
            than the cube corresponding to ``cube_specs[1]``.
        cube_templates: templates used to implement the respective entry in
            ``cube_specs``.
        pipe_type: the type of the pipe connecting the two cubes.
        has_spatial_up_or_down_pipe_in_timeslice: a flag indicating if a spatial
            pipe at the top or bottom of a spatial cube is executed on the same
            timeslice as this cube. This information is needed for the fixed
            boundary convention.
        at_temporal_hadamard_layer: flag indicating whether the pipe is a temporal
            pipe and there is a temporal Hadamard pipe at the same Z position
            in the block graph.

    """

    cube_specs: tuple[CubeSpec, CubeSpec]
    cube_templates: tuple[RectangularTemplate, RectangularTemplate]
    pipe_kind: PipeKind
    has_spatial_up_or_down_pipe_in_timeslice: bool = False
    at_temporal_hadamard_layer: bool = False
