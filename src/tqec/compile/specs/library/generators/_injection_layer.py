"""The layer and block the ``INJECTION`` cube builder returns.

The encoder itself lives in
:mod:`tqec.compile.specs.library.generators.injection`; this module wraps it as a
:class:`~tqec.compile.blocks.layers.atomic.raw.RawCircuitLayer` and as the
:class:`~tqec.compile.blocks.block.Block` the cube builders hand back.

An injection cube holds exactly **one** layer, the encoder. It runs no syndrome
extraction of its own: the memory rounds above it belong to the temporal pipe and
to the cube the pipe leads into. Two consequences shape this module.

**The pipe below is appended, not substituted.** For an ordinary cube, a temporal
pipe replaces the block's border layer --- ``Z_POSITIVE`` for the cube at the
pipe's lower end. A one-layer block has no layer to spare there, and the
substitution machinery would silently overwrite the encoder (a
``RawCircuitLayer`` is a ``BaseLayer``, so ``get_atomic_temporal_border`` returns
it without complaint). :class:`InjectionCubeBlock` appends instead, which is the
mirror image of what
:class:`~tqec.compile.specs.library.generators._ycube_circuit.YHalfCubeBlock`
does for the pipe at its ``Z_NEGATIVE`` border. It also means the memory round
that follows the encoder comes from the pipe, whose kind is derived from the ZX
cube above, so no boundary orientation is assumed here.

**The block is a cube with a constant temporal extent.** ``Block.is_cube``
requires all three dimensions to be scalable, and one layer is not, so
:class:`InjectionCubeBlock` states its cube-ness directly. The Y cap could
instead override ``scalable_timesteps`` because its ``k + 2`` rounds are real;
the encoder genuinely occupies a single timestep, and claiming otherwise would
corrupt the arithmetic that consumes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from typing_extensions import override

from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.compile.blocks.block import Block
from tqec.compile.blocks.enums import SpatialBlockBorder, TemporalBlockBorder
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.raw import RawCircuitLayer
from tqec.compile.blocks.layers.composed.base import BaseComposedLayer
from tqec.compile.specs.library.generators.injection import (
    INJECTION_ENCODER_MOMENTS,
    injection_encoder_circuit,
)
from tqec.compile.specs.library.generators.ycube import xtop_qubit_patch
from tqec.templates.base import RectangularTemplate
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

Coord = tuple[int, int]

_INJECTION_ELEMENT_SHAPE = PhysicalQubitScalable2D(LinearFunction(4, 5), LinearFunction(4, 5))
"""The encoder occupies the same physical footprint as a memory cube."""


class InjectionRawLayer(RawCircuitLayer):
    """The state-injection encoder, as a :class:`RawCircuitLayer`.

    Implements :class:`~tqec.compile.blocks.layers.atomic.raw.FlowSpecLayer` so
    the detector annotator can emit the detectors of the round that *follows*
    this one. The encoder measures nothing, so it carries no detectors itself;
    what it publishes instead is that it leaves every stabilizer of the patch
    prepared, each with an empty list of preparing measurements. The next round
    turns that into one single-record detector per ancilla.

    """

    def __init__(self, transposed: bool = False, proxy: bool = True) -> None:
        """Wrap the encoder as the single layer of an injection cube.

        Args:
            transposed: reflect the patch across its main diagonal, for an
                ``XZ*`` cube above. See
                :func:`~tqec.compile.specs.library.generators.injection.injection_encoder_circuit`.
            proxy: whether to inject through the Clifford proxy gate.

        """
        self._transposed = transposed
        self._proxy = proxy
        super().__init__(
            self._make_scheduled_circuit, _INJECTION_ELEMENT_SHAPE, INJECTION_ENCODER_MOMENTS
        )

    def _make_scheduled_circuit(self, k: int) -> ScheduledCircuit:
        return ScheduledCircuit.from_circuit(
            injection_encoder_circuit(2 * k + 1, transposed=self._transposed, proxy=self._proxy)
        )

    def start_spec(self, k: int) -> dict[Coord, list[Coord]]:
        """Return no stabilizers to close: nothing precedes the encoder.

        See
        :meth:`~tqec.compile.blocks.layers.atomic.raw.FlowSpecLayer.start_spec`.
        """
        return {}

    def end_spec(self, k: int) -> dict[Coord, list[Coord]]:
        """Return every stabilizer of the patch, prepared by no measurement.

        The encoder leaves each stabilizer in its ``+1`` eigenstate --- all of
        them, not just those of one basis, which is what a transversal reset
        would give --- so the next round's measurement of any stabilizer is
        deterministic on its own.

        See
        :meth:`~tqec.compile.blocks.layers.atomic.raw.FlowSpecLayer.end_spec`.
        """
        patch = xtop_qubit_patch(2 * k + 1, self._transposed)
        return {stabilizer.ancilla: [] for stabilizer in patch.stabilizers}

    def observable_spec(self, k: int) -> list[Coord] | None:
        """Return ``None``: the encoder measures nothing.

        See
        :meth:`~tqec.compile.blocks.layers.atomic.raw.FlowSpecLayer.observable_spec`.
        """
        return None

    def reconstruction_spec(self, k: int) -> dict[Coord, list[Coord]] | None:
        """Return ``None``: the encoder measures nothing.

        See
        :meth:`~tqec.compile.blocks.layers.atomic.raw.FlowSpecLayer.reconstruction_spec`.
        """
        return None


class InjectionCubeBlock(Block):
    """A state-injection cube: the encoder, and the pipe round it *gains*.

    See the module docstring for why the temporal pipe above is appended rather
    than substituted, and why this block declares its own cube-ness.

    ``template`` is the block's spatial footprint, needed to build the
    :class:`~tqec.compile.specs.base.PipeSpec` of the temporal pipe above (raw
    layers carry no template of their own). It describes shape only --- the
    orientation lives in the plaquettes, which the pipe supplies --- so carrying
    it here assumes nothing about the boundaries.
    """

    def __init__(
        self,
        layer_sequence: Sequence[BaseLayer | BaseComposedLayer],
        trimmed_spatial_borders: frozenset[SpatialBlockBorder] = frozenset(),
        template: RectangularTemplate | None = None,
    ) -> None:
        """Build an injection cube from its layers.

        Args:
            layer_sequence: the cube's layers: the encoder alone, plus the pipe's
                memory round once a temporal pipe has been substituted in.
            trimmed_spatial_borders: all the spatial borders that have been
                removed from the block. Always empty in practice --- an injection
                cube takes no spatial pipes.
            template: the block's spatial footprint. See the class docstring.

        """
        super().__init__(layer_sequence, trimmed_spatial_borders)
        self._template = template

    @property
    @override
    def declared_template(self) -> RectangularTemplate | None:
        return self._template

    @property
    @override
    def is_cube(self) -> bool:
        # ``Block.is_cube`` asks for three scalable dimensions. The encoder is a
        # single timestep whatever ``k`` is, so the temporal one is constant --
        # yet this is a cube, and occupies a cube's spacetime volume.
        return True

    @property
    @override
    def is_pipe(self) -> bool:
        # Would otherwise be ``True`` for the same reason ``is_cube`` is
        # ``False``: exactly two of the three dimensions scale.
        return False

    @override
    def with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> InjectionCubeBlock | None:
        # Keep the subclass, so a later substitution still appends.
        if not border_replacements:
            return self
        layers = self._layers_with_temporal_borders_replaced(border_replacements)
        if not layers:
            return None
        return InjectionCubeBlock(layers, self.trimmed_spatial_borders, self._template)

    @override
    def with_spatial_borders_trimmed(
        self, borders: Iterable[SpatialBlockBorder]
    ) -> InjectionCubeBlock:
        # Unreachable in a valid block graph: an injection cube is only ever
        # connected by a temporal pipe, so it has no spatial border to trim.
        # Kept consistent anyway, and keeping the subclass matters.
        return InjectionCubeBlock(
            self._layers_with_spatial_borders_trimmed(borders),
            self.trimmed_spatial_borders | frozenset(borders),
            self._template,
        )

    @override
    def _layers_with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> list[BaseLayer | BaseComposedLayer]:
        above = border_replacements.get(TemporalBlockBorder.Z_POSITIVE)
        remaining = {
            border: layer
            for border, layer in border_replacements.items()
            if border is not TemporalBlockBorder.Z_POSITIVE
        }
        layers = super()._layers_with_temporal_borders_replaced(remaining)
        if above is not None:
            layers.append(above)
        return layers


def make_injection_block(
    template: RectangularTemplate | None = None,
    transposed: bool = False,
    proxy: bool = True,
) -> InjectionCubeBlock:
    """Build the block an ``INJECTION`` cube lowers to.

    Args:
        template: the block's spatial footprint. See
            :class:`InjectionCubeBlock`.
        transposed: reflect the patch, for an ``XZ*`` cube above.
        proxy: whether to inject through the Clifford proxy gate.

    Returns:
        a one-layer block holding the encoder.

    """
    return InjectionCubeBlock([InjectionRawLayer(transposed, proxy)], template=template)
