from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Final, cast

from typing_extensions import override

from tqec.compile.blocks.enums import SpatialBlockBorder, TemporalBlockBorder
from tqec.compile.blocks.layers.atomic.base import BaseLayer
from tqec.compile.blocks.layers.atomic.layout import LayoutLayer
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.blocks.layers.composed.base import BaseComposedLayer
from tqec.compile.blocks.layers.composed.repeated import RepeatedLayer
from tqec.compile.blocks.layers.composed.sequenced import SequencedLayers
from tqec.compile.blocks.layers.merge import (
    contains_only_base_layers,
    contains_only_composed_layers,
    merge_base_layers,
    merge_composed_layers,
)
from tqec.compile.blocks.positioning import LayoutPosition2D
from tqec.utils.exceptions import TQECError
from tqec.utils.scale import LinearFunction, PhysicalQubitScalable2D

if TYPE_CHECKING:
    from tqec.computation.correlation import CorrelationSurface

_MEASUREMENT_INSTR_NAMES: Final[frozenset[str]] = frozenset(
    {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}
)


class Block(SequencedLayers):
    """Encodes the implementation of a block.

    This data structure is voluntarily very generic. It represents blocks as a
    sequence of layers that can be instances of either
    :class:`~tqec.compile.blocks.layers.atomic.base.BaseLayer` or
    :class:`~tqec.compile.blocks.layers.composed.base.BaseComposedLayer`.

    Depending on the stored layers, this class can be used to represent regular
    cubes (i.e. scaling in the 3 dimensions with ``k``) as well as pipes (i.e.
    scaling in only 2 dimension with ``k``).

    """

    @override
    def with_spatial_borders_trimmed(self, borders: Iterable[SpatialBlockBorder]) -> Block:
        return Block(
            self._layers_with_spatial_borders_trimmed(borders),
            self.trimmed_spatial_borders | frozenset(borders),
        )

    @override
    def with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> Block | None:
        if not border_replacements:
            return self
        layers = self._layers_with_temporal_borders_replaced(border_replacements)
        return Block(layers) if layers else None

    def get_atomic_temporal_border(self, border: TemporalBlockBorder) -> BaseLayer:
        """Get the layer at the provided temporal ``border``.

        This method is different to :meth:`get_temporal_layer_on_border` in that it raises when the
        border is not an atomic layer.

        Raises:
            TQECError: if the layer at the provided temporal ``border`` is not atomic (i.e., an
                instance of :class:`.BaseLayer`).

        """
        layer_index: int
        match border:
            case TemporalBlockBorder.Z_NEGATIVE:
                layer_index = 0
            case TemporalBlockBorder.Z_POSITIVE:
                layer_index = -1
        layer = self.layer_sequence[layer_index]
        if not isinstance(layer, BaseLayer):
            raise TQECError(
                "Expected to recover a temporal **border** (i.e. an atomic "
                f"layer) but got an instance of {type(layer).__name__} instead."
            )
        return layer

    @cached_property
    def dimensions(self) -> tuple[LinearFunction, LinearFunction, LinearFunction]:
        """Return the dimensions of ``self``.

        Returns:
            a 3-dimensional tuple containing the width for each of the
            ``(x, y, z)`` dimensions.

        """
        spatial_shape = self.scalable_shape
        return spatial_shape.x, spatial_shape.y, self.scalable_timesteps

    @property
    def is_cube(self) -> bool:
        """Return ``True`` if ``self`` represents a cube, else ``False``.

        A cube is defined as a block with all its 3 dimensions that are scalable.

        """
        return all(dim.is_scalable() for dim in self.dimensions)

    @property
    def is_pipe(self) -> bool:
        """Return ``True`` if ``self`` represents a pipe, else ``False``.

        A pipe is defined as a block with all but one of its 3 dimensions that are scalable.

        """
        return sum(dim.is_scalable() for dim in self.dimensions) == 2

    @property
    def is_temporal_pipe(self) -> bool:
        """Return ``True`` if ``self`` is a temporal pipe, else ``False``.

        A temporal pipe is a pipe (exactly 2 scalable dimensions) for which the non-scalable
        dimension is the third one (time dimension).

        """
        return self.is_pipe and self.dimensions[2].is_constant()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Block) and super().__eq__(value)

    def __hash__(self) -> int:
        raise NotImplementedError(f"Cannot hash efficiently a {type(self).__name__}.")


def _plaquette_layer_meas_signature(layer: PlaquetteLayer) -> dict[int, int]:
    """Per-plaquette-index measurement-instruction count for a plaquette layer."""
    return {
        idx: sum(
            len(inst.target_groups())
            for moment in plaquette.circuit.moments
            for inst in moment.instructions
            if inst.name in _MEASUREMENT_INSTR_NAMES
        )
        for idx, plaquette in layer.plaquettes.collection.items()
    }


def _block_meas_signature(block: Block) -> list[dict[int, int]]:
    """Per-layer plaquette-meas signature for a block.

    Recurses into :class:`RepeatedLayer` once; each entry in the returned list
    corresponds to one element in ``block.layer_sequence``.  Two blocks whose
    signatures match produce structurally identical measurement schedules
    under Canonical Emission Order.
    """
    sig: list[dict[int, int]] = []
    for layer in block.layer_sequence:
        if isinstance(layer, PlaquetteLayer):
            sig.append(_plaquette_layer_meas_signature(layer))
        elif isinstance(layer, RepeatedLayer):
            inner = layer.internal_layer
            if isinstance(inner, PlaquetteLayer):
                sig.append(_plaquette_layer_meas_signature(inner))
            else:
                sig.append({})
        else:
            sig.append({})
    return sig


class ConditionalBlock(Block):
    """Block whose execution depends on a runtime measurement outcome.

    Carries two sibling :class:`Block` instances --- one per branch of the
    enclosing :class:`~tqec.computation.cube.ConditionalLeafCubeKind`.
    Downstream emission code is expected to special-case this type to produce
    ``IF/ELSE`` wrapped Stim output.  Until that wiring lands, the inherited
    :class:`Block` API exposes the false-branch layer sequence so vanilla
    emission still produces a one-branch circuit.

    Construction enforces the Equal Measurement Count assumption structurally:
    both branches must share the same plaquette-meas signature per layer.
    """

    def __init__(
        self,
        block_if_zero: Block,
        block_if_one: Block,
        condition: CorrelationSurface,
    ) -> None:
        if len(block_if_zero.layer_sequence) != len(block_if_one.layer_sequence):
            raise TQECError(
                f"{type(self).__name__} requires both branches to have the "
                f"same number of layers.  Got {len(block_if_zero.layer_sequence)} "
                f"vs {len(block_if_one.layer_sequence)}."
            )
        sig_zero = _block_meas_signature(block_if_zero)
        sig_one = _block_meas_signature(block_if_one)
        if sig_zero != sig_one:
            raise TQECError(
                f"{type(self).__name__} requires both branches to share the "
                "same per-layer measurement signature (Equal Measurement Count "
                f"assumption).  zero={sig_zero}  one={sig_one}."
            )
        super().__init__(
            block_if_zero.layer_sequence, block_if_zero.trimmed_spatial_borders
        )
        self._block_if_zero = block_if_zero
        self._block_if_one = block_if_one
        self._condition = condition

    @property
    def block_if_zero(self) -> Block:
        """The Block executed when the condition evaluates to zero."""
        return self._block_if_zero

    @property
    def block_if_one(self) -> Block:
        """The Block executed when the condition evaluates to one."""
        return self._block_if_one

    @property
    def condition(self) -> CorrelationSurface:
        """Correlation surface whose Z outcome selects the active branch."""
        return self._condition

    @override
    def with_spatial_borders_trimmed(
        self, borders: Iterable[SpatialBlockBorder]
    ) -> ConditionalBlock:
        borders = tuple(borders)
        return ConditionalBlock(
            self._block_if_zero.with_spatial_borders_trimmed(borders),
            self._block_if_one.with_spatial_borders_trimmed(borders),
            self._condition,
        )

    @override
    def with_temporal_borders_replaced(
        self,
        border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None],
    ) -> ConditionalBlock | None:
        if not border_replacements:
            return self
        new_zero = self._block_if_zero.with_temporal_borders_replaced(border_replacements)
        new_one = self._block_if_one.with_temporal_borders_replaced(border_replacements)
        if new_zero is None or new_one is None:
            return None
        return ConditionalBlock(new_zero, new_one, self._condition)


def merge_parallel_block_layers(
    blocks_in_parallel: Mapping[LayoutPosition2D, Block],
    scalable_qubit_shape: PhysicalQubitScalable2D,
) -> list[LayoutLayer | BaseComposedLayer]:
    """Merge several stacks of layers executed in parallel into one stack of larger layers.

    Args:
        blocks_in_parallel: a 2-dimensional arrangement of blocks. Each of the
            provided block MUST have the exact same duration (also called
            "temporal footprint", or number of atomic layers).
        scalable_qubit_shape: scalable shape of a scalable qubit. Considered
            valid across the whole domain.

    Returns:
        a stack of layers representing the same slice of computation as the
        provided ``blocks_in_parallel``.

    Raises:
        TQECError: if two items from the provided ``blocks_in_parallel`` do
            not have the same temporal footprint.
        NotImplementedError: if the provided blocks cannot be merged due to a
            code branch not being implemented yet (and not due to a logical
            error making the blocks unmergeable).

    """
    if not blocks_in_parallel:
        return []
    internal_layers_schedules = frozenset(
        tuple(layer.scalable_timesteps for layer in block.layer_sequence)
        for block in blocks_in_parallel.values()
    )
    if len(internal_layers_schedules) != 1:
        raise NotImplementedError(
            "merge_parallel_block_layers only supports merging blocks that have "
            "layers with a matching temporal schedule. Found the following "
            "different temporal schedules in the provided blocks: "
            f"{internal_layers_schedules}."
        )
    schedule: Final = next(iter(internal_layers_schedules))
    merged_layers: list[LayoutLayer | BaseComposedLayer] = []
    for i in range(len(schedule)):
        layers = {pos: block.layer_sequence[i] for pos, block in blocks_in_parallel.items()}
        # Branch-``one`` alternates for ConditionalBlock cubes at this
        # timestep. ``layers[pos]`` already holds the zero-branch slice
        # via Block.__init__'s alias.
        conditional_one_layers: dict[LayoutPosition2D, BaseLayer | BaseComposedLayer] = {
            pos: block.block_if_one.layer_sequence[i]
            for pos, block in blocks_in_parallel.items()
            if isinstance(block, ConditionalBlock)
        }
        if contains_only_base_layers(layers):
            cond_base: dict[LayoutPosition2D, BaseLayer] = {}
            for pos, alt in conditional_one_layers.items():
                if not isinstance(alt, BaseLayer):
                    raise TQECError(
                        f"ConditionalBlock at {pos}: branch-one layer at "
                        f"timestep {i} is not a BaseLayer while the zero "
                        "side is. Both branches must share structure."
                    )
                cond_base[pos] = alt
            merged_layers.append(
                merge_base_layers(
                    cast(dict[LayoutPosition2D, BaseLayer], layers),
                    scalable_qubit_shape,
                    conditional_layers=cond_base or None,
                )
            )
        elif contains_only_composed_layers(layers):
            cond_composed: dict[LayoutPosition2D, BaseComposedLayer] = {}
            for pos, alt in conditional_one_layers.items():
                if not isinstance(alt, BaseComposedLayer):
                    raise TQECError(
                        f"ConditionalBlock at {pos}: branch-one layer at "
                        f"timestep {i} is not a BaseComposedLayer while "
                        "the zero side is."
                    )
                cond_composed[pos] = alt
            merged_layers.append(
                merge_composed_layers(
                    cast(dict[LayoutPosition2D, BaseComposedLayer], layers),
                    scalable_qubit_shape,
                    conditional_layers=cond_composed or None,
                )
            )
        else:
            raise RuntimeError(
                f"Found a mix of {BaseLayer.__name__} instances and "
                f"{BaseComposedLayer.__name__} instances in a single temporal "
                f"layer. This should be already checked before. This is a "
                "logical error in the code, please open an issue. Found layers:"
                f"\n{list(layers.values())}"
            )
    return merged_layers
