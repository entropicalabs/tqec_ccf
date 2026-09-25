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
from tqec.templates.base import RectangularTemplate
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

    @property
    def releases_its_qubits(self) -> bool:
        """Whether the block's last layer measures out every data qubit it owns.

        A block that does is finished when its layers run out: in a merged slice
        whose duration is set by a longer neighbour, its position can simply be
        **absent** from the trailing layers rather than padded with idle rounds.
        A block that does not (an ordinary memory cube, whose patch carries a
        logical state onwards to the next z-layer) must stay present for the
        whole slice and is padded instead.

        Defaults to ``False``, which is always the safe answer --- padding is
        physics-preserving either way, just longer.

        """
        return False

    @property
    def declared_template(self) -> RectangularTemplate | None:
        """The block's spatial footprint, when the block states it itself.

        A block's template is normally recovered from its
        :class:`~tqec.compile.blocks.layers.atomic.plaquettes.PlaquetteLayer`
        layers. A block built only from raw circuits (the Y-basis measurement
        cap) has none, so it declares the footprint here instead. ``None`` for
        every ordinary block.

        """
        return None

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


def _flatten_block_layers(block: Block, k: int) -> list[BaseLayer]:
    """Expand a block's layer sequence into one atomic layer per timestep.

    Every ``RepeatedLayer`` is unrolled at the concrete scaling factor ``k``.
    """
    flat: list[BaseLayer] = []
    for layer in block.layer_sequence:
        if isinstance(layer, RepeatedLayer):
            internal = layer.internal_layer
            if not isinstance(internal, BaseLayer):
                raise NotImplementedError(
                    "Flattening a RepeatedLayer whose internal layer is composed "
                    "is not supported for mismatched-schedule merges."
                )
            flat.extend([internal] * layer.repetitions.integer_eval(k))
        elif isinstance(layer, BaseLayer):
            flat.append(layer)
        else:
            raise NotImplementedError(
                f"Cannot flatten layer of type {type(layer).__name__} for a "
                "mismatched-schedule merge."
            )
    return flat


def _block_pad_body(block: Block) -> BaseLayer:
    """Return the bulk round used to pad a block shorter than the merged slice.

    That round is the internal layer of the block's (single) ``RepeatedLayer``.
    Padding with an extra copy of this round is physics-preserving: for a memory
    cube it is another memory round before the final measurement; for a Y cap it
    is another boundary (padding) round on the degenerate patch before the
    transversal final round.
    """
    repeated = [layer for layer in block.layer_sequence if isinstance(layer, RepeatedLayer)]
    if len(repeated) != 1:
        raise NotImplementedError(
            "Padding a block for a mismatched-schedule merge requires exactly one "
            f"RepeatedLayer to draw the bulk round from; found {len(repeated)}."
        )
    internal = repeated[0].internal_layer
    if not isinstance(internal, BaseLayer):
        raise NotImplementedError("RepeatedLayer internal layer must be atomic to pad.")
    return internal


def _merge_mismatched_block_layers(
    blocks_in_parallel: Mapping[LayoutPosition2D, Block],
    scalable_qubit_shape: PhysicalQubitScalable2D,
    k: int,
) -> list[LayoutLayer | BaseComposedLayer]:
    """Merge parallel blocks whose temporal schedules do not match.

    Each block is flattened at the concrete ``k`` and start-aligned. The merged
    slice runs for ``max`` rounds over the parallel blocks. A block shorter than
    the slice is handled one of two ways, according to
    :attr:`Block.releases_its_qubits`:

    - a block that measures out its data qubits (a Y-basis measurement cap) is
      finished when its layers run out, and is simply **absent** from the
      trailing merged layers;
    - a block that carries a logical state onwards (an ordinary memory cube) must
      stay present, and is **padded** with extra bulk rounds inserted just before
      its final border round so that round stays last.

    This lets a Y cap coexist with a continuing memory cube whatever their
    relative lengths: at small ``k`` the cap outlasts the column and the column is
    padded; at larger ``k`` the column outlasts the cap, which drops out and
    leaves the column to finish the slice alone.
    """
    flats = {pos: _flatten_block_layers(block, k) for pos, block in blocks_in_parallel.items()}
    duration = max(len(flat) for flat in flats.values())
    for pos, flat in flats.items():
        extra = duration - len(flat)
        if not extra or blocks_in_parallel[pos].releases_its_qubits:
            # A block that measures out its data qubits is done when its layers
            # run out; it is simply absent from the trailing merged layers.
            continue
        body = _block_pad_body(blocks_in_parallel[pos])
        flats[pos] = flat[:-1] + [body] * extra + flat[-1:]
    merged: list[LayoutLayer | BaseComposedLayer] = []
    for i in range(duration):
        layers = {pos: flat[i] for pos, flat in flats.items() if i < len(flat)}
        merged.append(merge_base_layers(layers, scalable_qubit_shape))
    return merged


def merge_parallel_block_layers(
    blocks_in_parallel: Mapping[LayoutPosition2D, Block],
    scalable_qubit_shape: PhysicalQubitScalable2D,
    k: int | None = None,
) -> list[LayoutLayer | BaseComposedLayer]:
    """Merge several stacks of layers executed in parallel into one stack of larger layers.

    Args:
        blocks_in_parallel: a 2-dimensional arrangement of blocks. Blocks that
            share the exact same temporal schedule are merged into a scalable
            structure. Blocks with mismatched schedules (a Y-basis measurement
            cap alongside a continuing memory cube) are flattened and merged at
            the concrete ``k`` (which must then be provided).
        scalable_qubit_shape: scalable shape of a scalable qubit. Considered
            valid across the whole domain.
        k: scaling factor. Only consulted when the provided blocks have
            mismatched temporal schedules, in which case it is required.

    Returns:
        a stack of layers representing the same slice of computation as the
        provided ``blocks_in_parallel``.

    Raises:
        NotImplementedError: if the provided blocks have mismatched schedules but
            no ``k`` was provided to flatten them.

    """
    if not blocks_in_parallel:
        return []
    internal_layers_schedules = frozenset(
        tuple(layer.scalable_timesteps for layer in block.layer_sequence)
        for block in blocks_in_parallel.values()
    )
    if len(internal_layers_schedules) != 1:
        if k is None:
            raise NotImplementedError(
                "merge_parallel_block_layers only supports merging blocks that "
                "have layers with a matching temporal schedule, unless a concrete "
                "k is provided to flatten them. Found the following different "
                f"temporal schedules in the provided blocks: {internal_layers_schedules}."
            )
        return _merge_mismatched_block_layers(blocks_in_parallel, scalable_qubit_shape, k)
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
