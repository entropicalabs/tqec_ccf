"""CubeBuilder support for ``ConditionalLeafCubeKind`` (Stage 4 + 5)."""

from __future__ import annotations

import pytest

from tqec.compile.blocks.block import Block, ConditionalBlock
from tqec.compile.blocks.layers.atomic.plaquettes import PlaquetteLayer
from tqec.compile.specs.base import CubeSpec
from tqec.compile.specs.library.fixed_boundary import FixedBoundaryCubeBuilder
from tqec.compile.specs.library.fixed_bulk import FixedBulkCubeBuilder
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.plaquette.compilation.base import IdentityPlaquetteCompiler
from tqec.plaquette.plaquette import Plaquette, Plaquettes
from tqec.plaquette.rpng.rpng import RPNGDescription
from tqec.plaquette.rpng.translators.default import DefaultRPNGTranslator
from tqec.templates.qubit import QubitTemplate
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.frozendefaultdict import FrozenDefaultDict
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction


_BLOCK_TEMPORAL_HEIGHT = LinearFunction(2, -1)
_TRANSLATOR = DefaultRPNGTranslator()


def _empty_plaquette() -> Plaquette:
    """Build a plaquette that performs no measurement."""
    return _TRANSLATOR.translate(RPNGDescription.empty())


def _measuring_plaquette() -> Plaquette:
    """Build a plaquette that performs one measurement."""
    return _TRANSLATOR.translate(RPNGDescription.from_string("-x1- -x2- -x3- -x4-"))


def _trivial_condition() -> CorrelationSurface:
    pos = Position3D(0, 0, 0)
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(pos, Basis.Z), ZXNode(pos, Basis.Z))])
    )


# Every pair is a temporal-basis swap -- the spatial boundaries match between
# branches, so both satisfy Equal Measurement Count.  ``ZXZ_ZXX`` used to be
# excluded here: its enum value was a typo, the spatial swap ``(ZXZ, XXZ)``,
# which changes the stabilizer set.  It has since been corrected to the
# ``(ZXZ, ZXX)`` its name and docstring describe.
_TEMPORAL_PAIRS = [k.name for k in ConditionalLeafCubeKind]


@pytest.mark.parametrize("pair_name", _TEMPORAL_PAIRS)
def test_fixed_bulk_builds_conditional_block(pair_name: str) -> None:
    builder = FixedBulkCubeBuilder(IdentityPlaquetteCompiler)
    condition = _trivial_condition()
    spec = CubeSpec(kind=ConditionalLeafCubeKind[pair_name], condition=condition)
    block = builder(spec, _BLOCK_TEMPORAL_HEIGHT)
    assert isinstance(block, ConditionalBlock)
    assert block.condition is condition


@pytest.mark.parametrize("pair_name", _TEMPORAL_PAIRS)
def test_fixed_boundary_builds_conditional_block(pair_name: str) -> None:
    builder = FixedBoundaryCubeBuilder(IdentityPlaquetteCompiler)
    condition = _trivial_condition()
    spec = CubeSpec(kind=ConditionalLeafCubeKind[pair_name], condition=condition)
    block = builder(spec, _BLOCK_TEMPORAL_HEIGHT)
    assert isinstance(block, ConditionalBlock)
    assert block.condition is condition


def test_missing_condition_raises() -> None:
    builder = FixedBulkCubeBuilder(IdentityPlaquetteCompiler)
    spec = CubeSpec(kind=ConditionalLeafCubeKind.XZX_XZZ, condition=None)
    with pytest.raises(TQECError, match="missing its CorrelationSurface"):
        builder(spec, _BLOCK_TEMPORAL_HEIGHT)


def test_conditional_block_rejects_unequal_measurement_counts() -> None:
    """``ConditionalBlock`` refuses branches whose measurement schedules differ.

    Equal Measurement Count is what lets downstream ``rec[-k]`` references stay
    branch-independent, so the guard is structural and enforced at construction.

    The mismatch is built directly rather than via a
    :class:`ConditionalLeafCubeKind` member. It used to be reached through
    ``ZXZ_ZXX``, whose value was a typo -- the spatial swap ``(ZXZ, XXZ)``, which
    changes the stabilizer set -- but that has been corrected, and no member is a
    spatial swap any more. Constructing the mismatch here keeps the guard covered
    without depending on a particular member staying wrong.
    """
    measuring = PlaquetteLayer(
        QubitTemplate(),
        Plaquettes(
            FrozenDefaultDict({1: _measuring_plaquette()}, default_value=_empty_plaquette())
        ),
    )
    empty = PlaquetteLayer(
        QubitTemplate(),
        Plaquettes(FrozenDefaultDict({1: _empty_plaquette()}, default_value=_empty_plaquette())),
    )
    with pytest.raises(TQECError, match="Equal Measurement Count"):
        ConditionalBlock(Block([measuring]), Block([empty]), _trivial_condition())


def test_conditional_block_rejects_unequal_layer_counts() -> None:
    """The cheaper half of the same structural guard."""
    empty = PlaquetteLayer(
        QubitTemplate(),
        Plaquettes(FrozenDefaultDict({}, default_value=_empty_plaquette())),
    )
    with pytest.raises(TQECError, match="same number of layers"):
        ConditionalBlock(Block([empty]), Block([empty, empty]), _trivial_condition())
