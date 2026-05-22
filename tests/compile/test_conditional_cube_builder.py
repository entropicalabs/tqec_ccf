"""CubeBuilder support for ``ConditionalLeafCubeKind`` (Stage 4 + 5)."""

from __future__ import annotations

import pytest

from tqec.compile.blocks.block import ConditionalBlock
from tqec.compile.specs.base import CubeSpec
from tqec.compile.specs.library.fixed_boundary import FixedBoundaryCubeBuilder
from tqec.compile.specs.library.fixed_bulk import FixedBulkCubeBuilder
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.plaquette.compilation.base import IdentityPlaquetteCompiler
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D
from tqec.utils.scale import LinearFunction


_BLOCK_TEMPORAL_HEIGHT = LinearFunction(0, 2)


def _trivial_condition() -> CorrelationSurface:
    pos = Position3D(0, 0, 0)
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(pos, Basis.Z), ZXNode(pos, Basis.Z))])
    )


# XZZ_XZX, ZXX_ZXZ, XZX_XZZ are temporal-basis swaps that satisfy Equal
# Measurement Count.  ZXZ_ZXX in the current cube enum is a spatial swap
# (typo upstream, see PR #829) and is expected to fail the structural check.
_TEMPORAL_PAIRS = ["XZZ_XZX", "ZXX_ZXZ", "XZX_XZZ"]


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


def test_spatial_swap_pair_violates_equal_meas_count() -> None:
    """ZXZ_ZXX is a typo upstream: its tuple is (ZXZ, XXZ), a spatial swap
    that changes the stabilizer set.  The ``ConditionalBlock`` Equal
    Measurement Count assertion catches this at construction time.
    """
    builder = FixedBulkCubeBuilder(IdentityPlaquetteCompiler)
    condition = _trivial_condition()
    spec = CubeSpec(kind=ConditionalLeafCubeKind.ZXZ_ZXX, condition=condition)
    with pytest.raises(TQECError, match="Equal Measurement Count"):
        builder(spec, _BLOCK_TEMPORAL_HEIGHT)
