"""Which measurements an ``IF`` condition actually reads, for a condition that resolves.

The other conditional fixtures use a condition whose measurements a temporal pipe
absorbs, so it resolves to nothing. These tests use one that does resolve: the
readout of a cube at ``z = 0`` that shares a spatial pipe with the column carrying
the conditional cube. They check the emitted text itself, without reusing the
resolver, so they would catch the resolver and the renderer disagreeing about
where an ``IF`` sits.
"""

from __future__ import annotations

import re

import pytest

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D

_KIND = ConditionalLeafCubeKind.ZXZ_ZXX
_MEASUREMENTS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}
_IF_RE = re.compile(r"IF\((.*)\) \{")
_REC_RE = re.compile(r"rec\[(-?\d+)\]")


def _graph(n_memory: int = 0) -> BlockGraph:
    """Build a side cube beside a column ending in the conditional cube.

    ``(0,0,0)`` and ``(1,0,0)`` share a spatial pipe; the column at ``x = 1``
    rises through ``n_memory`` memory cubes to the conditional cube. The
    condition is the ``z = 0`` part of the column's correlation surface, which
    reads the side cube's final data readout.
    """
    side, base = Position3D(0, 0, 0), Position3D(1, 0, 0)
    top = Position3D(1, 0, n_memory + 1)

    def build(leaf: ConditionalLeafCubeKind | str, condition: CorrelationSurface | None):
        g = BlockGraph("resolved condition")
        g.add_cube(side, "ZXZ")
        g.add_cube(base, _KIND.value[0])
        g.add_pipe(side, base)
        below = base
        for z in range(1, n_memory + 1):
            here = Position3D(1, 0, z)
            g.add_cube(here, _KIND.value[0])
            g.add_pipe(below, here)
            below = here
        g.add_cube(top, leaf, condition=condition)
        g.add_pipe(below, top)
        return g

    surface = build(_KIND.value[0], None).find_correlation_surfaces()[0]
    condition = CorrelationSurface(
        span=frozenset(e for e in surface.span if e.u.position.z == 0 and e.v.position.z == 0)
    )
    return build(_KIND, condition)


def _ifs_and_measurements(text: str) -> tuple[list[tuple[int, list[int]]], list[int]]:
    """Return each IF's (preceding measurement count, absolute indices) and the measured qubits.

    Measurements inside an ELSE arm are not counted: both arms perform the
    same number (Equal Measurement Count), so counting one arm is enough.
    """
    ifs: list[tuple[int, list[int]]] = []
    measured: list[int] = []
    in_else = False
    for raw in text.splitlines():
        line = raw.strip()
        match = _IF_RE.match(line)
        if match:
            offsets = [int(r) for r in _REC_RE.findall(match.group(1))]
            ifs.append((len(measured), sorted(len(measured) + r for r in offsets)))
            continue
        if line.startswith("} ELSE {"):
            in_else = True
            continue
        if line == "}":
            in_else = False
            continue
        if in_else or not line:
            continue
        name = line.split(" ")[0].split("(")[0]
        if name in _MEASUREMENTS:
            measured.extend(int(q) for q in line.split()[1:])
    return ifs, measured


@pytest.mark.parametrize("n_memory", [0, 1])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_every_if_of_a_cube_reads_the_same_measurements(k: int, n_memory: int) -> None:
    text = compile_block_graph(
        _graph(n_memory), FIXED_BULK_CONVENTION, observables=None
    ).generate_conditional_stim_text(k=k)
    ifs, _ = _ifs_and_measurements(text)
    assert len(ifs) > 1
    assert len({tuple(indices) for _, indices in ifs}) == 1


@pytest.mark.parametrize("n_memory", [0, 1])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_condition_reads_data_readouts_not_ancillas(k: int, n_memory: int) -> None:
    """Each qubit the condition reads is measured once before the IF: a data readout.

    An ancilla is measured every round, so an offset that lands on one (the
    symptom of a condition computed in the wrong frame, or of a repeated round
    counted once) fails this.
    """
    text = compile_block_graph(
        _graph(n_memory), FIXED_BULK_CONVENTION, observables=None
    ).generate_conditional_stim_text(k=k)
    ifs, measured = _ifs_and_measurements(text)
    for before, indices in ifs:
        assert indices
        assert all(0 <= i < before for i in indices)
        for i in indices:
            assert measured[:before].count(measured[i]) == 1


def test_a_condition_that_reads_nothing_is_rejected() -> None:
    """A condition a temporal pipe absorbs used to fall back to a placeholder ``rec[-1]``.

    The single ``Z`` node on the cube directly below the conditional cube names
    only data qubits the pipe carries on, so no measurement backs it.
    """
    base, top = Position3D(0, 0, 0), Position3D(0, 0, 1)
    g = BlockGraph("unresolvable condition")
    g.add_cube(base, _KIND.value[0])
    condition = CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(base, Basis.Z), ZXNode(base, Basis.Z))])
    )
    g.add_cube(top, _KIND, condition=condition)
    g.add_pipe(base, top)
    compiled = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    with pytest.raises(TQECError, match="reads no measurement"):
        compiled.generate_conditional_stim_text(k=1)
