"""The branching tree of ``notebooks/branching_tree.ipynb``: conditions and round trip.

A column of memory cubes where, at every layer ``z``, a lattice-surgery merge
with a side cube conditions a cube at ``z + 1``. Each condition is the outcome
of that merge, so it reads the seam's stabilizers rather than data readouts.
"""

from __future__ import annotations

import itertools
import re

import pytest
import stim

from tests.tools.resolve_test import _assert_circuits_equivalent_modulo_detector_order
from tools.resolve import resolve_if_else_by_measurement
from tqec.compile.compile import _resolve_conditional_cubes, compile_block_graph
from tqec.compile.graph import TopologicalComputationGraph
from tqec.compile.observables.abstract_observable import (
    AbstractObservable,
    compile_correlation_surface_to_abstract_observable,
)
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _branching_tree(n: int) -> BlockGraph:
    """Build the notebook's tree with ``n`` conditional cubes."""
    g = BlockGraph(f"tree_{n}")

    def add_branch_at_z(z: int) -> Position3D:
        s = 1 - 2 * (z % 2)
        p, b, c = Position3D(0, 0, z), Position3D(s, 0, z), Position3D(s, 0, z + 1)
        condition = CorrelationSurface(
            span=frozenset({ZXEdge(ZXNode(p, Basis.X), ZXNode(b, Basis.X))})
        )
        g.add_cube(p, "ZXZ")
        g.add_cube(b, "ZXZ")
        g.add_cube(c, ConditionalLeafCubeKind.ZXZ_ZXX, condition=condition)
        g.add_pipe(p, b)
        g.add_pipe(b, c)
        return p

    column = [add_branch_at_z(z) for z in range(1, n + 1)]
    bottom, top = Position3D(0, 0, 0), Position3D(0, 0, n + 1)
    g.add_cube(bottom, "ZXZ")
    g.add_cube(top, "ZXZ")
    for p1, p2 in itertools.pairwise([bottom, *column, top]):
        g.add_pipe(p1, p2)
    return g


def _if_conditions(text: str) -> list[tuple[int, ...]]:
    """Return the absolute measurement indices each IF condition names, in order."""
    done: list[str] = []
    out: list[tuple[int, ...]] = []
    in_else = False
    for raw in text.splitlines():
        line = raw.strip()
        match = re.match(r"IF\((.*)\) \{", line)
        if match:
            measured = stim.Circuit("\n".join(done)).num_measurements
            offsets = re.findall(r"rec\[(-?\d+)\]", match.group(1))
            out.append(tuple(sorted(measured + int(r) for r in offsets)))
        elif line.startswith("} ELSE {"):
            in_else = True
        elif line == "}":
            in_else = False
        elif line and not in_else:
            done.append(line)
    return out


def _observable_indices(circuit: stim.Circuit) -> tuple[int, ...]:
    """Return the absolute measurement indices ``OBSERVABLE_INCLUDE(0)`` XORs."""
    count = 0
    acc: set[int] = set()
    for inst in circuit.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            acc ^= {count + t.value for t in inst.targets_copy()}
        else:
            single = stim.Circuit()
            single.append(inst)
            count += single.num_measurements
    return tuple(sorted(acc))


def _swapped(
    g: BlockGraph, bits: tuple[int, ...], observables: list[AbstractObservable] | None = None
) -> TopologicalComputationGraph:
    """Compile ``g`` with each conditional cube, in z order, replaced by its chosen branch."""
    compiled = compile_block_graph(g, observables=None)
    by_z = sorted(compiled._conditional_blocks.items(), key=lambda item: item[0].z)
    for (pos, block), bit in zip(by_z, bits, strict=True):
        compiled._blocks[pos] = block.block_if_one if bit else block.block_if_zero
    compiled._conditional_blocks = {}
    if observables is not None:
        compiled._observables = observables
    return compiled


_CASES = [
    pytest.param(1, 1),
    pytest.param(2, 1),
    pytest.param(2, 2, marks=[pytest.mark.slow, pytest.mark.timeout(600)]),
    pytest.param(3, 1, marks=[pytest.mark.slow, pytest.mark.timeout(600)]),
]


@pytest.mark.parametrize(("n", "k"), _CASES)
def test_conditions_read_the_merge_outcome(n: int, k: int) -> None:
    """Each condition names exactly what the plain observable path reads for its surface.

    The plain path lowers a surface to measurements leaf by leaf, independently
    of the conditional resolver, so the two agreeing pins the records.
    """
    g = _branching_tree(n)
    text = compile_block_graph(g, observables=None).generate_conditional_stim_text(k=k)
    conditions = sorted(set(_if_conditions(text)))
    assert len(conditions) == n

    branch_zero = _resolve_conditional_cubes(g, 0)
    cubes = sorted((c for c in g.cubes if c.is_conditional), key=lambda c: c.position.z)
    for condition, cube in zip(conditions, cubes, strict=True):
        assert cube.condition is not None
        observable = compile_correlation_surface_to_abstract_observable(
            branch_zero, cube.condition, include_temporal_hadamard_pipes=True, _skip_validation=True
        )
        circuit = _swapped(g, (0,) * n, [observable]).generate_stim_circuit(k=k)
        assert condition == _observable_indices(circuit)


@pytest.mark.parametrize(("n", "k"), _CASES)
def test_every_branch_combination_round_trips(n: int, k: int) -> None:
    """Resolving the text for each outcome gives that branch combination's own circuit."""
    g = _branching_tree(n)
    text = compile_block_graph(g, observables=None).generate_conditional_stim_text(k=k)
    keys = [c[0] for c in sorted(set(_if_conditions(text)))]
    for bits in itertools.product((0, 1), repeat=n):
        resolved = resolve_if_else_by_measurement(text, dict(zip(keys, bits, strict=True)))
        reference = _swapped(g, bits).generate_stim_circuit(k=k)
        _assert_circuits_equivalent_modulo_detector_order(
            resolved.flattened(), reference.flattened()
        )
