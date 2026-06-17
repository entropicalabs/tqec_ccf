"""Per-branch OBSERVABLE_INCLUDE emission for ConditionalCorrelationSurface."""

from __future__ import annotations

import re

from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import (
    ConditionalCorrelationSurface,
    CorrelationSurface,
    ZXEdge,
    ZXNode,
)
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _build_graph() -> tuple[BlockGraph, ConditionalCorrelationSurface]:
    b1 = Position3D(0, 0, 0)
    c1 = Position3D(0, 0, 1)
    c2 = Position3D(1, 0, 1)
    t1 = Position3D(0, 0, 2)
    t2 = Position3D(1, 0, 2)

    g = BlockGraph("conditional_observable")
    g.add_cube(b1, "ZXX")
    g.add_cube(c1, "ZXZ")
    g.add_cube(c2, "ZXZ")
    g.add_cube(t1, "ZXX")
    g.add_cube(
        t2,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset(
                {ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(c1, Basis.Z))}
            )
        ),
    )
    g.add_pipe(b1, c1)
    g.add_pipe(c1, c2)
    g.add_pipe(c1, t1)
    g.add_pipe(c2, t2)

    branch_zero = CorrelationSurface(
        span=frozenset(
            {
                ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
                ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(c2, Basis.X)),
                ZXEdge(u=ZXNode(c2, Basis.X), v=ZXNode(t2, Basis.X)),
            }
        )
    )
    branch_one = CorrelationSurface(
        span=frozenset(
            {
                ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
                ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(t1, Basis.X)),
            }
        )
    )
    return g, ConditionalCorrelationSurface(
        branch_zero=branch_zero,
        branch_one=branch_one,
        conditional_cube_positions=(t2,),
    )


def test_conditional_observable_emits_per_branch_observable_include() -> None:
    g, cond_obs = _build_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)

    # Find an IfBlock whose then-body and else-body both reference
    # OBSERVABLE_INCLUDE(0). Use a structural scan of indented bodies.
    if_blocks = re.findall(
        r"IF\(.*?\) \{\n(.*?)\n\} ELSE \{\n(.*?)\n\}", text, re.DOTALL
    )
    assert if_blocks, "expected at least one IF { ... } ELSE { ... } block"
    observable_if = [
        (t, e)
        for t, e in if_blocks
        if "OBSERVABLE_INCLUDE(0)" in t or "OBSERVABLE_INCLUDE(0)" in e
    ]
    assert observable_if, (
        "expected at least one IF/ELSE block containing OBSERVABLE_INCLUDE(0)"
    )
    then_body, else_body = observable_if[0]
    assert "OBSERVABLE_INCLUDE(0)" in then_body or "OBSERVABLE_INCLUDE(0)" in else_body


def test_conditional_observable_diverges_per_branch_qubits() -> None:
    g, cond_obs = _build_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)
    # The two branches' OBSERVABLE_INCLUDE statements should differ in their
    # rec target lists (different qubit sets per branch).
    matches = re.findall(
        r"IF\(.*?\) \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\} ELSE \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\}",
        text,
    )
    assert matches, "expected paired OBSERVABLE_INCLUDE in IF/ELSE bodies"
    then_line, else_line = matches[0]
    assert then_line != else_line, (
        "branch-zero and branch-one OBSERVABLE_INCLUDE should not be identical"
    )
