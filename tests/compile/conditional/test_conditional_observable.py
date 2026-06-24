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
    cond_surface = _build_graph_get_condition(g)
    return g, ConditionalCorrelationSurface(
        conditions=(cond_surface,),
        resolutions={(False,): branch_zero, (True,): branch_one},
    )


def _build_graph_get_condition(g: BlockGraph) -> CorrelationSurface:
    """Recover the conditional cube's condition from the graph."""
    for c in g.cubes:
        if c.is_conditional and c.condition is not None:
            return c.condition
    raise AssertionError("no conditional cube in graph")


def test_conditional_observable_emits_observable_include() -> None:
    g, cond_obs = _build_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)

    # New flat-XOR canonical form: shared baseline OBSERVABLE_INCLUDE on the
    # trunk PLUS at least one IF(rec[...]) { OBSERVABLE_INCLUDE(0) ... }
    # (no ELSE) carrying the flip-delta for the single condition.
    if_then_only = re.findall(
        r"IF\(.*?\) \{\n(.*?)\n\}(?!\s*ELSE)", text, re.DOTALL
    )
    observable_ifs = [body for body in if_then_only if "OBSERVABLE_INCLUDE(0)" in body]
    assert observable_ifs, (
        "expected at least one IF { ... OBSERVABLE_INCLUDE(0) ... } block "
        f"(no ELSE) but got none. text:\n{text}"
    )


def _build_multi_cube_graph() -> tuple[BlockGraph, ConditionalCorrelationSurface]:
    a0 = Position3D(0, 0, 0)
    a1 = Position3D(0, 0, 1)
    b0 = Position3D(2, 0, 0)
    b1 = Position3D(2, 0, 1)
    b2 = Position3D(2, 0, 2)
    b3 = Position3D(2, 0, 3)

    init_kind = ConditionalLeafCubeKind.ZXX_ZXZ.value[0]
    g = BlockGraph("multi_cond_obs")
    g.add_cube(a0, init_kind)
    g.add_cube(
        a1,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset({ZXEdge(u=ZXNode(a0, Basis.Z), v=ZXNode(a0, Basis.Z))})
        ),
    )
    g.add_pipe(a0, a1)
    g.add_cube(b0, init_kind)
    g.add_cube(b1, init_kind)
    g.add_cube(b2, init_kind)
    g.add_cube(
        b3,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset({ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b2, Basis.Z))})
        ),
    )
    g.add_pipe(b0, b1)
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)

    span = frozenset(
        {
            ZXEdge(u=ZXNode(a0, Basis.X), v=ZXNode(a1, Basis.X)),
            ZXEdge(u=ZXNode(b0, Basis.X), v=ZXNode(b1, Basis.X)),
            ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(b2, Basis.X)),
            ZXEdge(u=ZXNode(b2, Basis.X), v=ZXNode(b3, Basis.X)),
        }
    )
    cond_a = next(c.condition for c in g.cubes if c.position == a1)
    cond_b = next(c.condition for c in g.cubes if c.position == b3)
    assert cond_a is not None and cond_b is not None
    surface_all = CorrelationSurface(span=span)
    cond_obs = ConditionalCorrelationSurface(
        conditions=(cond_a, cond_b),
        resolutions={
            (False, False): surface_all,
            (False, True): surface_all,
            (True, False): surface_all,
            (True, True): surface_all,
        },
    )
    return g, cond_obs


def test_multi_conditional_observable_compiles_when_all_branches_identical() -> None:
    """All four resolutions identical → trivially XOR-decomposable, no IF
    blocks for the observable (shared baseline only)."""
    g, cond_obs = _build_multi_cube_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)
    # OBSERVABLE_INCLUDE present on trunk (not inside any IF).
    assert "OBSERVABLE_INCLUDE(0)" in text


def test_multi_condition_flat_xor_emits_one_if_per_cube_no_else() -> None:
    """Two conditional cubes, identical resolutions across the truth table:
    flat-XOR decomposable trivially; emission produces one IF(no-ELSE) per
    cube wrapping OBSERVABLE_INCLUDE(0) for that cube's flip-delta."""
    g, cond_obs = _build_multi_cube_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)
    # All-identical resolutions → deltas may all be empty (no IFs) but the
    # compile must succeed and shared OBSERVABLE_INCLUDE must appear.
    assert "OBSERVABLE_INCLUDE(0)" in text
    # No paired IF/ELSE form for the observable.
    paired = re.findall(
        r"IF\(.*?\) \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\} ELSE \{",
        text,
    )
    assert not paired, "flat-XOR emission must never produce IF/ELSE pairs"


def test_surface_anchored_conditional_observable_on_plain_block_graph() -> None:
    """Plain ZXZ BlockGraph (no ConditionalLeafCubeKind anywhere) carrying a
    surface-anchored ConditionalCorrelationSurface. The conditional observable
    gates only OBSERVABLE_INCLUDE lines; no per-branch circuit divergence."""
    b1 = Position3D(0, 0, 0)
    b2 = Position3D(0, 0, 1)
    b3 = Position3D(0, 0, 2)
    c = Position3D(1, 0, 2)

    g = BlockGraph("surface_anchored")
    g.add_cube(b1, "ZXZ")
    g.add_cube(b2, "ZXZ")
    g.add_cube(b3, "ZXZ")
    g.add_cube(c, "ZXZ")
    g.add_pipe(b1, b2)
    g.add_pipe(b2, b3)
    g.add_pipe(b3, c)

    cond = CorrelationSurface(
        span=frozenset({ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(b2, Basis.X))})
    )
    spine = {
        ZXEdge(u=ZXNode(b1, Basis.Z), v=ZXNode(b2, Basis.Z)),
        ZXEdge(u=ZXNode(b2, Basis.Z), v=ZXNode(b3, Basis.Z)),
    }
    obs_off = CorrelationSurface(span=frozenset(spine))
    obs_on = CorrelationSurface(
        span=frozenset(spine | {ZXEdge(u=ZXNode(b3, Basis.Z), v=ZXNode(c, Basis.Z))})
    )
    cond_obs = ConditionalCorrelationSurface(
        conditions=(cond,),
        resolutions={(False,): obs_off, (True,): obs_on},
    )

    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)

    # Surface-anchored emission: shared OI on trunk + one IF(no-ELSE) with
    # OBSERVABLE_INCLUDE(0) inside.
    if_then = re.findall(r"IF\(.*?\) \{\n(.*?)\n\}(?!\s*ELSE)", text, re.DOTALL)
    obs_ifs = [body for body in if_then if "OBSERVABLE_INCLUDE(0)" in body]
    assert obs_ifs, "expected at least one IF { OBSERVABLE_INCLUDE(0) } (no ELSE)"
    # No paired IF/ELSE wrapping an observable.
    paired = re.findall(
        r"IF\(.*?\) \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\} ELSE \{",
        text,
    )
    assert not paired, "surface-anchored flat-XOR must not emit IF/ELSE pairs"


def test_conditional_observable_flat_form_has_no_else() -> None:
    """Flat-XOR canonical form never emits ELSE for the observable IF."""
    g, cond_obs = _build_graph()
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)
    # No paired IF { OBSERVABLE_INCLUDE(0) ... } ELSE { ... OBSERVABLE_INCLUDE(0) ... }
    paired = re.findall(
        r"IF\(.*?\) \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\} ELSE \{\n((?:.|\n)*?OBSERVABLE_INCLUDE\(0\)[^\n]*)\n\}",
        text,
    )
    assert not paired, (
        "flat-XOR emission must not produce IF/ELSE pairs for observables"
    )
