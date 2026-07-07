"""Verify spatial+temporal+detector locality of IF/ELSE bodies.

Builds the four-cubes graph from :mod:`scripts.test_four_cubes`, compiles
to conditional Stim text, and asserts that every instruction inside any
IF/ELSE block targets only qubits and detector coordinates belonging to
the conditional cube's spacetime tile.  Failure means canonical
emission order let branch-conditional ops leak into neighbouring cubes.

Run with ``uv run python scripts/verify_conditional_locality.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stim

from tools.resolve import _IfNode, _parse, resolve_if_else
from tqec.circuit.qubit_map import QubitMap
from tqec.compile.blocks.positioning import LayoutPosition3D
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION
from tqec.computation.block_graph import BlockGraph
from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.enums import Basis
from tqec.utils.position import Position3D


def _condition() -> CorrelationSurface:
    # TODO: cond cube here sits at z=0; mock at z=-1 to satisfy the
    # structural causality check at Cube.__post_init__. Resolver yields no
    # recs → placeholder rec[-1] fallback. Lift cond cube above z=0 for a
    # real condition.
    p = Position3D(0, 0, -1)
    return CorrelationSurface(span=frozenset([ZXEdge(ZXNode(p, Basis.Z), ZXNode(p, Basis.Z))]))


def _build_conditional_graph(pos_cond: Position3D) -> BlockGraph:
    g = BlockGraph("four_cubes_conditional")
    cubes = [
        (Position3D(0, 0, 0), "XZX"),
        (Position3D(0, 0, 1), "XZX"),
        (Position3D(1, 0, 0), "XZX"),
        (Position3D(1, 0, 1), "XZX"),
        (Position3D(0, 1, 0), "XZX"),
        (Position3D(0, 1, 1), "XZX"),
        (Position3D(1, 1, 1), "XZX"),
    ]
    pipes = [(0, 1), (2, 3), (4, 5)]
    for cube, kind in cubes:
        g.add_cube(cube, kind)
    for u, v in pipes:
        g.add_pipe(cubes[u][0], cubes[v][0])
    g.add_cube(pos_cond, ConditionalLeafCubeKind.XZX_XZZ, condition=_condition())
    g.add_pipe(pos_cond, cubes[6][0])
    return g


def _cluster_axis(values: list[int]) -> list[tuple[int, int]]:
    """Split a sorted unique coord list into contiguous clusters.

    Returns a list of ``(lo, hi)`` inclusive ranges, one per cube along
    that axis.  Gaps wider than 1 separate clusters.
    """
    if not values:
        return []
    out: list[tuple[int, int]] = []
    lo = prev = values[0]
    for v in values[1:]:
        if v - prev > 1:
            out.append((lo, prev))
            lo = v
        prev = v
    out.append((lo, prev))
    return out


def _footprint(qubit_map: QubitMap, pos_cond: Position3D) -> tuple[int, int, int, int]:
    """Return ``(x_lo, x_hi, y_lo, y_hi)`` of the cube tile at ``pos_cond``."""
    xs = sorted({q.x for q in qubit_map.i2q.values()})
    ys = sorted({q.y for q in qubit_map.i2q.values()})
    x_clusters = _cluster_axis(xs)
    y_clusters = _cluster_axis(ys)
    x_lo, x_hi = x_clusters[pos_cond.x]
    y_lo, y_hi = y_clusters[pos_cond.y]
    return x_lo, x_hi, y_lo, y_hi


def _flatten_if_bodies(entries: list, out: list[list]) -> None:
    for e in entries:
        if isinstance(e, _IfNode):
            out.append(e.then_body)
            if e.else_body is not None:
                out.append(e.else_body)
            _flatten_if_bodies(e.then_body, out)
            if e.else_body is not None:
                _flatten_if_bodies(e.else_body, out)


def _parse_line(line: str) -> stim.CircuitInstruction | None:
    try:
        c = stim.Circuit(line.strip())
    except Exception:
        return None
    if len(c) == 0:
        return None
    inst = c[0]
    if isinstance(inst, stim.CircuitInstruction):
        return inst
    return None


def _z_extent(cg, pos_cond: Position3D, k: int) -> int:
    from tqec.utils.position import BlockPosition3D

    layout = LayoutPosition3D.from_block_position(
        BlockPosition3D(pos_cond.x, pos_cond.y, pos_cond.z)
    )
    block = cg._blocks[layout]
    return block.dimensions[2].integer_eval(k)


def _walk_with_z(entries: list, z_offsets: list[tuple[int, list]]) -> None:
    """Top-level walk that tracks SHIFT_COORDS z-offset per IF block.

    Populates ``z_offsets`` with ``(z_at_if_open, if_node)`` pairs in
    document order.
    """
    z = 0
    for e in entries:
        if isinstance(e, _IfNode):
            z_offsets.append((z, e))
            continue
        if not isinstance(e, str):
            continue
        inst = _parse_line(e)
        if inst is None:
            continue
        if inst.name == "SHIFT_COORDS":
            args = inst.gate_args_copy()
            if len(args) >= 3:
                z += int(args[2])


def assert_spatial_locality(
    bodies: list[list],
    qubit_map: QubitMap,
    footprint: tuple[int, int, int, int],
) -> None:
    x_lo, x_hi, y_lo, y_hi = footprint
    violations: list[str] = []
    for body in bodies:
        for entry in body:
            if not isinstance(entry, str):
                continue
            inst = _parse_line(entry)
            if inst is None:
                continue
            if inst.name == "DETECTOR":
                continue  # checked in assert_detector_locality
            for tgt in inst.targets_copy():
                if not tgt.is_qubit_target:
                    continue
                qi = tgt.qubit_value
                if qi is None:
                    continue
                coord = qubit_map.i2q[qi]
                if not (x_lo <= coord.x <= x_hi and y_lo <= coord.y <= y_hi):
                    violations.append(
                        f"  {inst.name} q={qi} at ({coord.x},{coord.y}) "
                        f"outside tile x:[{x_lo},{x_hi}] y:[{y_lo},{y_hi}]"
                    )
    if violations:
        head = violations[:10]
        more = "" if len(violations) <= 10 else f"\n  ... and {len(violations) - 10} more"
        raise AssertionError(
            "Spatial locality violation: IF/ELSE-body gate targets leak "
            "into other cubes:\n" + "\n".join(head) + more
        )


def assert_detector_locality(
    text: str,
    footprint: tuple[int, int, int, int],
    z_lo: int,
    z_hi: int,
) -> None:
    """Walk top-level + nested, tracking SHIFT_COORDS z; assert DETECTORs inside IFs are local."""
    x_lo, x_hi, y_lo, y_hi = footprint
    entries, _ = _parse(text.splitlines(), 0, depth_indent=None)
    violations: list[str] = []

    def walk(entries_in: list, z: int, in_if: bool) -> int:
        for e in entries_in:
            if isinstance(e, _IfNode):
                walk(e.then_body, z, True)
                if e.else_body is not None:
                    walk(e.else_body, z, True)
                continue
            if not isinstance(e, str):
                continue
            inst = _parse_line(e)
            if inst is None:
                continue
            if inst.name == "SHIFT_COORDS":
                args = inst.gate_args_copy()
                if len(args) >= 3:
                    z += int(args[2])
                continue
            if in_if and inst.name == "DETECTOR":
                args = inst.gate_args_copy()
                if len(args) < 3:
                    continue
                dx, dy, dt = int(args[0]), int(args[1]), int(args[2])
                absolute_t = z + dt
                if not (x_lo <= dx <= x_hi and y_lo <= dy <= y_hi):
                    violations.append(
                        f"  DETECTOR({dx},{dy},{dt}) xy outside tile "
                        f"x:[{x_lo},{x_hi}] y:[{y_lo},{y_hi}]"
                    )
                if not (z_lo <= absolute_t < z_hi):
                    violations.append(
                        f"  DETECTOR({dx},{dy},{dt}) absolute t={absolute_t} "
                        f"outside z-window [{z_lo},{z_hi})"
                    )
        return z

    walk(entries, 0, False)

    if violations:
        head = violations[:10]
        more = "" if len(violations) <= 10 else f"\n  ... and {len(violations) - 10} more"
        raise AssertionError(
            "Detector locality violation: IF/ELSE-body DETECTORs leak "
            "into other cubes / layers:\n" + "\n".join(head) + more
        )


def assert_temporal_locality(text: str, z_lo: int, z_hi: int) -> None:
    """Every IF block must open while the running z-offset is in ``[z_lo, z_hi)``."""
    entries, _ = _parse(text.splitlines(), 0, depth_indent=None)
    z_offsets: list[tuple[int, _IfNode]] = []
    _walk_with_z(entries, z_offsets)
    violations: list[str] = []
    for z, _ in z_offsets:
        if not (z_lo <= z < z_hi):
            violations.append(f"  IF block opened at z={z} outside [{z_lo},{z_hi})")
    if violations:
        raise AssertionError(
            "Temporal locality violation: IF blocks straddle cube layers:\n"
            + "\n".join(violations)
        )


def main() -> None:
    pos_cond = Position3D(1, 1, 0)
    k = 1

    g = _build_conditional_graph(pos_cond)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(k=k)

    import re as _re

    m = _re.search(r"IF\(([^)]+)\)", text)
    assert m is not None
    rec = int(m.group(1).split("^")[0].strip().lstrip("rec[").rstrip("]"))
    branch_zero = resolve_if_else(text, conditions={rec: 0})
    qubit_map = QubitMap.from_circuit(branch_zero)
    footprint = _footprint(qubit_map, pos_cond)

    # z-window of conditional cube layer (in SHIFT_COORDS units).
    z_per_layer = _z_extent(cg, pos_cond, k)
    z_lo = pos_cond.z * z_per_layer
    z_hi = (pos_cond.z + 1) * z_per_layer

    # Collect every IF/ELSE body for spatial check (flat).
    entries, _ = _parse(text.splitlines(), 0, depth_indent=None)
    bodies: list[list] = []
    _flatten_if_bodies(entries, bodies)

    assert_spatial_locality(bodies, qubit_map, footprint)
    assert_temporal_locality(text, z_lo, z_hi)
    assert_detector_locality(text, footprint, z_lo, z_hi)

    print(
        f"OK: conditional IF/ELSE bodies are local to cube tile "
        f"x:[{footprint[0]},{footprint[1]}] y:[{footprint[2]},{footprint[3]}] "
        f"z:[{z_lo},{z_hi})."
    )


if __name__ == "__main__":
    main()
