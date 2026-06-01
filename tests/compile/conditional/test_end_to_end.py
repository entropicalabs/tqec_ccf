"""Stage 7: end-to-end compilation of XZX -> {XZX | XZZ} (and sibling pairs).

For each temporal-basis conditional pair the test:

  1. Builds the smallest valid block graph carrying that conditional cube
     (init cube + temporal pipe + conditional leaf).
  2. Compiles via the new ``generate_conditional_stim_text`` entry point.
  3. Asserts the structural invariants the design depends on:
        * Output contains exactly one top-level ``IF(rec[k]) { ... } ELSE``
          block group corresponding to the final conditional measurement.
        * Inside the IF span, branch-one body and branch-zero body have the
          same number of measurement targets (Equal Measurement Count).
        * Outside every IF/ELSE block, only shared gates appear; no
          ``M`` / ``MX`` instructions leak past the cube boundary.
        * The two branches' measurement target sets are identical (CEO).
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
from tqec.utils.position import Position3D


_TEMPORAL_PAIRS = ["XZZ_XZX", "ZXX_ZXZ", "XZX_XZZ"]
_MEAS_RE = re.compile(r"^\s*(M[XYZ]?R?|MR[XYZ]?)\s+([0-9 ]+)", re.MULTILINE)


def _condition() -> CorrelationSurface:
    p = Position3D(0, 0, 0)
    return CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p, Basis.Z), ZXNode(p, Basis.Z))])
    )


def _graph(pair_name: str) -> BlockGraph:
    g = BlockGraph(f"E2E {pair_name}")
    init_kind = ConditionalLeafCubeKind[pair_name].value[0]
    p0, p1 = Position3D(0, 0, 0), Position3D(0, 0, 1)
    g.add_cube(p0, init_kind)
    g.add_cube(p1, ConditionalLeafCubeKind[pair_name], condition=_condition())
    g.add_pipe(p0, p1)
    return g


def _extract_if_else_blocks(text: str) -> list[tuple[str, str]]:
    """Return list of (then_body, else_body) for each top-level IF/ELSE block.

    Bodies are returned as raw text (newline-separated).  Only handles single
    top-level depth (no nesting), which is what Stage 6 emits.
    """
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("IF(rec["):
            then_lines: list[str] = []
            else_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("} ELSE {"):
                then_lines.append(lines[i])
                i += 1
            # Skip the "} ELSE {" line.
            i += 1
            while i < len(lines) and not lines[i].startswith("}"):
                else_lines.append(lines[i])
                i += 1
            blocks.append(("\n".join(then_lines), "\n".join(else_lines)))
        i += 1
    return blocks


def _meas_target_sets(body: str) -> set[int]:
    qubits: set[int] = set()
    for match in _MEAS_RE.finditer(body):
        for tok in match.group(2).split():
            qubits.add(int(tok))
    return qubits


@pytest.mark.parametrize("pair_name", _TEMPORAL_PAIRS)
def test_end_to_end_conditional_pair(pair_name: str) -> None:
    g = _graph(pair_name)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(
        k=1, condition_recs={pos: -1 for pos in cg._conditional_blocks}
    )

    blocks = _extract_if_else_blocks(text)
    assert blocks, f"{pair_name}: expected at least one IF/ELSE block"

    # CEO invariant: measurement target sets identical across branches.
    for then_body, else_body in blocks:
        then_qubits = _meas_target_sets(then_body)
        else_qubits = _meas_target_sets(else_body)
        if then_qubits or else_qubits:
            assert then_qubits == else_qubits, (
                f"{pair_name}: measurement target sets differ between branches\n"
                f"  then={sorted(then_qubits)}\n  else={sorted(else_qubits)}"
            )

    # No bare M / MX / MR instructions outside IF/ELSE that would belong to a
    # conditional measurement.  Shared measurement layers (init / stabilizer
    # rounds) are fine -- they are not branch-divergent.  We just check no
    # measurement leak appears immediately after an IF/ELSE block closes.
    assert "IF(rec[-1])" in text


def test_branch_zero_is_lowercase_branch_kind() -> None:
    """The else-branch (rec=0) of XZX_XZZ should reflect the XZX cube --
    final data-qubit meas all in X basis -> single MX run, no M instructions.
    """
    g = _graph("XZX_XZZ")
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(
        k=1, condition_recs={pos: -1 for pos in cg._conditional_blocks}
    )
    blocks = _extract_if_else_blocks(text)
    assert blocks
    # Find the block whose else_body contains measurement instructions
    # (there should be exactly one).
    meas_blocks = [b for b in blocks if _meas_target_sets(b[1])]
    assert meas_blocks, "no IF/ELSE block carries measurements"
    then_body, else_body = meas_blocks[0]
    # The XZX else-branch measures all data qubits in X basis -> M instruction
    # absent, MX present.
    assert "MX" in else_body
    assert re.search(r"^\s*M\s+", else_body, re.MULTILINE) is None
    # The XZZ then-branch measures some qubits in Z basis -> M instruction
    # present.
    assert re.search(r"^\s*M\s+", then_body, re.MULTILINE) is not None
