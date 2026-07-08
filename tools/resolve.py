"""Resolve IF/ELSE-annotated Stim text into a vanilla ``stim.Circuit``.

Given the text dialect produced by
``ConditionalCircuit.to_stim_text`` and a mapping of condition outcomes,
parse the IF/ELSE blocks, pick the appropriate body per block, and emit
the resulting Stim text through ``stim.Circuit``.

``rec[-K]`` offsets inside the selected body remain valid: each branch
body was originally compiled with a self-consistent measurement count,
and the shared instructions surrounding the IF/ELSE block are
measurement-count-identical across branches (Equal Measurement Count +
CEO invariants enforced upstream).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import stim

# Matches a single- or multi-rec IF header:
#   IF(rec[-1]) {
#   IF(rec[-200]^rec[-198]^...) {
# Group 2 captures the whole XOR condition expression; the block is keyed by
# its first rec offset (multi-rec conditions all flip together as one XOR).
_IF_OPEN = re.compile(r"^(\s*)IF\((rec\[-?\d+\](?:\^rec\[-?\d+\])*)\)\s*\{\s*$")
_FIRST_REC = re.compile(r"rec\[(-?\d+)\]")
_ELSE_LINE = re.compile(r"^(\s*)\}\s*ELSE\s*\{\s*$")
_BLOCK_CLOSE = re.compile(r"^(\s*)\}\s*$")


@dataclass
class _IfNode:
    """Parsed ``IF(rec[k]) { then } ELSE? { else }`` block."""

    condition_rec: int
    then_body: list[_Entry] = field(default_factory=list)
    else_body: list[_Entry] | None = None


_Entry = str | _IfNode


def _parse(lines: list[str], pos: int, depth_indent: str | None) -> tuple[list[_Entry], int]:
    """Recursive-descent parse of an indented block body.

    Walks ``lines`` from ``pos``.  When ``depth_indent`` is None, parses
    until end of input (top-level).  Otherwise, parses until the matching
    closing ``}`` (a block body), and returns the index *after* that ``}``.
    """
    out: list[_Entry] = []
    while pos < len(lines):
        line = lines[pos]
        stripped = line.rstrip()
        if not stripped:
            pos += 1
            continue
        close_match = _BLOCK_CLOSE.match(stripped)
        if close_match and depth_indent is not None:
            return out, pos + 1
        else_match = _ELSE_LINE.match(stripped)
        if else_match and depth_indent is not None:
            # Caller (IF parser) handles the ELSE transition; bubble up.
            return out, pos
        if_match = _IF_OPEN.match(stripped)
        if if_match:
            indent = if_match.group(1)
            # Key the block by its first rec offset (a stable id for the
            # condition; a multi-rec XOR condition flips as a single unit).
            cond_rec = int(_FIRST_REC.search(if_match.group(2)).group(1))
            node = _IfNode(condition_rec=cond_rec)
            then_body, pos = _parse(lines, pos + 1, indent)
            node.then_body = then_body
            if pos < len(lines):
                cur = lines[pos].rstrip()
                if _ELSE_LINE.match(cur):
                    else_body, pos = _parse(lines, pos + 1, indent)
                    node.else_body = else_body
                if pos < len(lines) and _BLOCK_CLOSE.match(lines[pos].rstrip()):
                    pos += 1
            out.append(node)
            continue
        out.append(line)
        pos += 1
    return out, pos


def _expand(
    entries: list[_Entry],
    get_outcome: Callable[[_IfNode, int], int],
    sink: list[str],
    counter: list[int],
) -> None:
    """Walk the parsed AST, expanding each IF/ELSE via ``get_outcome``.

    ``get_outcome(node, index)`` returns 0/1 for the ``index``-th IF block
    encountered in program (pre-order) order.
    """
    for entry in entries:
        if isinstance(entry, _IfNode):
            index = counter[0]
            counter[0] += 1
            outcome = get_outcome(entry, index)
            if outcome not in (0, 1):
                raise ValueError(
                    f"Outcome must be 0 or 1; got {outcome} for "
                    f"IF(rec[{entry.condition_rec}]) (block #{index})."
                )
            body = entry.then_body if outcome == 1 else (entry.else_body or [])
            _expand(body, get_outcome, sink, counter)
        else:
            # Strip any indentation introduced by the IF/ELSE nesting so the
            # resulting text parses cleanly.  Internal Stim leading-space is
            # not semantically meaningful.
            sink.append(entry.lstrip())


def resolve_if_else(text: str, conditions: dict[int, int]) -> stim.Circuit:
    """Parse a Stim-text circuit with IF/ELSE blocks and resolve per ``conditions``.

    Args:
        text: Stim text produced by
            :meth:`tqec.compile.conditional.ConditionalCircuit.to_stim_text`,
            which may contain ``IF(rec[k]) { ... } ELSE { ... }`` blocks
            (``ELSE`` arm is optional).
        conditions: mapping from each IF block's ``rec`` offset (the negative
            integer inside ``IF(rec[k])``) to the chosen outcome --
            ``1`` selects the ``IF`` body, ``0`` selects the ``ELSE`` body
            (or empty, if no ``ELSE`` arm).  Every distinct ``rec`` offset
            appearing in the text MUST be present in this mapping. Blocks that
            share a ``rec`` offset all take the same outcome; use
            :func:`resolve_if_else_by_order` to drive them independently.

    Returns:
        A vanilla ``stim.Circuit`` representing the resolved branch.

    """

    def get_outcome(node: _IfNode, index: int) -> int:
        outcome = conditions.get(node.condition_rec)
        if outcome is None:
            raise ValueError(
                f"No outcome supplied for IF(rec[{node.condition_rec}])."
            )
        return outcome

    raw_lines = text.splitlines()
    entries, _ = _parse(raw_lines, 0, depth_indent=None)
    selected: list[str] = []
    _expand(entries, get_outcome, selected, [0])
    return stim.Circuit("\n".join(selected))


def resolve_if_else_by_order(text: str, outcomes: Sequence[int]) -> stim.Circuit:
    """Resolve IF/ELSE blocks by their position, not their ``rec`` offset.

    Use this when several IF blocks share the same ``rec`` offset (so they
    cannot be told apart by :func:`resolve_if_else`) and you need to drive them
    independently -- e.g. two conditional cubes whose relative condition offsets
    coincide but occur at different times.

    Args:
        text: Stim text with ``IF(rec[...]) { ... } ELSE { ... }`` blocks.
        outcomes: one outcome per IF block, in program (pre-order) order.
            ``outcomes[i]`` is 1 to take the i-th block's ``IF`` body, 0 for its
            ``ELSE`` body. Nested blocks in an un-taken branch are not consumed.

    Returns:
        A vanilla ``stim.Circuit`` representing the resolved branch.

    """

    def get_outcome(node: _IfNode, index: int) -> int:
        if index >= len(outcomes):
            raise ValueError(
                f"outcomes has {len(outcomes)} entries but the circuit has more "
                f"IF blocks (reached block #{index})."
            )
        return outcomes[index]

    raw_lines = text.splitlines()
    entries, _ = _parse(raw_lines, 0, depth_indent=None)
    selected: list[str] = []
    _expand(entries, get_outcome, selected, [0])
    return stim.Circuit("\n".join(selected))
