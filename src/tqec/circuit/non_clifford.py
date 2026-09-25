"""Rendering a circuit whose gates stim cannot all represent.

A state-injection cube can prepare a non-stabilizer state, which needs a gate
stim has no name for: ``stim.Circuit("T 0")`` raises. Such a gate is compiled as
the Clifford gate of the same rotation axis and half the angle, tagged with the
gate it stands in for --- see :mod:`tqec.utils.injection_state` --- so the whole
compile runs on an ordinary :class:`stim.Circuit`. What the tag buys is that the
real gate can be put back at the one point it matters: when the circuit is
serialised for something other than stim.

The result is *text*, deliberately. It is not parseable by
:class:`stim.Circuit`, which is the honest signal that it describes a
computation stim cannot simulate.

Compare :mod:`tqec.compile.conditional.circuit`, which solves the neighbouring
problem for ``IF``/``ELSE`` blocks. There the construct is structural and stim's
data model cannot hold it at all, so it needs its own container and renderer.
Here only one instruction's *name* differs, so a rewrite at the text boundary is
enough --- and it keeps the detector annotator, the observable builder and the
noise model all working on a real circuit.
"""

from __future__ import annotations

import re

import stim

from tqec.utils.injection_state import NON_CLIFFORD_INJECTION_TAGS

_TAGGED_LINE = re.compile(
    r"^(\s*)[A-Za-z_0-9]+\[("
    + "|".join(sorted(map(re.escape, NON_CLIFFORD_INJECTION_TAGS)))
    + r")\](.*)$"
)
"""A rendered instruction standing in for a gate stim cannot represent.

Anchored at the start of a line with the indentation captured, and restricted to
the known tags, so an unrelated tag added later is left alone.
"""


# stim indents a REPEAT body by four spaces; matching it exactly is what makes
# this renderer byte-identical to ``str(circuit)`` when nothing is tagged.
_INDENT = "    "


def _render(circuit: stim.Circuit, lines: list[str], indent: int) -> None:
    pad = _INDENT * indent
    for entry in circuit:
        if isinstance(entry, stim.CircuitRepeatBlock):
            lines.append(f"{pad}REPEAT {entry.repeat_count} {{")
            _render(entry.body_copy(), lines, indent + 1)
            lines.append(f"{pad}}}")
            continue
        rendered = str(entry)
        if entry.tag in NON_CLIFFORD_INJECTION_TAGS:
            # `S_DAG[T_DAG] 12` -> `T_DAG 12`. Substituting `name[tag]` is exactly
            # what stim renders, so this is not pattern-guessing.
            rendered = rendered.replace(f"{entry.name}[{entry.tag}]", entry.tag, 1)
        lines.extend(f"{pad}{line}" for line in rendered.splitlines())


def render_with_non_clifford_gates(circuit: stim.Circuit) -> str:
    """Render ``circuit`` as text, restoring the gates its tags stand in for.

    Per-instruction rendering is delegated to stim, so only the ``REPEAT``
    framing and the tagged names are this function's own. Byte-identical to
    ``str(circuit)`` when no instruction carries a known non-Clifford tag.

    Args:
        circuit: the circuit to render. Its tagged stand-in gates are replaced by
            the gates they stand in for.

    Returns:
        the circuit as stim text. If any substitution happened, the result is
        **not** parseable by :class:`stim.Circuit`.

    """
    lines: list[str] = []
    _render(circuit, lines, indent=0)
    return "\n".join(lines)


def rewrite_non_clifford_tags(text: str) -> str:
    """Restore stood-in-for gates in text that was rendered elsewhere.

    The line-wise counterpart of :func:`render_with_non_clifford_gates`, for text
    produced by
    :meth:`~tqec.compile.conditional.circuit.ConditionalCircuit.to_stim_text`,
    which renders each instruction with stim's own ``__str__`` and so preserves
    the tags.

    Args:
        text: stim text possibly containing tagged stand-in gates.

    Returns:
        the same text with each tagged stand-in replaced by the gate it stands
        in for.

    """
    return "\n".join(_TAGGED_LINE.sub(r"\1\2\3", line) for line in text.splitlines())


def has_non_clifford_gates(circuit: stim.Circuit) -> bool:
    """Return whether ``circuit`` carries any tagged stand-in gate."""
    return any(
        instruction.tag in NON_CLIFFORD_INJECTION_TAGS for instruction in circuit.flattened()
    )
