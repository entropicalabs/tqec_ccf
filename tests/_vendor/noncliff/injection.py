from __future__ import annotations
import stim


_Coord = tuple[int, int]
_Pair = tuple[_Coord, _Coord]


def _q(x: int, y: int, n: int) -> int:
    """
    Return qubut index from coordinates.

    Qubits are ordered sequentially x-first.

    Even coordinates: ancillas
    Odd coordinates: data qubits
    """
    # x // 2 * (2 * n + 1) + x % 2 * (n + 1) + y // 2
    return x * n + (x + 1) // 2 + y // 2


def _coord(q: int, n: int) -> _Coord:
    """Inverse of _q on the lattice where x % 2 == y % 2."""
    b, r = divmod(q, 2 * n + 1)
    if r <= n:  # even row: n+1 slots, y even
        x, k = 2 * b, r
    else:  # odd row: n slots, y odd
        x, k = 2 * b + 1, r - (n + 1)
    return x, 2 * k + (x % 2)


def _append_pairs(
    circuit: stim.Circuit, pairs: list[_Pair], n: int, gate: str = "CX"
) -> None:
    targets: list[int] = []
    for (x1, y1), (x2, y2) in pairs:
        targets.extend([_q(x1, y1, n), _q(x2, y2, n)])
    circuit.append(gate, targets)


def _add_coords(circuit: stim.Circuit, d: int) -> None:
    # (d+1) x (d+1) lattice at odd corrdinates for data qubits
    # d x d lattice at even coordiantes for ancillas
    coords = [
        (x, y) for x in range(0, 2 * d + 1, 2) for y in range(0, 2 * d + 1, 2)
    ] + [(x, y) for x in range(1, 2 * d, 2) for y in range(1, 2 * d, 2)]

    for x, y in coords:
        circuit.append("QUBIT_COORDS", [_q(x, y, d)], [x, y])


def _shell_layer_a(n: int, o: int = 0) -> list[_Pair]:
    """First CX layer of the shell that grows a patch from d-2 to d.

    ``o`` is the ring index counted inwards from the boundary of the ``n x n``
    patch; ``o = 0`` is the outermost ring.
    """
    lo, hi = 2 * o + 1, 2 * (n - o) - 1
    pairs: list[_Pair] = [((hi, lo + 2), (hi - 2, lo + 2))]
    pairs += [((x, lo), (x - 2, lo)) for x in range(hi - 2, lo + 3, -4)]
    pairs.append(((lo + 2, lo + 2), (lo + 2, lo)))
    pairs += [((lo, y), (lo, y - 2)) for y in range(lo + 4, hi - 3, 4)]
    pairs.append(((lo, hi - 2), (lo + 2, hi - 2)))
    pairs += [((x, hi), (x + 2, hi)) for x in range(lo + 2, hi - 3, 4)]
    pairs.append(((hi - 2, hi - 2), (hi - 2, hi)))
    pairs += [((hi, y), (hi, y + 2)) for y in range(hi - 4, lo + 3, -4)]
    return pairs


def _shell_layer_b(n: int, o: int = 0) -> list[_Pair]:
    """Second CX layer of the shell that grows a patch from d-2 to d."""
    lo, hi = 2 * o + 1, 2 * (n - o) - 1
    pairs: list[_Pair] = [((x, lo + 2), (x, lo)) for x in range(hi, lo + 3, -2)]
    pairs += [((lo, y), (lo + 2, y)) for y in range(lo, hi - 3, 2)]
    pairs += [((x, hi - 2), (x, hi)) for x in range(lo, hi - 3, 2)]
    pairs += [((hi, y), (hi - 2, y)) for y in range(lo + 4, hi + 1, 2)]
    return pairs


def _append_shell(circuit: stim.Circuit, n: int, o: int = 0) -> None:
    """Reset ring ``o`` of the ``n x n`` patch and entangle it with the interior."""
    layer_a = _shell_layer_a(n, o)
    layer_b = _shell_layer_b(n, o)
    lo, hi = 2 * o + 1, 2 * (n - o) - 1
    perimeter = {
        (x, y)
        for x in range(lo, hi + 1, 2)
        for y in range(lo, hi + 1, 2)
        if x in {lo, hi} or y in {lo, hi}
    }
    # Prepare each new boundary qubit in the basis it is first used in: control
    # of the first CX touching it -> |+>, target -> |0>.
    first_role: dict[_Coord, str] = {}
    for src, dst in layer_a + layer_b:
        if src in perimeter:
            first_role.setdefault(src, "RX")
        if dst in perimeter:
            first_role.setdefault(dst, "R")

    for basis in ("R", "RX"):
        circuit.append(
            basis,
            sorted(_q(x, y, n) for (x, y), role in first_role.items() if role == basis),
        )
    circuit.append("TICK")
    _append_pairs(circuit, layer_a, n)
    circuit.append("TICK")
    _append_pairs(circuit, layer_b, n)


def injection(
    d: int = 3, state_inst: str = "S_DAG", reset_inst: str = "RX"
) -> stim.Circuit:
    """Encode ``state_inst`` applied to the centre qubit into a d x d patch."""
    if d < 3 or d % 2 == 0:
        raise ValueError("d must be odd and at least 3")

    if d > 3:
        # Grow the d-2 patch: re-center it, then wrap it in one shell.
        circuit = stim.Circuit()
        _add_coords(circuit, d)
        circuit += shift_circuit(injection(d - 2, state_inst, reset_inst), d - 2, d, 2)
        circuit.append("TICK")
        _append_shell(circuit, d)
        return circuit

    circuit = stim.Circuit()

    _add_coords(circuit, d)

    circuit.append("TICK")
    circuit.append(reset_inst, [_q(d, d, d)])
    circuit.append("TICK")
    circuit.append(state_inst, _q(d, d, d))
    circuit.append("TICK")
    circuit.append("R", [_q(1, 1, 3), _q(1, 5, 3), _q(5, 1, 3), _q(5, 5, 3)])
    circuit.append("RX", [_q(3, 1, 3), _q(3, 5, 3), _q(1, 3, 3), _q(5, 3, 3)])
    circuit.append("TICK")
    _append_pairs(circuit, [((3, 1), (1, 1)), ((1, 3), (3, 3)), ((3, 5), (5, 5))], 3)
    circuit.append("TICK")
    _append_pairs(circuit, [((1, 3), (1, 5)), ((3, 3), (3, 1)), ((5, 3), (5, 1))], 3)
    circuit.append("TICK")
    _append_pairs(circuit, [((1, 3), (1, 1)), ((5, 1), (3, 1)), ((5, 3), (3, 3))], 3)
    circuit.append("TICK")
    _append_pairs(circuit, [((5, 3), (5, 5)), ((3, 3), (3, 5))], 3)

    return circuit


def shift_circuit(
    circuit: stim.Circuit,
    old_n: int,
    new_n: int,
    shift: int,
    include_coords: bool = False,
) -> stim.Circuit:
    """Shift circuit diagonally by shift onto a grid of size new_n x new_n"""
    shifted = stim.Circuit()
    if include_coords:
        _add_coords(shifted, new_n)
    for instruction in circuit.flattened():
        if instruction.name == "QUBIT_COORDS":
            continue
        targets = []
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                x, y = _coord(target.qubit_value, old_n)
                targets.append(stim.GateTarget(_q(x + shift, y + shift, new_n)))
            else:
                targets.append(target)
        shifted.append(instruction.name, targets, instruction.gate_args_copy())
    return shifted


# Sub-layer schedules of the two check types. Both walk the plaquette corners
# from -- to ++, X checks row-first and Z checks column-first, so hook errors
# run parallel to the boundary they can reach. The idle slot (``None``) keeps
# the two orders from ever contending for the same data qubit.
_CX_ORDER: tuple[_Coord | None, ...] = ((-1, -1), (1, -1), (-1, 1), None, (1, 1))
_CZ_ORDER: tuple[_Coord | None, ...] = ((-1, -1), None, (-1, 1), (1, -1), (1, 1))

_Stabilizer = tuple[str, _Coord, list[_Coord | None]]


def stabilizers(d: int) -> list[_Stabilizer]:
    """Checks of the rotated code as ``(gate, ancilla, schedule)``, ancilla-index order.

    ``gate`` is ``CX`` for an X check and ``CZ`` for a Z check; ``schedule[i]``
    is the data qubit the ancilla couples to in sub-layer ``i``, or ``None``
    where the plaquette has no corner there.
    """
    data = {(x, y) for x in range(1, 2 * d, 2) for y in range(1, 2 * d, 2)}
    stabs: list[_Stabilizer] = []
    for px in range(0, 2 * d + 1, 2):
        for py in range(0, 2 * d + 1, 2):
            # Checkerboard: even half-parity plaquettes measure Z, odd ones X.
            is_z = ((px + py) // 2) % 2 == 0
            support = [
                (px + dx, py + dy)
                for dx in (-1, 1)
                for dy in (-1, 1)
                if (px + dx, py + dy) in data
            ]
            # Two-body checks live only on the pair of edges matching their type.
            if len(support) != 4 and not (
                len(support) == 2 and (px in (0, 2 * d)) == is_z
            ):
                continue
            stabs.append(
                (
                    "CZ" if is_z else "CX",
                    (px, py),
                    [
                        (px + o[0], py + o[1])
                        if o is not None and (px + o[0], py + o[1]) in data
                        else None
                        for o in (_CZ_ORDER if is_z else _CX_ORDER)
                    ],
                )
            )
    return sorted(stabs, key=lambda stab: _q(*stab[1], d))


def _append_se_round(circuit: stim.Circuit, d: int, stabs: list[_Stabilizer]) -> None:
    """One syndrome extraction round: reset ancillas, five gate sub-layers, MX."""
    ancillas = [_q(*anc, d) for _, anc, _ in stabs]
    circuit.append("RX", ancillas)
    for slot in range(len(_CX_ORDER)):
        circuit.append("TICK")
        for gate in ("CX", "CZ"):
            targets = [
                q
                for g, anc, schedule in stabs
                if g == gate and schedule[slot] is not None
                for q in (_q(*anc, d), _q(*schedule[slot], d))
            ]
            if targets:
                circuit.append(gate, targets)
    circuit.append("TICK")
    circuit.append("MX", ancillas)


def injection_memory(
    d: int = 3,
    rounds: int = 3,
    state_inst: str = "S_DAG",
    reset_inst: str = "RX",
    basis: str = "X",
    observable: int = 0,
) -> stim.Circuit:
    """Inject into a d x d patch, hold it for ``rounds``, read out transversally.

    The injection is separated from the first round by an empty layer, which is
    what ``clifft_sim.get_empty_layer_idx`` keys on to leave it noiseless.

    ``reset_inst`` prepares the centre and ``basis`` picks the transversal
    readout; the two are independent. The default pair injects a state on the
    XY equator and reads out logical X, so a non-Clifford ``state_inst`` leaves
    the observable non-deterministic -- that is the measurement of interest,
    not a mistake. Set ``reset_inst="R", basis="Z"`` for a plain Z memory.
    """
    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    if rounds < 0:
        raise ValueError("rounds must be non-negative")

    circuit = injection(d, state_inst, reset_inst)
    stabs = stabilizers(d)
    n_anc = len(stabs)

    for r in range(rounds):
        circuit.append("TICK")
        circuit.append("TICK")  # empty layer
        _append_se_round(circuit, d, stabs)
        for i, (_, (ax, ay), _) in enumerate(stabs):
            # Injection leaves every check at +1, so round 0 is already deterministic.
            recs = [stim.target_rec(i - n_anc)]
            if r:
                recs.append(stim.target_rec(i - 2 * n_anc))
            circuit.append("DETECTOR", recs, [ax, ay, r])

    data = sorted(
        ((x, y) for x in range(1, 2 * d, 2) for y in range(1, 2 * d, 2)),
        key=lambda c: _q(*c, d),
    )
    rec_of = {c: i - len(data) for i, c in enumerate(data)}
    circuit.append("TICK")
    circuit.append("TICK")  # empty layer
    circuit.append("MX" if basis == "X" else "M", [_q(*c, d) for c in data])

    # Rebuild the same-basis checks from the data measurements.
    readout_gate = "CX" if basis == "X" else "CZ"
    for i, (gate, (ax, ay), schedule) in enumerate(stabs):
        if gate != readout_gate:
            continue
        recs = [stim.target_rec(rec_of[c]) for c in schedule if c is not None]
        if rounds:
            recs.append(stim.target_rec(i - n_anc - len(data)))
        circuit.append("DETECTOR", recs, [ax, ay, rounds])

    # Logical X is a column, logical Z a row; both are taken through the centre.
    logical = [c for c in data if (c[0] if basis == "X" else c[1]) == d]
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(rec_of[c]) for c in logical],
        [observable],
    )
    return circuit


# Crumble draws a check as a translucent quad; blue for Z, red for X, matching
# the hand-drawn circuits in circuits/.
_POLYGON_COLOR = {"CZ": "0,0,1,0.25", "CX": "1,0,0,0.25"}

# Crumble's URL codec, from Stim's glue/crumble/editor/sync_url_to_state.js.
# Order matters: the pragma prefix has to go before spaces become underscores.
_CRUMBLE_SUBS = (
    ("QUBIT_COORDS", "Q"),
    ("DETECTOR", "DT"),
    ("OBSERVABLE_INCLUDE", "OI"),
    ("#!pragma ", ""),
    (", ", ","),
    (") ", ")"),
    (" ", "_"),
    ("\n", ";"),
)


def polygon_pragmas(d: int) -> list[str]:
    """``#!pragma POLYGON`` lines outlining every check of the d x d patch."""
    lines = []
    for gate, (px, py), schedule in stabilizers(d):
        support = {c for c in schedule if c is not None}
        # Walk the corners as a cycle, so a four-body check renders as a square
        # rather than a bowtie. The schedule order would zig-zag.
        cycle = [
            (px + dx, py + dy)
            for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            if (px + dx, py + dy) in support
        ]
        qubits = " ".join(str(_q(x, y, d)) for x, y in cycle)
        lines.append(f"#!pragma POLYGON({_POLYGON_COLOR[gate]}) {qubits}")
    return lines


def crumble_url(circuit: stim.Circuit, d: int, every_layer: bool = False) -> str:
    """Crumble URL for ``circuit``, with the checks of a d x d patch outlined.

    Stim drops ``#!pragma`` lines when it parses a circuit, so the polygons
    cannot survive a round trip through :class:`stim.Circuit` and are spliced
    into the text here instead. They are layer-scoped in crumble; ``every_layer``
    repeats them so the patch stays visible while stepping through the rounds.
    """
    pragmas = polygon_pragmas(d)
    lines: list[str] = []
    placed = False
    for line in str(circuit).splitlines():
        # The coords block comes first; the polygons reference those qubits.
        if not placed and not line.startswith("QUBIT_COORDS"):
            lines += pragmas
            placed = True
        lines.append(line)
        if every_layer and line == "TICK":
            lines += pragmas
    if not placed:
        lines += pragmas

    text = "\n".join(lines)
    for old, new in _CRUMBLE_SUBS:
        text = text.replace(old, new)
    return "https://algassert.com/crumble#circuit=" + text


if __name__ == "__main__":
    print(crumble_url(injection_memory(7, state_inst="I", basis="X"), d=7))
