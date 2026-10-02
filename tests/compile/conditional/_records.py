"""Where the condition and observable records of an IF/ELSE circuit are measured."""

from __future__ import annotations

import re

import stim

from tools.resolve import resolve_if_else_by_order

_IF_OPEN = re.compile(r"^(\s*)IF\((rec\[-?\d+\](?:\^rec\[-?\d+\])*)\)\s*\{\s*$")
# A DETECTOR with this argument marks an IF header in the resolved circuit.
_MARK = -1.0


def measured_coordinates(text: str, outcome: int) -> list[tuple[str, frozenset[tuple[float, ...]]]]:
    """Return what every IF condition and OBSERVABLE_INCLUDE reads, in program order.

    Each IF block takes its ``IF`` arm when ``outcome`` is 1 and its ``ELSE`` arm
    when it is 0. Every entry is ``("IF", coordinates)`` or ``("OBS<i>",
    coordinates)``, with the set of coordinates of the qubits whose measurements
    the condition or the observable XORs. Two circuits reading the same
    measurements give the same list, however many other qubits they measure.
    """
    # Copy every IF header into a marker detector, outside the IF block, so that
    # it survives the resolution with its records.
    marked = []
    for line in text.splitlines():
        match = _IF_OPEN.match(line)
        if match:
            marked.append(f"{match.group(1)}DETECTOR({_MARK}) {match.group(2).replace('^', ' ')}")
        marked.append(line)
    n_blocks = sum(bool(_IF_OPEN.match(line)) for line in text.splitlines())
    circuit = resolve_if_else_by_order("\n".join(marked), [outcome] * n_blocks).flattened()

    coordinates = circuit.get_final_qubit_coordinates()
    measured: list[int] = []
    result: list[tuple[str, frozenset[tuple[float, ...]]]] = []
    for instruction in circuit:
        assert isinstance(instruction, stim.CircuitInstruction)
        if stim.gate_data(instruction.name).produces_measurements:
            measured.extend(t.value for t in instruction.targets_copy() if t.is_qubit_target)
            continue
        if instruction.name == "DETECTOR" and instruction.gate_args_copy() == [_MARK]:
            name = "IF"
        elif instruction.name == "OBSERVABLE_INCLUDE":
            name = f"OBS{int(instruction.gate_args_copy()[0])}"
        else:
            continue
        qubits = [measured[len(measured) + t.value] for t in instruction.targets_copy()]
        result.append((name, frozenset(tuple(coordinates[q]) for q in qubits)))
    return result
