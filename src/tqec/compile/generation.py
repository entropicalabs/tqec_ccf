"""Defines :meth:`.generate_circuit`, one of the core method of the :mod:`tqec` package.

This module defines two of core methods of the :mod:`tqec` package:

- :meth:`generate_circuit` that is the most convenient method for external users.
- :meth:`generate_circuit_from_instantiation` that gives more freedom to the
  user (and is used internally by :meth:`generate_circuit`) at the expense of
  often less convenient inputs.

Both of these methods are used to generate the ``stim.Circuit`` instance
representing **one** time slice (often equivalent to "one QEC round"). Users are
expected to call these methods several times and concatenate the output
``stim.Circuit`` instances in time to obtain a full QEC implementation.

Note that these methods do not work with ``REPEAT`` instructions.

"""

from __future__ import annotations

from collections.abc import Mapping

import numpy
import numpy.typing as npt

from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule import (
    ScheduledCircuit,
    merge_scheduled_circuits,
    merge_scheduled_circuits_per_branch,
    relabel_circuits_qubit_indices,
)
from tqec.compile.conditional.circuit import CircuitEntry
from tqec.plaquette.plaquette import Plaquettes
from tqec.templates.base import Template
from tqec.utils.array import to2dlist
from tqec.utils.position import BlockPosition2D, Shift2D


def generate_circuit(
    template: Template,
    k: int,
    plaquettes: Plaquettes,
    plaquette_to_block: Mapping[int, BlockPosition2D] | None = None,
) -> ScheduledCircuit:
    """Generate a quantum circuit from a template and its plaquettes.

    This is one of the core methods of the :mod:`tqec` package. It generates a
    quantum circuit from the description of the template that should be
    implemented as well as the plaquettes that should be used to instantiate the
    provided template.

    This function requires that a few pre-conditions on the inputs are met:

    1. the number of plaquettes provided should match the number of plaquettes
       required by the provided template.
    2. all the provided plaquettes should be implemented on
       :class:`~.qubit.GridQubit` instances **only**.

    If any of the above pre-conditions is not met, the inputs are considered
    invalid, in which case this function **might** raise an error.

    Args:
        template: spatial description of the quantum error correction experiment
            we want to implement.
        k: scaling parameter used to instantiate the provided ``template``.
        plaquettes: description of the computation that should happen at
            different time-slices of the quantum error correction experiment (or
            at least part of it).

    Returns:
        a :class:`~.schedule.circuit.ScheduledCircuit` instance implementing the
        (part of) quantum error correction experiment represented by the
        provided inputs.

    """
    # instantiate the template with the appropriate plaquette indices.
    # Index 0 is "no plaquette" by convention and should not be included here.
    _indices = list(range(1, template.expected_plaquettes_number + 1))
    template_plaquettes = template.instantiate(k, _indices)
    increments = template.get_increments()

    return generate_circuit_from_instantiation(
        template_plaquettes, plaquettes, increments, plaquette_to_block=plaquette_to_block
    )


def generate_circuit_from_instantiation(
    plaquette_array: npt.NDArray[numpy.int_],
    plaquettes: Plaquettes,
    increments: Shift2D,
    plaquette_to_block: Mapping[int, BlockPosition2D] | None = None,
) -> ScheduledCircuit:
    """Generate a quantum circuit from an array of plaquette indices and the associated plaquettes.

    This is one of the core methods of the :mod:`tqec` package. It generates a
    quantum circuit from a spatial description of where the plaquettes should be
    located as well as the actual plaquettes used.

    This function requires that a few pre-conditions on the inputs are met:

    1. the number of plaquettes provided should match the number of plaquettes
       required by the provided template.
    2. all the provided plaquettes should be implemented on
       :class:`~.qubit.GridQubit` instances **only**.

    If any of the above pre-conditions is not met, the inputs are considered
    invalid, in which case this function **might** raise an error.

    Args:
        plaquette_array: a 2-dimensional array of indices referencing
            :class:`~tqec.plaquette.plaquette.Plaquette` instances in the
            ``plaquettes`` argument.
        plaquettes: description of the computation that should happen at
            different time-slices of the quantum error correction experiment (or
            at least part of it).
        increments: the displacement between each plaquette origin.

    Returns:
        a :class:`~.schedule.circuit.ScheduledCircuit` instance implementing the
        (part of) quantum error correction experiment represented by the
        provided inputs.

    Raises:
        TQECError: if any index in ``plaquette_array`` is not correctly
            associated to a plaquette in ``plaquettes``.

    """
    all_scheduled_circuits, qubit_to_block, additional_mergeable_instructions = (
        _build_scheduled_circuits_for_plaquette_array(
            plaquette_array, plaquettes, increments, plaquette_to_block
        )
    )

    # Merge everything, but first make sure that the circuits are compatible.
    # Note that relabel_circuits_qubit_indices guarantees in its documentation
    # that the input circuits are not mutated but rather copied. This allows us
    # to not deepcopy the circuits earlier in the function.
    all_scheduled_circuits, qubit_map = relabel_circuits_qubit_indices(all_scheduled_circuits)
    return merge_scheduled_circuits(
        all_scheduled_circuits,
        qubit_map,
        additional_mergeable_instructions,
        qubit_to_block=qubit_to_block,
    )


def _build_scheduled_circuits_for_plaquette_array(
    plaquette_array: npt.NDArray[numpy.int_],
    plaquettes: Plaquettes,
    increments: Shift2D,
    plaquette_to_block: Mapping[int, BlockPosition2D] | None,
) -> tuple[list[ScheduledCircuit], dict[GridQubit, BlockPosition2D] | None, set[str]]:
    """Walk ``plaquette_array`` row-major and produce one mapped ``ScheduledCircuit``
    per non-zero entry. Returns the per-plaquette list together with the accumulated
    ``GridQubit -> BlockPosition2D`` ownership map (when requested) and the union of
    plaquette-declared mergeable instruction names.
    """
    indices = numpy.unique(plaquette_array)
    if indices[0] == 0:
        indices = indices[1:]

    plaquette_circuits = {0: ScheduledCircuit.empty()} | {
        i: plaquettes[i].circuit for i in indices.tolist()
    }

    all_scheduled_circuits: list[ScheduledCircuit] = []
    additional_mergeable_instructions: set[str] = set()
    qubit_to_block: dict[GridQubit, BlockPosition2D] | None = (
        {} if plaquette_to_block is not None else None
    )
    plaquette_array_list: list[list[int]] = to2dlist(plaquette_array)
    for row_index, line in enumerate(plaquette_array_list):
        for column_index, plaquette_index in enumerate(line):
            if plaquette_index == 0:
                continue
            plaquette = plaquettes[plaquette_index]
            qubit_offset = Shift2D(
                plaquette.origin.x + column_index * increments.x,
                plaquette.origin.y + row_index * increments.y,
            )
            scheduled_circuit = plaquette_circuits[plaquette_index]
            mapped_scheduled_circuit = scheduled_circuit.map_to_qubits(
                lambda q: q + qubit_offset, inplace_qubit_map=False
            )
            all_scheduled_circuits.append(mapped_scheduled_circuit)
            additional_mergeable_instructions |= plaquette.mergeable_instructions
            if qubit_to_block is not None and plaquette_to_block is not None:
                block_pos = plaquette_to_block.get(plaquette_index)
                if block_pos is not None:
                    for q in mapped_scheduled_circuit.qubits:
                        qubit_to_block.setdefault(q, block_pos)
    return all_scheduled_circuits, qubit_to_block, additional_mergeable_instructions


def generate_per_branch_circuit_from_instantiation(
    plaquette_array: npt.NDArray[numpy.int_],
    zero_plaquettes: Plaquettes,
    one_plaquettes: Plaquettes,
    increments: Shift2D,
    plaquette_to_block: Mapping[int, BlockPosition2D],
    condition_recs: list[int],
) -> tuple[list[list[CircuitEntry]], QubitMap]:
    """Per-branch sibling of :func:`generate_circuit_from_instantiation`.

    Walks ``plaquette_array`` twice, producing one ``ScheduledCircuit`` list per
    branch from the matching ``Plaquettes`` collection. Both lists share row-major
    ordering and the **same** qubit footprint at every slot (Equal Measurement
    Count + Canonical Emission Order assumption: branches differ only in gate
    *content*, not in *which* qubits they touch). The two lists are then handed to
    :func:`~tqec.circuit.schedule.merge_scheduled_circuits_per_branch`, which weaves
    :class:`IfBlock` entries at divergent CEO slots.

    Args:
        plaquette_array: instantiated template (shared across branches).
        zero_plaquettes: branch-zero plaquette collection.
        one_plaquettes: branch-one plaquette collection. May share entries with
            ``zero_plaquettes`` at non-conditional indices.
        increments: as in :func:`generate_circuit_from_instantiation`.
        plaquette_to_block: required; CEO needs block ownership.
        condition_recs: ``stim`` record offsets whose XOR selects the branch.

    Returns:
        ``(moments_entries, qubit_map)`` where ``moments_entries[i]`` is the
        ``list[CircuitEntry]`` for moment ``i``. The ``Schedule`` produced by
        :func:`merge_scheduled_circuits_per_branch` is collapsed at this layer
        (TICKs are positional in the Stim text); callers that need the explicit
        schedule should use the lower-level helpers.

    """
    zero_list, qubit_to_block_z, mergeable_z = _build_scheduled_circuits_for_plaquette_array(
        plaquette_array, zero_plaquettes, increments, plaquette_to_block
    )
    one_list, qubit_to_block_o, mergeable_o = _build_scheduled_circuits_for_plaquette_array(
        plaquette_array, one_plaquettes, increments, plaquette_to_block
    )
    assert qubit_to_block_z is not None and qubit_to_block_o is not None
    if qubit_to_block_z != qubit_to_block_o:
        from tqec.utils.exceptions import TQECError

        raise TQECError(
            "generate_per_branch_circuit_from_instantiation: per-branch qubit "
            "ownership maps differ; Equal Measurement Count + CEO assumption violated."
        )
    relabeled, qubit_map = relabel_circuits_qubit_indices(zero_list + one_list)
    zero_relabeled = relabeled[: len(zero_list)]
    one_relabeled = relabeled[len(zero_list) :]
    moments_entries, _schedule = merge_scheduled_circuits_per_branch(
        zero_relabeled,
        one_relabeled,
        qubit_map,
        condition_recs=condition_recs,
        mergeable_instructions=mergeable_z | mergeable_o,
        qubit_to_block=qubit_to_block_z,
    )
    return moments_entries, qubit_map
