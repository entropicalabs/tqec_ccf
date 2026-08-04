"""End-to-end compilation tests for the native Y-half cube (S-gate cap).

A single Y-capped memory column must compile all the way through
``compile_block_graph`` to a valid ``stim.Circuit`` whose detectors are all
deterministic, with no fictitious ``MPP`` elements (the physical-seam
property).
"""

from __future__ import annotations

import pytest

from tqec import BlockGraph, compile_block_graph
from tqec.computation.cube import LeafCubeKind, ZXCube
from tqec.utils.position import Position3D


def _y_capped_column() -> BlockGraph:
    g = BlockGraph("y_capped_column")
    g.add_cube(Position3D(0, 0, 0), ZXCube.from_str("ZXZ"))
    g.add_cube(Position3D(0, 0, 1), LeafCubeKind.Y_HALF_CUBE)
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))
    return g


@pytest.mark.parametrize("k", [1, 2])
def test_y_capped_column_compiles(k: int) -> None:
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=k)
    assert circuit.num_detectors > 0
    assert circuit.num_qubits > 0
    # A lone Y column has no closed correlation surface, so no observable.
    assert circuit.num_observables == 0


@pytest.mark.parametrize("k", [1, 2])
def test_y_capped_column_detectors_are_deterministic(k: int) -> None:
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=k)
    # Raises if any detector is non-deterministic -- exercises the seam detectors
    # the annotator emits across the transition round.
    circuit.detector_error_model(decompose_errors=False)


def test_y_capped_column_has_no_mpp() -> None:
    """The native Y cap uses no fictitious multi-qubit Pauli measurements."""
    circuit = compile_block_graph(_y_capped_column(), observables="auto").generate_stim_circuit(k=1)
    assert not any(inst.name == "MPP" for inst in circuit.flattened())
