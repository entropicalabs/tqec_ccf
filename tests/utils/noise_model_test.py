"""Test for `noise_model.py`.

# Important note:

This file has been recovered from https://zenodo.org/records/7487893
and is under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)
It is part of the code from the paper

    Gidney, C. (2022). Data for "Inplace Access to the Surface Code Y Basis".
    https://doi.org/10.5281/zenodo.7487893


Modifications to the original code:
1. Formatting with ruff.
2. Fixing typing issues and adapting a few imports to personal taste
3. Removing the line "DEPOLARIZE1(0.001) 0 1 2 3" from test_si_1000 and
   test_si_1000_repeat_block due to the removal of that noise from the main
   noise_model.py file.

"""

import pytest
import stim

from tqec.compile.compile import compile_block_graph
from tqec.gallery import memory
from tqec.utils.noise_model import (
    NoiseModel,
    _iter_split_op_moments,
    _measure_basis,
    occurs_in_classical_control_system,
)


def test_measure_basis() -> None:
    def f(code: str) -> str | None:
        first_instruction = stim.Circuit(code)[0]
        assert isinstance(first_instruction, stim.CircuitInstruction)
        return _measure_basis(split_op=first_instruction)

    assert f("H") is None
    assert f("H 0") is None
    assert f("R 0 1 2") is None

    assert f("MX") == "X"
    assert f("MX(0.01) 1") == "X"
    assert f("MY 0 1") == "Y"
    assert f("MZ 0 1 2") == "Z"
    assert f("M 0 1 2") == "Z"

    assert f("MRX") is None

    assert f("MPP X5") == "X"
    assert f("MPP X0*X2") == "XX"
    assert f("MPP Y0*Z2*X3") == "YZX"


def test_iter_split_op_moments() -> None:
    assert (
        list(
            _iter_split_op_moments(
                stim.Circuit(""""""),
                immune_qubits=set(),
            )
        )
        == []
    )

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""H 0"""),
            immune_qubits=set(),
        )
    ) == [[stim.CircuitInstruction("H", [0])]]

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""H 0\nTICK"""),
            immune_qubits=set(),
        )
    ) == [[stim.CircuitInstruction("H", [0])]]

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""H 0 1\nTICK"""),
            immune_qubits=set(),
        )
    ) == [[stim.CircuitInstruction("H", [0, 1])]]

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""H 0 1\nTICK"""),
            immune_qubits={3},
        )
    ) == [
        [stim.CircuitInstruction("H", [0]), stim.CircuitInstruction("H", [1])],
    ]

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""H 0\nTICK\nH 1"""),
            immune_qubits=set(),
        )
    ) == [
        [stim.CircuitInstruction("H", [0])],
        [stim.CircuitInstruction("H", [1])],
    ]

    assert list(
        _iter_split_op_moments(
            stim.Circuit("""
        CX rec[-1] 0 1 2 3 4
        MPP X5*X6 Y5
        CX 8 9 10 11
        TICK
        H 0
    """),
            immune_qubits=set(),
        )
    ) == [
        [
            stim.CircuitInstruction("CX", [stim.target_rec(-1), 0]),
            stim.CircuitInstruction("CX", [1, 2]),
            stim.CircuitInstruction("CX", [3, 4]),
            stim.CircuitInstruction(
                "MPP", [stim.target_x(5), stim.target_combiner(), stim.target_x(6)]
            ),
            stim.CircuitInstruction("MPP", [stim.target_y(5)]),
            stim.CircuitInstruction("CX", [8, 9, 10, 11]),
        ],
        [
            stim.CircuitInstruction("H", [0]),
        ],
    ]


def test_occurs_in_classical_control_system() -> None:
    assert not occurs_in_classical_control_system(op=stim.CircuitInstruction("H", [0]))
    assert not occurs_in_classical_control_system(op=stim.CircuitInstruction("CX", [0, 1, 2, 3]))
    assert not occurs_in_classical_control_system(op=stim.CircuitInstruction("M", [0, 1, 2, 3]))

    assert occurs_in_classical_control_system(
        op=stim.CircuitInstruction("CX", [stim.target_rec(-1), 0])
    )
    assert occurs_in_classical_control_system(
        op=stim.CircuitInstruction("DETECTOR", [stim.target_rec(-1)])
    )
    assert occurs_in_classical_control_system(op=stim.CircuitInstruction("TICK", []))
    assert occurs_in_classical_control_system(op=stim.CircuitInstruction("SHIFT_COORDS", []))


def test_si_1000() -> None:
    model = NoiseModel.si1000(1e-3)
    assert model.noisy_circuit(
        stim.Circuit("""
        R 0 1 2 3
        TICK
        ISWAP 0 1 2 3 4 5
        TICK
        H 4 5 6 7
        TICK
        M 0 1 2 3
    """)
    ) == stim.Circuit("""
        R 0 1 2 3
        X_ERROR(0.002) 0 1 2 3
        DEPOLARIZE1(0.0001) 4 5 6 7
        DEPOLARIZE1(0.002) 4 5 6 7
        TICK
        ISWAP 0 1 2 3 4 5
        DEPOLARIZE2(0.001) 0 1 2 3 4 5
        DEPOLARIZE1(0.0001) 6 7
        TICK
        H 4 5 6 7
        DEPOLARIZE1(0.0001) 4 5 6 7 0 1 2 3
        TICK
        M(0.005) 0 1 2 3
        DEPOLARIZE1(0.0001) 4 5 6 7
        DEPOLARIZE1(0.002) 4 5 6 7
    """)


###############################################################################
# Starting from here, code was not part of the original file. This means that #
# everything below that message is a modification of the original CC BY 4.0   #
# code                                                                        #
###############################################################################


def test_si_1000_repeat_block() -> None:
    model = NoiseModel.si1000(1e-3)
    assert model.noisy_circuit(
        stim.Circuit("""
        REPEAT 10 {
            R 0 1 2 3
            TICK
            ISWAP 0 1 2 3 4 5
            TICK
            H 4 5 6 7
            TICK
            M 0 1 2 3
        }
    """)
    ) == stim.Circuit("""
        REPEAT 10 {
            R 0 1 2 3
            X_ERROR(0.002) 0 1 2 3
            DEPOLARIZE1(0.0001) 4 5 6 7
            DEPOLARIZE1(0.002) 4 5 6 7
            TICK
            ISWAP 0 1 2 3 4 5
            DEPOLARIZE2(0.001) 0 1 2 3 4 5
            DEPOLARIZE1(0.0001) 6 7
            TICK
            H 4 5 6 7
            DEPOLARIZE1(0.0001) 4 5 6 7 0 1 2 3
            TICK
            M(0.005) 0 1 2 3
            DEPOLARIZE1(0.0001) 4 5 6 7
            DEPOLARIZE1(0.002) 4 5 6 7
        }
    """)


def test_a_tick_at_a_repeat_edge_opens_no_idle_moment() -> None:
    # tqec starts every REPEAT body with a TICK and follows a block with one.
    # Each closes a moment the block edge already closed, so neither may add a
    # moment of idling: the noisy circuit must be the noisy flattened circuit.
    model = NoiseModel.uniform_depolarizing(1e-3)
    circuit = stim.Circuit("""
        R 0 1 2
        TICK
        H 0
        REPEAT 3 {
            TICK
            CX 0 1
            TICK
            M 1
        }
        TICK
        H 2
    """)
    noisy = model.noisy_circuit(circuit)
    assert noisy.flattened() == model.noisy_circuit(circuit.flattened())
    assert noisy == stim.Circuit("""
        R 0 1 2
        X_ERROR(0.001) 0 1 2
        TICK
        H 0
        DEPOLARIZE1(0.001) 0 1 2
        REPEAT 3 {
            TICK
            CX 0 1
            DEPOLARIZE2(0.001) 0 1
            DEPOLARIZE1(0.001) 2
            TICK
            M(0.001) 1
            DEPOLARIZE1(0.001) 0 2
        }
        TICK
        H 2
        DEPOLARIZE1(0.001) 2 0 1
    """)


@pytest.mark.parametrize("k", [2, 3])
def test_noisy_memory_is_the_noisy_flattened_memory(k: int) -> None:
    model = NoiseModel.uniform_depolarizing(1e-3)
    circuit = compile_block_graph(memory()).generate_stim_circuit(k)
    assert model.noisy_circuit(circuit).flattened() == model.noisy_circuit(circuit.flattened())
