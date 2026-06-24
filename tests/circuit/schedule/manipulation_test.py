import pytest
import stim

from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.circuit.schedule.circuit import ScheduledCircuit
from tqec.circuit.schedule.manipulation import (
    _emit_moment_with_ceo,
    merge_instructions,
    merge_scheduled_circuits,
    merge_scheduled_circuits_per_branch,
    relabel_circuits_qubit_indices,
    remove_duplicate_instructions,
)
from tqec.compile.conditional.circuit import IfBlock
from tqec.utils.exceptions import TQECError, TQECWarning
from tqec.utils.position import BlockPosition2D


def test_remove_duplicate_instructions() -> None:
    instructions: list[stim.CircuitInstruction] = list(
        iter(stim.Circuit("H 0 0 0 2 0 0 0 0 1 1 2 2 0 0 0 3"))
    )  # type: ignore
    expected_instructions = set(iter(stim.Circuit("H 0 1 2 3")))  # type: ignore
    assert (
        set(remove_duplicate_instructions(instructions, frozenset(["H"]))) == expected_instructions
    )
    with pytest.warns(TQECWarning):
        # Several H gates are overlapping, which means that the returned instruction
        # list is not a valid Moment, which should raise a warning.
        assert remove_duplicate_instructions(instructions, frozenset()) == instructions

    instructions = list(iter(stim.Circuit("H 0 1 2 3 0 1 2 3\nM 4 5 6 4 5 6 4 5 6")))  # type: ignore
    with pytest.warns(TQECWarning):
        assert set(remove_duplicate_instructions(instructions, frozenset(["M"]))) == set(
            iter(stim.Circuit("H 0 1 2 3 0 1 2 3\nM 4 5 6"))
        )
    with pytest.warns(TQECWarning):
        assert set(remove_duplicate_instructions(instructions, frozenset(["H"]))) == set(
            iter(stim.Circuit("H 0 1 2 3\nM 4 5 6 4 5 6 4 5 6"))
        )
    assert set(remove_duplicate_instructions(instructions, frozenset(["H", "M"]))) == set(
        iter(stim.Circuit("H 0 1 2 3\nM 4 5 6"))
    )


def test_relabel_circuits_qubit_indices() -> None:
    # Any qubit target not defined by a QUBIT_COORDS instruction should raise
    # an exception.
    with pytest.raises(KeyError):
        relabel_circuits_qubit_indices(
            [
                ScheduledCircuit.from_circuit(stim.Circuit("H 0 1 2")),
                ScheduledCircuit.from_circuit(stim.Circuit("H 0 1 2")),
            ]
        )
    with pytest.raises(KeyError):
        relabel_circuits_qubit_indices(
            [
                ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")),
                ScheduledCircuit.from_circuit(stim.Circuit("H 0")),
            ]
        )
    circuits, q2i = relabel_circuits_qubit_indices(
        [
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")),
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(1, 1) 0\nX 0")),
        ]
    )
    assert q2i == QubitMap({0: GridQubit(0, 0), 1: GridQubit(1, 1)})
    assert len(circuits) == 2
    assert circuits[0].get_circuit() == stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")
    assert circuits[1].get_circuit() == stim.Circuit("QUBIT_COORDS(1, 1) 1\nX 1")


def test_merge_scheduled_circuits() -> None:
    # Any qubit target not defined by a QUBIT_COORDS instruction should raise
    # an exception.
    _circuits, _qubit_map = relabel_circuits_qubit_indices(
        [
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")),
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(1, 1) 0\nX 0")),
        ]
    )
    circuit = merge_scheduled_circuits(_circuits, _qubit_map)
    assert circuit.get_circuit() == stim.Circuit(
        "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 1) 1\nH 0\nX 1"
    )

    circuit = merge_scheduled_circuits(
        [
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")),
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")),
        ],
        global_qubit_map=QubitMap({0: GridQubit(0, 0)}),
        mergeable_instructions=["H"],
    )
    assert circuit.get_circuit() == stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0")

    _circuits, _qubit_map = relabel_circuits_qubit_indices(
        [
            ScheduledCircuit.from_circuit(
                stim.Circuit("QUBIT_COORDS(0, 0) 0\nH 0\nTICK\nM 0"), [0, 2]
            ),
            ScheduledCircuit.from_circuit(stim.Circuit("QUBIT_COORDS(1, 1) 0\nX 0"), 1),
        ]
    )
    circuit = merge_scheduled_circuits(_circuits, _qubit_map)
    assert circuit.get_circuit() == stim.Circuit(
        "QUBIT_COORDS(0, 0) 0\nQUBIT_COORDS(1, 1) 1\nH 0\nTICK\nX 1\nTICK\nM 0"
    )


def test_merge_instructions() -> None:
    circuit = stim.Circuit("H 0 1 2\nCX 3 4\nH 5 7")
    instructions: list[stim.CircuitInstruction] = []
    for instr in circuit:
        assert not isinstance(instr, stim.CircuitRepeatBlock)
        instructions.append(instr)
    merged_instructions = merge_instructions(instructions)
    assert len(merged_instructions) == 2
    assert {instr.name for instr in merged_instructions} == {"H", "CX"}
    for instr in merged_instructions:
        if instr.name == "H":
            assert sorted(t.value for t in instr.targets_copy()) == [0, 1, 2, 5, 7]
        if instr.name == "CX":
            assert instr.target_groups() == [[stim.GateTarget(3), stim.GateTarget(4)]]


def _moment_insts(text: str) -> list[stim.CircuitInstruction]:
    out: list[stim.CircuitInstruction] = []
    for inst in stim.Circuit(text):
        assert isinstance(inst, stim.CircuitInstruction)
        out.append(inst)
    return out


def test_emit_moment_with_ceo_no_branch_returns_plain_entries() -> None:
    q0, q1 = GridQubit(0, 0), GridQubit(1, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0), q1: BlockPosition2D(0, 0)}
    global_i2q = {0: q0, 1: q1}
    entries = _emit_moment_with_ceo(_moment_insts("R 0 1"), qubit_to_block, global_i2q)
    assert len(entries) == 1
    assert not isinstance(entries[0], IfBlock)
    assert entries[0].name == "R"
    assert sorted(t.value for t in entries[0].targets_copy()) == [0, 1]


def test_emit_moment_with_ceo_identical_branches_no_ifblock() -> None:
    q0, q1 = GridQubit(0, 0), GridQubit(1, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0), q1: BlockPosition2D(0, 0)}
    global_i2q = {0: q0, 1: q1}
    entries = _emit_moment_with_ceo(
        _moment_insts("R 0 1"),
        qubit_to_block,
        global_i2q,
        branch_merged_instructions=_moment_insts("R 0 1"),
        condition_recs=[-1],
    )
    assert all(not isinstance(e, IfBlock) for e in entries)
    assert len(entries) == 1
    assert entries[0].name == "R"


def test_emit_moment_with_ceo_weaves_ifblock_for_divergent_branch() -> None:
    q0, q1 = GridQubit(0, 0), GridQubit(1, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0), q1: BlockPosition2D(0, 0)}
    global_i2q = {0: q0, 1: q1}
    entries = _emit_moment_with_ceo(
        _moment_insts("R 0 1"),
        qubit_to_block,
        global_i2q,
        branch_merged_instructions=_moment_insts("RX 0 1"),
        condition_recs=[-1],
    )
    # R/RX are single-qubit gates -> two CEO slots (q0 and q1), each divergent
    # with matching (name, args) signatures across the two slots -> batched
    # into a single IfBlock with merged targets.
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, IfBlock)
    assert entry.condition_recs == [-1]
    assert len(entry.then_body) == 1 and len(entry.else_body or []) == 1
    then_inst = entry.then_body[0]
    else_inst = (entry.else_body or [])[0]
    assert then_inst.name == "RX"
    assert else_inst.name == "R"
    assert [t.value for t in then_inst.targets_copy()] == [0, 1]
    assert [t.value for t in else_inst.targets_copy()] == [0, 1]


def test_emit_moment_with_ceo_does_not_batch_across_gate_signatures() -> None:
    """Divergent slots with different per-branch (name, args) signatures used to
    stay in separate IfBlocks; the same-condition merge post-pass collapses them
    into a single IfBlock whose then/else bodies preserve per-slot ordering."""
    q0, q1 = GridQubit(0, 0), GridQubit(1, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0), q1: BlockPosition2D(0, 0)}
    global_i2q = {0: q0, 1: q1}
    entries = _emit_moment_with_ceo(
        _moment_insts("R 0\nH 1"),
        qubit_to_block,
        global_i2q,
        branch_merged_instructions=_moment_insts("RX 0\nS 1"),
        condition_recs=[-1],
    )
    if_blocks = [e for e in entries if isinstance(e, IfBlock)]
    assert len(if_blocks) == 1
    merged = if_blocks[0]
    then_names = [inst.name for inst in merged.then_body]
    else_names = [inst.name for inst in merged.else_body]
    assert then_names == ["RX", "S"]
    assert else_names == ["R", "H"]


def test_emit_moment_with_ceo_branch_requires_condition_recs() -> None:
    q0 = GridQubit(0, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0)}
    global_i2q = {0: q0}
    with pytest.raises(TQECError):
        _emit_moment_with_ceo(
            _moment_insts("R 0"),
            qubit_to_block,
            global_i2q,
            branch_merged_instructions=_moment_insts("RX 0"),
        )


def test_emit_moment_with_ceo_branch_length_mismatch_raises() -> None:
    q0, q1 = GridQubit(0, 0), GridQubit(1, 0)
    qubit_to_block = {q0: BlockPosition2D(0, 0), q1: BlockPosition2D(0, 0)}
    global_i2q = {0: q0, 1: q1}
    with pytest.raises(TQECError):
        _emit_moment_with_ceo(
            _moment_insts("R 0 1"),
            qubit_to_block,
            global_i2q,
            branch_merged_instructions=_moment_insts("R 0"),
            condition_recs=[-1],
        )


def _scheduled(text: str, schedule: int | list[int] = 0) -> ScheduledCircuit:
    return ScheduledCircuit.from_circuit(stim.Circuit(text), schedule)


def test_merge_scheduled_circuits_per_branch_identical_branches() -> None:
    circuits = [
        _scheduled("QUBIT_COORDS(0, 0) 0\nH 0"),
        _scheduled("QUBIT_COORDS(1, 0) 0\nH 0"),
    ]
    zero_circuits, qmap = relabel_circuits_qubit_indices(circuits)
    one_circuits, _ = relabel_circuits_qubit_indices(circuits)
    qubit_to_block = {
        GridQubit(0, 0): BlockPosition2D(0, 0),
        GridQubit(1, 0): BlockPosition2D(0, 0),
    }
    moments_entries, schedule = merge_scheduled_circuits_per_branch(
        zero_circuits,
        one_circuits,
        qmap,
        condition_recs=[-1],
        qubit_to_block=qubit_to_block,
    )
    assert list(schedule) == [0]
    assert len(moments_entries) == 1
    assert all(not isinstance(e, IfBlock) for e in moments_entries[0])


def test_merge_scheduled_circuits_per_branch_divergent_slot_emits_ifblock() -> None:
    zero_circuits, qmap = relabel_circuits_qubit_indices(
        [_scheduled("QUBIT_COORDS(0, 0) 0\nR 0")]
    )
    one_circuits, _ = relabel_circuits_qubit_indices(
        [_scheduled("QUBIT_COORDS(0, 0) 0\nRX 0")]
    )
    qubit_to_block = {GridQubit(0, 0): BlockPosition2D(0, 0)}
    moments_entries, schedule = merge_scheduled_circuits_per_branch(
        zero_circuits,
        one_circuits,
        qmap,
        condition_recs=[-3],
        qubit_to_block=qubit_to_block,
    )
    assert list(schedule) == [0]
    assert len(moments_entries) == 1
    ifs = [e for e in moments_entries[0] if isinstance(e, IfBlock)]
    assert len(ifs) == 1
    assert ifs[0].condition_recs == [-3]
    assert ifs[0].then_body[0].name == "RX"
    assert (ifs[0].else_body or [])[0].name == "R"


def test_merge_scheduled_circuits_per_branch_schedule_mismatch_raises() -> None:
    zero_circuits, qmap = relabel_circuits_qubit_indices(
        [_scheduled("QUBIT_COORDS(0, 0) 0\nH 0\nTICK\nM 0", [0, 2])]
    )
    one_circuits, _ = relabel_circuits_qubit_indices(
        [_scheduled("QUBIT_COORDS(0, 0) 0\nH 0\nTICK\nM 0", [0, 3])]
    )
    qubit_to_block = {GridQubit(0, 0): BlockPosition2D(0, 0)}
    with pytest.raises(TQECError):
        merge_scheduled_circuits_per_branch(
            zero_circuits,
            one_circuits,
            qmap,
            condition_recs=[-1],
            qubit_to_block=qubit_to_block,
        )
