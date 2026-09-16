"""Unit tests for ``PauliFrameTracker`` (Stage 1 stub: identity propagation)."""

from __future__ import annotations

from tqec.circuit.qubit import GridQubit
from tqec.compile.conditional.frame import ConditionId, PauliFrameTracker


_C0 = ConditionId(0)
_C1 = ConditionId(1)


def test_register_and_query_single_condition() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, timestep=3, conditions=frozenset({_C0}))
    assert t.frame_at(q, 3) == frozenset({_C0})
    assert t.frame_at(q, 5) == frozenset({_C0})


def test_frame_inactive_before_registration() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, timestep=10, conditions=frozenset({_C0}))
    assert t.frame_at(q, 9) == frozenset()


def test_independent_qubits() -> None:
    t = PauliFrameTracker()
    qa, qb = GridQubit(0, 0), GridQubit(1, 0)
    t.register_measurement(qa, 0, frozenset({_C0}))
    t.register_measurement(qb, 0, frozenset({_C1}))
    assert t.frame_at(qa, 0) == frozenset({_C0})
    assert t.frame_at(qb, 0) == frozenset({_C1})


def test_union_across_timesteps() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, 0, frozenset({_C0}))
    t.register_measurement(q, 5, frozenset({_C1}))
    assert t.frame_at(q, 10) == frozenset({_C0, _C1})


def test_empty_conditions_clears_entry() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, 3, frozenset({_C0}))
    t.register_measurement(q, 3, frozenset())
    assert t.frame_at(q, 3) == frozenset()


def test_propagate_through_is_noop_in_stub() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, 0, frozenset({_C0}))
    t.propagate_through("CX", [q, GridQubit(1, 0)], timestep=1)
    # Stub: target qubit does NOT inherit the frame.
    assert t.frame_at(GridQubit(1, 0), 1) == frozenset()
    # Source qubit still carries it.
    assert t.frame_at(q, 1) == frozenset({_C0})


def test_xor_records_for_known_conditions() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, 0, frozenset({_C0, _C1}))
    cond_to_rec = {_C0: -7, _C1: -3}
    # Sorted by condition id for determinism.
    assert t.xor_records_for(q, 5, cond_to_rec) == [-7, -3]


def test_xor_records_skips_unknown_conditions() -> None:
    t = PauliFrameTracker()
    q = GridQubit(0, 0)
    t.register_measurement(q, 0, frozenset({_C0, _C1}))
    cond_to_rec = {_C0: -7}  # _C1 not in mapping yet
    assert t.xor_records_for(q, 5, cond_to_rec) == [-7]
