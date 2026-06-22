"""Unit tests for ``tools.resolve_if_else``."""

from __future__ import annotations

import pytest
import stim

from tools.resolve import resolve_if_else


def test_no_if_else_passthrough() -> None:
    text = "H 0\nM 0"
    c = resolve_if_else(text, conditions={})
    assert len(c) == 2
    assert c[0].name == "H"
    assert c[1].name == "M"


def test_single_if_else_take_else() -> None:
    text = "\n".join(
        [
            "H 0",
            "IF(rec[-1]) {",
            "  M 7",
            "} ELSE {",
            "  MX 7",
            "}",
            "CX 1 2",
        ]
    )
    c = resolve_if_else(text, conditions={-1: 0})
    assert [inst.name for inst in c] == ["H", "MX", "CX"]


def test_single_if_else_take_if() -> None:
    text = "\n".join(
        [
            "H 0",
            "IF(rec[-1]) {",
            "  M 7",
            "} ELSE {",
            "  MX 7",
            "}",
            "CX 1 2",
        ]
    )
    c = resolve_if_else(text, conditions={-1: 1})
    assert [inst.name for inst in c] == ["H", "M", "CX"]


def test_if_without_else_outcome_one() -> None:
    text = "\n".join(
        [
            "H 0",
            "IF(rec[-5]) {",
            "  X 1",
            "  Y 2",
            "}",
            "CX 1 2",
        ]
    )
    c = resolve_if_else(text, conditions={-5: 1})
    assert [inst.name for inst in c] == ["H", "X", "Y", "CX"]


def test_if_without_else_outcome_zero_drops_body() -> None:
    text = "\n".join(
        [
            "H 0",
            "IF(rec[-5]) {",
            "  X 1",
            "  Y 2",
            "}",
            "CX 1 2",
        ]
    )
    c = resolve_if_else(text, conditions={-5: 0})
    assert [inst.name for inst in c] == ["H", "CX"]


def test_multiple_if_blocks_same_rec() -> None:
    text = "\n".join(
        [
            "IF(rec[-1]) {",
            "  X 1",
            "} ELSE {",
            "  Z 1",
            "}",
            "H 0",
            "IF(rec[-1]) {",
            "  Y 2",
            "} ELSE {",
            "  S 2",
            "}",
        ]
    )
    c = resolve_if_else(text, conditions={-1: 1})
    assert [inst.name for inst in c] == ["X", "H", "Y"]
    c2 = resolve_if_else(text, conditions={-1: 0})
    assert [inst.name for inst in c2] == ["Z", "H", "S"]


def test_missing_outcome_raises() -> None:
    text = "IF(rec[-1]) {\n  H 0\n}"
    with pytest.raises(ValueError, match="No outcome supplied"):
        resolve_if_else(text, conditions={})


def test_invalid_outcome_raises() -> None:
    text = "IF(rec[-1]) {\n  H 0\n} ELSE {\n  X 0\n}"
    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        resolve_if_else(text, conditions={-1: 2})


def test_resolve_matches_inplace_branch_compile() -> None:
    """End-to-end: resolving the IF/ELSE text must reproduce the per-branch
    compiled stim circuit instruction-for-instruction.

    The reference is the in-place swap (replace the ConditionalBlock in the
    compiled graph with one of its sub-blocks and re-compile) -- NOT a
    from-scratch rebuild, because the detector-annotation pass can pick
    different detector subsets for two topologically distinct graphs.
    """
    from tqec.compile.compile import compile_block_graph
    from tqec.compile.convention import FIXED_BULK_CONVENTION
    from tqec.computation.block_graph import BlockGraph
    from tqec.computation.correlation import CorrelationSurface, ZXEdge, ZXNode
    from tqec.computation.cube import ConditionalLeafCubeKind
    from tqec.utils.enums import Basis
    from tqec.utils.position import Position3D

    p_cond = Position3D(0, 0, 1)
    p_init = Position3D(0, 0, 0)
    # Causal surface (z=0, below cond cube at z=1) — Cube.__post_init__
    # rejects condition surfaces that reach z >= cube.z.
    cond = CorrelationSurface(
        span=frozenset([ZXEdge(ZXNode(p_init, Basis.Z), ZXNode(p_init, Basis.Z))])
    )
    pair = ConditionalLeafCubeKind.XZX_XZZ

    g = BlockGraph("resolve roundtrip")
    g.add_cube(p_init, pair.value[0].name)
    g.add_cube(p_cond, pair, condition=cond)
    g.add_pipe(p_init, p_cond)
    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=None)
    text = cg.generate_conditional_stim_text(k=1)
    # Extract the rec the resolver chose for branch-fixing below.
    import re as _re

    _m = _re.search(r"IF\(([^)]+)\)", text)
    assert _m is not None
    _rec = int(_m.group(1).split("^")[0].strip().lstrip("rec[").rstrip("]"))

    # In-place: temporarily replace the ConditionalBlock with one of its
    # sub-blocks and call the normal generate_stim_circuit; restore after.
    from tqec.compile.blocks.block import ConditionalBlock

    ((cond_pos, cblock),) = cg._conditional_blocks.items()
    assert isinstance(cblock, ConditionalBlock)

    def _inplace(branch: int) -> stim.Circuit:
        original_block = cg._blocks[cond_pos]
        original_cond = cg._conditional_blocks
        try:
            cg._conditional_blocks = {}
            cg._blocks[cond_pos] = (
                cblock.block_if_zero if branch == 0 else cblock.block_if_one
            )
            return cg.generate_stim_circuit(k=1)
        finally:
            cg._blocks[cond_pos] = original_block
            cg._conditional_blocks = original_cond

    resolved_zero = resolve_if_else(text, conditions={_rec: 0})
    reference_zero = _inplace(0)
    _assert_circuits_equivalent_modulo_detector_order(resolved_zero, reference_zero)

    resolved_one = resolve_if_else(text, conditions={_rec: 1})
    reference_one = _inplace(1)
    _assert_circuits_equivalent_modulo_detector_order(resolved_one, reference_one)


def _assert_circuits_equivalent_modulo_detector_order(
    actual: stim.Circuit, expected: stim.Circuit
) -> None:
    """The single-pass conditional compiler emits DETECTORs in a different order
    than the in-place per-branch compile (single-pass groups shared detectors
    first then divergent; in-place emits them in radius-2 lookback order).
    Both circuits define the same detector set; we assert per-measurement-block
    detector multisets agree and non-DETECTOR instructions match byte-for-byte.
    """

    def _split(circuit: stim.Circuit) -> tuple[list[str], list[frozenset[str]]]:
        non_det: list[str] = []
        detector_blocks: list[frozenset[str]] = []
        current_block: set[str] = set()
        for inst in circuit:
            text = str(inst)
            if inst.name == "DETECTOR":
                current_block.add(text)
                continue
            if current_block:
                detector_blocks.append(frozenset(current_block))
                current_block = set()
            non_det.append(text)
        if current_block:
            detector_blocks.append(frozenset(current_block))
        return non_det, detector_blocks

    actual_non, actual_blocks = _split(actual)
    expected_non, expected_blocks = _split(expected)
    assert actual_non == expected_non, "non-DETECTOR instructions differ"
    assert actual_blocks == expected_blocks, "DETECTOR multisets differ per block"


def _absolute_meas_index_sets(
    circuit: stim.Circuit,
) -> tuple[frozenset[frozenset[int]], dict[int, frozenset[int]], int]:
    """Walk flattened circuit. Return (detector_set, obs_parity_by_index,
    num_measurements). Detector set = frozenset of per-detector frozensets of
    absolute measurement indices. obs_parity_by_index = XOR'd absolute indices
    per OBSERVABLE_INCLUDE index."""
    m_count = 0
    detectors: set[frozenset[int]] = set()
    obs: dict[int, set[int]] = {}
    for inst in circuit.flattened():
        name = inst.name
        if name == "DETECTOR":
            ms = frozenset(
                m_count + t.value
                for t in inst.targets_copy()
                if t.is_measurement_record_target
            )
            detectors.add(ms)
        elif name == "OBSERVABLE_INCLUDE":
            idx = int(inst.gate_args_copy()[0])
            slot = obs.setdefault(idx, set())
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    slot.symmetric_difference_update({m_count + t.value})
        elif name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP"):
            m_count += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
    return (
        frozenset(detectors),
        {i: frozenset(s) for i, s in obs.items()},
        m_count,
    )


def _assert_circuits_semantically_equivalent(
    actual: stim.Circuit, expected: stim.Circuit
) -> None:
    """Compare two circuits via:
    - identical sequential measurement record (gate name + qubit per measurement),
    - identical detector parity-set collection (ignoring annotation order),
    - identical per-index observable measurement parity sets,
    - identical total measurement count.
    """
    actual_dets, actual_obs, actual_m = _absolute_meas_index_sets(actual)
    expected_dets, expected_obs, expected_m = _absolute_meas_index_sets(expected)
    assert actual_m == expected_m, (
        f"measurement count differs: actual={actual_m}, expected={expected_m}"
    )
    assert actual_dets == expected_dets, "detector parity sets differ"
    assert actual_obs == expected_obs, (
        f"observable parity differs: actual={actual_obs}, expected={expected_obs}"
    )


def test_resolve_matches_inplace_branch_compile_with_conditional_observable() -> None:
    """End-to-end with ConditionalCorrelationSurface: resolving the IF/ELSE
    text per branch must reproduce the per-branch in-place compiled circuit
    semantically (same detector parity sets, same OBSERVABLE_INCLUDE measurement
    parity, same measurement count).
    """
    from tqec.compile.blocks.block import ConditionalBlock
    from tqec.compile.compile import compile_block_graph
    from tqec.compile.convention import FIXED_BULK_CONVENTION
    from tqec.computation.block_graph import BlockGraph
    from tqec.computation.correlation import (
        ConditionalCorrelationSurface,
        CorrelationSurface,
        ZXEdge,
        ZXNode,
    )
    from tqec.computation.cube import ConditionalLeafCubeKind
    from tqec.utils.enums import Basis
    from tqec.utils.position import Position3D

    b1 = Position3D(0, 0, 0)
    c1 = Position3D(0, 0, 1)
    c2 = Position3D(1, 0, 1)
    t1 = Position3D(0, 0, 2)
    t2 = Position3D(1, 0, 2)

    g = BlockGraph("cond obs roundtrip")
    g.add_cube(b1, "ZXX")
    g.add_cube(c1, "ZXZ")
    g.add_cube(c2, "ZXZ")
    g.add_cube(t1, "ZXX")
    g.add_cube(
        t2,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset(
                {ZXEdge(u=ZXNode(c1, Basis.Z), v=ZXNode(c2, Basis.Z))}
            )
        ),
    )
    g.add_pipe(b1, c1)
    g.add_pipe(c1, c2)
    g.add_pipe(c1, t1)
    g.add_pipe(c2, t2)

    branch_zero_surface = CorrelationSurface(
        span=frozenset(
            {
                ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
                ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(c2, Basis.X)),
                ZXEdge(u=ZXNode(c2, Basis.X), v=ZXNode(t2, Basis.X)),
            }
        )
    )
    branch_one_surface = CorrelationSurface(
        span=frozenset(
            {
                ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
                ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(t1, Basis.X)),
            }
        )
    )
    cond_obs = ConditionalCorrelationSurface(
        branch_zero=branch_zero_surface,
        branch_one=branch_one_surface,
        conditional_cube_positions=(t2,),
    )

    cg = compile_block_graph(g, FIXED_BULK_CONVENTION, observables=[cond_obs])
    text = cg.generate_conditional_stim_text(k=1)
    # Mock condition resolves to rec[-1] placeholder; map every IF guard rec
    # to the chosen outcome.
    import re as _re

    rec_ids = {int(m) for m in _re.findall(r"IF\(rec\[(-?\d+)\]", text)}

    ((cond_pos, cblock),) = cg._conditional_blocks.items()
    assert isinstance(cblock, ConditionalBlock)

    def _inplace(branch: int) -> stim.Circuit:
        # Swap the ConditionalBlock for the chosen sub-block AND swap the
        # conditional observable for the matching plain surface so the
        # non-conditional pass emits the correct branch's observable.
        original_block = cg._blocks[cond_pos]
        original_cond_blocks = cg._conditional_blocks
        original_observables = cg._observables
        original_cond_obs = cg._conditional_abstract_observables
        try:
            cg._conditional_blocks = {}
            cg._blocks[cond_pos] = (
                cblock.block_if_zero if branch == 0 else cblock.block_if_one
            )
            # Re-compile the matching plain surface into an AbstractObservable
            # against a branch-resolved BlockGraph so we have a non-conditional
            # observable to annotate.
            from tqec.compile.compile import _resolve_conditional_cubes  # noqa: PLC0415
            from tqec.compile.observables.abstract_observable import (  # noqa: PLC0415
                compile_correlation_surface_to_abstract_observable,
            )

            resolved_bg = _resolve_conditional_cubes(g, branch)
            surface = branch_zero_surface if branch == 0 else branch_one_surface
            cg._observables = [
                compile_correlation_surface_to_abstract_observable(
                    resolved_bg, surface, include_temporal_hadamard_pipes=True
                )
            ]
            cg._conditional_abstract_observables = []
            return cg.generate_stim_circuit(k=1)
        finally:
            cg._blocks[cond_pos] = original_block
            cg._conditional_blocks = original_cond_blocks
            cg._observables = original_observables
            cg._conditional_abstract_observables = original_cond_obs

    for branch in (0, 1):
        conditions = {rid: branch for rid in rec_ids}
        resolved = resolve_if_else(text, conditions)
        reference = _inplace(branch)
        _assert_circuits_semantically_equivalent(resolved, reference)
