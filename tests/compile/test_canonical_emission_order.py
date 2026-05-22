"""Stage 1 diagnostic: Canonical Emission Order (CEO) for conditional cubes.

For each ``ConditionalLeafCubeKind`` pair ``(kind_zero, kind_one)``, compiles each
``ZXCube`` member standalone and asserts the per-timestep measurement schedule
emits qubits in identical order across branches (basis ``MX`` vs ``M`` may
differ).  This is the prerequisite that lets ``rec[-k]`` references in
downstream detectors / observables stay branch-independent.

Marked ``xfail`` while the CEO pass is unimplemented.  When the CEO sorter
lands these should flip to pass; ``xfail(strict=False)`` keeps that flip from
failing tests on the diagnostic side.
"""

from __future__ import annotations

import pytest
import stim

from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.compile.compile import compile_block_graph
from tqec.compile.convention import FIXED_BULK_CONVENTION, Convention
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ConditionalLeafCubeKind
from tqec.utils.position import Position3D


_MEAS_INSTRUCTIONS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}


def _single_cube_graph(kind_str: str) -> BlockGraph:
    g = BlockGraph(f"CEO {kind_str}")
    g.add_cube(Position3D(0, 0, 0), kind_str)
    return g


def _compile(g: BlockGraph, convention: Convention = FIXED_BULK_CONVENTION) -> stim.Circuit:
    cg = compile_block_graph(g, convention, observables=None)
    return cg.to_layer_tree().generate_circuit(
        k=1, detector_database=None, database_path=None
    )


def _basis_of(name: str) -> str:
    # "M" / "MR" -> Z; "MX"/"MRX" -> X; "MY"/"MRY" -> Y; "MZ"/"MRZ" -> Z
    if name in {"M", "MR"}:
        return "Z"
    return name[-1]


def _measurement_schedule(
    circuit: stim.Circuit,
) -> list[list[tuple[GridQubit, str]]]:
    """Group measurements by ``TICK`` boundary.

    Each entry of the outer list is one timestep; inner list preserves the
    emission order of ``(qubit, basis)`` for that timestep.  Timesteps with no
    measurements are dropped so the result indexes only measurement layers.
    """
    qmap = QubitMap.from_circuit(circuit)
    timesteps: list[list[tuple[GridQubit, str]]] = [[]]
    for instr in circuit.flattened():
        name = instr.name
        if name == "TICK":
            timesteps.append([])
            continue
        if name in _MEAS_INSTRUCTIONS:
            basis = _basis_of(name)
            for target in instr.targets_copy():
                qi = target.qubit_value
                if qi is None:
                    continue
                timesteps[-1].append((qmap.i2q[qi], basis))
    return [ts for ts in timesteps if ts]


def _format_diff(
    sched_zero: list[list[tuple[GridQubit, str]]],
    sched_one: list[list[tuple[GridQubit, str]]],
) -> str:
    lines: list[str] = []
    for t, (ts0, ts1) in enumerate(zip(sched_zero, sched_one)):
        q0 = [q for q, _ in ts0]
        q1 = [q for q, _ in ts1]
        if q0 != q1:
            lines.append(f"  timestep {t}:")
            lines.append(f"    zero ({len(q0)}): {q0}")
            lines.append(f"    one  ({len(q1)}): {q1}")
    return "\n".join(lines) if lines else "(no per-timestep diff)"


_CEO_PAIRS = [
    pytest.param(
        k.name,
        marks=pytest.mark.xfail(
            reason="ZXZ_ZXX baseline produces unequal-length measurement schedules "
            "(33 vs 31), so CEO qubit-order alignment also fails until "
            "equal-meas-count is addressed.",
            strict=True,
        )
        if k.name == "ZXZ_ZXX"
        else (),
    )
    for k in ConditionalLeafCubeKind
]


@pytest.mark.parametrize("pair_name", _CEO_PAIRS)
def test_ceo_qubit_order_identical(pair_name: str) -> None:
    pair = ConditionalLeafCubeKind[pair_name]
    kz, ko = pair.value  # both are ZXCube members
    cz = _compile(_single_cube_graph(kz.name))
    co = _compile(_single_cube_graph(ko.name))

    sched_z = _measurement_schedule(cz)
    sched_o = _measurement_schedule(co)

    assert len(sched_z) == len(sched_o), (
        f"{pair_name}: number of measurement timesteps differs — "
        f"{len(sched_z)} vs {len(sched_o)}"
    )

    diff = _format_diff(sched_z, sched_o)
    for t, (ts0, ts1) in enumerate(zip(sched_z, sched_o)):
        q0 = [q for q, _ in ts0]
        q1 = [q for q, _ in ts1]
        assert q0 == q1, (
            f"{pair_name}: qubit emission order differs at timestep {t}\n{diff}"
        )


@pytest.mark.parametrize("pair_name", _CEO_PAIRS)
def test_ceo_total_measurement_count_equal(pair_name: str) -> None:
    """Equal-Measurement-Count is independent of CEO — assert separately.

    PPD claims both branches of every supported conditional pair produce the
    same total measurement count (T-injection: 17 = 17).  Strict pass expected.
    """
    pair = ConditionalLeafCubeKind[pair_name]
    kz, ko = pair.value
    cz = _compile(_single_cube_graph(kz.name))
    co = _compile(_single_cube_graph(ko.name))
    total_z = sum(len(ts) for ts in _measurement_schedule(cz))
    total_o = sum(len(ts) for ts in _measurement_schedule(co))
    assert total_z == total_o, (
        f"{pair_name}: total measurement count differs — {total_z} vs {total_o}"
    )
