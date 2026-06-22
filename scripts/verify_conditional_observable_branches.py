"""Verify per-branch resolution of test_conditional_observable.py.

For each branch, build the equivalent **non-conditional** BlockGraph (t2
substituted with the branch's ZXCube kind) plus the branch's observable
surface; compile it to plain Stim text. Then resolve the conditional
Stim text from the conditional graph by selecting the matching branch
of every IF/ELSE block. Compare the two as ``stim.Circuit`` instances.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.resolve import resolve_if_else  # noqa: E402

from tqec.compile.compile import compile_block_graph  # noqa: E402
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


_branch_zero_surface = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
            ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(c2, Basis.X)),
            ZXEdge(u=ZXNode(c2, Basis.X), v=ZXNode(t2, Basis.X)),
        }
    )
)
_branch_one_surface = CorrelationSurface(
    span=frozenset(
        {
            ZXEdge(u=ZXNode(b1, Basis.X), v=ZXNode(c1, Basis.X)),
            ZXEdge(u=ZXNode(c1, Basis.X), v=ZXNode(t1, Basis.X)),
        }
    )
)


def _build_conditional() -> tuple[BlockGraph, ConditionalCorrelationSurface]:
    g = BlockGraph("conditional")
    g.add_cube(b1, "ZXX")
    g.add_cube(c1, "ZXZ")
    g.add_cube(c2, "ZXZ")
    g.add_cube(t1, "ZXX")
    g.add_cube(
        t2,
        ConditionalLeafCubeKind.ZXX_ZXZ,
        condition=CorrelationSurface(
            span=frozenset({ZXEdge(u=ZXNode(c1, Basis.Z), v=ZXNode(c2, Basis.Z))})
        ),
    )
    g.add_pipe(b1, c1)
    g.add_pipe(c1, c2)
    g.add_pipe(c1, t1)
    g.add_pipe(c2, t2)
    cond = ConditionalCorrelationSurface(
        branch_zero=_branch_zero_surface,
        branch_one=_branch_one_surface,
        conditional_cube_positions=(t2,),
    )
    return g, cond


def _build_reference(branch: str) -> tuple[BlockGraph, CorrelationSurface]:
    """``branch`` in {"zero", "one"} resolves t2 to ZXX or ZXZ."""
    g = BlockGraph(f"reference_{branch}")
    g.add_cube(b1, "ZXX")
    g.add_cube(c1, "ZXZ")
    g.add_cube(c2, "ZXZ")
    g.add_cube(t1, "ZXX")
    g.add_cube(t2, "ZXX" if branch == "zero" else "ZXZ")
    g.add_pipe(b1, c1)
    g.add_pipe(c1, c2)
    g.add_pipe(c1, t1)
    g.add_pipe(c2, t2)
    surface = _branch_zero_surface if branch == "zero" else _branch_one_surface
    return g, surface


def _compile_conditional_text() -> str:
    g, cond = _build_conditional()
    cg = compile_block_graph(g, observables=[cond])
    return cg.generate_conditional_stim_text(k=1)


def _compile_reference_text(branch: str) -> str:
    g, surface = _build_reference(branch)
    cg = compile_block_graph(g, observables=[surface])
    return str(cg.generate_stim_circuit(k=1))


def _resolve(text: str, condition_value: int):
    import re

    import stim  # noqa: PLC0415

    rec_ids = {int(m) for m in re.findall(r"IF\(rec\[(-?\d+)\]", text)}
    conditions = {rid: condition_value for rid in rec_ids}
    circuit = resolve_if_else(text, conditions)
    assert isinstance(circuit, stim.Circuit)
    return circuit


def _measurement_records(circuit):
    """Return the ordered list of (instruction_name, qubit_index) for every
    measurement in the flattened circuit. Two circuits with identical lists
    produce identical measurement records under the same shot."""
    records: list[tuple[str, int]] = []
    for inst in circuit.flattened():
        name = inst.name
        if name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"):
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    records.append((name, t.qubit_value))
        elif name == "MPP":
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    records.append((name, t.qubit_value))
    return records


def _absolute_sets(circuit):
    """Walk flattened circuit; collect (detector_set, observable_meas_set,
    num_measurements). Detector set is a frozenset of frozensets, each inner
    set being the absolute measurement indices that XOR into one detector.
    Observable set is the XOR'd set of absolute measurement indices entering
    OBSERVABLE_INCLUDE(0). Ignores instruction order entirely."""
    m_count = 0
    detectors: set[frozenset[int]] = set()
    obs: set[int] = set()
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
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    obs.symmetric_difference_update({m_count + t.value})
        elif name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP"):
            m_count += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
    return frozenset(detectors), frozenset(obs), m_count


def main() -> None:
    import stim  # noqa: PLC0415

    conditional_text = _compile_conditional_text()
    all_ok = True
    for branch, condition_value in (("zero", 0), ("one", 1)):
        resolved = _resolve(conditional_text, condition_value)
        reference = stim.Circuit(_compile_reference_text(branch))

        det_r, obs_r, m_r = _absolute_sets(resolved)
        det_g, obs_g, m_g = _absolute_sets(reference)
        rec_r = _measurement_records(resolved)
        rec_g = _measurement_records(reference)
        det_eq = det_r == det_g
        obs_eq = obs_r == obs_g
        m_eq = m_r == m_g
        rec_eq = rec_r == rec_g
        ok = det_eq and obs_eq and m_eq and rec_eq
        all_ok &= ok
        status = "OK" if ok else "DIFF"
        print(
            f"branch_{branch}: {status}  detectors={det_eq} "
            f"(|R|={len(det_r)},|G|={len(det_g)})  observable_meas={obs_eq} "
            f"(|R|={len(obs_r)},|G|={len(obs_g)})  num_M={m_eq} (R={m_r},G={m_g})  "
            f"records_in_order={rec_eq}"
        )
        if not rec_eq:
            first_diff = next(
                (i for i in range(min(len(rec_r), len(rec_g))) if rec_r[i] != rec_g[i]),
                min(len(rec_r), len(rec_g)),
            )
            print(
                f"  first diff at record idx {first_diff}: "
                f"resolved={rec_r[first_diff] if first_diff < len(rec_r) else 'END'}  "
                f"reference={rec_g[first_diff] if first_diff < len(rec_g) else 'END'}"
            )
        if not det_eq:
            print(f"  detectors only in resolved: {sorted(det_r - det_g)[:3]}")
            print(f"  detectors only in reference: {sorted(det_g - det_r)[:3]}")
        if not obs_eq:
            print(f"  obs symmetric diff: {sorted(obs_r ^ obs_g)}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
