from typing import List

from tests._vendor.midout import gen
from tests._vendor.midout.circuits.steps._patches import make_xtop_qubit_patch, make_stability_patch
from tests._vendor.midout.circuits.steps._stability_to_memory_transition_round import make_memory_to_stability_transition_round


def make_xz_memory_experiment_chunks(
        *,
        distance: int,
        basis: str,
        memory_rounds: int,
        boundary_rounds: int,
) -> List[gen.Chunk]:
    if boundary_rounds > 0:
        return make_stability_padded_xz_memory_experiment_chunks(
            distance=distance,
            basis=basis,
            memory_rounds=memory_rounds,
            boundary_rounds=boundary_rounds,
        )

    qubit_patch = make_xtop_qubit_patch(distance=distance)
    xs = {q for q in qubit_patch.data_set if q.real == 0}
    zs = {q for q in qubit_patch.data_set if q.imag == 0}
    assert len(xs & zs) % 2 == 1
    obs = gen.PauliString({q: basis for q in (xs if basis == 'X' else zs)})
    assert memory_rounds > 0
    if memory_rounds == 1 and boundary_rounds == 0:
        return [
            gen.standard_surface_code_chunk(
                qubit_patch,
                init_data_basis=basis,
                measure_data_basis=basis,
                obs=obs)
        ]

    if boundary_rounds > 0:
        boundary_patch = make_stability_patch(distance=distance, basis=basis)

        return [
            gen.standard_surface_code_chunk(
                boundary_patch,
                init_data_basis=basis,
                obs=obs,
            ),
            gen.standard_surface_code_chunk(
                boundary_patch,
                obs=obs,
            ).with_repetitions(boundary_rounds - 1),
            gen.standard_surface_code_chunk(
                qubit_patch,
                obs=obs,
            ).with_repetitions(memory_rounds),
            gen.standard_surface_code_chunk(
                boundary_patch,
                obs=obs,
            ).with_repetitions(boundary_rounds - 1),
            gen.standard_surface_code_chunk(
                boundary_patch,
                measure_data_basis=basis,
                obs=obs,
            ),
        ]

    return [
        gen.standard_surface_code_chunk(
            qubit_patch,
            init_data_basis=basis,
            obs=obs,
        ),
        gen.standard_surface_code_chunk(
            qubit_patch,
            obs=obs,
        ).with_repetitions(memory_rounds - 2),
        gen.standard_surface_code_chunk(
            qubit_patch,
            measure_data_basis=basis,
            obs=obs,
        ),
    ]


def make_stability_padded_xz_memory_experiment_chunks(
        *,
        distance: int,
        basis: str,
        memory_rounds: int,
        boundary_rounds: int,
) -> List[gen.Chunk]:
    assert memory_rounds > 0
    boundary_rounds = max(boundary_rounds, 2)

    mem = make_xtop_qubit_patch(distance=distance)
    stab = make_stability_patch(distance=distance, basis=basis)
    xs = {q for q in mem.data_set if q.real == 0}
    zs = {q for q in mem.data_set if q.imag == 0}
    qs = xs if basis == 'X' else zs
    obs_mem = gen.PauliString({q: basis for q in qs})

    stab_from_mem = make_memory_to_stability_transition_round(
        distance=distance,
        basis=basis,
    )
    hold_stab = gen.standard_surface_code_chunk(stab)
    hold_mem = gen.standard_surface_code_chunk(mem, obs=obs_mem)
    init_stab = gen.standard_surface_code_chunk(
        stab,
        init_data_basis=basis,
    )
    return [
        init_stab,
        hold_stab * (boundary_rounds - 2),
        stab_from_mem.inverted(),
        hold_mem * memory_rounds,
        stab_from_mem,
        hold_stab * (boundary_rounds - 2),
        init_stab.inverted(),
    ]
