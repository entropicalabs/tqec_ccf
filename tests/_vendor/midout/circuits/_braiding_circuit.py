from typing import List

from tests._vendor.midout import gen
from tests._vendor.midout.circuits.steps._measure_y_transition_round import make_y_transition_round_nesw_xzxz_to_xzzx
from tests._vendor.midout.circuits.steps._patches import make_xtop_qubit_patch, make_ztop_yboundary_patch


def make_y_braiding_experiment_chunks(
        *,
        distance: int,
        boundary_rounds: int,
        memory_rounds: int,
) -> List[gen.Chunk]:
    boundary_patch = make_ztop_yboundary_patch(distance=distance)
    qubit_patch = make_xtop_qubit_patch(distance=distance)
    qubit_obs = gen.PauliString({
        0: 'Y',
        **{q: 'Z' for q in range(1, distance)},
        **{q*1j: 'X' for q in range(1, distance)},
    })

    memory_round = gen.standard_surface_code_chunk(qubit_patch, obs=qubit_obs)
    qubit_to_boundary_round = make_y_transition_round_nesw_xzxz_to_xzzx(distance=distance)
    qubit_to_boundary_round_2 = make_y_transition_round_nesw_xzxz_to_xzzx(distance=distance, obs_along_bottom=True)
    boundary_round = gen.standard_surface_code_chunk(boundary_patch)
    final_round = gen.standard_surface_code_chunk(
        boundary_patch,
        measure_data_basis={q: 'Z' if q.real + q.imag < distance else 'X' for q in boundary_patch.data_set},
    )

    def quarter_turn(q: complex) -> complex:
        center = (distance - 1) * (1 + 1j) / 2
        q -= center
        q *= 1j
        q += center
        return q

    def vertical_mirror(q: complex) -> complex:
        middle = (distance - 1) / 2
        q -= middle*1j
        q = q.real - 1j*q.imag
        q += middle*1j
        return q

    def opposite_diag(chunk: gen.Chunk) -> gen.Chunk:
        if distance % 2 == 1:
            return chunk.inverted().with_xz_flipped().with_transformed_coords(quarter_turn)
        else:
            return chunk.inverted().with_transformed_coords(vertical_mirror)

    return [
        opposite_diag(final_round),
        opposite_diag(boundary_round).with_repetitions(boundary_rounds),
        opposite_diag(qubit_to_boundary_round_2),
        memory_round.with_repetitions(memory_rounds // 2),
        memory_round.inverted().with_repetitions(memory_rounds - memory_rounds // 2),
        qubit_to_boundary_round,
        boundary_round.with_repetitions(boundary_rounds),
        final_round,
    ]
