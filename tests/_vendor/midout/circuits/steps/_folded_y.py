from typing import List, Tuple, Dict

import stim

from tests._vendor.midout import gen
from tests._vendor.midout.circuits.steps._patches import rectangular_surface_code_patch, UR, DR, UL, DL


def folded_patch(*, distance: int) -> Tuple[gen.Patch, Dict[str, gen.PauliString]]:
    def order_func(m: complex) -> List[complex]:
        order_S = [UR, UL, DR, DL]
        order_N = [UR, DR, UL, DL]
        if gen.checkerboard_basis(m) == 'X':
            return order_S
        else:
            return order_N

    patch = rectangular_surface_code_patch(
        width=distance + 1,
        height=distance,
        top_basis='X',
        bot_basis='X',
        left_basis='Z',
        right_basis='Z',
        order_func=order_func,
    )
    obs_x = gen.PauliString({q: 'X' for q in patch.data_set if q.real == 0})
    obs_z = gen.PauliString({q: 'Z' for q in patch.data_set if q.imag == 0})
    assert obs_x.anticommutes(obs_z)
    obs_y = obs_x * obs_z
    return patch, {'X': obs_x, 'Y': obs_y, 'Z': obs_z}


def folded_x2y_chunk(*, distance: int, init: bool) -> gen.Chunk:
    patch, obs = folded_patch(distance=distance)

    def partner(q: complex) -> complex:
        r, i = q.real, q.imag
        if r <= i:
            return i + r*1j + 1
        else:
            return i + r*1j - 1j

    builder = gen.Builder.for_qubits(patch.used_set)
    measure_xs = gen.Patch([tile for tile in patch.tiles if tile.basis == 'X'])
    measure_zs = gen.Patch([tile for tile in patch.tiles if tile.basis == 'Z'])
    builder.gate("RX", measure_xs.measure_set)
    if init:
        builder.gate("RX", patch.data_set)
    builder.gate("R", measure_zs.measure_set)
    builder.tick()

    num_layers, = {len(tile.ordered_data_qubits) for tile in patch.tiles}
    for k in range(num_layers):
        builder.gate2('CX', [
            (tile.measurement_qubit, tile.ordered_data_qubits[k])[::-1 if tile.basis == 'Z' else +1]
            for tile in patch.tiles
            if tile.ordered_data_qubits[k] is not None
        ])
        builder.tick()

    builder.gate2('CZ', [
        (q, partner(q))
        for q in patch.data_set
        if q.real < partner(q).real
    ])
    builder.tick()
    builder.gate('S', [0j] + [k * (1 + 1j) + 1 for k in range(distance)])
    builder.tick()

    builder.measure(measure_xs.measure_set, basis='X', save_layer='solo')
    builder.measure(measure_zs.measure_set, basis='Z', save_layer='solo')

    flows = []
    for tile in patch.tiles:
        m = tile.measurement_qubit
        p = gen.PauliString.from_tile_data(tile)
        measurements = [m]

        if init:
            if tile.basis == 'X':
                flows.append(gen.Flow(
                    center=m,
                    measurement_indices=builder.tracker.measurement_indices([
                        gen.AtLayer(k, layer='solo')
                        for k in [m]
                    ]),
                ))
        else:
            flows.append(gen.Flow(
                start=p,
                center=m,
                measurement_indices=builder.tracker.measurement_indices([
                    gen.AtLayer(k, layer='solo')
                    for k in [m]
                ]),
            ))

        if tile.basis == 'X' and m != 0.5 - 0.5j:
            measurements.append(partner(m))
        flows.append(gen.Flow(
            end=p,
            center=m,
            measurement_indices=builder.tracker.measurement_indices([
                gen.AtLayer(k, layer='solo')
                for k in measurements
            ]),
        ))
    flows.append(gen.Flow(
        start=None if init else obs['X'],
        end=obs['Y'],
        center=0,
        obs_index=0,
    ))

    return gen.Chunk(
        circuit=builder.circuit,
        q2i=builder.q2i,
        flows=flows,
    )


def folded_surface_code_memory_y_chunks(*, distance: int, rounds: int) -> List[gen.Chunk]:
    patch, obs = folded_patch(distance=distance)
    x2y_chunk = folded_x2y_chunk(distance=distance, init=True)
    return [
        x2y_chunk,
        gen.standard_surface_code_chunk(patch, obs=obs['Y']).with_repetitions(rounds),
        x2y_chunk.inverted(),
    ]
