from tests._vendor.midout import gen
from tests._vendor.midout.circuits.steps._patches import make_xtop_qubit_patch, make_stability_patch


def make_memory_to_stability_transition_round(*, distance: int, basis: str) -> gen.Chunk:
    mem = make_xtop_qubit_patch(distance=distance)
    stab = make_stability_patch(distance=distance, basis=basis)
    out = gen.Builder.for_qubits(mem.used_set | stab.used_set)
    gen.build_surface_code_round_circuit(
        patch=stab,
        measure_data_basis={q: basis for q in mem.data_set - stab.data_set},
        save_layer='solo',
        out=out,
    )

    flows = []
    discarded = []

    # Annotate input stabilizers that get measured.
    for tile in mem.tiles:
        m = tile.measurement_qubit
        if m in stab.measure_set:
            measurements = [m] + [q for q in tile.data_set if q not in stab.used_set]
        else:
            discarded.append(gen.PauliString.from_tile_data(tile))
            continue
        flows.append(gen.Flow(
            start=gen.PauliString.from_tile_data(tile),
            center=m,
            measurement_indices=out.tracker.measurement_indices([
                gen.AtLayer(k, layer='solo')
                for k in measurements
            ]),
        ))
    for tile in stab.tiles:
        if tile.measurement_qubit not in mem.measure_set:
            discarded.append(gen.PauliString.from_tile_data(tile))

    # Annotate output stabilizers that get prepared.
    for tile in stab.tiles:
        m = tile.measurement_qubit
        measurements = [m]
        flows.append(gen.Flow(
            end=gen.PauliString.from_tile_data(tile),
            center=m,
            measurement_indices=out.tracker.measurement_indices([
                gen.AtLayer(k, layer='solo')
                for k in measurements
            ]),
        ))

    # Annotate how observable flows through the system.
    xs = {q for q in mem.data_set if q.real == 0}
    zs = {q for q in mem.data_set if q.imag == 0}
    qs = xs if basis == 'X' else zs
    start_obs = gen.PauliString({q: basis for q in qs})
    obs_measurements = [
        tile.measurement_qubit
        for tile in stab.tiles
        if tile.basis == basis
        if (basis == 'Z' or tile.measurement_qubit.real > 0)
        if (basis == 'X' or tile.measurement_qubit.imag > 0)
    ] + [
        q for q in start_obs.qubits.keys() if q not in stab.used_set
    ]
    flows.append(
        gen.Flow(
            center=0,
            start=start_obs,
            measurement_indices=out.tracker.measurement_indices([
                gen.AtLayer(m, layer='solo')
                for m in obs_measurements
            ]),
            obs_index=0,
        )
    )

    return gen.Chunk(
        circuit=out.circuit,
        q2i=out.q2i,
        flows=flows,
        discarded_inputs=discarded,
    )
