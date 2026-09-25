import pathlib
from typing import Union, Any, Optional

import stim

from tests._vendor.midout import gen
from tests._vendor.midout.circuits._braiding_circuit import make_y_braiding_experiment_chunks
from tests._vendor.midout.circuits._idle_memory_circuit import make_idle_memory_with_magic_before_and_after
from tests._vendor.midout.circuits._xz_memory_circuits import make_xz_memory_experiment_chunks
from tests._vendor.midout.circuits._y_memory_circuit import make_y_memory_experiment_chunks, \
    make_y_memory_transition_chunks_with_magic_before_and_after, make_y_measure_chunks_with_magic_before
from tests._vendor.midout.circuits.steps._folded_y import folded_surface_code_memory_y_chunks


def _write(path: Any, content: Any):
    path = pathlib.Path(path)
    with open(path, "w") as f:
        print(content, file=f)
    print(f'wrote file://{path.absolute()}')


def make_circuit(
    *,
    basis: str,
    noise: Optional[gen.NoiseModel],
    boundary_rounds: int,
    memory_rounds: int,
    distance: int,
    verify_chunks: bool = False,
    debug_out_dir: Union[None, str, pathlib.Path] = None,
    convert_to_cz: bool = True,
):
    skip_mpp_head = False
    skip_mpp_tail = False
    if basis == 'Y':
        chunks = make_y_memory_experiment_chunks(
            distance=distance,
            boundary_rounds=boundary_rounds,
            memory_rounds=memory_rounds,
        )
    elif basis == 'X' or basis == 'Z':
        chunks = make_xz_memory_experiment_chunks(
            distance=distance,
            memory_rounds=memory_rounds,
            boundary_rounds=boundary_rounds,
            basis=basis,
        )
    elif basis == 'Y_folded':
        assert boundary_rounds == 0
        chunks = folded_surface_code_memory_y_chunks(
            distance=distance,
            rounds=memory_rounds,
        )
    elif basis == 'Y_braid':
        chunks = make_y_braiding_experiment_chunks(
            distance=distance,
            boundary_rounds=boundary_rounds,
            memory_rounds=memory_rounds,
        )
    elif basis == 'Y_magic_measure':
        chunks = make_y_measure_chunks_with_magic_before(
            distance=distance,
            boundary_rounds=boundary_rounds,
            memory_rounds=memory_rounds,
        )
        skip_mpp_head = True
    elif basis == 'Y_magic_transition':
        chunks = make_y_memory_transition_chunks_with_magic_before_and_after(
            distance=distance,
            boundary_rounds=boundary_rounds,
            memory_rounds=memory_rounds,
        )
        skip_mpp_head = True
        skip_mpp_tail = True
    elif basis == 'Y_magic_idle':
        chunks = make_idle_memory_with_magic_before_and_after(
            distance=distance,
            memory_rounds=memory_rounds,
            basis='Y',
        )
        skip_mpp_head = True
        skip_mpp_tail = True
    elif basis == 'Z_magic_idle':
        chunks = make_idle_memory_with_magic_before_and_after(
            distance=distance,
            memory_rounds=memory_rounds,
            basis='Z',
        )
        skip_mpp_head = True
        skip_mpp_tail = True
    elif basis == 'X_magic_idle':
        chunks = make_idle_memory_with_magic_before_and_after(
            distance=distance,
            memory_rounds=memory_rounds,
            basis='X',
        )
        skip_mpp_head = True
        skip_mpp_tail = True
    else:
        raise NotImplementedError(f'{basis=}')

    if debug_out_dir is not None:
        patches = [chunk.end_patch() for chunk in chunks[:-1]]
        changed_patches = [patches[k] for k in range(len(patches)) if k == 0 or patches[k] != patches[k-1]]
        allowed_qubits = {q for patch in changed_patches for q in patch.used_set}
        _write(debug_out_dir / "patch.svg", gen.patch_svg_viewer(
            changed_patches,
            show_order=False,
            available_qubits=allowed_qubits,
        ))

    if verify_chunks:
        for chunk in chunks:
            chunk.verify()

    if debug_out_dir is not None:
        ignore_errors_ideal_circuit = gen.compile_chunks_into_circuit(chunks, ignore_errors=True)
        _write(debug_out_dir / "ideal_circuit.html", gen.stim_circuit_html_viewer(
            ignore_errors_ideal_circuit,
            patch={k: chunks[k].end_patch() for k in range(len(chunks))},
        ))
        _write(debug_out_dir / "ideal_circuit.stim", ignore_errors_ideal_circuit)
        _write(debug_out_dir / "ideal_circuit_dets.svg", ignore_errors_ideal_circuit.diagram("time+detector-slice-svg"))

    body = gen.compile_chunks_into_circuit(chunks)
    mpp_indices = [
        k
        for k, inst in enumerate(body)
        if isinstance(inst, stim.CircuitInstruction) and inst.name == 'MPP'
    ]
    body_start = mpp_indices[0] + 2 if skip_mpp_head else 0
    body_end = mpp_indices[1] if skip_mpp_tail else len(body)
    magic_head = body[:body_start]
    magic_tail = body[body_end:]
    body = body[body_start:body_end]

    if convert_to_cz:
        body = gen.to_z_basis_interaction_circuit(body)
        if debug_out_dir is not None:
            ideal_circuit = magic_head + body + magic_tail
            _write(debug_out_dir / "ideal_cz_circuit.html", gen.stim_circuit_html_viewer(
                ideal_circuit,
                patch=chunks[0].end_patch(),
            ))
            _write(debug_out_dir / "ideal_cz_circuit.stim", ideal_circuit)
            _write(debug_out_dir / "ideal_cz_circuit_dets.svg", ideal_circuit.diagram("time+detector-slice-svg"))

    if noise is not None:
        body = noise.noisy_circuit(body)
    noisy_circuit = magic_head + body + magic_tail

    if debug_out_dir is not None:
        _write(debug_out_dir / "noisy_circuit.html", gen.stim_circuit_html_viewer(
            noisy_circuit,
            patch=chunks[0].end_patch(),
        ))
        _write(debug_out_dir / "noisy_circuit.stim", noisy_circuit)
        _write(debug_out_dir / "noisy_circuit_dets.svg", noisy_circuit.diagram("time+detector-slice-svg"))

    return noisy_circuit
