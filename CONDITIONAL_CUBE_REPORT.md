# Conditional-Cube Compilation: Implementation Report

**Branch**: `ccf-compile` (tqec_ccf fork)
**Base**: `b77ca994` (tip of upstream PR #829, `feat/conditional-cube`)
**Commits added**: 16
**Final test state**: 670 pass, 8 skip, 2 xfail (upstream typo), 0 regressions

---

## Goal

Implement compilation pass that PR #829 left as `NotImplementedError`. Emit single IF/ELSE-annotated Stim text from a `BlockGraph` containing conditional cubes — one pass, no branch pre-resolution, no exponential blowup.

## Design source

`/Users/isaac/Documents/entropica/loom-weave/PPD_tqec_conditional_circuits.md` — three coordinated containment mechanisms:

1. **Equal Measurement Count** (compile-time assert)
2. **Canonical Emission Order** (target-order alignment across branches)
3. **Pauli Frame Tracking + XOR `rec[-K]`** (sign propagation w/o IF/ELSE bloat)

---

## Commits

### `76f4d471` — Canonical Emission Order

**Files**: `manipulation.py`, `generation.py`, `layout.py`, `tests/compile/test_canonical_emission_order.py`

Replaces dict-insertion-order target emission inside multi-qubit Stim instructions with template-grouped sort. Target order now `(block.y, block.x, qubit_idx)`; same-`(name, args)` runs collapse into single instructions. Applies to all multi-qubit instructions (measurements + gates).

Plumbing:

```
LayoutLayer.to_circuit
  -> builds plaquette_to_block from LayoutTemplate.get_indices_map_for_instantiation()
  -> generate_circuit (forwards)
  -> generate_circuit_from_instantiation (accumulates qubit_to_block)
  -> merge_scheduled_circuits (dispatches to _emit_moment_with_ceo)
```

Detector substructure callers (`detectors/{compute,database}.py`) skip CEO (pass `None`).

CEO test: 3/4 conditional pairs now strict-pass. ZXZ_ZXX still fails (upstream typo flagged separately).

### `63ecdd01` — `ConditionalBlock` + CubeBuilder wiring

**Files**: `block.py`, `specs/base.py`, `fixed_bulk.py`, `fixed_boundary.py`, `test_conditional_cube_builder.py`

Replaces NotImplementedError stubs in both convention CubeBuilders. For `ConditionalLeafCubeKind`:

1. Splits to `kind_zero, kind_one = kind.value`.
2. Builds two CubeSpecs via `dataclasses.replace(spec, kind=branch, condition=None)`.
3. Recurses `_call_impl` for each → `Block_zero`, `Block_one`.
4. Wraps in `ConditionalBlock(block_zero, block_one, condition)`.

`ConditionalBlock.__init__` enforces Equal Measurement Count via per-layer plaquette-meas signature equality. Catches ZXZ_ZXX (spatial swap → different stabilizer set), admits the 3 temporal-basis swaps.

`CubeSpec.condition: CorrelationSurface | None` added; `CubeSpec.from_cube` propagates it from `Cube.condition`.

### `721502f9` — `PauliFrameTracker` stub

**Files**: `src/tqec/compile/conditional/{__init__,frame}.py`, `tests/.../test_frame.py`

API:

- `register_measurement(qubit, timestep, conditions)`
- `propagate_through(gate_name, qubits, timestep)` — no-op stub
- `frame_at(qubit, timestep) -> frozenset[ConditionId]`
- `xor_records_for(qubit, timestep, condition_to_rec) -> list[int]`

Identity Clifford propagation only. Real propagation table deferred until emission code starts driving the tracker through gates.

### `989d72ef` — graph → tree metadata threading

**Files**: `graph.py`, `tree/tree.py`, `test_conditional_metadata_threading.py`

`TopologicalComputationGraph` keeps side-channel `_conditional_blocks: dict[LayoutPosition3D, ConditionalBlock]` populated by `add_cube`. Survives `merge_parallel_block_layers` flattening at `to_layer_tree()`.

`LayerTree.__init__` accepts `conditional_blocks` kwarg; exposes via `conditional_blocks` property. Non-conditional graphs default to empty mapping.

### `dcdabc53` — `ConditionalCircuit` / `IfBlock` / `branch_diff`

**Files**: `circuit.py`, `test_circuit.py`

Stim 1.15 has no IF/ELSE primitive → custom wrapper:

- `IfBlock(condition_rec, then_body, else_body?)` dataclass
- `ConditionalCircuit` mirrors subset of `stim.Circuit.append` + `append_if`
- `to_stim_text()` serializes with `IF(rec[k]) { ... } ELSE { ... }` syntax
- `to_stim_circuit_strict()` round-trips to vanilla `stim.Circuit` (raises if IfBlock present)

`branch_diff(branch_zero, branch_one, condition_rec)` walks two parallel branch circuits via `difflib.SequenceMatcher`. Matching spans emit unconditionally; unmatched spans collapse into single `IfBlock`. Handles count mismatch (branch-only detectors).

### `8a3b383f` — Two-pass IF/ELSE emission

**Files**: `graph.py`, `circuit.py` (diff upgraded), `test_emission.py`

`TopologicalComputationGraph.generate_conditional_stim_text(k, condition_rec, ...)`:

1. No conditional blocks → returns `str(generate_stim_circuit(k))` (passthrough).
2. Exactly one conditional block → transient swap to `block_if_zero`, compile; swap to `block_if_one`, compile; `branch_diff` the two `stim.Circuit` outputs; serialize via `ConditionalCircuit.to_stim_text()`.
3. Multi-conditional graphs raise `NotImplementedError` (two-pass strategy is 2^N).

Caller supplies `condition_rec` until frame tracker drives resolution.

### `e49fb729` — End-to-end test

**Files**: `test_end_to_end.py`

Per pair in `{XZZ_XZX, ZXX_ZXZ, XZX_XZZ}`:

- Builds minimal graph: init cube + temporal pipe + conditional leaf
- Compiles via `generate_conditional_stim_text(k=1, condition_rec=-1)`
- Asserts: ≥1 top-level IF/ELSE block; per-branch measurement target sets identical (CEO); for XZX_XZZ, else-arm is MX-only and then-arm contains Z-basis measurements

### `e93570b6` — Propagate ConditionalBlock through pipe-border transforms

**Files**: `compile/blocks/block.py`, `compile/graph.py`

`add_pipe` calls `with_temporal_borders_replaced` / `with_spatial_borders_trimmed` on the affected cube and writes result back into `_blocks[pos]`. Base `Block` overrides returned plain `Block`, dropping `block_if_zero` / `block_if_one` / `condition`. `_conditional_blocks[pos]` still held the untransformed original, so two-pass emission swapped in sub-blocks whose pipe-facing border was never substituted — top cube re-initialised all data qubits and dropped every boundary-crossing detector.

Fix:

- `ConditionalBlock.with_temporal_borders_replaced` / `with_spatial_borders_trimmed` overrides apply transform to both branches and rebuild a `ConditionalBlock`. Equal-layer-count + equal-meas-signature invariants preserved because the same transform is applied to both branches.
- `_set_cube_block` helper mirrors writes into `_conditional_blocks`. `_trim_cube_spatial_borders` + `_replace_temporal_border` route through it.

Validated by `scripts/test_conditional_two_cubes.py` (XZX bottom + XZX_XZZ conditional top, joined by temporal pipe — branch_zero must equal `XZX → XZX` baseline, branch_one must equal `XZX → XZZ` baseline). Both asserts pass after fix.

### `db986014` — Per-target locality in `branch_diff` IF/ELSE wrap

**Files**: `compile/conditional/circuit.py`, `scripts/{verify_conditional_locality,test_four_cubes}.py`

When CEO merges init/measurement layer across all cubes in a slice, one branch may emit a single big instruction (e.g. `RX <all qubits>`) while the other emits the CEO-canonical interleaved form (`RX(...); R(...); RX(...); R(...); ...`). Previous `branch_diff` collapsed the whole span into one `IfBlock` — pulling neighbouring cubes' qubits into the conditional cube's branch body.

`_split_unequal_span` walks the structured side's instruction boundaries (`drive_by_one = len(one_span) >= len(zero_span)`). Per S-instruction it splits targets into common / only-one / only-zero buckets, emits common pieces unconditionally and wraps only divergent targets in narrow IF/ELSE. Stim's adjacent-same-name auto-merge on parse reproduces the baseline instruction structure on resolution, so strict text equality with the non-conditional baseline is preserved.

Measurement gates excluded from per-target splitting via `_PER_QUBIT_GATES` whitelist (R*, single-qubit Cliffords). Equal-measurement-count invariant guarantees they should never appear in unequal spans, and splitting them would shift `rec[-k]` offsets.

New scripts:

- `scripts/test_four_cubes.py` — 2x2x2 graph: 6 plain XZX cubes + conditional `XZX_XZZ` at `(1,1,0)` joined to `(1,1,1)` via temporal pipe. Asserts branch_zero / branch_one text-eq with the non-conditional baselines `(1,1,0)=XZX` and `(1,1,0)=XZZ`.
- `scripts/verify_conditional_locality.py` — Parses IF/ELSE bodies via `tools/resolve._parse`. Asserts every gate target maps to a qubit coord inside the conditional cube's tile (derived empirically from `QubitMap` clustering); every DETECTOR `(x, y, t)` triple inside IF/ELSE has `(x,y)` in tile and `t` in the cube's z-window; every IF block opens while the running `SHIFT_COORDS` z-counter sits in the cube z-window.

### `145dc902` — Guard against spatial-pipe-from-conditional cube

**Files**: `compile/graph.py`

`ConditionalBlock.__init__` aliases `self.layer_sequence` to `block_if_zero.layer_sequence`. When a conditional cube has a spatial pipe already attached (`trimmed_spatial_borders` non-empty) and a temporal pipe is then added, `_replace_temporal_border` would feed the substituted layer into `_substitute_part_of_spatial_pipe` using state derived only from the zero branch — silently losing the one-branch boundary. Spatial-pipe substitution would also store a single (non-conditional) pipe block that cannot encode per-branch boundary differences.

Conservative guard added at the top of `_replace_temporal_border`: raise `NotImplementedError` if the target cube is a `ConditionalBlock` with `trimmed_spatial_borders`. Until per-branch spatial pipe substitution is wired, conditional cubes must be connected via temporal pipes only. No current code path triggers the guard (all integration scripts use temporal pipes to/from conditional cubes); all 42 conditional tests still pass.

### `e85bb72e` + `82628331` — Co-compile prep: plumb `conditional_layers` down the layer tree

**Files**: `compile/blocks/layers/atomic/layout.py`, `compile/blocks/layers/merge.py`, `compile/blocks/block.py`

First two commits of the planned single-pass co-compilation refactor (see `~/.claude/plans/how-do-you-suggest-radiant-bentley.md`).

- `LayoutLayer.__init__` gains an opt-in `conditional_layers: dict[LayoutPosition2D, BaseLayer] | None` parameter. It carries the branch-`one` slice for cube positions that came from a `ConditionalBlock`; `layers[pos]` keeps the zero-branch slice via `Block.__init__`'s alias. Field also participates in `__eq__`.
- `merge_base_layers` gains a matching pass-through.
- `merge_composed_layers`, `merge_repeated_layers`, `merge_sequenced_layers`, and `merge_repeated_and_sequenced_layers` all thread `conditional_layers` recursively. The `RepeatedLayer` LCM-expansion path raises `NotImplementedError` if conditional alternates are present (won't fire on current fixtures).
- `merge_parallel_block_layers` detects `ConditionalBlock` positions and passes `block.block_if_one.layer_sequence[i]` per timestep. The branch-one slice lands inside the resulting `LayoutLayer.conditional_layers` dict.

Probe: compiling a 2-cube graph (`XZX` + `XZX_XZZ` joined by temporal pipe) produces 6 `LayoutLayer`s; the 3 sitting at the conditional cube's z-layer carry a populated `conditional_layers` map. 42 conditional tests still pass.

Field is populated but not yet read — emission still goes through the existing two-pass `branch_diff` path. Subsequent commits will (a) teach `_emit_moment_with_ceo` to weave per-branch slots into `IfBlock` entries, (b) teach the detector annotator to do per-branch detector computation with structural-equality categorisation, and (c) collapse `generate_conditional_stim_text` to a thin wrapper.

### Co-compile Stage A commit 2 — `_emit_moment_with_ceo` accepts per-branch input

**Files**: `circuit/schedule/manipulation.py`, `tests/circuit/schedule/manipulation_test.py`

Promotes the per-moment buffer in `merge_scheduled_circuits` to a returned `list[stim.CircuitInstruction | IfBlock]`. Refactor steps:

- Extract pure `_stage_ceo_entries(merged, qubit_to_block, global_i2q) -> (passthrough, ceo_entries)` helper from the previous in-place emitter — same split-and-sort logic, reused for both branches.
- `_emit_moment_with_ceo` returns a `list[CircuitEntry]` instead of mutating an external `stim.Circuit`. Non-conditional path: collapses same-`(name, args)` CEO runs into plain `stim.CircuitInstruction`s; byte-equivalent to the pre-refactor emitter.
- New kw-only params `branch_merged_instructions` + `condition_rec`. When given, both branches are staged independently. Slot count + passthrough equality are sanity-checked (EMC + CEO assertion). For each CEO slot index, identical `(block_pos, qids, name, args, targets)` → joined into a collapsed plain instruction (multi-slot run); divergent → `IfBlock(condition_rec, then=[branch_one], else=[branch_zero])`.
- `merge_scheduled_circuits` caller iterates returned entries; non-`IfBlock` appended to the moment's `stim.Circuit` as today. An `IfBlock` (impossible until commit 3 wires branch input) raises `NotImplementedError` — sentinel for next step.

Convention: first positional `merged_instructions` is treated as branch-zero, `branch_merged_instructions` as branch-one (matches `branch_diff(zero, one, …)` ordering: `then_body=one`, `else_body=zero`).

5 new unit tests cover: no-branch baseline parity, identical branches produce no IfBlock, divergent `R`/`RX` weave produces per-slot IfBlocks, missing `condition_rec` raises, slot-count mismatch raises. Full suite: 670 pass, 0 regressions.

### Co-compile Stage A commit 3a — `merge_scheduled_circuits_per_branch` helper

**Files**: `circuit/schedule/manipulation.py`, `circuit/schedule/__init__.py`, `tests/circuit/schedule/manipulation_test.py`

Adds a sibling of `merge_scheduled_circuits` that consumes two parallel `list[ScheduledCircuit]` (one per branch), walks moments in lockstep with a shared `global_qubit_map`, and returns `(list[list[CircuitEntry]], Schedule)`. Each per-moment bucket is the output of `_emit_moment_with_ceo` with both branches threaded in — divergent CEO slots surface as `IfBlock(condition_rec, then=branch_one, else=branch_zero)`, identical slots collapse to plain `stim.CircuitInstruction`.

Errors: moment-count divergence, schedule mismatch between branches, CEO slot count mismatch (the last delegated to `_emit_moment_with_ceo`).

No downstream callers yet — the generator/tree/detector annotators still use the single-branch path. Commit 3b will wire `LayoutLayer.to_circuit` (and the detector annotator) onto this helper when `conditional_layers` is non-empty.

3 new unit tests: identical-branches produce no IfBlock, R/RX divergence emits IfBlock(condition_rec=-3), schedule mismatch raises. Full suite: 673 pass, 0 regressions.

---

## Test outcomes

| Layer | Tests added | Status |
|---|---|---|
| CEO diagnostic | 8 | 6 pass, 2 xfail (ZXZ_ZXX) |
| Conditional CubeBuilder | 8 | all pass (incl. ZXZ_ZXX rejection) |
| Metadata threading | 2 | pass |
| PauliFrameTracker | 8 | pass |
| ConditionalCircuit / branch_diff | 8 | pass |
| Stage 6 emission | 3 | pass |
| End-to-end | 4 | pass |
| **Total new** | **41** | |
| **Baseline** | 617 | unchanged |
| **Final** | **656 pass + 8 skip + 2 xfail** | **0 regressions** |

---

## Known issues / deferred

1. **ZXZ_ZXX upstream typo** — enum tuple is `(ZXCube.ZXZ, ZXCube.XXZ)` per PR #829 but name + docstring imply `(ZXCube.ZXZ, ZXCube.ZXX)`. Spatial vs temporal swap. Rejected by Equal-Meas-Count guard. User decision: ignore for now.

2. **Two-pass scaling** — single conditional cube only. Multi-cube graphs require true single-pass per-layer diff (not 2 full compiles per cube).

3. **Frame tracker is stub** — no Clifford propagation. Downstream detectors that should XOR-cancel branch dependence currently end up wrapped in IF/ELSE by `branch_diff`. Works but suboptimal text size. (After `db986014` per-target locality, only genuinely divergent targets sit in IF/ELSE — the residual IF/ELSE bloat is exactly the detectors the frame tracker would collapse.)

4. **Manual `condition_rec`** — caller supplies; should be auto-resolved from `CorrelationSurface.condition` once frame tracker drives emission.

5. **No nested IF / chained conditions** — explicitly out of scope per PPD.

---

## Files added

- `src/tqec/compile/conditional/__init__.py`
- `src/tqec/compile/conditional/circuit.py` — `IfBlock`, `ConditionalCircuit`, `branch_diff`, `_split_unequal_span` (per-target locality)
- `src/tqec/compile/conditional/frame.py` — `PauliFrameTracker`, `ConditionId`
- `src/tqec/compile/blocks/block.py` — `ConditionalBlock` class added inside existing file
- `tools/resolve.py` — IF/ELSE-text → `stim.Circuit` resolver (per-branch flatten)
- `scripts/test_conditional_two_cubes.py` — two-cube temporal-pipe baseline test
- `scripts/test_four_cubes.py` — 2x2x2 four-cubes baseline test
- `scripts/verify_conditional_locality.py` — IF/ELSE-body locality assertions
- 6 new test files under `tests/compile/{,conditional/}` + `tests/tools/test_resolve.py`

## Files modified

| File | Why |
|---|---|
| `src/tqec/circuit/schedule/manipulation.py` | CEO sort + new `_emit_moment_with_ceo` |
| `src/tqec/compile/generation.py` | Threads `plaquette_to_block` |
| `src/tqec/compile/blocks/layers/atomic/layout.py` | Builds `plaquette_to_block` |
| `src/tqec/compile/blocks/block.py` | `ConditionalBlock` + Equal-Meas-Count helpers |
| `src/tqec/compile/specs/base.py` | `CubeSpec.condition` field |
| `src/tqec/compile/specs/library/fixed_bulk.py` | Conditional CubeBuilder branch |
| `src/tqec/compile/specs/library/fixed_boundary.py` | Conditional CubeBuilder branch |
| `src/tqec/compile/graph.py` | `_conditional_blocks` + `generate_conditional_stim_text` + `_set_cube_block` helper |
| `src/tqec/compile/tree/tree.py` | `conditional_blocks` kwarg + property |
| `src/tqec/compile/blocks/block.py` | `ConditionalBlock.with_spatial_borders_trimmed` / `with_temporal_borders_replaced` overrides |
| `src/tqec/compile/conditional/circuit.py` | `_split_unequal_span` per-target diff (replaces `_split_per_target`) |
