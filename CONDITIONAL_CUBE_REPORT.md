# Conditional-Cube Compilation: Implementation Report

**Branch**: `ccf-compile` (tqec_ccf fork)
**Base**: `b77ca994` (tip of upstream PR #829, `feat/conditional-cube`)
**Commits added**: 19
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

### Co-compile Stage A commit 3b — `LayoutLayer.to_conditional_circuit`

**Files**: `circuit/schedule/manipulation.py` (none — already shipped), `compile/blocks/layers/atomic/layout.py`, `compile/generation.py`, `tests/compile/blocks/layers/atomic/layout_test.py`

End of the per-branch emission chain at the `LayoutLayer` boundary.

- `_compute_template_and_plaquettes(layers)` extracted from `to_template_and_plaquettes`. New `_branch_one_layers()` builds the branch-one layer map (`conditional_layers[pos] if present, else layers[pos]`).
- `generation._build_scheduled_circuits_for_plaquette_array` extracted from `generate_circuit_from_instantiation` (pure refactor; same byte-output).
- New `generation.generate_per_branch_circuit_from_instantiation(plaquette_array, zero_plaquettes, one_plaquettes, increments, plaquette_to_block, condition_rec)` walks the plaquette array twice (once per branch), relabels both lists against a shared `QubitMap` (concat-then-split), and delegates to `merge_scheduled_circuits_per_branch`. Returns `(list[list[CircuitEntry]], QubitMap)`. Raises `TQECError` if per-branch qubit ownership maps disagree.
- New `LayoutLayer.to_conditional_circuit(k, condition_rec, reschedule_measurements=True) -> ConditionalCircuit`. Requires non-empty `conditional_layers`. Computes per-branch templates (must match exactly), per-branch plaquettes, builds `plaquette_to_block` from the template's `get_indices_map_for_instantiation`, runs the per-branch emission, then assembles a `ConditionalCircuit` with `QUBIT_COORDS` preamble (qubits shifted into the layer frame as in `to_circuit`) and `TICK` separators between moments. Stabiliser-round LayoutLayers (where both branches' plaquettes happen to be equal) produce a `ConditionalCircuit` with no `IfBlock` entries — exactly what the design predicts.
- New `_reschedule_per_branch_measurements()` extends `reschedule_measurements` to plaquettes that live only in `conditional_layers`.

No call site rewired yet: `LayerTree`, the detector annotator, and `generate_conditional_stim_text` still go through the existing `to_circuit` / two-pass `branch_diff` path. Commit 3c will wire the new method into the tree-level annotation.

2 new unit tests: `to_conditional_circuit` raises on empty `conditional_layers`; on a compiled 2-cube XZX_XZZ graph, at least one of the 3 `LayoutLayer`s carrying `conditional_layers` surfaces `IfBlock` entries with the supplied `condition_rec`. Full suite: 675 pass, 0 regressions.

### Co-compile Stage A commit 3c — Annotator wires `to_conditional_circuit`

**Files**: `compile/tree/annotations.py`, `compile/tree/node.py`, `compile/tree/annotators/circuit.py`, `compile/tree/tree.py`, `tests/compile/tree/annotators/circuit_test.py`

`AnnotateCircuitOnLayerNode` gains an optional `condition_rec: int | None` ctor parameter; `LayerTree._annotate_circuits` / `LayerTree._generate_annotations` thread it through. When `condition_rec` is supplied **and** a leaf's `LayoutLayer.conditional_layers` is non-empty, the annotator additionally calls `to_conditional_circuit(...)` and stores the result on the new `LayerNodeAnnotations.conditional_circuit: ConditionalCircuit | None` field. The branch-zero `ScheduledCircuit` annotation is always populated (unchanged); downstream detector annotation keeps reading from it.

New `LayerNode.set_conditional_circuit_annotation(k, conditional_circuit)` mirror to the existing setter.

No tree-level assembly yet: `LayerTree.generate_circuit` still returns a plain `stim.Circuit` from the branch-zero annotations; the per-leaf `conditional_circuit` is annotated but not yet consumed by any caller. The follow-up commit will introduce a `LayerNode.generate_conditional_circuit(k, qubit_map) -> ConditionalCircuit` walker that picks up the new annotation; commit 4 takes per-branch detector annotation and retires `branch_diff`.

2 new unit tests on a compiled 2-cube `XZX_XZZ` graph: with `condition_rec=None`, no leaf gets a `conditional_circuit`; with `condition_rec=-7`, every leaf whose `LayoutLayer` has `conditional_layers` carries a populated `ConditionalCircuit`, and at least one of those surfaces an `IfBlock` entry. Full suite: 677 pass, 0 regressions.

### Co-compile Stage A commit 3d — Tree-level assembly into a single `ConditionalCircuit`

**Files**: `compile/conditional/circuit.py`, `compile/blocks/layers/atomic/layout.py`, `compile/tree/node.py`, `compile/tree/tree.py`, `tests/compile/tree/conditional_circuit_test.py`

Closes the tree-level half of the single-pass refactor. Per-leaf `ConditionalCircuit` annotations laid down by commit 3c are now consumed by a tree walker that produces one assembled `ConditionalCircuit` for the whole computation.

- `ConditionalCircuit` gains an optional `qubit_map: QubitMap | None` attribute (`.qubit_map` property + `set_qubit_map(...)`). `append_instruction_or_if(entry)` convenience for callers iterating a mixed `CircuitEntry` sequence. New module-level `remap_entry_qubit_indices(entry, qubit_index_remap) → CircuitEntry` walks a single entry (recursing into `IfBlock` bodies) and returns a copy with qubit-target indices remapped; non-qubit targets pass through unchanged.
- `LayoutLayer.to_conditional_circuit` builds a *shifted* `QubitMap` (each `GridQubit` shifted by the layer's `mincube * (eshape - 1)` to match the global frame) and attaches it to the returned `ConditionalCircuit`. The `QUBIT_COORDS` preamble inside the returned circuit also uses the shifted coordinates, so a downstream consumer that strips the preamble loses no information.
- `LayerNode.has_conditional_descendant(k) → bool` — `True` only when at least one descendant leaf's annotated `ConditionalCircuit` actually surfaces an `IfBlock`. Stabiliser-round leaves that carry a `conditional_layers` map but happen to be byte-identical across branches are intentionally reported as non-conditional so that the `RepeatedLayer` fast path can still expand them as plain stim.
- `LayerNode.generate_conditional_circuit(k, global_qubit_map) → ConditionalCircuit`:
  - `LayoutLayer` leaf: when the leaf has a `conditional_circuit` annotation, walk its entries (skipping the local `QUBIT_COORDS` preamble) and apply `remap_entry_qubit_indices` with the local→global lookup derived from the local + global `QubitMap`. Otherwise reuse the existing branch-zero `ScheduledCircuit` path (`map_qubit_indices`, `get_circuit(include_qubit_coords=False)`, ingest as plain `stim.CircuitInstruction` entries). Detector + observable annotations and a trailing `SHIFT_COORDS(0,0,1)` are appended at the leaf in the same positions as the existing emitter — they still come from the branch-zero annotation (per-branch detector annotation is commit 4 territory).
  - `SequencedLayers`: walk children, insert a `TICK` between adjacent fragments (same rule as `generate_circuits_with_potential_polygons`: skip the `TICK` when the next child is a `RepeatedLayer`).
  - `RepeatedLayer`: raise `TQECError` if any descendant has an actual `IfBlock` annotation; otherwise reuse the plain `generate_circuit` (returning `stim.Circuit`) and flatten its instructions into the `ConditionalCircuit`. `stim.CircuitRepeatBlock` is rendered as a flat repetition since `ConditionalCircuit` has no native `REPEAT` primitive at this stage. **Known limitation**: a multi-cube graph whose conditional cube sits *inside* a repeated stabiliser-round wrapper will fail here; current fixtures never hit this case.
- `LayerTree.generate_conditional_circuit(k, condition_rec, ...)` is the new top-level entry point. Reuses the database-resolution prelude from `generate_circuit`, drives `_generate_annotations(..., condition_rec=condition_rec)`, then assembles via the new walker. `include_qubit_coords=True` prepends the global `QUBIT_COORDS` map once at the top of the returned `ConditionalCircuit`. Existing `generate_circuit` (plain `stim.Circuit` return) is unchanged.

The path is **not** yet wired into `TopologicalComputationGraph.generate_conditional_stim_text`; the public entry point still goes through the two-pass `branch_diff` strategy. Commit 4 will collapse that wrapper onto `LayerTree.generate_conditional_circuit` and replace branch-zero-only detector annotation with per-branch annotation.

2 new unit tests: on a compiled 2-cube `XZX_XZZ` graph, `LayerTree.generate_conditional_circuit(k=1, condition_rec=-1)` returns a `ConditionalCircuit` with at least one top-level `IfBlock` (and no duplicate `QUBIT_COORDS`); the rendered `to_stim_text()` output contains the expected `IF(rec[-1]) { ... } ELSE { ... }` block syntax. Full suite: 679 pass, 0 regressions.

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
