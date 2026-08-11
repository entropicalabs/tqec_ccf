# Vendored code notice (test oracle only)

The contents of `tests/_vendor/midout/` are vendored from Craig Gidney's
`midout` package, published alongside:

> C. Gidney, "Inplace Access to the Surface Code Y Basis", 2024.
> DOI: [10.5281/zenodo.7487893](https://doi.org/10.5281/zenodo.7487893)

Source obtained from the author-provided artifact archive
(`Gidney-Y-meas-code`, `src/midout/`), licensed Apache License 2.0.

## Purpose in this repository

This code is used **only as a verification oracle for tests** — it is the
ground-truth generator for the surface-code Y-basis measurement circuits and
their stabilizer flows, against which tqec's own native Y-half-cube
implementation (`src/tqec/compile/specs/library/generators/ycube.py` and
`_ycube_circuit.py`) is cross-checked. **It is never imported by `src/tqec/`
and never ships on the compile path.** It lives under `tests/` for this reason.

## Modifications

Vendored verbatim from Gidney's `src/midout/`, with only these mechanical,
behavior-preserving changes:

1. **Import-path rewrite** (all files): `from midout...` / `import midout`
   rewritten to `from tests._vendor.midout...` so the package resolves at its
   new location. No behavior change.
2. **`numpy.bool8` -> `numpy.bool_`** (`gen/_flow_verifier.py`): `numpy.bool8`
   was removed in NumPy 2.0; `np.bool_` is the identical dtype under a name
   that still exists. No behavior change.

The vendored `*_test.py` self-tests were intentionally omitted (they would be
collected by this repo's pytest run and are not needed — the oracle is
exercised by tqec's own Y-cube tests).

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
