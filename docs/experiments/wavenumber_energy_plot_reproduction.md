# Reproduce the E5, E8 and E10 energy comparison

`scripts/plot_wavenumber_energy_comparison.py` reads saved trajectory arrays.
It does not import the model, launch Modal work, train or evaluate checkpoints.
Use a Python environment containing NumPy and Matplotlib. Pillow is only needed
for the optional pixel comparison described below.

## Required saved evidence

Supply the root of `wavenumber_full_window_lr1e4_e10_20261005`, containing:

- `manifest.json`, which identifies the heldout cases, fundamentals, amplitudes,
  mode weights and phases.
- `evaluation/update_00002960/*.npz` for E5, inherited from the parent run.
- `evaluation/update_00004736/*.npz` for E8.
- `evaluation/update_00005920/*.npz` for E10.

Each evaluation must include all 21 cases. The script reads `times`,
`teacher_energy` and `model_energy`, verifies finite positive model energy,
and checks that reference arrays and time samples agree across checkpoints.
It does not require the large model checkpoints themselves.

The local root used for the completed run is
`/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e10_20261005`.
The remote volume is `vpml-wavenumber-generalization-v2-20260925`, with the same
run directory name. Artifacts are retained outside Git.

## Latest figure, with formulas and without footer text

From the repository root, using the scientific Python environment:

```bash
python scripts/plot_wavenumber_energy_comparison.py \
  --root /Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e10_20261005 \
  --output /tmp/vpml_e10_reproduction/energy_formulas.png
```

Choose a new output path. The script refuses to replace either the PNG or its
adjacent `_provenance.json` record. The provenance includes all 63 source array
file hashes, manifest and generator hashes, panel placements, full precision
normalized coefficients, phases and the formula displayed in each panel.

Use `--footer` to restore the two explanatory lines. Use `--style families
--footer` to reproduce the earlier version with isolated and mixture labels.
The default style is `formulas`, with no footer.

## Panel interpretation

Columns correspond to fundamental wavenumbers 0.30, 0.35, 0.40, 0.45 and 0.50.
Rows align linear, weak and strong amplitudes, first for single cosines and then
for sums of cosines. Empty positions indicate that the dataset contains no
heldout single cosine at that fundamental. All 21 cases appear exactly once.

The header defines `theta = k0*x`. Panel formulas use coefficients normalized
by the sum of absolute mode weights. Coefficients and phases are displayed to
three decimal places, with an approximation sign for the rounded expressions.
Phases are in radians. Curve data are never rounded or recomputed.

The shared color key is blue for the kinetic reference, ochre for E5, teal for
E8 and purple for E10. Energy uses a logarithmic vertical axis, with separate
vertical limits per panel and a common time range of 0 to 120.

Mixtures at familiar fundamentals use J={1,2,4}; mixtures at unseen fundamentals
use J={1,2,3,4}. Their weights and phases also vary. Comparing columns therefore
does not isolate a change in fundamental alone.

## Verification when committing the generator

On 2026-10-06, all three existing five-column figure variants were regenerated
into a temporary directory. Decoded RGBA arrays matched the original PNGs
exactly for the family labels, formulas with footer and formulas without footer.
The latest rendering was also visually inspected. All original PNG hashes
remained unchanged, all 21 cases and 63 input files were accounted for, and a
second render to each existing output path was rejected.

The original scripts and images remain under the run's `analysis/final_e10`
directory. The repository generator consolidates their plotting behavior with
explicit input and output arguments. Pixel equality was checked in the existing
local scientific environment; other Matplotlib or font versions may render
text differently without changing the data.
