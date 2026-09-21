# Matched linear-only closure experiment

Status: implementation and preflight, before launch.

The preserved history and latent models had inaccurate late linear damping.
This controlled pair asks whether removing nonlinear examples from optimization
lets the same models learn the existing multimode linear family. It does not
test nonlinear generalization or establish an architectural impossibility if
training fails.

The historical E40 launch implementation was committed as `e8a5763` before
development. The new branch is `exp/linear-only-history-latent`. The original
research checkout at `/Users/armin/Documents/NYU/vpml` is not being cleaned up or
committed as part of this experiment.

## Fixed comparison

- Existing 16 linear training ICs and 4 linear validation ICs.
- Existing kinetic cache and split, unchanged.
- Random seed 1729, common shared parameter initialization.
- Burles history FNO, width 64, depth 8, 32 spectral modes.
- History-only has zero latent channels. Memory model has six latent channels.
- Nx 128, dt 0.025, five-unit history, five-unit detached autonomous burn-in,
  ten-unit scored rollout, existing global relative trajectory loss.
- Every valid linear anchor once per epoch, shuffled without replacement.
- Batch 16 linear windows, one update per batch. This retains the previous
  number of linear examples per update, but removes the 32 nonlinear examples.
- Expected 8,416 linear windows and 526 updates per epoch, derived and checked
  from the anchor table. E20 therefore has 10,520 updates per model.
- AdamW initial learning rate 0.001, final 0.00001 on the original 1000-epoch
  schedule. No gradient clipping, update cap, backtracking or loss-based update
  rejection. The authorized stopping point is E20.
- Fixed normalization copied from the preserved history E0 statistics only.
  No old neural parameters initialize either run.
- All epoch checkpoints evaluated autonomously from t = 0 through T = 120.
  An E1 stop/resume permits early trajectory inspection and preserves optimizer,
  RNG and schedule state. It is not a new training phase.

## Known limitation retained deliberately

The detached burn-in disconnects the latent initializer-specific projection
head from trajectory-loss gradients. Shared features can still change the
effective initialization. A bounded numerical preflight records the zero
gradient and compares an identical forward computation with the initialization
gradient path connected. This confirms a training limitation without silently
changing the architecture comparison. Any repair is a separately reported
proposal, not an undisclosed part of these runs.

## Preservation

Remote run directories are
`/mnt/vpml/runs/linear_only_pair_20260921_v1_history_E20` and
`/mnt/vpml/runs/linear_only_pair_20260921_v1_latent_E20`.
Creation refuses existing directories unless explicit resume is requested with
matching source/configuration hashes. Original checkpoint fingerprints are
checked before and after the run. Each process has a private derived-cache
metadata directory while reading the preserved reference arrays.

Local records are under
`/Users/armin/Documents/NYU/vpml/out_bench/linear_only_pair_20260921_v1`.
Downloaded checkpoints and evaluations will be checked against remote hashes.
`out_bench` remains Git ignored.

## Launch

From the new source worktree, calculate the source tree SHA256 with
`model.train.modal_burles_latent_fno._source_tree_sha256(Path.cwd())` and record
`git rev-parse HEAD`. Invoke once per model:

```sh
modal run --detach --timestamps model/train/modal_linear_closure_pair.py::run \
  --model history --source-tree-sha256 <source-hash> --source-commit <commit>

modal run --detach --timestamps model/train/modal_linear_closure_pair.py::run \
  --model latent --source-tree-sha256 <same-source-hash> --source-commit <same-commit>
```

## Interpretation

Compare raw complete-epoch training loss and held-out window validation loss,
then judge full autonomous damping, frequency, envelopes and phase separately.
If both models improve, mixed-regime optimization is implicated. If their
linear behavior differs, inspect the added memory coupling. If both fail,
investigate shared scaling, closure timing, loss sensitivity and optimization.
None of these outcomes alone proves that additional memory or more epochs are
necessary or sufficient. A smooth loss curve is not the acceptance metric.
