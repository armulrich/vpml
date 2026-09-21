# Matched linear-only closure experiment

Status: stopped on 21 September 2026 after workspace budget exhaustion.
History completed E13, latent completed E15. No E20 result exists. Both Modal
apps are stopped with zero tasks. No further remote compute is planned.

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
- E0 and E1 were evaluated remotely from t = 0 through T = 120. An E1
  stop/resume preserved optimizer, RNG and schedule state. Later evaluation
  was scheduled after E20 and was not reached. Selected frozen checkpoints
  were instead evaluated locally after the budget stop. Not every saved epoch
  has an autonomous evaluation.

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
matching source/configuration hashes. After the interrupted run, client-side Volume reads verified all 126
historical checkpoint/configuration/state fingerprints against the pre-run list. Each process has a private derived-cache
metadata directory while reading the preserved reference arrays.

Local records are under
`/Users/armin/Documents/NYU/vpml/out_bench/linear_only_pair_20260921_v1`.
All 132 downloaded files matched SHA256 hashes computed from remote Volume
reads after stopping the apps. This launched no remote compute.
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


## Budget-stop qualification

| Model | Complete epoch | Complete updates | Partial state | Durable updates |
|---|---:|---:|---|---:|
| History | 13 | 6,838 | E14: 450/526 | 7,288 |
| Latent | 15 | 7,890 | E16: 400/526 | 8,290 |

The durable `training_state.npz` contains the complete loss history and exact
partial-epoch resume state. The separately exported `training_metrics.npz` is
stale at E1 because orderly finalization was interrupted. Loss figures use the
state history and omit incomplete epoch averages. Run directory names ending in
E20 record the intended target, not the achieved epoch.

| Checkpoint | Window validation | Full field error | Log-energy RMSE (decades) |
|---|---:|---:|---:|
| History E2 | 0.03237 | 0.48278 | 2.34116 |
| History E13 | 0.71888 | 1.19047 | 7.90653 |
| Latent E13 | 0.03056 | 0.40703 | 2.35614 |
| Latent E15 | 0.02783 | 0.42225 | 1.44012 |

All values average four linear validation ICs. Field and energy diagnostics use
the complete autonomous T=0 to 120 interval. E13 provides equal update exposure.
History E2 and latent E15 minimize their completed window-validation histories,
not necessarily autonomous error across every saved checkpoint. All evaluated
rollouts were finite with zero density/pressure limiter activation.

The latent model retains damping longer at E15 but still develops an incorrect
late floor. History deteriorated after a large finite E8 gradient/loss spike.
Latent also spiked at E6 and E10, then recovered. Removing nonlinear examples
therefore did not remove optimization instability or guarantee late damping.
No clipping or update cap was active. Adam's retained second moment can still
reduce effective step sizes after large gradients.

The initializer-specific output head has zero trajectory-loss gradient under
detached preparation. Its six Adam m/v arrays remain exactly zero through saved
update 8,290. A synthetic counterfactual restored a gradient norm of 8.3416e-7
without any forward or loss change. This establishes the disconnected path,
not that it explains all damping errors. Shared initializer features still train.
History-only failure requires an additional explanation.

Recommended next work is a targeted gradient-path repair and bounded checks of
late-time field sensitivity and coupled linear response. No further paid run,
update cap, architecture change or optimizer reset was performed. This is one
seed and an interrupted comparison, not an impossibility result for history-only
closure or proof that memory solves linear kinetics.

The full report and figures are preserved at:
`/Users/armin/Documents/NYU/vpml/out_bench/linear_only_pair_20260921_v1/REPORT.md`.
The artifact directory also includes per-case metrics, optimizer accounting,
CPU/GPU evaluation parity, all downloads and the storage integrity manifest.
CPU and GPU E1 evaluations differ by at most 3.4e-5 relative field L2 and 0.00050
decades energy RMSE, much smaller than the reported checkpoint differences.

Launched source commit: `af7ad4f3eec5da0872102a0dd97f96158fb230d7`.
Model/vpml source hash:
`21fd6a60a96c16e475e0675024491816e03a1164cd4f5a53d0238c40bb0b784a`.
