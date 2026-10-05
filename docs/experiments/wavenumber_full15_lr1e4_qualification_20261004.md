# Full-window learning-rate qualification

This experiment tests whether a smaller global AdamW learning rate can train the
15-unit objective without the initial overshoot or the later autonomous failure.
It does not introduce a new architecture, IC weighting, gradient cap, or loss term.

## Evidence motivating the run

The preceding full-window attempt stopped before accepting any update. Its first
AdamW proposal at learning rate 0.001 increased the same-batch loss from
0.6078956598 to 0.6301582615. The model and gradients were finite.

A preserved directional check found a directional derivative of
-0.1359747551 along that proposal. Finite differences agreed, with maximum relative
error 1.35e-8 across the tested small steps. Taking 0.1 times the proposal reduced
loss to 0.5945270753. This supports initial step overshoot. It does not establish
that the later trajectory instability has the same cause.

The earlier 5-unit preparation plus 10-unit scoring run reached 592 accepted
updates, then failed on 3 of 21 autonomous T120 cases. Its saved fixed-weight
diagnosis found artificial closure-driven harmonic growth before admissibility
limits activated. That remains a distinct qualification concern.

## Frozen experiment

- Output namespace: `wavenumber_full_window_lr1e4_e5_20261004`.
- Local root: `/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e5_20261004`.
- Same remote root in volume `vpml-wavenumber-generalization-v2-20260925`.
- 75 existing references: 54 training and 21 held-out ICs. No regeneration or old dataset cases.
- Same random initialization, seed 1729, FNO and six-channel latent model.
- Float64 dynamics, Nx=128, dt=0.025, 50 history samples at spacing 0.1.
- All 600 steps in each 15-unit training window are scored and differentiated.
  There is no unscored or detached preparation interval. Windows still start
  from reference moments at their anchors, with the learned initializer.
- Same per-window relative L2 objective and normalization. No case-specific scaling.
- AdamW initial rate 0.0001, final rate 0.00001 over the same planned 1000 epochs.
  No gradient clipping, update cap, or backtracking.
- 28,404 windows and 592 accepted updates per complete epoch.
- Frozen scientific source SHA256:
  `39f23173cdce9ac1a895985c96b432fab6e11d0258bf896ee22925da17773cc6`.
- Protocol SHA256:
  `a344037e0a424e41232776270f21d04cfa333d100b35256e49bb1197abb706d5`.
- Full protocol, initialization, manifest, evaluator and normalization hashes are
  stored in the immutable run setup. Prior result directories remain intact.

The E0 evaluation was reused only after verifying identical weights, data,
normalization, model and evaluation configurations. Only optimizer learning rates
were excluded from that fixed-weight comparison. Cached evaluation files were
individually hashed.

## Bounded checks

The rate was fixed before testing batches 1, 2, 37 and 40. Each test started from
the same E0 model and zero Adam state, and its trial weights were discarded.

| Batch | Loss before | Loss after |
|---|---:|---:|
| 1 | 0.60789566 | 0.59452708 |
| 2 | 0.32053972 | 0.30163443 |
| 37 | 0.40791583 | 0.39153034 |
| 40 | 0.21720869 | 0.20150722 |

All four had finite gradients and nonzero initializer gradients. The first actual
update was also repeated from the persisted initial checkpoint with bitwise-equal
outputs. Subsequent training does not reject finite steps merely because a
minibatch loss increases.

## Update-50 review

| Accepted updates | Finite T120 cases | Full15 validation | Mean field error | Mean log-energy RMSE |
|---|---:|---:|---:|---:|
| 0 | 21/21 | 0.5917913 | 4.0434886 | 3.5882059 |
| 3 | 21/21 | 0.5792135 | 3.6946314 | 3.5162016 |
| 50 | 21/21 | 0.3943342 | 1.2240339 | 2.5873839 |

At update 50, all 21 field errors and 19 of 21 log-energy errors improved relative
to update 3. All seven linear cases had negative late log-energy slopes over
t=60 to 120. The energy panels still show excess late retention, missing nonlinear
recovery, and incorrect dominant spatial harmonics. The isolated weak k=0.35 case
and strong mixture k0=0.30 have worse log-energy error than at update 3.

At the same 50 updates the previous preparation/scoring run had field error
1.1926644 and log-energy RMSE 2.6897449. The new run has not established superior
performance or sustained stability. The review permits the unchanged lineage to
continue through E1, with intermediate T120 evaluations at updates 150 and 300.

Saved plot reconstruction uses the raw batch loss and the full15 validation loss.
The older ten-unit validation objective is reported separately. Only complete
dataset sweeps receive an epoch mean. No EMA is used.

## E1 review

At 592 accepted updates, all 21 autonomous T120 trajectories remain finite,
including the three cases that failed at E1 in the preceding preparation/scoring
run. The first complete epoch has mean training loss 0.13787306. Full15 validation
is 0.16562232, mean field error is 0.44358850, and mean log-energy RMSE is 1.61068585.

The metrics disagree about late behavior. Update 300 has lower log-energy RMSE,
1.32354663, despite higher field error, 0.49767085. Between update 300 and E1,
15 of 21 field errors improve, but only 9 of 21 log-energy errors improve.
Several linear and weak trajectories retain excessive late energy. Harmonic 25
dominates several high-wavenumber linear predictions, and the two k0=0.45 linear
cases have small positive late energy slopes. Finite evolution is therefore not
evidence that the damping problem is solved.

The saved E1 review permits the same authorized E5 qualification to continue
unchanged, with autonomous evaluation at each epoch. Update 300 remains the best
log-energy checkpoint at this point. E1 is the best field-error checkpoint.

## Final E5 result

The authorized run completed 2,960 accepted updates and 142,020 training windows.
All 21 held-out autonomous T120 trajectories were finite at every saved evaluation.
The following epoch means use complete dataset sweeps without EMA.

| Epoch | Mean training loss | Full15 validation | Mean field error | Mean log-energy RMSE |
|---|---:|---:|---:|---:|
| 1 | 0.137873 | 0.165622 | 0.443588 | 1.610686 |
| 2 | 0.077666 | 0.145583 | 0.381280 | 1.490643 |
| 3 | 0.064890 | 0.127911 | 0.299489 | 1.413468 |
| 4 | 0.057308 | 0.124392 | 0.334184 | 1.650524 |
| 5 | 0.052361 | 0.116600 | 0.298396 | 1.358093 |

Both epoch training and validation losses decline across E1–E5. Autonomous
accuracy does not improve monotonically: E4 regresses in both trajectory metrics.
E5 has the lowest mean field error, only marginally below E3. Update 300 remains
the lowest mean log-energy checkpoint (1.323547). The lowest mean early fitted
growth-rate error is E2 (0.054306), compared with 0.152529 at E5. These are three
different selection criteria, and the full-energy plots must not be interpreted
as the early-rate scalar.

The k0=0.35 held-out linear cases and k0=0.45 linear cases show useful damping
over much of T120. Several k0=0.40/0.50 linear and weak cases retain false late
energy floors. Strong k0=0.50 damps where the reference regrows. Other nonlinear
phase and envelope errors persist. E5 improves field error on 12 of 21 cases
relative to E3 and log-energy error on 10 of 21. This is a completed stable
qualification, not uniform trajectory convergence or evidence that more epochs
alone will remove the remaining errors.

The final report and all figures are in
`/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e5_20261004/FINAL_REPORT.md`.
Clean current/reference panels for update 300, E3 and E5 are under
`analysis/update_00002960/clean/`. Raw batch loss and complete-epoch training and
validation plots are saved separately. Each panel uses one checkpoint for all cases.

## Execution and preservation

The first Modal application lost its local heartbeat when the laptop entered
clamshell sleep. The last durable checkpoint was update 3, while updates through
17 had appeared in the log. A second application expired before dispatch while
the laptop was asleep. Both were confirmed stopped before resumption.

The resumed run reproduced the logged loss, gradient norm and update norm exactly
for updates 4 through 17. It uses detached execution and a process-scoped idle
sleep assertion. Detached local entrypoints protect an already dispatched remote
call, not the entire sequence of future calls. No scientific setting changed.

During E2, a local DNS/connection failure interrupted orchestration. The remote
call was subsequently cancelled after logging update 721, with update 700 the
last durable state. Its remote and local checkpoint hashes matched. After the
old app stopped with zero tasks, the same frozen launcher resumed from update
700. The loss, gradient norm and update norm at updates 701 through 721 matched
the interrupted execution exactly, and checkpoint 750 was saved and mirrored.
Replayed work is not counted as additional unique training exposure.

Snapshots include parameters, Adam moments, update count, data cursor and loss
history. Remote snapshots and evaluations are mirrored with SHA256 checks and
overwrite refusal. The combined spending cap remains $350, with a reported-cost
stop at $300 to reserve for billing lag. `STOPPED.json` is a scientific stop and
must not trigger an automatic retry.

The current endpoint and review decisions are in `PLAN.json`,
`REVIEW_UPDATE_00000050.json`, and subsequent endpoint records in the run root.
E5 is complete. All 68 checkpoints and 466 remote files were mirrored and hash
verified. Remote checkpoint sidecars match local snapshots. The app stopped with
zero tasks, the local controller exited, and the monitoring heartbeat was paused.
No training beyond E5 or another scientific experiment was launched.

Reported cost is $34.53 for this run and $111.73 combined since September 25,
within the $350 cap. Billing may lag and these figures are not a finalized invoice.
Prior artifacts remain intact, and `out_bench` remains ignored.

## Git preservation and reproduction

The implementation is preserved on `exp/broader-wavenumber-latent` in three
focused commits, followed by this experiment record:

- `3525de2`: finite derivatives at equilibrium without changing forward values.
- `d3f91ce`: relative L2 loss, precision propagation and full-window/resume tests.
- `7ae76d9`: the exact frozen controller, autonomous evaluator and endpoint tests.

All 50 Python files in the run's scientific source match the committed source
byte for byte. The controller and evaluator also match their frozen copies.
The optional `initial_rms` diagnostic remains available in that source but is
inactive here. The protocol selects `window_rms`. No scientific code or model
weights were changed while making these commits.

The exact configuration is tracked in
[`wavenumber_full15_lr1e4_20261004/protocol.json`](wavenumber_full15_lr1e4_20261004/protocol.json).
The accompanying [provenance record](wavenumber_full15_lr1e4_20261004/provenance.json)
contains source hashes, checkpoint hashes, epoch losses, autonomous evaluations,
selection rules and original artifact locations. All 68 local checkpoint hashes
were checked again against their sidecars during preservation. Checkpoints,
reference arrays and figures remain in the existing artifact directory rather
than in Git. Earlier exploratory scripts remain in the worktree.

The historical launch/resume command was run from the artifact root, using its
immutable `setup` and `frozen_source` directories:

```bash
cd /Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e5_20261004
VPML_RUN_NAME=wavenumber_full_window_lr1e4_e5_20261004 \
  caffeinate -i /Users/armin/miniforge3/bin/modal run --detach \
  setup/modal_precision_e5.py::run --stop-epoch 5
```

This records the completed E5 command, not a new launch or approval to extend it.
The controller requires the preserved references, normalization, setup files
and matching hashes. The normalization comes from the recorded
`wavenumber_initial_rms_e5_20261002` namespace. That is a normalization artifact
dependency, not the training IC source or an active initial-RMS objective.
The dataset remains the 75-case `expanded_v2` manifest.

The following local checks passed during preservation:

```bash
cd /Users/armin/Documents/NYU/vpml-generalization
/Users/armin/Documents/NYU/vpml/.venv/bin/python -m unittest \
  tests.test_low_moment_ad tests.test_wavenumber_training \
  tests.test_wavenumber_precision tests.test_wavenumber_full_window -v
VPML_RUN_NAME=wavenumber_full_window_lr1e4_e5_20261004 \
  /Users/armin/miniforge3/bin/python3.12 -m unittest \
  tests.test_precision_controller_endpoints -v
```

These are 24 numerical/training tests and 2 controller bookkeeping tests.
They cover floor derivatives, per-example loss definitions, complete IC exposure,
precision, initializer gradients, chunked forward parity, exact resume and
overwrite refusal. The controller tests do not launch remote work.

`scripts/analyze_full_window_qualification.py` reconstructs the loss plots and
checkpoint comparisons from saved exports. Its `energy_panel` function generates
clean current/reference panels when `comparison=None` and a fresh output directory
is supplied. `scripts/summarize_saved_wave_feedback.py` summarizes saved spectra
and late energy slopes. Both analysis entry points refuse to overwrite existing
output files. No model rollout or training is needed to read these saved results.
