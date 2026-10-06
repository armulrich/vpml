# Completed full15 latent qualification through E10

The exact E5 continuation completed at 5920 accepted updates, adding 2960 updates
with unchanged scientific settings. All 21 autonomous heldout trajectories were
finite through T120 at every evaluated epoch. Finiteness does not establish
correct late damping or nonlinear recovery.

## Complete epoch results

| Epoch | Update | Training mean | Full15 validation | Mean field error | Mean log-energy error |
|---|---:|---:|---:|---:|---:|
| E5 | 2960 | 0.05236113 | 0.11659991 | 0.29839557 | 1.35809303 |
| E6 | 3552 | 0.04827227 | 0.11520435 | 0.28347255 | 1.41835885 |
| E7 | 4144 | 0.04597257 | 0.11102912 | 0.66823032 | 1.67661441 |
| E8 | 4736 | 0.04420189 | 0.11026465 | 0.29032698 | 1.24174767 |
| E9 | 5328 | 0.04163258 | 0.10746194 | 0.29315711 | 1.28762600 |
| E10 | 5920 | 0.04086862 | 0.10722245 | 0.25601042 | 1.37850774 |

Training means average batches with changing weights over each complete epoch.
Validation evaluates fixed endpoint weights. Loss plots use raw batches and
complete epoch means without EMA smoothing.

## Checkpoint selection and trajectory interpretation

- Best evaluated mean field error: E10, update 5920, 0.2560104199.
- Best evaluated mean full trajectory log-energy error: E8, update 4736, 1.2417476698.
- Best evaluated mean early growth-rate error: E2, update 1184, 0.0543059057.
- Relative to E5, E10 lowers training mean by 21.95%, validation by 8.04% and
  mean field error by 14.20%. Individual field errors improve in 15/21 cases;
  log-energy errors improve in 13/21 cases.
- E7 has a substantial finite false-growth excursion in the weak k0=0.45 mixture.
  E8 recovers from that excursion and has the best aggregate log-energy score.
- E10 improves strong isolated cases, but the k0=0.45 linear mixture shows
  spurious late growth. Harmonic 2 carries 89.3% of its late field energy.
  False growth also persists at k0=0.50. Phase and nonlinear recovery errors remain.

Lower training or validation loss is not uniform improvement in autonomous
physics. Selection must name the metric; there is no single best checkpoint.

## Integrity and cost at completion

The completion audit verified all 592 remote files against the local mirror.
The final checkpoint is `training/state_00005920.npz`, SHA256
`9922e0659fcd50e202425064c5c57c9ea0a75851e23a338a1736e663760f210a`.
The parent E5 artifacts remain unchanged.

Reported combined spending was $147.04603165 against the $350 cap. Continuation
apps accounted for $35.27605657, including infrastructure restarts and replay.
Reported charges can lag the final invoice. The last owned app was confirmed
stopped with zero tasks, and the monitoring heartbeat was deleted.

## Evidence and limitations

The adjacent JSON retains epoch metrics, best checkpoint selections, all 21 E10
versus E5 case comparisons, the completion audit and hashes of supporting records.
It is a compact repository record; large checkpoints and trajectories stay in
the artifact store rather than Git.

Local artifact root:
`/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e10_20261005`.
Remote volume: `vpml-wavenumber-generalization-v2-20260925`, under the same run name.

The run root contains `analysis/final_e10`, raw and complete epoch loss plots in
`analysis/update_00005920`, and `analysis/spectral_feedback_00005920.json`.
The parent analysis and earlier comparison figures remain preserved.

This is one initialization and one split. Repeatedly inspected heldout cases
are development evidence. The results do not isolate the effect of data breadth
or latent memory because the training formulation and precision were repaired
before this lineage. No training beyond E10 was performed or authorized.
