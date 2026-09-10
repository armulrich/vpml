# Faithful September 6 lineage replay

This directory records the commands and immutable provenance for the
September 6 coupled-latent checkpoint replay performed on 2026-09-09 from
commit `bbdb8c609f308586fa20b5979655f1766cea22cb`.

The replay reproduced the random rank-32 shared stages, compressed refinement,
rank-32 to rank-60 lift, tail updates, and ordinary-expert updates. The final
checkpoint combines the reproduced 33-tensor base with 17 cyclic/router tensors
from the archived September 6 branch. Those 17 tensors were not retrained, so
this is provenance and trajectory-parity evidence rather than a random-start
reproduction of the complete specialist-learning procedure.

The large checkpoints, logs, and plots remain immutable under
`out_bench/reproductions/sep6_faithful_20260909`. `REPLAY_SUMMARY.json` records
their absolute paths, hashes, stage comparisons, and the twelve-case Metric 1
panel. `provenance.json` records source and data hashes.

The shell files in `commands/` are the exact saved stage invocations. They are
records of the replay, not a recommended general training pipeline.
