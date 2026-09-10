# Controlled random rank-60 base and learned specialization

Status is recorded in `out_bench/controlled_rank60_plan_20260910.json`.

1. Preserve the faithful September 6 replay on `repro/sep6-faithful`.
2. Train one random shared rank-60 base through update 20 on the fixed broad48
   family and fixed full-trajectory objective.
3. At update 20 classify the base as still improving, recovered-and-conflicted,
   or plateaued-before-recovery. Continue shared training only for the first
   outcome; introduce experts only for the second; stop for the third.
4. Convert the qualified base to four small zero-mean random residual experts
   and a randomly initialized causal router. Jointly train shared, expert, and
   router tensors through post-training update 10.
5. Evaluate official Metric 1, integrated field error, full envelopes,
   significant transitions, router activation, and full T=120 trajectories.
6. Stop before E200. Produce the exact resume command only if E10 passes the
   frozen September-frontier gate.

The historical replay, current shared-network pilot, and all existing output
artifacts remain immutable comparisons. No manually selected expert identities,
case labels, rebound counts, sign edits, or checkpoint composition are allowed.
