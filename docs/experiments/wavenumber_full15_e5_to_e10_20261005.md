# Exact full-window latent continuation from E5 to E10

The user authorized five additional epochs after reviewing the saved E5 energy
panels and losses. This continues the same learned model and optimizer. It does
not restart training or change the objective.

Status: completed on 2026-10-06 at 5920 accepted updates. This file records
the continuation protocol and execution history. Final metrics and limitations
are in `wavenumber_full15_e10_results_20261006.md`.

## Preserved training state

- Parent: `wavenumber_full_window_lr1e4_e5_20261004`, complete at 2960 accepted
  updates and 142020 windows.
- Parent checkpoint SHA256:
  `750d0e616ce5cb481386b16b712be8dfb4d0ca1af0409da4139368d3955c0eea`.
- Child: `wavenumber_full_window_lr1e4_e10_20261005` in the existing
  `vpml-wavenumber-generalization-v2-20260925` volume.
- Local child root:
  `/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_full_window_lr1e4_e10_20261005`.
- Scientific source remains
  `39f23173cdce9ac1a895985c96b432fab6e11d0258bf896ee22925da17773cc6`.
- Continuation controller SHA256:
  `83be63774412a703a816ce2af338ebbe057eaa28acadcf518f8a691ce9e701e3`.
- Child protocol SHA256:
  `4acbccc3029a8af925761e3dd1505ad22f44e7f05ce45d1035d1912bad6f101a`.

Only the execution endpoint and artifact namespace change. The controller checks
identical configuration, source, evaluator, references, normalization, initial
seed, optimizer schedule, exposure and validation definitions. It preserves the
parent's evaluation files and restores the model, Adam moments, optimizer step,
epoch/batch cursor and full loss prefix. The checkpoint provenance hash changes
to identify the child protocol. Every other saved array is compared byte for byte.

The next batch is epoch 6, batch 1. Batch order is deterministically derived from
the unchanged seed and epoch. The learning rate at accepted update 2960 is
0.00009999444844291987, continuing the original 1000-epoch schedule. There is no
new warm-up, loss-descent acceptance rule, clipping or update cap.

Both the local temporary restore and the remote restore passed exact array and
optimizer parity. Five controller tests passed, including rejection of changed
scientific configuration, skipped evaluation gates and incorrect resume cursors.
The first continued GPU update, 2961, was repeated after loading the saved state.
`continuation_first_update_check.json` records a successful bitwise replay before
acceptance. `CONTINUATION_PARITY.json` records 2960 preserved loss rows and the
restored zero based cursor at epoch 6, batch 0.

## Execution and qualification

From the child artifact root, the historical launch command was:

```bash
VPML_RUN_NAME=wavenumber_full_window_lr1e4_e10_20261005 \
  caffeinate -i /Users/armin/miniforge3/bin/modal run --detach \
  setup/modal_precision_e5.py::run --stop-epoch 10
```

The initial app was `ap-KE4oXWO2LczRbLObOL5RTu`. `PLAN.json`, the launch and
resume logs, `budget_status.json` and checkpoint sidecars retain execution
evidence. The monitoring heartbeat was deleted after completion. This archived
command is not authorization for another run.

E6 through E10 end at updates 3552, 4144, 4736, 5328 and 5920. Each epoch evaluates
all 21 autonomous trajectories through T120 and the same full15 validation
objective. Checkpoints are also saved at most 50 updates apart and mirrored with
hash checks. Nonfinite training or autonomous evaluation stops the run. An
infrastructure interruption can resume only the exact saved state after checking
that the previous remote call is no longer active.

The combined spending cap remains $350, with a reported-cost threshold of $300
to reserve for billing lag. Before launch, reported combined spending was
$111.76930978. The prior median warm update took 8.81 seconds, implying about
7.2 training hours for the additional 2960 updates, excluding evaluation,
compilation and transfer.

The comparison tracks per-case late damping and nonlinear recovery alongside
loss. At launch, E5's field error was 0.2983955670 and log-energy error was
1.3580930255. Update 300 then held the lower mean log-energy score of
1.3235466291; E8 subsequently improved it to 1.2417476698. No training beyond
E10 is authorized.

## Infrastructure recovery and completion

Network failures, cancellations and local sleep interruptions required exact
checkpoint resumes. Detached Modal execution protected an already dispatched
call, not the entire local controller. Each manual recovery checked that the
previous app had stopped before resuming the same frozen controller. Recovery
records and logs remain under `infrastructure/` and the run root.

Logged but unsaved updates were replayed from durable checkpoints. The replay
comparisons for 3201–3230, 4551–4564, 4601–4620 and 4751–4795 matched the logged
loss, gradient norm and update norm exactly. Replays are not additional accepted
updates. The last interval followed automatic worker preemption within the same
app. An incomplete E7 evaluation was preserved locally and remotely at
`infrastructure/partial_e7_20261005T234448` before its replacement completed.

The final app, `ap-beI1lVairAfRTldwsK7JUA`, stopped with zero tasks on
2026-10-06 at 09:22:15 UTC. `COMPLETE.json` records epoch 10 and 5920 accepted
updates. The continuation added exactly 2960 accepted updates. A saved next
epoch cursor of 11 does not mean that epoch 11 was trained.

The original E5 outputs and datasets remain unchanged. The continuation's setup
directory freezes the exact controller, evaluator, scientific source and protocol
used at launch. The scientific Python files match committed repository source;
the exact continuation controller is preserved by commit `6b64abf`.
