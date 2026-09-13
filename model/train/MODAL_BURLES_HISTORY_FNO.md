# Modal history-only FNO qualification

This runner preserves the local E1 and all prior `out_bench` artifacts. It starts a new random-seed-1729 history-only lineage on one A100 80 GB GPU and uses the same complete-anchor Burles-style exposure as the corrected local run.

## One-time staging

The source kinetic cache is 217 GB. The trainer reads only C0:C3, and the Nx=128 evaluator needs only the first 65 electric-field Fourier modes. The staging command creates an approximately 15 GB lossless-for-this-experiment cache under ignored `out_bench`, records parent and output hashes, creates a dedicated Modal Volume v2 if absent, and uploads the compact cache only when a matching remote manifest is absent.

```bash
model/train/stage_modal_burles_history_fno.sh
```

The existing Modal CLI profile is already authenticated. No API key belongs in the repository or shell script.

## Read-only remote preflight

This checks the 60-case manifest and confirms that JAX sees exactly one CUDA GPU. It does not train.

```bash
modal run model/train/modal_burles_history_fno.py::preflight
```

## E20 launch

Run only after explicit approval:

```bash
model/train/run_modal_burles_history_fno_E20.sh
```

The detached job trains or exactly resumes to E20 with a schedule planned through E1000, then evaluates autonomous T=120 held-out trajectories at E0, E1, E2, E5, E10, E15, and E20. Checkpoints, optimizer/RNG state, logs, reports, and Metric 1 plots persist on the Volume. Rerunning the command resumes the same lineage and preserves completed evaluations.

Evaluation is remote because it reuses the uploaded reference cache and GPU checkpoint directly. Only the resulting checkpoints, reports, and plots are downloaded:

```bash
model/train/download_modal_burles_history_fno.sh
```

## Fixed resources

- GPU: one A100 80 GB
- CPU: 16 physical cores
- Memory: 64 GiB
- Function limit: 24 hours
- Persistent volume: `vpml-low-moment-burles`
- Remote run: `/runs/history_fno_random1729_full_anchor_E1000`
- Trainer state: `/runs/history_fno_random1729_full_anchor_E1000/training`

At current public rates, the requested A100 costs about $2.50 per GPU-hour. CPU and 64 GiB memory add approximately $1.27 per hour if fully billed at their requested values, for a conservative total near $3.77 per running hour. The approximately 15 GB compact cache is within Modal's included Volume allowance on current plans. Actual wall time must be measured from the first remote epoch; no speedup claim is assumed in advance.
