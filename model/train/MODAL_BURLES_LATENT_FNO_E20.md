# Modal Burles-style latent FNO E20 qualification

## Scope

This experiment is the persistent-memory member of the matched low-moment
closure comparison. It starts every neural parameter from seed 1729, evolves
only the three physical moments in the fluid solver, and augments the causal
history FNO with six learned persistent spatial channels.

The label *Burles-style* refers to the causal 50-sample history, FNO width 64,
eight FNO blocks, 32 retained Fourier modes, predicted closure history,
online rollout loss, AdamW optimization, and complete anchor exposure described
by [Burles et al.](https://arxiv.org/html/2607.29364v1). It is not an exact
reproduction. This experiment uses the VPML broad multimode IC family, spectral
spatial operators, `Nx=128`, `dt=0.025`, ten-unit training rollouts, and the
additional persistent latent state.

No Hermite coefficients, projected kinetic basis, case identity, future state,
or teacher closure enter autonomous inference. The six latent channels are
learned neural memory rather than a numerical Hermite solver.

## Preserved command

The tracked launch command is:

```bash
model/train/run_modal_burles_latent_fno_E20.sh
```

It resolves the source-tree hash and invokes:

```bash
modal run --detach --timestamps \
  model/train/modal_burles_latent_fno.py::train_and_evaluate_e20 \
  --source-tree-sha256 <computed-source-tree-sha256>
```

The runner trains or exactly resumes the dedicated latent lineage and evaluates
autonomous `T=120` trajectories at E0, E1, E2, E5, E10, E15, and E20. Existing
checkpoints and evaluations are preserved.

## Fixed configuration

- Modal app: `vpml-burles-latent-fno`
- Modal volume: `vpml-low-moment-burles`
- Remote run: `/runs/history_latent6_random1729_full_anchor_resumable_E1000`
- Matched control: `/runs/history_fno_random1729_full_anchor_E1000`
- GPU: one A100 80 GB
- Random seed: 1729; no initialization checkpoint
- Training ICs: 48, balanced across linear, weak nonlinear, and strong nonlinear regimes
- Training exposure: one complete shuffled anchor sweep per epoch
- Training horizon: 400 solver steps at `dt=0.025`, or ten physical units
- History: 50 samples at stride 4, spanning five physical units
- Model: width 64, depth 8, 32 Fourier modes, six persistent latent channels
- Optimizer: AdamW, cosine schedule from `1e-3` to `1e-5`, weight decay `1e-4`
- Gradient clipping: disabled
- Batch size: 16; 526 accepted updates per epoch
- Full-trajectory evaluation: 12 held-out ICs through `T=120`
- Within-epoch checkpoint interval: 50 accepted updates
- Planned schedule prefix: E1000; qualification stopped at E20

## E20 result

All 21 checkpoints from E0 through E20 and every scheduled evaluation were
downloaded and hashed. E20 completed 20 full-anchor sweeps and 10,520 accepted
AdamW updates. All 12 held-out rollouts remained bounded through `T=120`.

| Quantity | History-only E20 | Latent-6 E20 | Relative improvement |
|---|---:|---:|---:|
| Mean `epsilon_E` | 0.508409 | 0.381937 | 24.9% |
| Mean `epsilon_grow` | 0.416223 | 0.183130 | 56.0% |
| Worst `epsilon_E` | 1.008817 | 0.754305 | 25.2% |
| Strong ic08 `epsilon_E` | 0.570192 | 0.389314 | 31.7% |
| Strong ic17 `epsilon_E` | 1.008817 | 0.754305 | 25.2% |

The persistent state is active rather than inert. Resetting it every accepted
step changes mean `epsilon_E` from 0.3820 to 0.4393 at `T=120`; disabling the
learned closure changes it to 6.3219. The E20 latent state remains bounded, is
case dependent, and places more than 99.9% of its spectral power in modes 0-32.

## Effective mean closure

The run intentionally uses `--allow-uniform-heating`. Burles et al. train the
network to predict an effective discretization-dependent closure and retain the
spatial mean needed to reproduce mean pressure evolution. In that sense, the
nonzero mean is part of the Burles-style formulation and is not evidence of an
implementation bug.

It does leave an open conservation question for VPML. Enforcing zero mean only
at inference worsens latent E20 mean `epsilon_E` from 0.3820 to 0.4214 and
strong ic17 from 0.7548 to 1.0592, so that ablation is out of distribution and
does not qualify as a corrected checkpoint. A conservative closure would need
its own controlled training run.

## Limitations and decision

The latent model materially improves the matched history-only control, but it
does not recover the September 5-6 trajectory frontier. Linear trajectories
develop late damping floors or reversal, strong ic12 retains artificial energy,
and strong ic17 remains inaccurate in phase and envelope. The long-horizon
audit also finds that E20 improves integrated `T=120` field error over E15 while
worsening the ten-unit error and the `T=120` log-energy-envelope error.

The evidence therefore supports persistent unresolved memory as useful, but it
does not support continuing this ten-unit-objective checkpoint to E1000. The
next controlled experiment must expose the same random-start latent model to a
long-horizon autonomous objective while preserving this E20 lineage unchanged.

## Artifact identities

- Reference manifest SHA-256: `503a47ee5be541a5b16ee32f018eab6ee6a44ee123f98fab3935599add017e52`
- Training source-tree SHA-256: `5118eb0a0e5c65c4e5fe04c6cdffae380a0d15c70a25095ea0f3184b5bffb7e0`
- E20 checkpoint SHA-256: `ac395a6c0a497667b453d66d39cc3771941dc00d538de0e73de44bd70e060396`
- E20 evaluation summary SHA-256: `a21ac46290ad49a59ca749f3fe88c83c957991557eef7ff7133f51e812bbd00d`
- E20 Metric 1 panel SHA-256: `0a5a7d2c3030de5e4c81db29d6eca34a0e4a2cd3be29c442640ad00adb8c5d17`
- Final training state SHA-256: `4eaed7fa0d9260199636d4ac63bfe8aa8b758e103f606dcf471c0f1777cad956`
- Long-horizon audit summary SHA-256: `e49a8723c381673f6da7245cb6517f153af944965854391cb877f1519b62c5bd`
- Local run root: `out_bench/modal_downloads/history_latent6_random1729_full_anchor_resumable_E1000`
- Local audit root: `out_bench/diagnostics/latent6_E15_E20_long_horizon_audit_20260914`

`out_bench` remains ignored by Git. The files above are preserved locally and
on the Modal volume and are referenced here by immutable hashes rather than
stored as Git objects.
