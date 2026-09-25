# Broader-wavenumber latent FNO: controlled E5 run

The study trains one random-start three-moment plus six-channel memory FNO.
Its architecture, history length, numerical fluid solver, optimizer, and
global relative ten-unit trajectory objective follow the preserved E40 latent
model. The five-unit autonomous preparation is differentiated in this run,
whereas the original E40 trainer detached it. This is a controlled correction
to initializer gradient flow, not evidence that the change will help.

The v2 manifest has 54 training cases: fundamentals 0.30, 0.40, and 0.50;
three amplitudes; and four isolated harmonics plus two mixtures for every
fundamental/amplitude pair. Its 21 development cases include isolated and
four-mode mixtures at held-out fundamentals 0.35 and 0.45, and a new mixture
on each training fundamental. All cases use periodic domain `2*pi/k0`, so
harmonics are exact Fourier modes. The original 48/12 panel is preserved
separately for later secondary evaluation; it is not part of this v2 training
set. The old 210/63 v1 pilot manifest and results remain untouched.

The kinetic reference uses Nx=1024, Nv=8192, dt=0.01, and T=120, with the
established cubic spline semi-Lagrangian solver. Only the velocity spline's
tridiagonal solve is batched across spatial columns. On an A100, 100-step
timing fell from 17.78 to 1.39 seconds; after 200 steps the largest
distribution difference was 1.51e-14 and the largest saved-moment difference
was 1.16e-10. The complete case's persisted target is the three low moments
on Nx=128 at dt=0.025. The reference worker commits a kinetic checkpoint and
moment block every 100 steps, retains the two most recent kinetic states, and
rejects a resume if its case/configuration hash differs. A completed case is
mirrored locally with SHA-256 verification.

Each epoch sweeps all 54*526 training anchors exactly once, with 592 balanced
updates. E5 is 2,960 accepted updates. The training runner saves full model,
Adam state, exposure cursor, and losses at most 50 updates apart; each returned
snapshot is committed remotely and mirrored locally. Evaluation is scheduled
at E0 through E5, including autonomous T=120 trajectories and a fixed
held-out window objective. The 1000-epoch learning-rate prefix is retained,
but this authorized run stops at E5. Lower window loss is not an acceptance
criterion for autonomous generalization.

Remote volume: `vpml-wavenumber-generalization-v2-20260925`.
Remote root: `wavenumber_generalization_v2_e5_20260925`.
Local root: `/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_generalization_v2_e5_20260925`.
The original low-moment volume is mounted only for the original manifest and
frozen normalization. Previous checkpoints and reference outputs are not
overwritten or removed.

The preflight source hash was `5cebeb44d1bf7656bb263ddac36ab05861ef84479fe891c47b29a4fc8b87f4c0`.
The production reference source hash is
`ad8cabb58635341993b24385b550f950f6d7398c69c7193b50ea820c4b19e88e`.
The preflight-to-production source difference is in the local orchestration
entrypoint, not the spline or kinetic equations. The manifest hash from the
actual original input is `8ff10ac4ecbd726fc7e176ab9972ede8455b92de52ab86068cf088c4f89be71c`.
The frozen normalization SHA-256 is
`3b9483428177a4c4671c9197a5d4c7225e2a52b27a1a92b90d6bf195ddfdf34e`.

The current reference source hash in the launched image is
`ad8cabb58635341993b24385b550f950f6d7398c69c7193b50ea820c4b19e88e`.
If the run stops for budget or infrastructure reasons, invoke the same
`run_references` entrypoint with that hash and local root to resume uncompleted
cases. It validates completed local cases against the saved SHA and resumes
remote partial cases from their latest committed kinetic state. The
`run_training` entrypoint likewise resumes its latest full optimizer snapshot.
Any source change requires a new recorded source hash and a provenance review.
