# Broader wavelength experiment: progress and frozen protocol

## Progress

- [x] Back up dirty legacy source without changing it; hash 107 preserved checkpoints.
- [x] Commit and push preserved Hermite E100 diagnostics and archive tag.
- [x] Integrate maintained history FNO, latent FNO and Hermite paths into main; test and push.
- [x] Freeze domain-family manifest and train/development separation.
- [x] Implement per-example geometry, differentiable preparation, complete anchor exposure and exact resume.
- [x] Pass bounded gradient, geometry, finite-update, exact-resume and artifact-protection checks.
- [ ] Full-duration reference convergence and production evaluation qualification (pending budget gate).
- [x] Run bounded A100 pilot (maximum $25); measured total projection exceeds $750 cap.
- [ ] BLOCKED by cost gate: generate new references if qualification and cost permit.
- [ ] BLOCKED by cost gate: complete production orchestration, train through at most E20, and download/verify checkpoints and figures.

## Preservation

The new artifact namespace is `wavenumber_generalization_20260924_v1`, under the original checkout's ignored `out_bench`. Remote writes use a separate volume and the same namespace. Existing run directories are never reused as output. Historical code is tagged `archive/research-before-wavenumber-expansion-20260924` at dfaa18d. Legacy changes are preserved in a binary patch and source-only tar archive. Reference inputs may be read but never changed.

## Dataset

Use the regular fundamental grid 0.25, 0.27, ..., 0.49. Hold out every third point: 0.29, 0.35, 0.41, 0.47. Each domain has L=2*pi/k0 with harmonics j*k0, j=1..4. Each of nine training domains adds 12 isolated cases (four harmonics at amplitudes 0.001, 0.03, 0.3) and six mixtures (two per regime, with two, three or four modes). Retain the original 48 training cases: 210 total. Development contains original 12 plus 27 fresh mixtures on training domains and 24 cases on held-out domains: 63 total. A held-out fundamental does not imply all its harmonics are absent from training. In particular report physical-wavenumber overlap explicitly.

## Training

Reuse the unchanged six-channel latent FNO, width 64, eight blocks, 32 modes, seed 1729, solver Nx128 and dt0.025, history 50 samples every0.1, five-unit preparation and ten-unit scoring. Train every valid anchor exactly once per epoch in balanced minibatches of 48 (last partial batch retained). There are 110460 windows and 2302 updates per epoch. Use the existing global relative resolved-trajectory objective and AdamW with the same 1000-epoch cosine schedule, no gradient clipping, update cap or backtracking. All neural weights start randomly. No curriculum.

The new five-unit preparation is differentiable with rematerialization. This repairs the disconnected initializer gradient and is a second explicit intervention in addition to broader coverage. A fixed-batch forward-parity/gradient control compares the connected and detached versions. This experiment cannot attribute every gain exclusively to data coverage.

## Frozen observation and stop rules

Evaluate autonomous T=120 at E0, E1, E2, E5, E10, E15, E20. Use a fixed small panel at quarter and half of E1 and intervening epochs. Record raw window loss, fixed window validation, field error, log-energy error, early growth error, per-case trajectories, gradients, updates and exposure. Compare against preserved checkpoints as well as E0. The checkpoint minimizing field error need not minimize energy error.

Stop immediately on nonfinite parameters, optimizer or trajectories, inconsistent geometry/data/resume hashes, or projected cost above $750. Numerical rollout failure is recorded without silently skipping cases. Do not use a raw minibatch-loss spike alone as a stopping reason. At E5 and later, persistent deterioration means both field and energy mean error worsen by over 20% from the best prior evaluated checkpoint at two successive full evaluations; stop and classify from telemetry.

Early success may stop training only after two successive full evaluations meet all of: new-domain mean field and log-energy errors improve at least 20% over frozen E40 evaluated on the same new panel; at least 75% of new cases improve in field error; each regime's original-panel mean field and energy error stays within 10% of frozen E40. Include k0=0.35 separately. These are operational qualification criteria, not claims of general success or untouched final evidence.

The pilot must measure gradient throughput, peak memory, reference throughput and evaluation cost before reference generation or E20 is launched. The $750 cap includes pilot, references, training and evaluations. Do not top up account billing or automatically increase any workspace budget. If access/budget blocks allocation, preserve the prepared source and report the exact blocker.

## Completed pilot and decision

The maintained closure integration passed 90 tests and was pushed to main at `9a80a77`. The new branch is `exp/broader-wavenumber-latent`. Launched source commit `ad3aab0`, model/vpml SHA256 `d64da671c08316f7bc43ace5dcc96e1c052614e8e6265f2072969540e9ee1e94`. The pilot used Modal app `ap-abFcM9uc1Urzc7wSaBlkWl` and a new volume `vpml-wavenumber-generalization-20260924`. It completed and stopped with zero tasks on 24 September 2026.

| Observation | Measured result |
|---|---:|
| Fixed-batch loss before update | 60.257500 |
| Same-batch loss after update | 47.731987 |
| Initializer-specific gradient norm | 0.0524022 |
| Uncapped parameter update norm | 1.624089 |
| Connected vs detached forward loss relative difference | 0.000000886 |
| Stop/resume next update | Bitwise identical parameters and Adam moments |
| Median update time, batch 48, 600 differentiated solver steps | 6.0130 s |
| Peak JAX live allocation | 4.58 GiB |
| Kinetic reference step, Nx1024/Nv8192, dt0.01 | 0.19365 s |

The loss control uses 48 real windows from three preserved original training cases, at multiple times. It tests wiring and throughput, not expanded-family learning or generalization. One update is saved. Repeated fixed-state benchmarks and a second-update resume comparison are diagnostic computations, not an epoch. GPU forward values agree to numerical roundoff; the local small test is tighter. No update clipping or backtracking was introduced.

At 2302 updates/epoch, the training projection is 3.845 hours/epoch, or 76.90 hours through E20. Generating 213 new kinetic reference trajectories at the unchanged resolution projects to 137.49 hours. These are extrapolations from short benchmarks, not completed work. At the verified Modal resource rate of $3.764448/hour, training plus references and pilot project to $807.21 before evaluation. A 20 percent reserve plus $50 gives $1018.66. This exceeds the approved $750 limit even before the reserve. Reference generation and E20 were therefore not launched.

The decision is computational, not evidence against the expanded dataset. The next decision concerns reducing reference-generation cost while validating unchanged physical accuracy, or increasing the budget. Neither reference resolution nor the scientific training specification was silently reduced. Full-duration convergence, reference-output parity, production orchestration and E20 qualification remain pending. The available segmented trainer and evaluator are components, not a validated ready-to-launch E20 pipeline.

All 93 shared/adaptation tests pass; a separate chunked/uninterrupted evaluation test also passes (94 total). The common `vpml.low_moment` architecture source is unchanged. The pilot has 5 individually downloaded files with matching hashes in `pilot_verified/`. A first recursive CLI transfer produced an unusable combined local stream; it is retained separately and is not used as evidence. All 107 previously hashed local checkpoint files are unchanged. Key remote E20/E30/E40 files are compared separately to their preserved local copies.

Artifact root: `/Users/armin/Documents/NYU/vpml/out_bench/wavenumber_generalization_20260924_v1`. Results and data remain ignored by Git. Price source checked on 24 September: https://modal.com/pricing.
