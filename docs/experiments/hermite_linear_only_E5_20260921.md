# Local linear-only Hermite E5 pilot

Completed one random-start five-epoch run on the local CPU. No Modal run or resources were used.

Artifacts: `/Users/armin/Documents/NYU/vpml/out_bench/hermite_linear_only_E5_20260921/`.
The artifact `REPORT.md` records configuration, controls, source hashes, all metrics and commands.

The canonical Hermite trainer and physical solver remain unchanged. The new runner reuses their initialization, loss, sampler and Adam functions, selects the 16 linear training ICs and recomputes training-only normalization. Four complete linear ICs remain held out. Seed 0, width 128, two residual blocks, six tail inputs, cutoff cycle 6/7/12/20/36/64, batch 64, H128, dt0.01, Nx256, 30 updates per epoch. Historical Adam at 1e-4 and gradient clip0.5 are retained. There is no update cap or backtracking. The original Nx1024/Nv8192 reference and quadrature4096 are reused without modification.

All 150 updates were finite and accepted. The 9600 sampled windows contain 8842 unique cutoff/anchor pairs. Optimizer-loop time was 586.3 seconds. E0 through E5 retain checkpoint, Adam state, RNG state and exposure counts.

Training epoch mean decreases from 41.9397 at E1 to 0.141724 at E5. Fixed heldout q-window loss decreases from 91.4136 at E0 to 0.318090 at E5. The zero-q validation control is 0.403942, so the final improvement against that control is only 21.25%. The loss has not plateaued by E5.

E5 is finite through T120 on all four heldouts, while the evaluated E0/E1/E2 checkpoints become nonfinite. E5 nevertheless develops artificial retention and growth. With historical evaluation dt0.005 and dealiasing, its mean field error on normalized physical modes1–4 is 16.5851. The preserved linear-only history E2 and latent E15 values on the same common grid are 0.481797 and 0.421542. These runs have different objectives, exposure and architecture, so this is an achieved-behavior comparison, not matched-compute evidence.

The dt0.01 unfiltered evaluation and a linearized-RHS control also show the late failure. Historical fitted mixed-regime Hermite E400 has a much lower integrated field error of 0.043875, but a log-energy RMSE of 3.6431 decades versus latent E15's 1.4453. No conclusion that the Hermite architecture is incapable follows from this short pilot.

At the Maxwellian IC, the true omitted Hermite interface is zero. E5 predicts nonzero initial interface values, with 95.8–98.3% of their power outside spatial modes1–4. Its homogeneous-equilibrium output is exactly zero. This establishes a closure prediction error, but not a unique causal explanation of late growth.

Fifteen existing Hermite tests and two pilot optimizer tests pass. The lightweight evaluator matches the production solver within 1.74e-18 on the checked short trajectories with closure on/off and both evaluation settings. Parseval checks pass. All 22 reference fingerprints are unchanged. Prior trained results and remote outputs were not overwritten. No continuation was launched.

## Preserved continuation through E100

The unchanged E5 run was continued locally to E100 in a separate directory,
`out_bench/hermite_linear_only_E100_continue_20260921`. E4-to-E5 exact replay
passed before continuing. All 3000 optimizer updates and checkpoint/RNG states
were retained. The original E0-E5 artifacts remain unchanged.

The best evaluated T120 energy checkpoint was E20 (mean log-energy RMSE
4.7080 decades). E60 minimized modes-1-4 field error (0.084109), with a worse
energy discrepancy of 6.3610 decades. E80 had only 1/4 finite autonomous cases,
and E100 had 0/4. These are separate metric rankings. Decreasing interface
validation loss did not establish stable full-trajectory improvement.

The saved `REPORT.md`, `continuation_summary.json`, and integrity records in
that directory contain the complete evidence. The continuation used no Modal
resources.
