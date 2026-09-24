# Maintained closure models

The canonical Hermite trainer is `model.train.interface_flux_rollout`. The history FNO and six-channel persistent-memory FNO share `vpml.low_moment` and `model.train.low_moment_closure`; use `--latent-memory-dim 0` or `6` with the Burles backend. The E20/E30/E40 launchers preserve exact historical commands.

Source and diagnostics before consolidation are preserved at tag `archive/research-before-wavenumber-expansion-20260924` (commit dfaa18d). Failed continuous training and projected-rank experiments remain on their existing branches. They are not alternative maintained trainers. No historical results were deleted. `out_bench` remains ignored.

E40 is a completed continuation, not the best checkpoint for every observable. E33 minimized saved mean field error, E29 minimized saved complete-interval energy discrepancy, and E10 minimized the early growth-rate metric. History E20 field error was 0.5084; latent E20 was 0.3819 and E30 was 0.3667. These figures concern the original mixture family. All parameters of the upcoming wider-family experiment start from random initialization.

The legacy dirty checkout is unchanged. Its binary patch, untracked source archive, and 107 historical checkpoint SHA256 records are preserved in `out_bench/wavenumber_generalization_20260924_v1/preservation` in the original VPML checkout.
