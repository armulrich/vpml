import numpy as np
import matplotlib.pyplot as plt

# --- vpml data (truncation, SW-matched parameters) ---
# Generate with:
#   python fh_benchmarks_2412_07073_jax.py linear_landau --method truncation --outdir out_landau_trunc
vp = np.load("/Users/juliannestratton/vpml/out_landau_trunc/linear_landau_truncation.npz")
t_vp = vp["times"]              # (Nt_vp,)
E_vp = vp["E_abs_k0p5"]         # (Nt_vp,)

# --- SW data: |E_k(t)| for the k≈1 mode (truncation closure, Nx=50, Nv=10, L=2π) ---
sw = np.load(
    "/Users/juliannestratton/SW-Conservative-Closure/data/SW/langmuir/"
    "E_k_10_truncation_closure.npz"
)
t_sw = sw["times"]              # (Nt_sw,)
E_sw = sw["E_abs_k0p5"]         # (Nt_sw,)

# Normalize by initial value to compare damping rate only (both curves start at 1)
E_vp_norm = E_vp / E_vp[0]
E_sw_norm = E_sw / E_sw[0] if E_sw[0] != 0 else E_sw

plt.figure(figsize=(6, 4))
plt.semilogy(t_vp, E_vp_norm, label="vpml (truncation, k≈1)")
plt.semilogy(t_sw, E_sw_norm, "o--", label="SW (truncation, k≈1)")

plt.xlabel("t")
plt.ylabel(r"$|E_k(t)| / |E_k(0)|$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()