import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    """
    Compare two vpml linear Landau runs on the same plot.

    Expected files (generate with fh_benchmarks_2412_07073_jax.py):

      truncation:
        python benchmarks/fh_benchmarks_2412_07073_jax.py linear_landau \\
            --method truncation --outdir out_landau_trunc

      hyper:
        python benchmarks/fh_benchmarks_2412_07073_jax.py linear_landau \\
            --method hyper --outdir out_landau_hyper

    This script loads both NPZ files and plots |E_{k=0.5}(t)| and the
    normalized curves |E_k(t)| / |E_k(0)| for each run.
    """

    repo_root = Path(__file__).resolve().parents[1]

    # --- truncation run ---
    trunc_path = repo_root / "out_landau_trunc" / "linear_landau_truncation.npz"
    hyper_path = repo_root / "out_landau_hyper" / "linear_landau_hyper.npz"

    if not trunc_path.is_file():
        raise FileNotFoundError(f"Truncation NPZ not found at {trunc_path}")
    if not hyper_path.is_file():
        raise FileNotFoundError(f"Hyper NPZ not found at {hyper_path}")

    tr = np.load(trunc_path)
    hy = np.load(hyper_path)

    t_tr = np.asarray(tr["times"]).ravel()
    E_tr = np.asarray(tr["E_abs_k0p5"]).ravel()

    t_hy = np.asarray(hy["times"]).ravel()
    E_hy = np.asarray(hy["E_abs_k0p5"]).ravel()

    print("truncation: times.shape =", t_tr.shape, "E.shape =", E_tr.shape, "E[0] =", E_tr[0])
    print("hyper:      times.shape =", t_hy.shape, "E.shape =", E_hy.shape, "E[0] =", E_hy[0])
    if t_tr.shape == t_hy.shape and E_tr.shape == E_hy.shape:
        same = np.allclose(t_tr, t_hy) and np.allclose(E_tr, E_hy)
        print("Same data in both files?", same)
    else:
        print("Different lengths — cannot compare; likely different runs.")

    # Raw amplitudes (distinct styles so both visible)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t_tr, E_tr, "C0-", lw=2, label="truncation |E_k(t)|")
    ax.semilogy(t_hy, E_hy, "C1--", lw=2, label="hyper |E_k(t)|")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|E_k(t)|$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Normalized damping (distinct styles)
    E_tr_norm = E_tr / E_tr[0] if E_tr[0] != 0 else E_tr
    E_hy_norm = E_hy / E_hy[0] if E_hy[0] != 0 else E_hy

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t_tr, E_tr_norm, "C0-", lw=2, label="truncation (normalized)")
    ax.semilogy(t_hy, E_hy_norm, "C1--", lw=2, label="hyper (normalized)")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|E_k(t)| / |E_k(0)|$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

