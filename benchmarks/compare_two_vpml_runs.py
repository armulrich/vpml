import numpy as np
import matplotlib.pyplot as plt


def load_run(path, key="E_abs_k0p5"):
    data = np.load(path)
    t = data["times"]
    E = data[key]
    return t, E


def main():
    # Edit these paths to point to the two vpml runs you want to compare.
    # Example 1: original paper-style run (k=0.5,1.5, L=4π, Nv=20)
    path1 = "/Users/juliannestratton/vpml/out_landau_trunc/linear_landau_truncation.npz"
    label1 = "vpml truncation (paper params)"

    # Example 2: SW-matched run (e.g. L=2π, Nx=50, Nv=10, k≈1)
    path2 = "/Users/juliannestratton/vpml/out_landau_trunc_swmatch/linear_landau_truncation.npz"
    label2 = "vpml truncation (SW-matched)"

    # Load both runs (k=0.5 channel by default)
    t1, E1 = load_run(path1, key="E_abs_k0p5")
    t2, E2 = load_run(path2, key="E_abs_k0p5")

    # Normalize by initial value to compare damping/growth only
    E1_norm = E1 / E1[0]
    E2_norm = E2 / E2[0]

    plt.figure(figsize=(6, 4))
    plt.semilogy(t1, E1_norm, label=label1)
    plt.semilogy(t2, E2_norm, "--", label=label2)

    plt.xlabel("t")
    plt.ylabel(r"$|E_k(t)| / |E_k(0)|$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

