"""Construct one reproducible Gaussian initial distribution without evolution."""
import numpy as np

from vpml.datasets.initial_conditions import (
    DEFAULT_AMPLITUDES, draw_coefficients, fields_from_coefficients,
    gaussian_distribution,
)


def main():
    coefficients = draw_coefficients(seed=17, modes=3, decay=2.0)
    density, velocity, temperature = fields_from_coefficients(
        coefficients, nx=64, strength=0.3, amplitudes=DEFAULT_AMPLITUDES,
    )
    velocity_grid = np.linspace(-8, 8, 256)
    distribution = gaussian_distribution(density, velocity, temperature, velocity_grid)
    print(f"Distribution shape (velocity, position): {distribution.shape}")
    print(f"Density range: {density.min():.6f} to {density.max():.6f}")
    print(f"Temperature range: {temperature.min():.6f} to {temperature.max():.6f}")


if __name__ == "__main__":
    main()
