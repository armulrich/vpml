"""Compatibility imports for :mod:`vpml.datasets.wavenumber_reference`.

New callers should import the maintained implementation directly.
"""

from vpml.datasets.wavenumber_reference import (
    atomic_write,
    PhysicalGridVlasovPoissonConfig,
    build_physical_grid_ops,
    semilagrangian_vlasov_poisson_step,
    normalize_density_on_grid,
    gaussian_pdf,
    evaluate_manifest_case,
    sha256_json,
    sha256_file,
    reference_kernel,
    benchmark_reference,
    _atomic_npy,
    _atomic_json,
    generate_reference_resumable,
)
