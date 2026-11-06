"""Array names for the Gaussian reference output contract."""

OUTPUT_SCHEMA = "gaussian-fields-v1"
FIELD_NAMES = ("n", "u", "p", "Q", "dQ_dx")
RAW_MOMENT_NAMES = ("M0", "M1", "M2")

DIAGNOSTIC_NAMES = (
    "mass", "total_energy", "electric_energy", "min_f",
    "negative_mass_fraction", "boundary_abs_fraction", "min_n", "charge_imbalance",
)
LINEAR_DIAGNOSTIC_NAMES = ("linear_electric_energy", "delta_e2")



def validate_completed_progress(progress: dict, numerics, linearized: bool) -> None:
    """Tie generic storage completion to this solver's final time and schema."""
    import numpy as np

    if progress["step"] != numerics.nsteps:
        raise ValueError("Completed case does not reach the final solver step")
    diagnostics = len(DIAGNOSTIC_NAMES) + (len(LINEAR_DIAGNOSTIC_NAMES) if linearized else 0)
    expected = {
        "times": ((numerics.nsamples,), "float64"),
        "diagnostic_times": ((numerics.nsteps + 1,), "float64"),
        "diagnostics": ((numerics.nsteps + 1, diagnostics), "float64"),
        "fields": ((numerics.nsamples, len(FIELD_NAMES), numerics.output_nx), numerics.storage_dtype),
        "raw_moments": ((numerics.nsamples, len(RAW_MOMENT_NAMES), numerics.output_nx), numerics.storage_dtype),
        "electric": ((numerics.nsamples, numerics.output_nx), numerics.storage_dtype),
    }
    actual = progress["outputs"]["arrays"]
    if actual.keys() != expected.keys() or any(
        tuple(actual[key]["shape"]) != shape or np.dtype(actual[key]["dtype"]) != np.dtype(dtype)
        for key, (shape, dtype) in expected.items()
    ):
        raise ValueError("Completed output schema disagrees with the numerical protocol")
    for name in ("fields", "raw_moments", "electric", "diagnostics"):
        if sum(block["observations"]["arrays"][name]["shape"][0]
               for block in progress["blocks"]) != numerics.nsteps + 1:
            raise ValueError("Completed history length disagrees with the numerical protocol")


