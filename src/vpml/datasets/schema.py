"""Array names for the Gaussian reference output contract."""

OUTPUT_SCHEMA = "gaussian-moments"
LEGACY_OUTPUT_SCHEMA = "gaussian-fields-v1"
SUPPORTED_OUTPUT_SCHEMAS = (OUTPUT_SCHEMA, LEGACY_OUTPUT_SCHEMA)
FIELD_NAMES = ("n", "u", "p", "Q", "dQ_dx")
RAW_MOMENT_NAMES = ("M0", "M1", "M2")

DIAGNOSTIC_NAMES = (
    "mass", "total_energy", "electric_energy", "min_f",
    "negative_mass_fraction", "boundary_abs_fraction", "min_n", "charge_imbalance",
)
LINEAR_DIAGNOSTIC_NAMES = ("linear_electric_energy", "delta_e2")



def validate_completed_progress(
    progress: dict, numerics, linearized: bool, output_schema: str = OUTPUT_SCHEMA,
) -> None:
    """Tie generic storage completion to this solver's final time and schema."""
    import numpy as np

    if progress["step"] != numerics.nsteps:
        raise ValueError("Completed case does not reach the final solver step")
    if output_schema not in SUPPORTED_OUTPUT_SCHEMAS:
        raise ValueError("Unsupported output schema")
    expected = {
        "times": ((numerics.nsamples,), "float64"),
        "fields": ((numerics.nsamples, len(FIELD_NAMES), numerics.output_nx), numerics.storage_dtype),
        "raw_moments": ((numerics.nsamples, len(RAW_MOMENT_NAMES), numerics.output_nx), numerics.storage_dtype),
        "electric": ((numerics.nsamples, numerics.output_nx), numerics.storage_dtype),
    }
    history_names = ["fields", "raw_moments", "electric"]
    if output_schema == LEGACY_OUTPUT_SCHEMA:
        diagnostics = len(DIAGNOSTIC_NAMES) + (len(LINEAR_DIAGNOSTIC_NAMES) if linearized else 0)
        expected["diagnostic_times"] = ((numerics.nsteps + 1,), "float64")
        expected["diagnostics"] = ((numerics.nsteps + 1, diagnostics), "float64")
        history_names.append("diagnostics")
    elif linearized:
        expected["linear_electric"] = ((numerics.nsamples, numerics.output_nx), numerics.storage_dtype)
        history_names.append("linear_electric")
    actual = progress["outputs"]["arrays"]
    if actual.keys() != expected.keys() or any(
        tuple(actual[key]["shape"]) != shape or np.dtype(actual[key]["dtype"]) != np.dtype(dtype)
        for key, (shape, dtype) in expected.items()
    ):
        raise ValueError("Completed output schema disagrees with the numerical protocol")
    for name in history_names:
        if sum(block["observations"]["arrays"][name]["shape"][0]
               for block in progress["blocks"]) != numerics.nsteps + 1:
            raise ValueError("Completed history length disagrees with the numerical protocol")


def observation_schema(output_schema: str = OUTPUT_SCHEMA) -> dict:
    """JSON-compatible definitions included in dataset provenance."""
    if output_schema not in SUPPORTED_OUTPUT_SCHEMAS:
        raise ValueError("Unsupported output schema")
    return {
        "version": output_schema,
        "axes": ["time", "field", "position"],
        "fields": list(FIELD_NAMES),
        "raw_moments": list(RAW_MOMENT_NAMES),
        "raw_moment_definition": "M_j = integral v**j f dv; no equilibrium subtraction",
        "pressure_definition": "p = integral (v-u)**2 f dv",
        "heat_flux_definition": "Q = integral (v-u)**3 f dv",
        "spatial_order": "evaluate moments on kinetic grid, then Fourier restrict each field",
        "temporal_order": "linear interpolation of saved observables from integration-step times",
        "units": "nondimensional Vlasov-Poisson units",
    }
