"""Lift a coupled equilibrium-CNN checkpoint to a larger Hermite latent rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from vpml.kinetic_latent import init_state_conditioned_latent_operator


ARRAY_KEYS = {
    "basis", "linear_propagator", "resolved_center", "resolved_scale",
    "latent_center", "latent_scale", "latent_residual_scale",
    "closure_residual_scale", "closure_correction_bound", "correction_bounds", "k_arr",
}


def _copy_conditioned_inputs(target, source, old_rank: int, new_rank: int):
    target[:, :3] = source[:, :3]
    target[:, 3 : 3 + old_rank] = source[:, 3 : 3 + old_rank]
    target[:, 3 + new_rank : 3 + new_rank + old_rank] = source[:, 3 + old_rank :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--template-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    source_report = json.loads((args.source_run / "report.json").read_text())
    template_report = json.loads((args.template_run / "report.json").read_text())
    source_config = source_report["configuration"]
    template_config = template_report["configuration"]
    old_rank = int(source_config["latent_rank"])
    new_rank = int(template_config["latent_rank"])
    if new_rank <= old_rank:
        raise ValueError("template latent rank must exceed source rank")
    if source_config["latent_readout_mode"] != "equilibrium_cnn":
        raise ValueError("source checkpoint must use equilibrium_cnn")
    with np.load(args.source_run / "best_coupled_low_moment_latent.npz") as payload:
        source = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(args.template_run / "best_coupled_low_moment_latent.npz") as payload:
        arrays = {key: np.asarray(payload[key]) for key in ARRAY_KEYS if key in payload.files}

    params = {
        key: np.asarray(value).copy()
        for key, value in init_state_conditioned_latent_operator(
            jax.random.PRNGKey(int(source_config["seed"])),
            resolved_channels=3,
            latent_rank=new_rank,
            width=int(source_config["width"]),
            depth=int(source_config["depth"]),
            spectral_modes=int(source_config["operator_modes"]),
            operator_rank=int(source_config["operator_rank"]),
            kernel_size=int(source_config["kernel_size"]),
            output_projection_init_scale=0.0,
            conditioner_output_channels=new_rank,
            conditioner_output_init_scale=0.0,
            latent_delay_input=bool(source_config["latent_delay_input"]),
        ).items()
    }
    for key in params:
        params[key][...] = 0.0
    for key in source:
        if key.startswith("block_") or key in {"lift_bias"}:
            params[key][...] = source[key]
    _copy_conditioned_inputs(params["lift_kernel"], source["lift_kernel"], old_rank, new_rank)
    _copy_conditioned_inputs(
        params["operator_v_real"], source["operator_v_real"], old_rank, new_rank,
    )
    _copy_conditioned_inputs(
        params["operator_v_imag"], source["operator_v_imag"], old_rank, new_rank,
    )
    params["operator_u_real"][:, :, :old_rank] = source["operator_u_real"]
    params["operator_u_imag"][:, :, :old_rank] = source["operator_u_imag"]
    params["output_kernel"][:old_rank] = source["output_kernel"]
    params["output_bias"][:old_rank] = source["output_bias"]

    config = dict(source_config)
    config.update(
        latent_rank=new_rank,
        projected_cache=template_config["projected_cache"],
        latent_state_bound=32.0,
        latent_excursion_limit=32.0,
        init_checkpoint=None,
        resume_training_state=None,
        outdir=str(args.outdir),
    )
    args.outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.outdir / "best_coupled_low_moment_latent.npz", **params, **arrays
    )
    (args.outdir / "report.json").write_text(
        json.dumps({"configuration": config, "history": [], "lift": {
            "source_run": str(args.source_run), "template_run": str(args.template_run),
            "old_rank": old_rank, "new_rank": new_rank,
        }}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
