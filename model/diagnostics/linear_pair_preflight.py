"""Bounded structural audit; no optimizer steps or changes to training defaults.

Run with ``python -m model.diagnostics.linear_pair_preflight --backend cpu``.
Use --backend gpu to require a GPU, --out for a NEW output file, and
--archive-dir to attach the preserved local_closure_audit_20260920 evidence.
A passing audit explicitly reproduces the known detached initializer limitation;
it is not a claim that the paired E20 experiment is free of numerical bugs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from vpml.jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np

from model.train.low_moment_closure import (
    _rollout_window_memory,
    _unpack_window_memory,
    make_loss_function,
)
from vpml.low_moment import (
    DYNAMIC_INPUT_SCALING,
    init_burles_latent_fno_params,
    primitive_fields,
)


INIT_PREFIX = "compact_latent_init_"
LIMITATION = (
    "Detached burn-in gives the three initializer-head tensors zero trajectory-loss "
    "gradient. Shared features can still train and weight decay can change the head. "
    "A successful history0/latent6 pair does not establish absence of bugs, "
    "long-horizon stability, or kinetic accuracy."
)


def _norm(tree):
    return float(np.sqrt(sum(
        np.sum(np.asarray(v, dtype=np.float64) ** 2)
        for v in jax.tree_util.tree_leaves(tree)
    )))


def _finite(tree):
    return all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(tree))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    """Keep a failed nonfinite probe machine-readable (checks still fail)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def archive_evidence(directory):
    """Read prior evidence, without rerunning or altering the archived probes."""
    directory = Path(directory).resolve()
    summary = json.loads((directory / "gradient_summary.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    rows = [{k: row[k] for k in (
        "case", "anchor", "training_loss", "finite_gradient", "parameter_group_gradient_l2"
    )} for row in summary["rows"]]
    expected = {(case, anchor) for case in (
        "linear_landau_ic00", "nonlinear_landau_weak_ic00", "nonlinear_landau_strong_ic00"
    ) for anchor in (0, 100)}
    verified = (
        len(rows) == 6
        and {(r["case"], r["anchor"]) for r in rows} == expected
        and all(r["finite_gradient"] and np.isfinite(r["training_loss"])
                and r["parameter_group_gradient_l2"]["initializer"] == 0 for r in rows)
    )
    return {
        "evidence_kind": "existing report, not a new E20 checkpoint replay",
        "directory": str(directory),
        "file_sha256": {name: _sha(directory / name) for name in (
            "gradient_summary.json", "REPORT.md", "provenance.json"
        )},
        "recorded_provenance": provenance,
        "recorded_scope": summary["scope"],
        "e20_rows": rows,
        "e20_six_zero_initializer_probes": bool(verified),
        "separate_e40_optimizer_moments": summary["saved_E40_initializer_optimizer_moments_norms"],
        "checkpoint_hash_reverified": False,
    }


def _fixture():
    # One small linear-regime perturbation, not a kinetic reference or accuracy test.
    nx, horizon, burnin, memory_steps = 16, 4, 2, 2
    x = jnp.arange(nx, dtype=jnp.float32) * (2 * jnp.pi / nx)
    density = 1e-3 * (jnp.cos(x) + 0.2 * jnp.cos(2 * x + 0.3))
    initial = jnp.stack((density, jnp.zeros_like(density), density))[None]
    batch = {
        "initial": initial,
        "targets": jnp.repeat(initial[:, None], horizon, axis=1),
        "memory": jnp.zeros((1, memory_steps, 3, nx), dtype=jnp.float32),
        "heat_flux_gradient_history": jnp.zeros((1, memory_steps, nx), dtype=jnp.float32),
        "amplitude": jnp.asarray([1e-3], dtype=jnp.float32),
        "regime_index": jnp.asarray([0], dtype=jnp.int32),
    }
    common = dict(
        dt=0.025, memory_stride=4, input_scale=jnp.ones(4, dtype=jnp.float32),
        heat_flux_gradient_scale=0.1, amplitude_center=-3.0, amplitude_scale=1.0,
        poisson_sign=1.0, normalized_heat_flux_bound=128.0,
        density_floor=1e-4, pressure_floor=1e-4, scan_unroll=1,
        input_scaling=DYNAMIC_INPUT_SCALING, dynamic_amplitude_floor=1e-6,
        allow_uniform_heating=True, closure_history_input=True,
        memory_backend="burles_latent_fno",
    )
    k = jnp.asarray(2 * np.pi * np.fft.rfftfreq(nx, d=4 * np.pi / nx), dtype=jnp.float32)
    return batch, k, common, horizon, burnin, memory_steps


def _prediction(params, batch, k, common, horizon, burnin, *, initializer_only):
    # Counterfactual: only the three initializer tensors carry parameter tangents
    # during burn-in. All scored-window parameters are constants. Preserve state
    # tangents into the scored window, including history and carried closure.
    burn_params = ({name: value if name.startswith(INIT_PREFIX)
                    else jax.lax.stop_gradient(value) for name, value in params.items()}
                   if initializer_only else params)
    states, memory = _rollout_window_memory(
        burn_params, batch["initial"], batch["memory"], jnp.asarray(0, jnp.int32),
        batch["amplitude"], k, horizon=burnin, **common,
    )
    state = states[:, -1]
    if not initializer_only:
        state, memory = jax.tree_util.tree_map(jax.lax.stop_gradient, (state, memory))
    history, closure, encoded, counter, previous, latent = _unpack_window_memory(
        memory, common["memory_backend"]
    )
    scored_params = (jax.tree_util.tree_map(jax.lax.stop_gradient, params)
                     if initializer_only else params)
    return _rollout_window_memory(
        scored_params, state, history, counter, batch["amplitude"], k,
        horizon=horizon, compact_latent=latent, previous_heat_flux_gradient=previous,
        heat_flux_gradient_history=closure, encoded_history=encoded, **common,
    )[0]


def run_preflight(*, backend="auto", archive_dir=None):
    start = time.monotonic()
    devices = jax.devices() if backend == "auto" else jax.devices(backend)
    device = devices[0]  # Explicit GPU requests fail rather than silently use CPU.
    with jax.default_device(device):
        batch, k, common, horizon, burnin, memory_steps = _fixture()
        init_options = dict(width=4, spectral_modes=3, memory_steps=memory_steps,
                            depth=1, input_channels=5, dtype=jnp.float32)
        pair = {dim: init_burles_latent_fno_params(
            jax.random.PRNGKey(1729), latent_dim=dim, **init_options
        ) for dim in (0, 6)}
        shared = sorted(name for name in pair[0]
                        if not name.startswith("compact_latent_")
                        and name in pair[6] and pair[0][name].shape == pair[6][name].shape)
        mismatches = [name for name in shared
                      if not np.array_equal(np.asarray(pair[0][name]), np.asarray(pair[6][name]))]
        actual_loss = make_loss_function(
            k_arr=np.asarray(k), width=4, horizon=horizon,
            regime_scales=np.ones((3, 4), dtype=np.float32),
            global_relative_trajectory_loss=True, autonomous_burnin_steps=burnin, **common,
        )
        actual_grad = jax.jit(jax.value_and_grad(actual_loss, has_aux=True))
        rows = {}
        for dim, params in pair.items():
            (loss, aux), gradient = actual_grad(params, batch)
            rows[f"history{dim}" if dim == 0 else f"latent{dim}"] = {
                "loss": float(loss), "finite_loss_aux_gradient": bool(_finite((loss, aux, gradient))),
                "gradient_l2": _norm(gradient),
                "initializer_tensor_gradient_l2": {
                    name: _norm(value) for name, value in gradient.items() if name.startswith(INIT_PREFIX)
                },
            }

        def reconstructed(params, *, initializer_only):
            prediction = _prediction(params, batch, k, common, horizon, burnin,
                                     initializer_only=initializer_only)
            fields = primitive_fields(prediction.reshape(-1, 3, 16), k).reshape(1, horizon, 4, 16)
            target = primitive_fields(batch["targets"].reshape(-1, 3, 16), k).reshape(1, horizon, 4, 16)
            loss = jnp.mean(jnp.sum((fields - target) ** 2, axis=(1, 2, 3))
                            / jnp.sum(target ** 2, axis=(1, 2, 3)))
            return loss, prediction

        detached_loss, detached_prediction = jax.jit(
            lambda p: reconstructed(p, initializer_only=False)
        )(pair[6])
        (counter_loss, counter_prediction), counter_grad = jax.jit(jax.value_and_grad(
            lambda p: reconstructed(p, initializer_only=True), has_aux=True
        ))(pair[6])
        counter_init = {name: value for name, value in counter_grad.items() if name.startswith(INIT_PREFIX)}
        counter_other = {name: value for name, value in counter_grad.items() if not name.startswith(INIT_PREFIX)}
        checks = {
            "shared_nonlatent_initialization_identical": bool(shared and not mismatches),
            "paired_actual_loss_and_gradients_finite_nonzero": all(
                r["finite_loss_aux_gradient"] and r["gradient_l2"] > 0 for r in rows.values()),
            "actual_latent6_initializer_gradient_exactly_zero": all(
                v == 0 for v in rows["latent6"]["initializer_tensor_gradient_l2"].values()),
            "reconstruction_matches_actual_loss": bool(np.isclose(
                float(detached_loss), rows["latent6"]["loss"], rtol=2e-5, atol=1e-10)),
            "counterfactual_forward_trajectory_unchanged": bool(np.allclose(
                detached_prediction, counter_prediction, rtol=2e-6, atol=1e-9)),
            "counterfactual_loss_unchanged": bool(np.isclose(
                float(detached_loss), float(counter_loss), rtol=2e-5, atol=1e-10)),
            "forward_and_counterfactual_gradient_finite": bool(_finite(
                (detached_prediction, counter_prediction, counter_loss, counter_grad))),
            "counterfactual_initializer_gradient_positive": _norm(counter_init) > 0,
            "counterfactual_other_parameter_gradients_zero": _norm(counter_other) == 0,
        }
        evidence = archive_evidence(archive_dir) if archive_dir else None
        if evidence is not None:
            checks["archived_e20_initializer_limitation_confirmed"] = evidence["e20_six_zero_initializer_probes"]
        root = Path(__file__).resolve().parents[2]
        return {
            "schema_version": 1, "passed": all(checks.values()),
            "status": "checks_passed_with_known_initializer_limitation" if all(checks.values()) else "failed",
            "known_limitation": LIMITATION, "checks": checks,
            "scope": "Synthetic linear-regime fixture; static non-kinetic targets; no optimizer steps or training.",
            "configuration": {"seed": 1729, "nx": 16, "horizon": horizon, "burnin": burnin,
                              "width": 4, "depth": 1, "spectral_modes": 3, "memory_steps": memory_steps,
                              "memory_stride": 4, "dt": common["dt"], "dtype": "float32"},
            "runtime": {"backend": device.platform, "device": str(device), "jax_version": jax.__version__},
            "source_sha256": {name: _sha(root / name) for name in (
                "model/train/low_moment_closure.py", "vpml/low_moment.py"
            )},
            "shared_parameter_names": shared, "mismatched_parameter_names": mismatches,
            "paired_actual_loss": rows,
            "counterfactual": {
                "gradient_path": "burn-in initializer parameters only; scored parameter gradients stopped",
                "detached_reconstruction_loss": float(detached_loss), "loss": float(counter_loss),
                "trajectory_max_abs_difference": float(jnp.max(jnp.abs(detached_prediction - counter_prediction))),
                "initializer_gradient_l2": _norm(counter_init),
                "initializer_tensor_gradient_l2": {name: _norm(v) for name, v in counter_init.items()},
                "other_parameter_gradient_l2": _norm(counter_other),
            },
            "archive_evidence": evidence, "elapsed_seconds": time.monotonic() - start,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--out", "--json-out", dest="json_out", type=Path,
                        help="New output file; existing files are never overwritten")
    parser.add_argument("--archive-dir", type=Path, help="Optional preserved local closure audit directory")
    args = parser.parse_args(argv)
    if args.json_out and args.json_out.exists():
        parser.error("--out already exists; choose a new file")
    try:
        report = run_preflight(backend=args.backend, archive_dir=args.archive_dir)
    except Exception as exc:
        report = {"schema_version": 1, "passed": False, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}", "known_limitation": LIMITATION}
    payload = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    if args.json_out:
        with args.json_out.open("x") as output:
            output.write(payload)
    else:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
