"""Train a spline-grid online rollout residual beside the Fourier-Hermite path.

This is an independent experimental trainer. It does not produce or consume the
Fourier-Hermite learned-interface checkpoints used by the existing online
rollout sweeps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from vpml.jax_runtime import bootstrap_jax_runtime

bootstrap_jax_runtime()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model.train.train import (
    REGIME_LINEAR,
    REGIME_STRONG,
    REGIME_WEAK,
    parse_float_tuple,
    parse_str_tuple,
    sample_initial_condition,
)
from vpml.physical_grid import PhysicalGridVlasovPoissonConfig, run_semilagrangian_vlasov_poisson
from vpml.rollout.spline_fem import (
    init_spline_residual_params,
    maxwellian_on_grid,
    restrict_history_to_grid,
    spline_fem_rollout_loss,
)

try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


SPLINE_FEM_CACHE_FORMAT = "spline_fem_online_rollout_reference_v2"


def _cache_mismatch(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape:
        return True
    if actual.dtype.kind in {"U", "S", "O"} or expected.dtype.kind in {"U", "S", "O"}:
        return tuple(actual.astype(str).reshape(-1)) != tuple(expected.astype(str).reshape(-1))
    return not np.array_equal(actual, expected)


def build_metadata(args: argparse.Namespace, regimes: Sequence[str]) -> Dict[str, np.ndarray]:
    return {
        "dataset_format": np.array([SPLINE_FEM_CACHE_FORMAT], dtype=np.str_),
        "regimes": np.asarray(tuple(regimes), dtype=np.str_),
        "target_vgrid": np.array([int(args.target_vgrid)], dtype=np.int32),
        "low_Nx": np.array([int(args.low_Nx)], dtype=np.int32),
        "teacher_Nx": np.array([int(args.teacher_Nx)], dtype=np.int32),
        "teacher_Nv": np.array([int(args.teacher_Nv)], dtype=np.int32),
        "teacher_L": np.array([float(args.teacher_L)], dtype=np.float64),
        "teacher_vmin": np.array([float(args.teacher_vmin)], dtype=np.float64),
        "teacher_vmax": np.array([float(args.teacher_vmax)], dtype=np.float64),
        "teacher_dt": np.array([float(args.teacher_dt)], dtype=np.float64),
        "linear_T": np.array([float(args.linear_T)], dtype=np.float64),
        "linear_eps": np.array([float(args.linear_eps)], dtype=np.float64),
        "linear_modes": np.asarray(
            tuple(float(v) for v in parse_float_tuple(args.linear_modes)),
            dtype=np.float64,
        ),
        "linear_num_samples": np.array([int(args.linear_num_samples)], dtype=np.int32),
        "linear_seed": np.array([int(args.linear_seed)], dtype=np.int32),
        "nonlinear_T": np.array([float(args.nonlinear_T)], dtype=np.float64),
        "nonlinear_k0": np.array([float(args.nonlinear_k0)], dtype=np.float64),
        "weak_eps": np.asarray(
            tuple(float(v) for v in parse_float_tuple(args.weak_eps)),
            dtype=np.float64,
        ),
        "strong_eps": np.asarray(
            tuple(float(v) for v in parse_float_tuple(args.strong_eps)),
            dtype=np.float64,
        ),
        "val_fraction": np.array([float(args.val_fraction)], dtype=np.float64),
    }


def load_dataset_cache(path: Path, metadata: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            for key, expected in metadata.items():
                if key not in data.files or _cache_mismatch(np.asarray(data[key]), np.asarray(expected)):
                    raise ValueError(key)
            return {
                "train_histories": np.asarray(data["train_histories"], dtype=np.float64),
                "val_histories": np.asarray(data["val_histories"], dtype=np.float64),
                "train_regimes": np.asarray(data["train_regimes"], dtype=np.str_),
                "val_regimes": np.asarray(data["val_regimes"], dtype=np.str_),
            }
    except ValueError as exc:
        print(f"[spline-fem] ignoring incompatible dataset cache {path}: metadata mismatch {exc}")
        return None


def save_dataset_cache(path: Path, dataset: Dict[str, np.ndarray], metadata: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **metadata, **dataset)


def split_histories(
    histories: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    val_fraction: float,
) -> Dict[str, np.ndarray]:
    if not histories:
        raise ValueError("No histories were generated")
    if len(histories) == 1:
        train_histories = histories
        train_labels = labels
        val_histories = histories
        val_labels = labels
    else:
        n_val = max(1, int(round(len(histories) * float(val_fraction))))
        n_val = min(n_val, len(histories) - 1)
        train_histories = histories[:-n_val]
        train_labels = labels[:-n_val]
        val_histories = histories[-n_val:]
        val_labels = labels[-n_val:]
    return {
        "train_histories": np.stack(train_histories, axis=0).astype(np.float64),
        "val_histories": np.stack(val_histories, axis=0).astype(np.float64),
        "train_regimes": np.asarray(train_labels, dtype=np.str_),
        "val_regimes": np.asarray(val_labels, dtype=np.str_),
    }


def build_low_history(
    *,
    teacher_config: PhysicalGridVlasovPoissonConfig,
    low_config: PhysicalGridVlasovPoissonConfig,
    perturbation_x: np.ndarray,
) -> np.ndarray:
    equilibrium = maxwellian_on_grid(teacher_config.v)
    f0 = equilibrium[:, None] * (1.0 + jnp.asarray(perturbation_x, dtype=jnp.float64)[None, :])
    raw = run_semilagrangian_vlasov_poisson(
        teacher_config,
        f0,
        history_stride=1,
        return_state_history=True,
    )
    low_history = restrict_history_to_grid(
        jnp.asarray(raw["state_history"], dtype=jnp.float64),
        teacher_config,
        low_config,
    )
    return np.asarray(low_history, dtype=np.float64)


def build_dataset(args: argparse.Namespace, regimes: Sequence[str]) -> Dict[str, np.ndarray]:
    if not math.isclose(float(args.linear_T), float(args.nonlinear_T)):
        raise ValueError("spline FEM rollout dataset currently expects linear_T == nonlinear_T")

    teacher_config = PhysicalGridVlasovPoissonConfig(
        Nx=int(args.teacher_Nx),
        Nv=int(args.teacher_Nv),
        Lx=float(args.teacher_L),
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.teacher_dt),
        T=float(args.linear_T),
        poisson_sign=float(args.teacher_poisson_sign),
        snapshot_times=(),
    )
    low_config = PhysicalGridVlasovPoissonConfig(
        Nx=int(args.low_Nx),
        Nv=int(args.target_vgrid),
        Lx=float(args.teacher_L),
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.teacher_dt),
        T=float(args.linear_T),
        poisson_sign=float(args.teacher_poisson_sign),
        snapshot_times=(),
    )
    histories: List[np.ndarray] = []
    labels: List[str] = []

    if REGIME_LINEAR in regimes:
        rng = np.random.default_rng(int(args.linear_seed))
        x = np.asarray(teacher_config.x, dtype=np.float64)
        modes = parse_float_tuple(args.linear_modes)
        for _ in range(int(args.linear_num_samples)):
            perturb = sample_initial_condition(rng, x, modes, float(args.linear_eps))
            histories.append(
                build_low_history(
                    teacher_config=teacher_config,
                    low_config=low_config,
                    perturbation_x=np.asarray(perturb, dtype=np.float64),
                )
            )
            labels.append(REGIME_LINEAR)

    nonlinear_x = np.asarray(teacher_config.x, dtype=np.float64)
    nonlinear_template = np.cos(float(args.nonlinear_k0) * nonlinear_x)
    for regime_name, eps_values in (
        (REGIME_WEAK, parse_float_tuple(args.weak_eps)),
        (REGIME_STRONG, parse_float_tuple(args.strong_eps)),
    ):
        if regime_name not in regimes:
            continue
        for eps in eps_values:
            histories.append(
                build_low_history(
                    teacher_config=teacher_config,
                    low_config=low_config,
                    perturbation_x=float(eps) * nonlinear_template,
                )
            )
            labels.append(regime_name)

    return split_histories(histories, labels, val_fraction=float(args.val_fraction))


def make_low_config(args: argparse.Namespace) -> PhysicalGridVlasovPoissonConfig:
    return PhysicalGridVlasovPoissonConfig(
        Nx=int(args.low_Nx),
        Nv=int(args.target_vgrid),
        Lx=float(args.teacher_L),
        vmin=float(args.teacher_vmin),
        vmax=float(args.teacher_vmax),
        dt=float(args.teacher_dt),
        T=float(args.linear_T),
        poisson_sign=float(args.teacher_poisson_sign),
        snapshot_times=(),
    )


def loss_on_batch(
    params: Dict[str, object],
    batch: jnp.ndarray,
    config: PhysicalGridVlasovPoissonConfig,
    *,
    rollout_horizon: int,
    rollout_anchor_samples: int,
) -> jnp.ndarray:
    losses = jax.vmap(
        lambda hist: spline_fem_rollout_loss(
            params,
            hist,
            config,
            rollout_horizon=int(rollout_horizon),
            rollout_anchor_samples=int(rollout_anchor_samples),
        )
    )(batch)
    return jnp.mean(losses)


def adam_init(params: Dict[str, object]) -> Dict[str, object]:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"step": jnp.asarray(0, dtype=jnp.int32), "m": zeros, "v": zeros}


def adam_step(
    params: Dict[str, object],
    grads: Dict[str, object],
    state: Dict[str, object],
    lr: float,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
    grad_clip: float | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if grad_clip is not None and float(grad_clip) > 0.0:
        sq_norm = sum(jnp.sum(jnp.abs(g) ** 2) for g in jax.tree_util.tree_leaves(grads))
        norm = jnp.sqrt(jnp.maximum(sq_norm, jnp.asarray(1.0e-30, dtype=jnp.float64)))
        scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.float64), float(grad_clip) / norm)
        grads = jax.tree_util.tree_map(lambda g: scale * g, grads)
    step = state["step"] + jnp.asarray(1, dtype=jnp.int32)
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (jnp.abs(g_i) ** 2), state["v"], grads)
    step_f = step.astype(jnp.float64)
    m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1.0 - beta1**step_f), m)
    v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1.0 - beta2**step_f), v)
    new_params = jax.tree_util.tree_map(
        lambda p, m_i, v_i: p - float(lr) * m_i / (jnp.sqrt(v_i) + float(eps)),
        params,
        m_hat,
        v_hat,
    )
    return new_params, {"step": step, "m": m, "v": v}


def save_checkpoint(
    path: Path,
    params: Dict[str, object],
    *,
    args: argparse.Namespace,
    train_loss: Sequence[float],
    val_loss: float,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "target_vgrid": np.array([int(args.target_vgrid)], dtype=np.int32),
        "low_Nx": np.array([int(args.low_Nx)], dtype=np.int32),
        "hidden_width": np.array([int(args.hidden_width)], dtype=np.int32),
        "res_blocks": np.array([int(args.res_blocks)], dtype=np.int32),
        "rollout_horizon": np.array([int(args.rollout_horizon)], dtype=np.int32),
        "rollout_anchor_samples": np.array([int(args.rollout_anchor_samples)], dtype=np.int32),
        "teacher_dt": np.array([float(args.teacher_dt)], dtype=np.float64),
        "train_loss": np.asarray(train_loss, dtype=np.float64),
        "val_loss": np.array([float(val_loss)], dtype=np.float64),
        "W0": np.asarray(params["W0"]),
        "b0": np.asarray(params["b0"]),
        "Wout": np.asarray(params["Wout"]),
        "bout": np.asarray(params["bout"]),
    }
    for i, block in enumerate(params["blocks"]):
        payload[f"block{i}_W1"] = np.asarray(block["W1"])
        payload[f"block{i}_b1"] = np.asarray(block["b1"])
        payload[f"block{i}_W2"] = np.asarray(block["W2"])
        payload[f"block{i}_b2"] = np.asarray(block["b2"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def save_loss_plot(path: Path, train_loss: Sequence[float], val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    epochs = np.arange(1, len(train_loss) + 1)
    ax.semilogy(epochs, np.maximum(np.asarray(train_loss), 1.0e-30), label="train")
    ax.axhline(max(float(val_loss), 1.0e-30), color="#b45309", linestyle="--", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative spline-grid perturbation rollout loss")
    ax.set_title("Spline/FEM Online Rollout Residual")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def train(args: argparse.Namespace) -> Dict[str, object]:
    regimes = tuple(
        regime
        for regime in parse_str_tuple(args.regimes)
        if regime in {REGIME_LINEAR, REGIME_WEAK, REGIME_STRONG}
    )
    if not regimes:
        raise ValueError("At least one valid regime is required")
    metadata = build_metadata(args, regimes)
    dataset = None
    if args.dataset_cache is not None:
        dataset = load_dataset_cache(args.dataset_cache, metadata)
    if dataset is None:
        dataset = build_dataset(args, regimes)
        if args.dataset_cache is not None:
            save_dataset_cache(args.dataset_cache, dataset, metadata)
            print(f"[spline-fem] saved dataset cache to {args.dataset_cache}")
    if args.build_dataset_only:
        print("[spline-fem] dataset built; skipping training")
        return {"dataset": dataset, "train_loss": [], "val_loss": math.nan}

    train_histories = jnp.asarray(dataset["train_histories"], dtype=jnp.float64)
    val_histories = jnp.asarray(dataset["val_histories"], dtype=jnp.float64)
    config = make_low_config(args)
    params = init_spline_residual_params(
        jax.random.PRNGKey(int(args.seed)),
        hidden_width=int(args.hidden_width),
        res_blocks=int(args.res_blocks),
    )
    opt_state = adam_init(params)

    def batch_loss(params_i, batch_i):
        return loss_on_batch(
            params_i,
            batch_i,
            config,
            rollout_horizon=int(args.rollout_horizon),
            rollout_anchor_samples=int(args.rollout_anchor_samples),
        )

    @jax.jit
    def train_step(params_i, opt_state_i, batch_i):
        loss, grads = jax.value_and_grad(batch_loss)(params_i, batch_i)
        params_o, opt_state_o = adam_step(
            params_i,
            grads,
            opt_state_i,
            float(args.lr),
            grad_clip=float(args.grad_clip),
        )
        return params_o, opt_state_o, loss

    val_loss_fn = jax.jit(lambda params_i, batch_i: batch_loss(params_i, batch_i))

    rng = np.random.default_rng(int(args.seed))
    train_loss: List[float] = []
    n_train = int(train_histories.shape[0])
    batch_size = max(1, int(args.online_case_batch_size))
    steps_per_epoch = max(1, int(args.steps_per_epoch))
    for epoch in range(1, int(args.epochs) + 1):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            idx = rng.choice(n_train, size=batch_size, replace=n_train < batch_size)
            batch = train_histories[np.asarray(idx, dtype=np.int32)]
            params, opt_state, loss = train_step(params, opt_state, batch)
            epoch_losses.append(float(loss))
        update_loss = float(np.mean(epoch_losses))
        mean_loss = float(val_loss_fn(params, train_histories))
        train_loss.append(mean_loss)
        if int(args.log_every) > 0 and (
            epoch == 1
            or epoch % int(args.log_every) == 0
            or epoch == int(args.epochs)
        ):
            print(
                f"[spline-fem] epoch={epoch:04d} "
                f"train_loss={mean_loss:.6e} update_loss={update_loss:.6e}"
            )

    val_loss = float(val_loss_fn(params, val_histories))
    checkpoint = args.checkpoint or (args.outdir / "spline_fem_residual.npz")
    loss_plot = args.loss_plot or checkpoint.with_suffix(".loss.png")
    save_checkpoint(checkpoint, params, args=args, train_loss=train_loss, val_loss=val_loss)
    save_loss_plot(loss_plot, train_loss, val_loss)
    summary = {
        "checkpoint": str(checkpoint),
        "loss_plot": str(loss_plot),
        "target_vgrid": int(args.target_vgrid),
        "low_Nx": int(args.low_Nx),
        "train_cases": int(train_histories.shape[0]),
        "val_cases": int(val_histories.shape[0]),
        "train_loss_final": float(train_loss[-1]),
        "val_loss": float(val_loss),
        "rollout_horizon": int(args.rollout_horizon),
        "rollout_anchor_samples": int(args.rollout_anchor_samples),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[spline-fem] saved checkpoint to {checkpoint}")
    print(f"[spline-fem] saved loss plot to {loss_plot}")
    print(f"[spline-fem] val_loss={val_loss:.6e}")
    return {"dataset": dataset, "train_loss": train_loss, "val_loss": val_loss, "summary": summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an independent spline/FEM-style online rollout residual"
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--loss-plot", type=Path, default=None)
    parser.add_argument("--dataset-cache", type=Path, default=None)
    parser.add_argument("--build-dataset-only", action="store_true")
    parser.add_argument(
        "--target-vgrid",
        "--target-Nv",
        dest="target_vgrid",
        type=int,
        required=True,
    )
    parser.add_argument("--low-Nx", type=int, default=200)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=5)
    parser.add_argument("--online-case-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--rollout-anchor-samples", type=int, default=32)
    parser.add_argument(
        "--regimes",
        type=str,
        default="linear_landau,nonlinear_landau_weak,nonlinear_landau_strong",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--teacher-Nx", type=int, default=256)
    parser.add_argument("--teacher-Nv", type=int, default=512)
    parser.add_argument("--teacher-L", type=float, default=4.0 * math.pi)
    parser.add_argument("--teacher-vmin", type=float, default=-8.0)
    parser.add_argument("--teacher-vmax", type=float, default=8.0)
    parser.add_argument("--teacher-dt", type=float, default=0.01)
    parser.add_argument("--teacher-poisson-sign", type=float, default=1.0)
    parser.add_argument("--linear-T", type=float, default=10.0)
    parser.add_argument("--linear-eps", type=float, default=0.01)
    parser.add_argument("--linear-modes", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--linear-num-samples", type=int, default=8)
    parser.add_argument("--linear-seed", type=int, default=0)
    parser.add_argument("--nonlinear-T", type=float, default=10.0)
    parser.add_argument("--nonlinear-k0", type=float, default=0.5)
    parser.add_argument("--weak-eps", type=str, default="0.03,0.05,0.07,0.1,0.15")
    parser.add_argument("--strong-eps", type=str, default="0.15,0.25,0.35,0.5,0.65")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
