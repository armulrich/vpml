"""Modal runner for the random-start, history-only Burles-style FNO.

This module defines remote resources but performs no work until invoked with
``modal run``.  The E20 entrypoint resumes exactly from the Volume checkpoint
when present and evaluates the fixed T=120 held-out panel remotely.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import modal


APP_NAME = "vpml-burles-history-fno"
VOLUME_NAME = "vpml-low-moment-burles"
VOLUME_ROOT = Path("/mnt/vpml")
CACHE_NAME = "e376aa1efa28e754b5f6-low-moment-nx128"
REFERENCE_CACHE = VOLUME_ROOT / "reference" / CACHE_NAME
RUN_NAME = "history_fno_random1729_full_anchor_E1000"
RUN_DIR = VOLUME_ROOT / "runs" / RUN_NAME
TRAIN_DIR = RUN_DIR / "training"
REPO_ROOT = Path("/root/vpml")


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for package in ("model", "vpml"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda13]==0.10.0",
        "numpy==2.4.4",
        "matplotlib==3.10.9",
        "scipy",
    )
    .add_local_dir("vpml", str(REPO_ROOT / "vpml"), copy=True)
    .add_local_dir("model", str(REPO_ROOT / "model"), copy=True)
)


def history_train_arguments(stop_epoch: int) -> list[str]:
    if not 1 <= int(stop_epoch) <= 1000:
        raise ValueError("stop_epoch must be in [1, 1000]")
    arguments = [
        sys.executable,
        "-m",
        "model.train.low_moment_closure",
        "--reference-cache",
        str(REFERENCE_CACHE),
        "--outdir",
        str(TRAIN_DIR),
        "--rollout-Nx",
        "128",
        "--solver-dt",
        "0.025",
        "--rollout-horizon",
        "400",
        "--training-schedule",
        "random_windows",
        "--full-anchor-sweep",
        "--memory-backend",
        "burles_latent_fno",
        "--latent-memory-dim",
        "0",
        "--memory-steps",
        "50",
        "--memory-stride",
        "4",
        "--closure-history-input",
        "--autonomous-history-burnin-steps",
        "200",
        "--input-scaling",
        "current_density_rms_arcsinh",
        "--allow-uniform-heating",
        "--batch-size",
        "16",
        "--steps-per-epoch",
        "526",
        "--training-passes-per-epoch",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--data-parallel-devices",
        "1",
        "--no-translation-augmentation",
        "--width",
        "64",
        "--spectral-modes",
        "32",
        "--fno-depth",
        "8",
        "--epochs",
        str(int(stop_epoch)),
        "--planned-epochs",
        "1000",
        "--learning-rate",
        "1e-3",
        "--final-learning-rate",
        "1e-5",
        "--weight-decay",
        "1e-4",
        "--grad-clip",
        "0",
        "--global-relative-trajectory-loss",
        "--validation-every",
        "1",
        "--training-diagnostic-cases-per-regime",
        "1",
        "--diagnostic-start-times",
        "0,20,40,60,80,100",
        "--seed",
        "1729",
        "--skip-evaluation",
    ]
    if (TRAIN_DIR / "training_state.npz").exists():
        arguments.extend(("--resume-run", str(TRAIN_DIR)))
    return arguments


def evaluation_arguments(epoch: int) -> list[str]:
    checkpoint = TRAIN_DIR / f"epoch{int(epoch):03d}_low_moment_closure.npz"
    evaluation_dir = RUN_DIR / "evaluations" / f"epoch{int(epoch):03d}"
    return [
        sys.executable,
        "-m",
        "model.train.low_moment_closure",
        "--reference-cache",
        str(REFERENCE_CACHE),
        "--outdir",
        str(evaluation_dir),
        "--evaluate-checkpoint",
        str(checkpoint),
        "--evaluation-chunk-steps",
        "250",
    ]


def _run_logged(arguments: Iterable[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[modal] command:", " ".join(arguments), flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            list(arguments),
            cwd=REPO_ROOT,
            env={**os.environ, "VPML_JAX_BACKEND": "gpu", "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    subprocess.run(("sync", str(VOLUME_ROOT)), check=True)
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(arguments))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_remote_inputs(*, verify_hashes: bool = False) -> dict[str, object]:
    completion_path = REFERENCE_CACHE / "COMPACT_CACHE_MANIFEST.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Missing compact cache at {completion_path}; run the one-time staging command"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if int(completion.get("case_count", 0)) != 60:
        raise ValueError("Compact cache must contain all 60 ICs")
    if verify_hashes:
        metadata_path = REFERENCE_CACHE / "metadata.json"
        if _sha256_file(metadata_path) != completion["metadata_sha256"]:
            raise ValueError("Compact cache metadata hash mismatch")
        for index, record in enumerate(completion["cases"], start=1):
            case_id = record["case_id"]
            coefficient_path = REFERENCE_CACHE / "cases" / f"{case_id}.npy"
            snapshot_path = REFERENCE_CACHE / "snapshots" / f"{case_id}.npz"
            if _sha256_file(coefficient_path) != record["coefficient_sha256"]:
                raise ValueError(f"Compact coefficient hash mismatch for {case_id}")
            if _sha256_file(snapshot_path) != record["snapshot_sha256"]:
                raise ValueError(f"Compact snapshot hash mismatch for {case_id}")
            if index % 10 == 0:
                print(f"[modal] verified compact cache {index}/60", flush=True)
    import jax

    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError(f"Expected exactly one JAX GPU, got {jax.devices()}")
    print(f"[modal] JAX={jax.__version__} devices={jax.devices()}", flush=True)
    print(
        f"[modal] cache cases={completion['case_count']} "
        f"manifest={completion['manifest_sha256']}",
        flush=True,
    )
    return completion


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def train_and_evaluate_e20(
    source_tree_sha256: str,
    stop_epoch: int = 20,
    evaluation_epochs: str = "0,1,2,5,10,15,20",
) -> str:
    """Train/resume to E20 and evaluate selected full T=120 checkpoints."""
    _verify_remote_inputs()
    if int(stop_epoch) != 20:
        raise ValueError("The qualified launch target is fixed at E20")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = RUN_DIR / "modal_provenance.json"
    if not provenance_path.exists():
        provenance_path.write_text(
            json.dumps(
                {
                    "app": APP_NAME,
                    "volume": VOLUME_NAME,
                    "run_name": RUN_NAME,
                    "gpu": "A100-80GB",
                    "planned_epochs": 1000,
                    "qualification_stop_epoch": 20,
                    "history_only": True,
                    "latent_memory_dim": 0,
                    "seed": 1729,
                    "source_tree_sha256": source_tree_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    final_checkpoint = TRAIN_DIR / "epoch020_low_moment_closure.npz"
    if final_checkpoint.exists():
        print("[modal] E20 checkpoint already exists; preserving it", flush=True)
    else:
        _run_logged(history_train_arguments(20), RUN_DIR / "train.log")
    requested = tuple(int(value) for value in evaluation_epochs.split(",") if value)
    if requested != (0, 1, 2, 5, 10, 15, 20):
        raise ValueError("Evaluation checkpoints are fixed at 0,1,2,5,10,15,20")
    for epoch in requested:
        summary = RUN_DIR / "evaluations" / f"epoch{epoch:03d}" / "heldout_cases" / "summary.json"
        if summary.exists():
            print(f"[modal] evaluation E{epoch} already exists; preserving it", flush=True)
            continue
        _run_logged(
            evaluation_arguments(epoch),
            RUN_DIR / "evaluations" / f"epoch{epoch:03d}" / "evaluation.log",
        )
    subprocess.run(("sync", str(VOLUME_ROOT)), check=True)
    return str(RUN_DIR)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def preflight() -> dict[str, object]:
    """Verify the uploaded cache and CUDA-backed JAX without training."""
    completion = _verify_remote_inputs(verify_hashes=True)
    return {
        "jax_backend": "gpu",
        "case_count": completion["case_count"],
        "manifest_sha256": completion["manifest_sha256"],
        "run_dir": str(RUN_DIR),
    }


@app.local_entrypoint()
def launch() -> None:
    local_repo_root = Path(__file__).resolve().parents[2]
    print(
        train_and_evaluate_e20.remote(
            source_tree_sha256=_source_tree_sha256(local_repo_root)
        )
    )
