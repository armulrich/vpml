"""Isolated, matched linear-only history and latent E20 qualification.

Old volume paths are read only. Each model writes to a new path, with explicit
resume authorization and matching configuration/source hashes required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import modal

_path = Path(__file__).with_name("modal_burles_latent_fno.py")
if not _path.is_file():
    _path = Path("/root/vpml/model/train/modal_burles_latent_fno.py")
_spec = importlib.util.spec_from_file_location("linear_pair_support", _path)
assert _spec and _spec.loader
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)

app = modal.App("vpml-linear-only-pair-20260921")
image, volume = _support.image, _support.volume
ROOT, REPO, CACHE = _support.VOLUME_ROOT, _support.REPO_ROOT, _support.REFERENCE_CACHE
PAIR_NAME = "linear_only_pair_20260921_v1"
NORMALIZATION = ROOT / "runs/history_fno_random1729_full_anchor_E1000/training/epoch000_low_moment_closure.npz"
NORMALIZATION_SHA = "3b9483428177a4c4671c9197a5d4c7225e2a52b27a1a92b90d6bf195ddfdf34e"
SOURCE_RUNS = (
    "history_fno_random1729_full_anchor_E1000",
    "history_latent6_random1729_full_anchor_resumable_E1000",
    "history_latent6_random1729_full_anchor_E20_to_E30_20260915",
    "history_latent6_random1729_full_anchor_E30_to_E40_20260918",
)


def run_path(model: str) -> Path:
    if model not in ("history", "latent"):
        raise ValueError("model must be history or latent")
    return ROOT / "runs" / f"{PAIR_NAME}_{model}_E20"


def training_arguments(model: str, epoch: int, train: Path) -> list[str]:
    # Reuse the preserved optimizer/model setup, changing only regime selection
    # and the latent dimension. Data order is identical between this pair.
    args = _support.latent_train_arguments(epoch, train_dir=train)
    args[args.index("--reference-cache") + 1] = str(private_cache_path(model))
    args[args.index("--latent-memory-dim") + 1] = "0" if model == "history" else "6"
    # The full sweep derives its actual count from the selected anchor table.
    ix = args.index("--steps-per-epoch")
    del args[ix:ix + 2]
    args.extend([
        "--training-regimes", "linear_landau",
        "--normalization-checkpoint", str(NORMALIZATION),
        "--update-norm-cap", "0",
    ])
    return args


def private_cache_path(model: str) -> Path:
    return Path("/tmp") / f"vpml_{PAIR_NAME}_{model}_cache"


def prepare_private_cache(model: str) -> Path:
    """Share immutable arrays, isolate any derived-cache metadata writes."""
    dst = private_cache_path(model)
    if dst.exists():
        return dst
    dst.mkdir()
    for source in CACHE.iterdir():
        if source.name == "derived":
            def copy_derived(src, target):
                if Path(src).suffix == ".npy":
                    Path(target).symlink_to(Path(src).resolve())
                    return target
                return shutil.copy2(src, target)
            shutil.copytree(source, dst / source.name, copy_function=copy_derived)
        elif source.is_dir():
            (dst / source.name).symlink_to(source, target_is_directory=True)
        else:
            shutil.copy2(source, dst / source.name)
    return dst


def previous_fingerprints() -> dict[str, str]:
    paths = [NORMALIZATION]
    for name in SOURCE_RUNS:
        train = ROOT / "runs" / name / "training"
        paths.extend(sorted(train.glob("epoch*_low_moment_closure.npz")))
        for filename in ("training_state.npz", "run_configuration.json", "training_metrics.npz"):
            if (train / filename).is_file():
                paths.append(train / filename)
    return {str(p): _support._sha256_file(p) for p in sorted(set(paths))}


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def audit_epoch_zero(train: Path, model: str) -> dict:
    import numpy as np
    # Keep the orchestration process off the GPU. Loading through the trainer
    # would create device arrays and reserve memory needed by the next subprocess.
    with np.load(train / "epoch000_low_moment_closure.npz", allow_pickle=False) as payload:
        params = {k.removeprefix("param_"): np.asarray(payload[k]) for k in payload.files if k.startswith("param_")}
        stats = {k.removeprefix("stat_"): np.asarray(payload[k]) for k in payload.files if k.startswith("stat_")}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).reshape(-1)[0]))
    with np.load(NORMALIZATION, allow_pickle=False) as payload:
        old_stats = {k.removeprefix("stat_"): np.asarray(payload[k]) for k in payload.files if k.startswith("stat_")}
    # Canonical regime scales can be selected down to one row. All input/output
    # and amplitude normalization constants must remain bitwise unchanged.
    keys = ("input_scale", "heat_flux_gradient_scale", "amplitude_center", "amplitude_scale")
    for key in keys:
        np.testing.assert_array_equal(np.asarray(stats[key]), np.asarray(old_stats[key]))
    shared = {}
    for key, value in params.items():
        if key.startswith("compact_latent"):
            continue
        a = np.asarray(value)
        shared[key] = hashlib.sha256(a.tobytes()).hexdigest()
    return {
        "model": model,
        "epoch_zero_sha256": _support._sha256_file(train / "epoch000_low_moment_closure.npz"),
        "shared_parameter_hashes": shared,
        "normalization_equal_to_preserved_E0": True,
        "parameter_count": sum(int(np.asarray(v).size) for v in params.values()),
        "metadata": metadata,
    }


@app.function(image=image, gpu="A100-80GB", cpu=16, memory=65536,
              timeout=24 * 60 * 60, startup_timeout=30 * 60,
              volumes={str(ROOT): volume})
def run(model: str, source_tree_sha256: str, source_commit: str, resume: bool = False) -> str:
    """Run exactly E20, with early E0/E1 inspection and every epoch evaluated."""
    dst = run_path(model)
    train = dst / "training"
    completion = _support._verify_remote_inputs()
    actual_source = _support._source_tree_sha256(REPO)
    if actual_source != source_tree_sha256:
        raise ValueError("Uploaded source tree differs from recorded launch hash")
    if _support._sha256_file(NORMALIZATION) != NORMALIZATION_SHA:
        raise ValueError("Preserved normalization checkpoint hash differs")
    expected = {
        "pair": PAIR_NAME, "model": model, "latent_dim": 0 if model == "history" else 6,
        "source_tree_sha256": source_tree_sha256, "source_commit": source_commit,
        "manifest_sha256": completion["manifest_sha256"],
        "normalization_source": str(NORMALIZATION), "normalization_sha256": NORMALIZATION_SHA,
        "training_regimes": ["linear_landau"], "seed": 1729,
        "stop_epoch": 20, "planned_epochs": 1000, "batch_size": 16,
        "initialization": "random neural parameters; preserved statistics only",
        "burnin_gradient": "detached, unchanged from preserved controls",
        "gpu": "A100-80GB", "evaluations": list(range(21)),
    }
    if dst.exists():
        if not resume:
            raise FileExistsError(f"Use explicit resume for this experiment only: {dst}")
        if json.loads((dst / "provenance.json").read_text()) != expected:
            raise ValueError("Refusing resume with different provenance")
        originals = json.loads((dst / "original_hashes.json").read_text())
    else:
        if resume:
            raise FileNotFoundError("Cannot resume absent experiment")
        dst.mkdir(parents=True, exist_ok=False)
        originals = previous_fingerprints()
        write_json(dst / "provenance.json", expected)
        write_json(dst / "original_hashes.json", originals)
    if previous_fingerprints() != originals:
        raise ValueError("A preserved source artifact changed before this run")
    prepare_private_cache(model)
    write_json(dst / "status.json", {"status": "preflight", "model": model})
    volume.commit()

    def evaluate(epoch: int) -> None:
        out = dst / "evaluations" / f"epoch{epoch:03d}"
        if (out / "heldout_cases/summary.json").is_file():
            return
        # Evaluation reads the regime selection saved in the checkpoint.
        arguments = _support.evaluation_arguments(epoch, train_dir=train, run_dir=dst)
        arguments[arguments.index("--reference-cache") + 1] = str(private_cache_path(model))
        _support._run_logged(arguments, out / "evaluation.log")
        write_json(dst / "status.json", {"status": "evaluating", "model": model, "evaluated_epoch": epoch})
        volume.commit()

    try:
        if not (dst / "preflight.json").is_file():
            _support._run_logged(
                [sys.executable, "-m", "model.diagnostics.linear_pair_preflight", "--out", str(dst / "preflight.json")],
                dst / "preflight.log",
            )
        # This stop/resume does not change the 1000-epoch optimizer schedule.
        # It exposes initial autonomous behavior before the remaining training.
        for stop in (1, 20):
            if not (train / f"epoch{stop:03d}_low_moment_closure.npz").is_file():
                write_json(dst / "status.json", {"status": "training", "model": model, "stop_epoch": stop})
                _support._run_logged(training_arguments(model, stop, train), dst / "train.log")
                volume.commit()
            if not (dst / "initialization_audit.json").is_file():
                write_json(dst / "initialization_audit.json", audit_epoch_zero(train, model))
            if stop == 1:
                evaluate(0)
                evaluate(1)
        for epoch in range(2, 21):
            evaluate(epoch)
        if previous_fingerprints() != originals:
            raise ValueError("A preserved source artifact changed during this run")
        outputs = {
            str(p.relative_to(dst)): _support._sha256_file(p)
            for p in sorted(dst.rglob("*"))
            if p.is_file() and p.suffix in (".npz", ".json", ".png") and p.name not in ("artifact_hashes.json", "status.json")
        }
        write_json(dst / "artifact_hashes.json", outputs)
        write_json(dst / "status.json", {"status": "complete", "model": model, "completed_epoch": 20, "originals_unchanged": True})
        volume.commit()
        return str(dst)
    except BaseException as exc:
        write_json(dst / "status.json", {"status": "failed", "model": model, "error": repr(exc)})
        volume.commit()
        raise
