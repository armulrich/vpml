"""Guarded exact continuation of the preserved latent Burles E30 run to E40.

The source E30 run is treated as immutable.  Before training, this runner
verifies the source checkpoint and exact training state, copies the complete
training directory into a new Modal volume namespace, and resumes only inside
that copy.  E31--E40 are evaluated independently through T=120.
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess

import modal


_LOCAL_SUPPORT_PATH = Path(__file__).with_name("modal_burles_latent_fno.py")
_IMAGE_SUPPORT_PATH = Path("/root/vpml/model/train/modal_burles_latent_fno.py")
_SUPPORT_PATH = (
    _LOCAL_SUPPORT_PATH if _LOCAL_SUPPORT_PATH.is_file() else _IMAGE_SUPPORT_PATH
)
_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "vpml_modal_burles_latent_support", _SUPPORT_PATH
)
if _SUPPORT_SPEC is None or _SUPPORT_SPEC.loader is None:
    raise ImportError(f"Cannot load Modal support module from {_SUPPORT_PATH}")
_support = importlib.util.module_from_spec(_SUPPORT_SPEC)
_SUPPORT_SPEC.loader.exec_module(_support)

REFERENCE_CACHE = _support.REFERENCE_CACHE
REPO_ROOT = _support.REPO_ROOT
VOLUME_ROOT = _support.VOLUME_ROOT
_run_logged = _support._run_logged
_sha256_file = _support._sha256_file
_source_tree_sha256 = _support._source_tree_sha256
_verify_remote_inputs = _support._verify_remote_inputs
evaluation_arguments = _support.evaluation_arguments
image = _support.image
latent_train_arguments = _support.latent_train_arguments
volume = _support.volume


APP_NAME = "vpml-burles-latent-fno-e40"
SOURCE_RUN_NAME = "history_latent6_random1729_full_anchor_E20_to_E30_20260915"
SOURCE_RUN_DIR = VOLUME_ROOT / "runs" / SOURCE_RUN_NAME
SOURCE_TRAIN_DIR = SOURCE_RUN_DIR / "training"
RUN_NAME = "history_latent6_random1729_full_anchor_E30_to_E40_20260918"
RUN_DIR = VOLUME_ROOT / "runs" / RUN_NAME
TRAIN_DIR = RUN_DIR / "training"
EVALUATION_EPOCHS = tuple(range(31, 41))

SOURCE_HASHES = {
    "epoch030_low_moment_closure.npz": (
        "ec78a3935d7416e709f02c38c02926f9536fb725cdcc605bcc83b77d8d596639"
    ),
    "training_state.npz": (
        "1b7a71d65c9d30104a437170f87d4c971ba879bc8d3348becf019f17afde8aaf"
    ),
}


app = modal.App(APP_NAME)


def _verify_source_e30() -> dict[str, str]:
    if RUN_DIR == SOURCE_RUN_DIR:
        raise ValueError("Continuation and source run directories must differ")
    observed: dict[str, str] = {}
    for name, expected in SOURCE_HASHES.items():
        path = SOURCE_TRAIN_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing preserved E30 artifact: {path}")
        observed[name] = _sha256_file(path)
        if observed[name] != expected:
            raise ValueError(
                f"Preserved E30 artifact hash mismatch for {name}: "
                f"{observed[name]} != {expected}"
            )
    return observed


def _prepare_continuation(source_tree_sha256: str) -> dict[str, object]:
    completion = _verify_remote_inputs()
    observed = _verify_source_e30()
    expected = {
        "app": APP_NAME,
        "run_name": RUN_NAME,
        "source_run": str(SOURCE_RUN_DIR),
        "source_completed_epoch": 30,
        "continuation_stop_epoch": 40,
        "planned_epochs": 1000,
        "evaluation_epochs": list(EVALUATION_EPOCHS),
        "reference_manifest_sha256": completion["manifest_sha256"],
        "source_artifact_sha256": observed,
        "continuation_source_tree_sha256": source_tree_sha256,
    }
    provenance_path = RUN_DIR / "modal_provenance.json"
    if RUN_DIR.exists():
        if not provenance_path.is_file():
            raise FileExistsError(
                f"Refusing to reuse unprovenanced continuation directory: {RUN_DIR}"
            )
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Existing E40 continuation provenance does not match")
        if not (TRAIN_DIR / "training_state.npz").is_file():
            raise FileNotFoundError("Continuation directory lacks training_state.npz")
        return expected

    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir()
    # Preserve even incomplete destination artifacts for postmortem inspection.
    # A failed copy has no provenance and will be rejected on a later launch.
    shutil.copytree(SOURCE_TRAIN_DIR, TRAIN_DIR)
    shutil.copy2(
        SOURCE_RUN_DIR / "modal_provenance.json",
        RUN_DIR / "source_modal_provenance.json",
    )
    for name, expected_hash in SOURCE_HASHES.items():
        copied_hash = _sha256_file(TRAIN_DIR / name)
        if copied_hash != expected_hash:
            raise ValueError(f"Copied E30 artifact hash mismatch for {name}")
    provenance_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run(("sync", str(VOLUME_ROOT)), check=True)
    return expected


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    volumes={str(VOLUME_ROOT): volume},
)
def continue_and_evaluate_e40(source_tree_sha256: str) -> str:
    """Fork immutable E30 state, resume exactly to E40, and evaluate E31--E40."""
    _prepare_continuation(source_tree_sha256)
    final_checkpoint = TRAIN_DIR / "epoch040_low_moment_closure.npz"
    if final_checkpoint.exists():
        print("[modal] E40 checkpoint already exists; preserving it", flush=True)
    else:
        _run_logged(
            latent_train_arguments(40, train_dir=TRAIN_DIR),
            RUN_DIR / "train_E30_to_E40.log",
        )
    for epoch in EVALUATION_EPOCHS:
        summary = (
            RUN_DIR
            / "evaluations"
            / f"epoch{epoch:03d}"
            / "heldout_cases"
            / "summary.json"
        )
        if summary.exists():
            print(f"[modal] evaluation E{epoch} already exists; preserving it", flush=True)
            continue
        _run_logged(
            evaluation_arguments(epoch, train_dir=TRAIN_DIR, run_dir=RUN_DIR),
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
def preflight_e40(source_tree_sha256: str) -> dict[str, object]:
    """Verify immutable E30 inputs and that the E40 namespace is safe to use."""
    completion = _verify_remote_inputs(verify_hashes=True)
    source_hashes = _verify_source_e30()
    if RUN_DIR.exists():
        provenance = RUN_DIR / "modal_provenance.json"
        if not provenance.is_file():
            raise FileExistsError(f"Unprovenanced destination already exists: {RUN_DIR}")
        destination_status = "existing_provenanced_continuation"
    else:
        destination_status = "absent_ready_to_create"
    return {
        "source_tree_sha256": source_tree_sha256,
        "manifest_sha256": completion["manifest_sha256"],
        "source_hashes": source_hashes,
        "source_run": str(SOURCE_RUN_DIR),
        "destination_run": str(RUN_DIR),
        "destination_status": destination_status,
    }


@app.local_entrypoint()
def launch() -> None:
    local_repo_root = Path(__file__).resolve().parents[2]
    print(
        continue_and_evaluate_e40.remote(
            source_tree_sha256=_source_tree_sha256(local_repo_root)
        )
    )
