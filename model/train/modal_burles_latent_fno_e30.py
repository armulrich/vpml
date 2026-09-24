"""Guarded exact continuation of the preserved latent Burles E20 run to E30.

The source E20 run is treated as immutable.  Before training, this runner
verifies the source checkpoint and exact training state, copies the complete
training directory into a new Modal volume namespace, and resumes only inside
that copy.  E21--E30 are evaluated independently through T=120.
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
SOURCE_RUN_DIR = _support.RUN_DIR
SOURCE_TRAIN_DIR = _support.TRAIN_DIR
VOLUME_ROOT = _support.VOLUME_ROOT
_run_logged = _support._run_logged
_sha256_file = _support._sha256_file
_source_tree_sha256 = _support._source_tree_sha256
_verify_remote_inputs = _support._verify_remote_inputs
evaluation_arguments = _support.evaluation_arguments
image = _support.image
latent_train_arguments = _support.latent_train_arguments
volume = _support.volume


APP_NAME = "vpml-burles-latent-fno-e30"
RUN_NAME = "history_latent6_random1729_full_anchor_E20_to_E30_20260915"
RUN_DIR = VOLUME_ROOT / "runs" / RUN_NAME
TRAIN_DIR = RUN_DIR / "training"
EVALUATION_EPOCHS = tuple(range(21, 31))

SOURCE_HASHES = {
    "epoch020_low_moment_closure.npz": (
        "ac395a6c0a497667b453d66d39cc3771941dc00d538de0e73de44bd70e060396"
    ),
    "training_state.npz": (
        "4eaed7fa0d9260199636d4ac63bfe8aa8b758e103f606dcf471c0f1777cad956"
    ),
}


app = modal.App(APP_NAME)


def _verify_source_e20() -> dict[str, str]:
    if RUN_DIR == SOURCE_RUN_DIR:
        raise ValueError("Continuation and source run directories must differ")
    observed: dict[str, str] = {}
    for name, expected in SOURCE_HASHES.items():
        path = SOURCE_TRAIN_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing preserved E20 artifact: {path}")
        observed[name] = _sha256_file(path)
        if observed[name] != expected:
            raise ValueError(
                f"Preserved E20 artifact hash mismatch for {name}: "
                f"{observed[name]} != {expected}"
            )
    return observed


def _prepare_continuation(source_tree_sha256: str) -> dict[str, object]:
    completion = _verify_remote_inputs()
    observed = _verify_source_e20()
    expected = {
        "app": APP_NAME,
        "run_name": RUN_NAME,
        "source_run": str(SOURCE_RUN_DIR),
        "source_completed_epoch": 20,
        "continuation_stop_epoch": 30,
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
            raise ValueError("Existing E30 continuation provenance does not match")
        if not (TRAIN_DIR / "training_state.npz").is_file():
            raise FileNotFoundError("Continuation directory lacks training_state.npz")
        return expected

    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir()
    try:
        shutil.copytree(SOURCE_TRAIN_DIR, TRAIN_DIR)
        shutil.copy2(
            SOURCE_RUN_DIR / "modal_provenance.json",
            RUN_DIR / "source_modal_provenance.json",
        )
        for name, expected_hash in SOURCE_HASHES.items():
            copied_hash = _sha256_file(TRAIN_DIR / name)
            if copied_hash != expected_hash:
                raise ValueError(f"Copied E20 artifact hash mismatch for {name}")
        provenance_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        subprocess.run(("sync", str(VOLUME_ROOT)), check=True)
    except BaseException:
        # A failed initial copy is never accepted as a resumable lineage.
        shutil.rmtree(RUN_DIR, ignore_errors=True)
        subprocess.run(("sync", str(VOLUME_ROOT)), check=False)
        raise
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
def continue_and_evaluate_e30(source_tree_sha256: str) -> str:
    """Fork immutable E20 state, resume exactly to E30, and evaluate E21--E30."""
    _prepare_continuation(source_tree_sha256)
    final_checkpoint = TRAIN_DIR / "epoch030_low_moment_closure.npz"
    if final_checkpoint.exists():
        print("[modal] E30 checkpoint already exists; preserving it", flush=True)
    else:
        _run_logged(
            latent_train_arguments(30, train_dir=TRAIN_DIR),
            RUN_DIR / "train_E20_to_E30.log",
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
def preflight_e30(source_tree_sha256: str) -> dict[str, object]:
    """Verify immutable E20 inputs and that the E30 namespace is safe to use."""
    completion = _verify_remote_inputs(verify_hashes=True)
    source_hashes = _verify_source_e20()
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
        continue_and_evaluate_e30.remote(
            source_tree_sha256=_source_tree_sha256(local_repo_root)
        )
    )
