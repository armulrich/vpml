"""Rescore archived Gaussian checkpoints on a supplied panel of held-out windows."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np

from vpml.datasets.gaussian_training import GaussianTrainingData
from vpml.io.serialization import file_digest, json_digest, read_json
from vpml.training.gaussian_checkpoints import write_json
from vpml.training.gaussian_validation import make_validation_report
from vpml.training.wavenumber_training import Configuration, initialize, read_checkpoint_parameters


def rescore(dataset, run, normalization, panel, steps, output, cache=None):
    """Evaluate each requested snapshot with the same compiled window loss."""
    dataset, run, output = (Path(path).resolve() for path in (dataset, run, output))
    if any(output.is_relative_to(path) for path in (dataset, run)):
        raise ValueError("Choose an evaluation output directory outside the training run and dataset")
    if not steps or len(set(steps)) != len(steps) or any(step < 0 for step in steps):
        raise ValueError("Supply distinct, nonnegative checkpoint steps")
    if cache and any(Path(cache).resolve().is_relative_to(path) for path in (dataset, run)):
        raise ValueError("Choose a shared training cache outside the run and dataset")
    protocol = read_json(run / "training-protocol.json")
    if file_digest(normalization) != protocol["normalization_sha256"]:
        raise ValueError("Normalization differs from training")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    config = Configuration(**protocol["model"])
    with np.load(normalization, allow_pickle=False) as arrays:
        stats = {key: arrays[key] for key in arrays.files}
    selection = read_json(panel)
    protocol_sha = json_digest(protocol)
    template = initialize(config)
    with TemporaryDirectory(prefix="vpml-validation-") as scratch:
        data = (GaussianTrainingData.open_indexed(dataset, cache, config) if cache
                else GaussianTrainingData(dataset, scratch, config))
        if (data.manifest_sha256 != protocol["dataset_sha256"]
                or data.training_outputs_sha256 != protocol["training_outputs_sha256"]):
            raise ValueError("Reference trajectories differ from training")
        evaluate = make_validation_report(data, config, stats, batch_size=protocol["microbatch_size"],
                                          panel=selection)
        output.mkdir(parents=True, exist_ok=True)
        for step in steps:
            name = f"step-{step:08d}"
            snapshot = run / "snapshots" / name
            destination = output / f"{name}.json"
            if destination.exists():
                raise FileExistsError(destination)
            receipt = read_json(snapshot / "receipt.json")
            weights, stored = read_checkpoint_parameters(snapshot / "checkpoint.npz", receipt["sha256"],
                                                         protocol_sha, template)
            if stored != step or receipt["step"] != step:
                raise ValueError(f"Archived checkpoint disagrees with requested update {step}")
            report = evaluate(weights)
            write_json(destination, dict(step=step, protocol_sha256=protocol_sha,
                                         checkpoint_sha256=receipt["sha256"],
                                         selection_sha256=json_digest(selection), **report), replace=False)
            print(f"{destination}: mean relative L2 = {report['mean_relative_l2']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run", "normalization", "panel", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--cache", type=Path, help="Existing indexed training cache, if available")
    rescore(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
