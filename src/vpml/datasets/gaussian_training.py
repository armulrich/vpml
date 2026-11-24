"""Indexed windows of conserved moment histories for the Gaussian dataset."""

from collections import Counter
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from vpml.io.atomic import atomic_write
from vpml.io.serialization import file_digest, json_digest

from .manifest import Numerics, read_manifest
from .packed_moments import open_packed, publish_packed
from .reader import _read_current_case
from .training_index import open_training_index, read_completed_digest


SAMPLED_ROUND_UPDATES = 25


def _draw_without_replacement(rng, population, count):
    """Partial Fisher–Yates shuffle with storage proportional to the draw."""
    remaining = population - np.arange(count, dtype=np.int64)
    slots = rng.integers(remaining)
    swaps = {}
    result = np.empty(count, dtype=np.int64)
    for i, slot in enumerate(slots):
        last = remaining[i] - 1
        result[i] = swaps.get(slot, slot)
        swaps[slot] = swaps.get(last, last)
    return result


def _reference_moments(root, manifest, manifest_digest, case, case_digest, output_digest):
    arrays = _read_current_case(root, manifest, case, manifest_sha256=manifest_digest,
                                case_sha256=case_digest)
    if read_completed_digest(root, case["case_id"]) != output_digest:
        raise ValueError(f"Reference output changed during training: {case['case_id']}")
    states = np.array(arrays["raw_moments"], dtype=np.float64, copy=True, order="C")
    states[:, 0] -= 1
    states[:, 2] -= 1
    return states


def _packed_layout(cache, output_digest, numerics):
    return cache / "packed" / output_digest, (numerics.nsamples, 3, numerics.output_nx)


def pack_training_cache(reference_root, cache_root, *, target_bytes=256 * 1024**2):
    """Prepare multitrajectory shards without expanding all windows or running a solver."""
    root = Path(reference_root)
    metadata, cases = open_training_index(root, cache_root)
    manifest = dict(metadata["metadata"], cases=cases)
    cache = Path(cache_root) / "conserved_moments" / metadata["manifest_sha256"]
    digest = metadata["training_outputs_sha256"]
    directory, shape = _packed_layout(cache, digest, Numerics(**manifest["numerics"]))

    def load_case(index):
        case = cases[index]
        output = cases.records["output_sha256"][index].decode("ascii")
        if read_completed_digest(root, case["case_id"]) != output:
            raise ValueError("Source output disagrees with the published training index")
        path = cache / f"{case['case_id']}-{output}.npy"
        receipt = path.with_suffix(".sha256")
        if path.is_file() and receipt.is_file() and file_digest(path) == receipt.read_text().strip():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        case_digest = cases.records["case_sha256"][index].decode("ascii")
        return _reference_moments(root, manifest, metadata["manifest_sha256"], case, case_digest, output)

    return publish_packed(directory, digest, cases, shape, target_bytes, load_case)


class GaussianTrainingData:
    """Read one verified case at a time, then serve its windows from a mmap.

    A row is (index in the immutable manifest, index on the anchor grid).
    The sampler works on these indices alone, so shuffling and seeking to a
    saved batch do not decompress, copy or materialize any trajectory windows.
    """

    @classmethod
    def open_indexed(cls, reference_root, cache_root, config):
        """Open published data for training without a metadata scan on each restart."""
        index = open_training_index(reference_root, cache_root)
        return cls(reference_root, cache_root, config, index=index)

    def __init__(self, reference_root, cache_root, config, *, index=None):
        self.reference_root = Path(reference_root)
        self._index = index
        if index is None:
            self.manifest = read_manifest(self.reference_root / "manifest.json")
            self.manifest_sha256 = json_digest(self.manifest)
        else:
            metadata, case_table = index
            self.manifest = dict(metadata["metadata"], cases=case_table)
            self.manifest_sha256 = metadata["manifest_sha256"]
        self.config = config
        self.numerics = Numerics(**self.manifest["numerics"])
        if config.nx != self.numerics.output_nx or config.dt != self.numerics.save_dt:
            raise ValueError("Trainer resolution and step must match the saved observations")
        stride = 0.2 / config.dt
        if not np.isclose(stride, round(stride), rtol=0, atol=1e-10) or stride < 1:
            raise ValueError("Training step must divide the 0.2 anchor spacing")
        self.anchor_stride = round(stride)
        steps = config.preparation_steps + config.horizon
        if config.preparation_steps < 0 or config.horizon < 1 or steps >= self.numerics.nsamples:
            raise ValueError("Reference does not cover a complete training window")
        if config.relative_normalization != "window_rms":
            raise ValueError("Gaussian training uses the E10 window_rms objective")
        self.anchors_per_case = (self.numerics.nsamples - steps - 1) // self.anchor_stride + 1
        self.cases = self.manifest["cases"]
        if index is None:
            splits = np.asarray([case["split"] for case in self.cases], dtype="S10")
            strengths = np.asarray([case["strength"] for case in self.cases])
        else:
            splits = case_table.records["split"]
            strengths = case_table.records["strength"]
        self.case_counts = Counter({split: int(np.count_nonzero(splits == split.encode()))
                                    for split in ("train", "validation", "test")})
        self.strengths = tuple(float(s) for s in np.unique(strengths))
        train = splits == b"train"
        self._training_cases = tuple(np.flatnonzero(train & (strengths == s))
                                     for s in self.strengths)
        self._sample_round = lru_cache(maxsize=1)(self._round_rows)
        self.cache = Path(cache_root) / "conserved_moments" / self.manifest_sha256
        self.cache.mkdir(parents=True, exist_ok=True)
        self._packed = None
        if index is not None:
            directory, shape = _packed_layout(self.cache, self.training_outputs_sha256, self.numerics)
            self._packed = open_packed(directory, self.training_outputs_sha256, self.cases, shape)
        self._verified_paths = {}
        self._source_digests = {}
        # Bound the number of open mappings on both laptops and compute nodes.
        self._mapped = lru_cache(maxsize=64)(lambda path: np.load(path, mmap_mode="r", allow_pickle=False))

    def _round_rows(self, round_index, seed, round_size):
        rng = np.random.default_rng(np.random.SeedSequence([seed, round_index]))
        rows = np.empty((round_size, 2), dtype=np.int64)
        groups = len(self.strengths)
        for slot in range(groups):
            cases = self._training_cases[(round_size * round_index + slot) % groups]
            selected = rows[slot::groups]
            draws = _draw_without_replacement(rng, len(cases) * self.anchors_per_case, len(selected))
            case_offsets, anchors = np.divmod(draws, self.anchors_per_case)
            selected[:, 0] = cases[case_offsets]
            selected[:, 1] = anchors
        rows.flags.writeable = False
        return rows

    def round_updates(self, batch_size):
        if batch_size < 1 or not self.case_counts["train"]:
            raise ValueError("Sampling requires a positive batch size and training data")
        windows = self.case_counts["train"] * self.anchors_per_case
        return min(SAMPLED_ROUND_UPDATES, (windows + batch_size - 1) // batch_size)

    def sample_rows(self, step, seed, batch_size):
        """Return one minibatch by its zero-based accepted-update number.

        A round has at most 25 minibatches. Small corpora exhaust every window
        once per round and can have a short final batch. Only one round's
        indices are retained, independently of the number of possible windows.
        """
        if step < 0 or batch_size < 1 or not self.case_counts["train"]:
            raise ValueError("Sampling requires a nonnegative step, positive batch size and training data")
        round_size = min(batch_size * SAMPLED_ROUND_UPDATES,
                         self.case_counts["train"] * self.anchors_per_case)
        steps_per_round = self.round_updates(batch_size)
        round_index, offset = divmod(step, steps_per_round)
        start = offset * batch_size
        return self._sample_round(round_index, seed, round_size)[start:start + batch_size]

    def epoch_rows(self, epoch, seed, split="train"):
        """Shuffle all eligible windows once per epoch, interleaving strengths.

        Every consecutive group of len(strengths) contains every strength
        once. Slicing the stream into batches of 48 rotates the extra seats.
        """
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown dataset split: {split}")
        if not self.case_counts[split]:
            raise ValueError(f"No cases in split: {split}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        pool_size = self.case_counts[split] * self.anchors_per_case
        rows = np.empty((pool_size, 2), dtype=np.int32)
        for slot, strength in enumerate(self.strengths):
            indices = np.array([i for i, case in enumerate(self.cases)
                                if case["split"] == split and case["strength"] == strength])
            order = rng.permutation(len(indices) * self.anchors_per_case)
            case_offsets, anchors = np.divmod(order, self.anchors_per_case)
            rows[slot::len(self.strengths), 0] = indices[case_offsets]
            rows[slot::len(self.strengths), 1] = anchors
        return rows

    def _read_output_digest(self, case_index):
        case_id = self.cases[case_index]["case_id"]
        digest = read_completed_digest(self.reference_root, case_id)
        if self._index is not None:
            expected = self.cases.records["output_sha256"][case_index].decode("ascii")
            if expected and expected != digest:
                raise ValueError(f"Reference output disagrees with the published training index: {case_id}")
        return digest

    def _output_digest(self, case_index):
        if case_index not in self._source_digests:
            self._source_digests[case_index] = self._read_output_digest(case_index)
        return self._source_digests[case_index]

    @cached_property
    def training_outputs_sha256(self):
        """Pin the actual train and validation outputs across restarts and machines."""
        if self._index is not None:
            return self._index[0]["training_outputs_sha256"]
        return json_digest([[case["case_id"], self._output_digest(i)]
                            for i, case in enumerate(self.cases)
                            if case["split"] in ("train", "validation")])

    def _cache_case(self, case_index):
        case = self.cases[case_index]
        digest = self._output_digest(case_index)
        path = self.cache / f"{case['case_id']}-{digest}.npy"
        receipt = path.with_suffix(".sha256")
        if path.is_file() and receipt.is_file() and file_digest(path) == receipt.read_text().strip():
            return path
        # This reader verifies the trajectory's checksum, time coordinates,
        # shape and protocol before deriving a cache from its full raw moments.
        case_digest = (None if self._index is None else
                       self.cases.records["case_sha256"][case_index].decode("ascii"))
        states = _reference_moments(self.reference_root, self.manifest, self.manifest_sha256,
                                    case, case_digest, digest)
        atomic_write(path, lambda handle: np.save(handle, states, allow_pickle=False))
        checksum = file_digest(path).encode("ascii")
        atomic_write(receipt, lambda handle: handle.write(checksum))
        return path

    def state(self, case_index):
        """Get a read-only (time, 3, x) array of (M0 - 1, M1, M2 - 1)."""
        if not 0 <= case_index < len(self.cases):
            raise IndexError("Manifest case index out of bounds")
        if self._packed is not None:
            case = self._packed.state(case_index)
            if case is not None:
                return case
        if case_index not in self._verified_paths:
            self._verified_paths[case_index] = self._cache_case(case_index)
        return self._mapped(self._verified_paths[case_index])

    def prepare(self, split="train"):
        """Prebuild or validate the on-disk caches for one manifest split."""
        if not self.case_counts[split]:
            raise ValueError(f"No cases in split: {split}")
        for index, case in enumerate(self.cases):
            if case["split"] == split:
                self.state(index)

    def batch(self, rows):
        """Materialize only one batch, preserving the case-specific physical grid."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != 2 or not len(rows) or rows.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty integer array of (case, anchor) rows")
        if np.any(rows[:, 1] < 0) or np.any(rows[:, 1] >= self.anchors_per_case):
            raise ValueError("Anchor out of bounds")
        config = self.config
        count = len(rows)
        dtype = np.dtype(config.compute_dtype)
        initial = np.empty((count, 3, config.nx), dtype=dtype)
        targets = np.empty((count, config.horizon, 3, config.nx), dtype=dtype)
        waves = np.empty((count, config.nx // 2 + 1), dtype=dtype)
        modes = np.arange(waves.shape[1])
        for slot, (case_index, anchor_index) in enumerate(rows):
            states = self.state(int(case_index))
            anchor = int(anchor_index) * self.anchor_stride
            first_target = anchor + config.preparation_steps + 1
            initial[slot] = states[anchor]
            targets[slot] = states[first_target:first_target + config.horizon]
            waves[slot] = modes * self.cases[case_index]["k0"]
        return {
            "initial": initial, "targets": targets, "k_arr": waves,
            "memory": np.zeros((count, config.memory_steps, 3, config.nx), dtype=dtype),
            "heat_flux_gradient_history": np.zeros((count, config.memory_steps, config.nx), dtype=dtype),
            # E10 derives its amplitude feature from the instantaneous state.
            # Its global relative loss is independent of legacy regime labels.
            "amplitude": np.sqrt(np.mean(np.square(initial[:, 0]), axis=-1)),
            "regime_index": np.zeros(count, dtype=np.int32),
        }
