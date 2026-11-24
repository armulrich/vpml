"""Optional shared, indexed shards of read-only float64 moment histories."""
import fcntl
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, file_digest, read_json


def _paths(root, shard):
    return root / f"{shard:08d}.f64", root / f"{shard:08d}.json"


class PackedMoments:
    def __init__(self, root, metadata, indices, shape):
        self.root = root
        self.shape = tuple(shape)
        self.per_shard = metadata["cases_per_shard"]
        self.byte_count = 8 * int(np.prod(self.shape))
        self.receipts = metadata["receipt_hashes"]
        self.indices = indices
        self._open = lru_cache(maxsize=16)(self._open_shard)
        self._verified = set()

    def _open_shard(self, shard):
        data, receipt = _paths(self.root, shard)
        if file_digest(receipt) != self.receipts[shard]:
            raise ValueError(f"Packed moment receipt checksum mismatch: {receipt}")
        checksums = read_json(receipt)["cases"]
        count = min(self.per_shard, len(self.indices) - shard * self.per_shard)
        if len(checksums) != count or data.stat().st_size != self.byte_count * count:
            raise ValueError(f"Incomplete packed moment shard: {data}")
        return np.memmap(data, mode="r", dtype="<f8", shape=(count, *self.shape)), checksums

    def state(self, case_index):
        rank = int(np.searchsorted(self.indices, case_index))
        if rank == len(self.indices) or self.indices[rank] != case_index:
            return None
        shard, offset = divmod(rank, self.per_shard)
        states, checksums = self._open(shard)
        case = states[offset]
        if rank not in self._verified:
            if hashlib.sha256(case).hexdigest() != checksums[offset]:
                raise ValueError(f"Packed moment checksum mismatch at case index {case_index}")
            self._verified.add(rank)
        return case


def open_packed(root, source_digest, cases, shape):
    header = root / "packed.json"
    if not header.exists():
        return None
    metadata = read_json(header)
    indices = np.flatnonzero(cases.records["output_sha256"] != b"")
    per_shard = metadata["cases_per_shard"]
    if (metadata["training_outputs_sha256"] != source_digest or tuple(metadata["shape"]) != shape
            or type(per_shard) is not int or per_shard < 1
            or len(metadata["receipt_hashes"]) != (len(indices) + per_shard - 1) // per_shard):
        raise ValueError("Packed moments disagree with the selected training catalog")
    return PackedMoments(root, metadata, indices, shape)


def publish_packed(root, source_digest, cases, shape, target_bytes, load_case):
    """Write one trajectory at a time. A verified shard receipt permits a retry."""
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    per_shard = max(1, target_bytes // (8 * int(np.prod(shape))))
    root.mkdir(exist_ok=True, parents=True)
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if open_packed(root, source_digest, cases, shape) is not None:
            return root
        indices = np.flatnonzero(cases.records["output_sha256"] != b"")
        receipt_hashes = []
        for start in range(0, len(indices), per_shard):
            shard, receipt = _paths(root, start // per_shard)
            selected = indices[start:start + per_shard]
            # A changed packing size cannot reuse an earlier shard's layout.
            identity = {"first": int(selected[0]), "last": int(selected[-1]),
                        "count": len(selected), "source": source_digest}
            if receipt.exists() and shard.exists():
                prior = read_json(receipt)
                if (prior["identity"] == identity and file_digest(shard) == prior["sha256"]):
                    receipt_hashes.append(file_digest(receipt))
                    continue
            checksums = []
            shard_checksum = hashlib.sha256()

            def write(handle):
                for index in selected:
                    values = np.ascontiguousarray(load_case(int(index)), dtype="<f8")
                    if values.shape != shape:
                        raise ValueError("Unexpected moment shape during packing")
                    checksums.append(hashlib.sha256(values).hexdigest())
                    shard_checksum.update(values)
                    handle.write(memoryview(values).cast("B"))

            atomic_write(shard, write)
            record = {"identity": identity, "cases": checksums, "sha256": shard_checksum.hexdigest()}
            atomic_write(receipt, lambda f: f.write(canonical_json_bytes(record, newline=True)))
            receipt_hashes.append(file_digest(receipt))
        metadata = {"training_outputs_sha256": source_digest, "cases_per_shard": per_shard,
                    "shape": shape, "receipt_hashes": receipt_hashes}
        atomic_write(root / "packed.json", lambda f: f.write(canonical_json_bytes(metadata, newline=True)))
    return root
