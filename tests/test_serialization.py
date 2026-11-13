"""Storage compatibility and failure behavior on small local files."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from vpml.io.atomic import atomic_write
from vpml.io.serialization import canonical_json_bytes, file_digest, read_json, source_tree_digest


class SerializationTests(unittest.TestCase):
    def test_source_digest_matches_flat_upload_and_src_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layout in ("checkout", "upload"):
                (root / layout / "model").mkdir(parents=True)
                (root / layout / "model/driver.py").write_text("value = 1\n")
            (root / "checkout/src/vpml").mkdir(parents=True)
            (root / "upload/vpml").mkdir()
            for path in (root / "checkout/src/vpml/solver.py", root / "upload/vpml/solver.py"):
                path.write_text("value = 2\n")
            self.assertEqual(source_tree_digest(root / "checkout"), source_tree_digest(root / "upload"))
            (root / "checkout/src/vpml/solver.py").write_text("value = 3\n")
            self.assertNotEqual(source_tree_digest(root / "checkout"), source_tree_digest(root / "upload"))

    def test_canonical_bytes_preserve_existing_hash_conventions(self):
        value = {"z": 1.0, "a": "λ"}
        expected = b'{"a":"\\u03bb","z":1.0}'
        self.assertEqual(canonical_json_bytes(value), expected)
        self.assertEqual(canonical_json_bytes(value, newline=True), expected + b"\n")

    def test_failed_write_and_exclusive_publication_preserve_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record"
            atomic_write(path, lambda handle: handle.write(b"old"), replace=False)

            def fail(handle):
                handle.write(b"partial")
                raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                atomic_write(path, fail)
            with self.assertRaises(FileExistsError):
                atomic_write(path, lambda handle: handle.write(b"replacement"), replace=False)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_write(path, lambda handle: handle.write(b"new"))
            self.assertEqual(file_digest(path, chunk_bytes=1), hashlib.sha256(b"new").hexdigest())

    def test_strict_reader_rejects_ambiguous_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for invalid in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                path.write_text(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    read_json(path)


if __name__ == "__main__":
    unittest.main()
