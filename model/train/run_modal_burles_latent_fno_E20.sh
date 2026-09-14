#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_TREE_SHA256="$(python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("model/train/modal_burles_latent_fno.py").resolve()
spec = importlib.util.spec_from_file_location("vpml_modal_latent", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module._source_tree_sha256(Path.cwd()))
PY
)"

modal run --detach --timestamps \
  model/train/modal_burles_latent_fno.py::train_and_evaluate_e20 \
  --source-tree-sha256 "${SOURCE_TREE_SHA256}"
