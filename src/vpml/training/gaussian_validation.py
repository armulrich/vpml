"""Compare checkpoints on fixed, paired windows from the validation split."""
from collections import defaultdict

import jax
import numpy as np

from vpml.training.wavenumber_training import loss_factory


def validation_windows(manifest, anchors_per_case):
    """Choose one matched shape per spatial family, at the same time for every strength."""
    families = defaultdict(lambda: defaultdict(dict))
    for index, case in enumerate(manifest["cases"]):
        if case["split"] == "validation":
            family = (case["k0"], case["mode_count"], case["decay"])
            families[family][case["shape_id"]][case["strength"]] = index
    if not families or anchors_per_case < 1:
        raise ValueError("Validation requires eligible windows")
    strengths = {case["strength"] for case in manifest["cases"]}
    anchors = np.linspace(0, anchors_per_case - 1, len(families)).round().astype(int)
    rows = []
    for anchor, family in zip(anchors, sorted(families)):
        shapes = families[family]
        cases = shapes[min(shapes)]
        if set(cases) != strengths:
            raise ValueError(f"Incomplete strength comparison for validation family {family}")
        rows.extend((strength, cases[strength], int(anchor)) for strength in sorted(cases))
    return rows


def make_validation_report(data, config, stats):
    """Build one forward-only loss kernel and reuse it for each checkpoint."""
    rows = validation_windows(data.manifest, data.anchors_per_case)
    score = jax.jit(loss_factory(config, stats))

    def report(params):
        values = defaultdict(list)
        for strength, case, anchor in rows:
            batch = data.batch([(case, anchor)])
            values[strength].append(float(score(params, batch)))
        # A divergent validation rollout is visible in JSON independently of training.
        return {
            "validation_windows": len(rows),
            "by_strength": {
                str(strength): {
                    "windows": len(losses),
                    "finite": sum(bool(np.isfinite(loss)) for loss in losses),
                    "mean_relative_l2": float(np.mean(losses)) if np.all(np.isfinite(losses)) else None,
                }
                for strength, losses in sorted(values.items())
            },
        }

    return report
