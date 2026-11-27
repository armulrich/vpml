"""Compare checkpoints on fixed, paired windows from the validation split."""
from collections import defaultdict

import jax
import numpy as np

from vpml.io.serialization import json_digest
from vpml.training.wavenumber_training import window_loss_factory


def validation_windows(manifest, anchors_per_case):
    """Choose one matched shape per spatial family, at the same time for every strength."""
    families = {}
    strengths = set()
    for index, case in enumerate(manifest["cases"]):
        strength = case["strength"]
        strengths.add(strength)
        if case["split"] != "validation":
            continue
        family = (case["k0"], case["mode_count"], case["decay"])
        shape = case["shape_id"]
        if family not in families or shape < families[family][0]:
            families[family] = (shape, {})
        if shape == families[family][0]:
            families[family][1][strength] = index
    if not families or anchors_per_case < 1:
        raise ValueError("Validation requires eligible windows")
    anchors = np.linspace(0, anchors_per_case - 1, len(families)).round().astype(int)
    rows = []
    for anchor, family in zip(anchors, sorted(families)):
        _, cases = families[family]
        if set(cases) != strengths:
            raise ValueError(f"Incomplete strength comparison for validation family {family}")
        rows.extend((strength, cases[strength], int(anchor)) for strength in sorted(cases))
    return rows


def _external_rows(data, config, panel):
    """Resolve a frozen list of held-out windows against the active case catalog."""
    rows = []
    seen = set()
    cases = data.manifest["cases"]
    for row in panel["rows"]:
        index, anchor = row["case_index"], row["anchor_index"]
        if (type(index) is not int or not 0 <= index < len(cases)
                or type(anchor) is not int or not 0 <= anchor < data.anchors_per_case):
            raise ValueError("Validation panel case index or anchor is outside the dataset")
        case = cases[index]
        if case["split"] != "validation" or row["case_id"] != case["case_id"]:
            raise ValueError("Validation panel requires matching held-out case IDs")
        for field in ("strength", "shape_id", "k0", "mode_count", "decay"):
            if field in row and row[field] != case[field]:
                raise ValueError(f"Validation panel disagrees with case {field}")
        if ("start_time" in row and not np.isclose(row["start_time"], anchor * data.anchor_stride * config.dt,
                                                   rtol=0, atol=1e-9)):
            raise ValueError("Validation window start time disagrees with anchor")
        if "time_band" in row and "strata" in panel:
            lo, hi = panel["strata"][row["time_band"]]
            if not lo <= anchor <= hi:
                raise ValueError("Validation anchor is outside its stated time band")
        if ("source_output_sha256" in row
                and row["source_output_sha256"] != data._output_digest(index)):
            raise ValueError(f"Validation reference output changed at {case['case_id']}")
        if (index, anchor) in seen:
            raise ValueError("Duplicate validation window")
        seen.add((index, anchor))
        rows.append((case["strength"], index, anchor))
    if not rows:
        raise ValueError("Validation panel is empty")
    return rows


def make_validation_report(data, config, stats, *, batch_size=12, panel=None):
    """Score the fixed panel with a bounded number of forecast windows in memory."""
    if batch_size < 1:
        raise ValueError("Validation batch size must be positive")
    rows = (validation_windows(data.manifest, data.anchors_per_case) if panel is None
            else _external_rows(data, config, panel))
    panel_sha256 = json_digest([
        [data.manifest["cases"][case]["case_id"], anchor] for _, case, anchor in rows
    ])
    score_windows = jax.jit(window_loss_factory(config, stats))

    def report(params):
        values = defaultdict(list)
        window_values = []
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = data.batch([(case, anchor) for _, case, anchor in selected])
            losses = np.asarray(score_windows(params, batch))
            for (strength, _, _), loss in zip(selected, losses, strict=True):
                value = float(loss) if np.isfinite(loss) else None
                values[strength].append(value)
                window_values.append(value)
        all_losses = [loss for losses in values.values() for loss in losses]
        result = {
            "panel_sha256": panel_sha256,
            "validation_windows": len(rows),
            "mean_relative_l2": float(np.mean(all_losses)) if None not in all_losses else None,
            "by_strength": {
                str(strength): {
                    "windows": len(losses),
                    "finite": len(losses) - losses.count(None),
                    "mean_relative_l2": float(np.mean(losses)) if None not in losses else None,
                    "relative_l2": losses,
                }
                for strength, losses in sorted(values.items())
            },
        }
        if panel is not None:
            # Aligned to the frozen input rows, including per-shape and time-band metadata.
            result["window_relative_l2"] = window_values
        return result

    report.panel_sha256 = panel_sha256
    return report
