"""Compare verified rollout summaries; fail closed on missing required evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.train.coupled_run_support import atomic_json, file_digest


def compare_frontier(baseline, candidate, *, numerical_tolerance=1e-6):
    """Development gate, not a certificate of independent generalization.

    Metric 1 is checked per case. Regime field means and every previously
    detected significant transition must not regress. Whole-envelope means
    must improve along with strong field error. No transition count is imposed
    on the model; the reference supplies whichever transitions are present.
    """
    excluded = {"nonlinear_landau_weak_ic16"}
    failures = []
    def index(payload):
        rows = [r for r in payload.get("cases", []) if r["case_id"] not in excluded]
        result = {r["case_id"]: r for r in rows}
        if len(rows) != len(result):
            raise ValueError("duplicate case identity in evaluation")
        return result
    old, new = index(baseline), index(candidate)
    if not old or old.keys() != new.keys():
        return {"passed": False, "failures": ["case sets differ or are empty"], "regimes": {}}
    for key in ("metric_protocol", "reference_manifest_sha256", "projected_metadata_sha256"):
        if key in baseline or key in candidate:
            if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
                failures.append(f"evaluation provenance mismatch: {key}")
    def number(row, key):
        value = row.get(key)
        return float(value) if value is not None and np.isfinite(value) else None
    for case_id, before in old.items():
        after = new[case_id]
        if before.get("regime") != after.get("regime"):
            failures.append(f"{case_id}: regime changed")
        if not after.get("bounded_to_final_time", False):
            failures.append(f"{case_id}: rollout not bounded")
        for key in ("minimum_density", "minimum_pressure"):
            value = number(after, key)
            if value is None or value <= 0:
                failures.append(f"{case_id}: invalid {key}")
        previous_rate, rate = number(before, "epsilon_grow"), number(after, "epsilon_grow")
        if previous_rate is None or rate is None or rate > previous_rate + numerical_tolerance:
            failures.append(f"{case_id}: Metric 1 missing or regressed")
        if "epsilon_grow_matched_cadence" in before or "epsilon_grow_matched_cadence" in after:
            previous_matched = number(before, "epsilon_grow_matched_cadence")
            matched = number(after, "epsilon_grow_matched_cadence")
            if previous_matched is None or matched is None or matched > previous_matched + numerical_tolerance:
                failures.append(f"{case_id}: matched-cadence Metric 1 missing or regressed")
        saturation = after.get("latent_saturation", {}).get("saved_state_saturation_fraction")
        if saturation is None or not np.isfinite(saturation) or saturation > 1e-4:
            failures.append(f"{case_id}: saturation missing or sustained")
        transitions = before.get("significant_transitions", [])
        following = after.get("significant_transitions", [])
        if transitions:
            old_field, new_field = number(before, "epsilon_E"), number(after, "epsilon_E")
            if old_field is None or new_field is None or new_field > old_field + numerical_tolerance:
                failures.append(f"{case_id}: rebound-case field error missing or regressed")
        if len(transitions) != len(following):
            failures.append(f"{case_id}: reference transitions changed")
        for left, right in zip(transitions, following):
            for key in ("start_time", "peak_time", "teacher_factor"):
                if not np.isclose(left[key], right[key], rtol=1e-6, atol=1e-8):
                    failures.append(f"{case_id}: reference transition definition changed")
            old_factor, factor = number(left, "model_factor"), number(right, "model_factor")
            reference = number(left, "teacher_factor")
            if old_factor is None or factor is None or reference is None or min(old_factor, factor, reference) <= 0:
                failures.append(f"{case_id}: invalid transition factor")
                continue
            if abs(np.log(factor / reference)) > abs(np.log(old_factor / reference)) + numerical_tolerance:
                failures.append(f"{case_id}: transition error regressed")
    regimes = {}
    for regime in ("linear_landau", "nonlinear_landau_weak", "nonlinear_landau_strong"):
        keys = [c for c in old if old[c]["regime"] == regime]
        if not keys:
            failures.append(f"{regime}: missing cases")
            continue
        result = {}
        for metric in ("epsilon_E", "log_envelope_rmse"):
            values = []
            for panel in (old, new):
                rows = [number(panel[k] if metric == "epsilon_E" else panel[k].get("full_envelope", {}), metric)
                        for k in keys]
                values.append(float(np.mean(rows)) if all(v is not None for v in rows) else None)
            previous, current = values
            result[metric] = {"baseline": previous, "candidate": current}
            if previous is None or current is None:
                failures.append(f"{regime}: {metric} evidence missing")
            elif current > previous + numerical_tolerance:
                failures.append(f"{regime}: {metric} regressed")
            elif regime == "nonlinear_landau_strong" and current >= 0.95 * previous:
                failures.append(f"{regime}: {metric} has not improved by 5 percent")
        regimes[regime] = result
    return {"passed": not failures, "failures": failures, "regimes": regimes,
            "scope": "development frontier only; independent panel and seed repetition still required",
            "numerical_tolerance": numerical_tolerance}


def compare_frontier_set(baselines, candidate, *, numerical_tolerance=1e-6):
    """Require improvement against each named checkpoint, never a synthetic model."""
    comparisons = {name: compare_frontier(panel, candidate, numerical_tolerance=numerical_tolerance)
                   for name, panel in baselines.items()}
    case_failures = []
    new = {r["case_id"]: r for r in candidate.get("cases", [])}
    for name, panel in baselines.items():
        for row in panel.get("cases", []):
            case_id = row["case_id"]
            if case_id == "nonlinear_landau_weak_ic16":
                continue
            after = new.get(case_id, {})
            for metric, before_value, after_value in (
                ("field", row.get("epsilon_E"), after.get("epsilon_E")),
                ("envelope", row.get("full_envelope", {}).get("log_envelope_rmse"),
                 after.get("full_envelope", {}).get("log_envelope_rmse")),
            ):
                if (before_value is None or after_value is None
                        or not np.isfinite(before_value) or not np.isfinite(after_value)
                        or after_value > before_value + numerical_tolerance):
                    case_failures.append(f"{name}: {case_id}: {metric} missing or regressed")
    return {"passed": bool(comparisons) and all(r["passed"] for r in comparisons.values())
            and not case_failures, "comparisons": comparisons, "case_failures": case_failures,
            "scope": "named development checkpoint frontier; independent evaluation still required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, action="append")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    if len(args.baseline) == 1:
        result = compare_frontier(json.loads(args.baseline[0].read_text()), candidate)
    else:
        result = compare_frontier_set({str(p.resolve()): json.loads(p.read_text())
                                       for p in args.baseline}, candidate)
    result["input_sha256"] = {str(p): file_digest(p) for p in (*args.baseline, args.candidate)}
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
