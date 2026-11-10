"""Compare verified retained dataset fields without qualifying either solver.

Only completed current-format outputs are accepted. Comparisons may vary the
kinetic grid, timestep, checkpoint cadence and output precision. By default,
norms use saved fields, including storage rounding; an explicit alternative
samples verified float64 observations at the same output times.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from vpml.datasets.checkpoints import CheckpointStore
from vpml.datasets.manifest import Numerics, read_manifest
from vpml.datasets.reader import read_case
from vpml.datasets.runner import sampled_outputs
from vpml.io.serialization import file_digest, json_digest, read_json


_FIELDS = (("n-1", "n", 1.0), ("u", "u", 0.0), ("p-1", "p", 1.0),
           ("Q", "Q", 0.0), ("dQ_dx", "g", 0.0), ("E", "E", 0.0))
_VARIABLE_SETTINGS = {"nx", "nv", "dt", "block_steps", "storage_dtype"}
_REPRESENTATIONS = ("saved", "observations-float64")


def _comparison_fields(root, case, numerics, representation):
    if representation == "saved":
        return {key: case[key] for _, key, _ in _FIELDS}, None
    directory = root / "cases" / case["case_id"]
    progress_digest = file_digest(directory / "progress.json")
    protocol = read_json(directory / "protocol.json")
    store = CheckpointStore(directory, protocol, read_only=True)
    progress = store.read_progress()
    if progress["status"] != "complete" or protocol["numerics"] != numerics.to_dict():
        raise ValueError("Observation protocol changed after completed-output verification")
    history = store.histories()  # Checks every observation, restart and output bundle.
    expected = {"fields": (numerics.nsteps + 1, 5, numerics.output_nx),
                "raw_moments": (numerics.nsteps + 1, 3, numerics.output_nx),
                "electric": (numerics.nsteps + 1, numerics.output_nx),
                "diagnostics": (numerics.nsteps + 1, len(case["diagnostic_names"]))}
    if set(history) != set(expected) or any(
            history[key].shape != shape or history[key].dtype != np.dtype("float64")
            for key, shape in expected.items()):
        raise ValueError("Observation history does not match its float64 numerical schema")
    provenance_digests = set()
    previous = 0
    for index, block in enumerate(progress["blocks"]):
        record = block["observations"]
        count = 1 if index == 0 else block["step"] - previous
        if any(array["shape"][0] != count for array in record["arrays"].values()):
            raise ValueError("Observation sample count disagrees with checkpoint steps")
        previous = block["step"]
        digest = record.get("provenance_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("Observation bundle has no valid provenance digest")
        provenance_digests.add(digest)
    for digest in provenance_digests:
        provenance = read_json(root / "provenance" / f"{digest}.json")
        if (json_digest(provenance) != digest
                or provenance["source"]["compatibility"] != protocol["source"]
                or provenance["runtime"] != protocol["runtime"]):
            raise ValueError("Observation provenance disagrees with its case protocol")
    arrays = sampled_outputs(history, replace(numerics, storage_dtype="float64"))
    if (not np.array_equal(arrays["times"], case["times"])
            or file_digest(directory / "progress.json") != progress_digest):
        raise ValueError("Observation sampling or progress changed during comparison")
    fields = {key: arrays["fields"][:, index]
              for index, key in enumerate(("n", "u", "p", "Q", "g"))}
    fields["E"] = arrays["electric"]
    return fields, sorted(provenance_digests)


def _spatial_rms(values: np.ndarray) -> np.ndarray:
    """Scale before squaring, so tiny nonzero references remain nonzero."""
    scale = np.max(np.abs(values), axis=1)
    normalized = np.divide(values, scale[:, None], out=np.zeros_like(values),
                           where=scale[:, None] != 0)
    return scale * np.sqrt(np.mean(normalized ** 2, axis=1))


def _window_rms(values, times, start, end):
    lower = max(0, int(np.searchsorted(times, start, side="right")) - 1)
    upper = min(len(times), int(np.searchsorted(times, end, side="left")) + 1)
    values, times = values[lower:upper], times[lower:upper]
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    squares = np.mean((values / scale) ** 2, axis=1)
    interior = (times > start) & (times < end)
    coordinates = np.concatenate(([start], times[interior], [end]))
    integrand = np.concatenate(([np.interp(start, times, squares)], squares[interior],
                                [np.interp(end, times, squares)]))
    return float(scale * np.sqrt(np.trapezoid(integrand, coordinates) / (end - start)))


def _boundary_row(values, times, point):
    upper = int(np.searchsorted(times, point))
    if upper == 0 or times[upper] == point:
        return values[upper]
    fraction = (point - times[upper - 1]) / (times[upper] - times[upper - 1])
    return (1 - fraction) * values[upper - 1] + fraction * values[upper]


def _field_window(candidate, reference, times, start, end, tolerance):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        difference = candidate - reference
        error_rms = _window_rms(difference, times, start, end)
        reference_rms = _window_rms(reference, times, start, end)
        selected = (times >= start) & (times <= end)
        errors = _spatial_rms(difference[selected]) if selected.any() else np.empty(0)
        references = _spatial_rms(reference[selected]) if selected.any() else np.empty(0)
        nonzero = references != 0
        zero_mismatch_count = int(np.count_nonzero((~nonzero) & (errors != 0)))
        ratios = errors[nonzero] / references[nonzero]
        relative = error_rms / reference_rms if reference_rms else None
        max_relative = float(np.max(ratios)) if len(ratios) else None
        edge_errors = np.stack((_boundary_row(difference, times, start),
                                _boundary_row(difference, times, end)))
        maximum = float(np.max(np.abs(edge_errors)))
        if selected.any():
            maximum = max(maximum, float(np.max(np.abs(difference[selected]))))
    if not all(np.isfinite(value) for value in (error_rms, reference_rms, maximum)):
        raise ValueError("Nonfinite comparison metrics")
    if any(value is not None and not np.isfinite(value) for value in (relative, max_relative)):
        raise ValueError("Nonfinite relative comparison metrics")
    exact_zero = reference_rms == 0 and error_rms == 0
    inconclusive = (reference_rms == 0 and not exact_zero) or zero_mismatch_count > 0
    passed = not inconclusive and (exact_zero or relative <= tolerance)
    status = ("inconclusive" if inconclusive else "exact_zero_match" if exact_zero
              else "passed" if passed else "exceeds_tolerance")
    return {
        "relative_L2": relative, "absolute_RMS": error_rms, "absolute_max": maximum,
        "reference_RMS": reference_rms, "max_per_time_relative_L2": max_relative,
        "per_time_sample_count": int(selected.sum()),
        "per_time_nonzero_reference_count": int(nonzero.sum()),
        "per_time_zero_reference_count": int((~nonzero).sum()),
        "per_time_zero_reference_nonzero_error_count": zero_mismatch_count,
        "tolerance_passed": bool(passed), "status": status,
    }


def compare_dataset_cases(candidate_root, reference_root, case_ids, tolerance=0.005,
                          *, representation="saved") -> dict:
    """Return strict-JSON sensitivity metrics; never certify reference accuracy.

    Relative L2 acceptance applies separately to the full horizon and each
    third. Per-time maxima are additional diagnostics. A zero reference at a
    saved time with a nonzero error makes its windows inconclusive; exact zero
    matches pass without inventing a relative denominator.

    ``observations-float64`` bypasses output storage rounding by resampling the
    original verified histories. Neither mode modifies manifests or relaxes
    checkpoint compatibility for continuation.
    """
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"representation must be one of {_REPRESENTATIONS}")
    if isinstance(case_ids, (str, bytes)):
        raise ValueError("case_ids must be a nonempty sequence of unique IDs")
    case_ids = list(case_ids)
    if (not case_ids or any(not isinstance(value, str) for value in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise ValueError("case_ids must be a nonempty sequence of unique IDs")
    roots = (Path(candidate_root), Path(reference_root))
    manifests = [read_manifest(root / "manifest.json") for root in roots]
    identities = [{key: value for key, value in manifest.items() if key != "numerics"}
                  for manifest in manifests]
    if identities[0] != identities[1]:
        raise ValueError("Frozen manifests disagree on initial conditions, amplitudes or schema")
    numerics = [manifest["numerics"] for manifest in manifests]
    fixed = [{key: value for key, value in settings.items() if key not in _VARIABLE_SETTINGS}
             for settings in numerics]
    if fixed[0] != fixed[1]:
        raise ValueError("Numerical protocols may differ only in nx, nv, dt, block_steps and storage_dtype")
    differences = {key: {"candidate": numerics[0][key], "reference": numerics[1][key]}
                   for key in sorted(_VARIABLE_SETTINGS) if numerics[0][key] != numerics[1][key]}
    known = {case["case_id"] for case in manifests[0]["cases"]}
    if not set(case_ids) <= known:
        raise ValueError("Unknown manifest case ID")
    final_time = numerics[0]["final_time"]
    bounds = np.linspace(0.0, final_time, 4)
    windows = [("full", 0.0, final_time)] + [
        (f"third_{index + 1}", float(bounds[index]), float(bounds[index + 1]))
        for index in range(3)]
    results = []
    for case_id in case_ids:
        candidate, reference = [read_case(root, case_id) for root in roots]
        if (not candidate["output_integrity_verified"] or not reference["output_integrity_verified"]
                or candidate["format"] != reference["format"]
                or candidate["diagnostic_names"] != reference["diagnostic_names"]
                or not np.array_equal(candidate["times"], reference["times"])
                or not np.array_equal(candidate["x"], reference["x"])):
            raise ValueError("Verified output schemas or coordinates disagree")
        times = np.asarray(reference["times"], dtype=np.float64)
        if times[0] != 0 or times[-1] != final_time or np.any(np.diff(times) <= 0):
            raise ValueError("Saved times do not cover the full horizon")
        (left_fields, left_provenance), (right_fields, right_provenance) = [
            _comparison_fields(root, case, Numerics(**settings), representation)
            for root, case, settings in zip(roots, (candidate, reference), numerics, strict=True)]
        fields = []
        for label, key, background in _FIELDS:
            left = np.asarray(left_fields[key], dtype=np.float64) - background
            right = np.asarray(right_fields[key], dtype=np.float64) - background
            expected_shape = (len(times), numerics[0]["output_nx"])
            if (left.shape != expected_shape or right.shape != expected_shape
                    or not np.isfinite(left).all() or not np.isfinite(right).all()):
                raise ValueError("Invalid retained field shape or values")
            fields.append((label, left, right))
        comparisons = []
        for name, start, end in windows:
            try:
                metrics = {label: _field_window(left, right, times, start, end, tolerance)
                           for label, left, right in fields}
            except (FloatingPointError, OverflowError) as exc:
                raise ValueError("Nonfinite comparison metrics") from exc
            comparisons.append({"name": name, "start": start, "end": end, "fields": metrics,
                                "all_tolerances_passed": all(m["tolerance_passed"] for m in metrics.values())})
        results.append({
            "case_id": case_id, "windows": comparisons,
            "all_tolerances_passed": all(window["all_tolerances_passed"] for window in comparisons),
            "candidate_diagnostic_checks_passed": bool(candidate["report"]["checks_passed"]),
            "reference_diagnostic_checks_passed": bool(reference["report"]["checks_passed"]),
            "candidate_diagnostic_failures": list(candidate["report"]["failures"]),
            "reference_diagnostic_failures": list(reference["report"]["failures"]),
            "candidate_provenance_sha256": candidate["provenance_sha256"],
            "reference_provenance_sha256": reference["provenance_sha256"],
            "candidate_observation_provenance_sha256": left_provenance,
            "reference_observation_provenance_sha256": right_provenance,
        })
    inconclusive = any(metric["status"] == "inconclusive" for case in results
                       for window in case["windows"] for metric in window["fields"].values())
    passed = all(case["all_tolerances_passed"] for case in results)
    return {
        "schema_version": 1, "candidate_root": str(roots[0].resolve()),
        "reference_root": str(roots[1].resolve()), "case_ids": case_ids,
        "candidate_manifest_sha256": json_digest(manifests[0]),
        "reference_manifest_sha256": json_digest(manifests[1]),
        "candidate_numerics": numerics[0], "reference_numerics": numerics[1],
        "protocol_differences": differences, "representation": representation,
        "saved_times": {"start": 0.0, "end": final_time,
                        "spacing": numerics[0]["save_dt"],
                        "count": Numerics(**numerics[0]).nsamples},
        "tolerance": float(tolerance), "cases": results,
        "all_tolerances_passed": passed,
        "status": "inconclusive" if inconclusive else "passed" if passed else "exceeds_tolerance",
        "reference_uncertainty_assessed": False, "training_qualified": False,
        "output_integrity_verified": True,
        "observation_history_integrity_verified": representation == "observations-float64",
        "scope": "Retained-field differences only; neither dataset is certified as an accurate reference.",
        "norm_definition": "Trapezoid time integral of spatial mean square, divided by window duration; equal spatial weights. Window edges interpolate mean-square values between saved times.",
        "zero_reference_policy": "No denominator floor. Relative errors are null for zero reference norms. Exact zero matches pass; nonzero differences are inconclusive.",
        "acceptance": "Every field and window must meet relative L2 tolerance, with no nonzero error at a zero-reference saved time. Per-time maxima are reported separately.",
        "storage_rounding_included": representation == "saved",
        "storage_rounding_error_separately_assessed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--case-id", required=True, action="append", dest="case_ids")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--representation", choices=_REPRESENTATIONS, default="saved",
                        help="Saved outputs (default) or verified float64 observations sampled at output times")
    parser.add_argument("--out", "--out-report", type=Path, dest="out")
    args = parser.parse_args(argv)
    report = compare_dataset_cases(args.candidate, args.reference, args.case_ids, args.tolerance,
                                   representation=args.representation)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(payload)
    else:
        print(payload, end="")
    return 0 if report["all_tolerances_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
