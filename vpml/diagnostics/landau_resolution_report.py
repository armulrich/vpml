"""Combine physical-grid and projection-quadrature Landau diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


def _first_projection_grid_passing_all_cases(
    refinement_summary: Dict[str, Dict[str, Dict[str, object]]],
) -> Optional[int]:
    if not refinement_summary:
        return None
    common_grids = set.intersection(
        *(
            {int(grid) for grid in by_grid}
            for by_grid in refinement_summary.values()
        )
    )
    ordered_grids = sorted(common_grids)
    for grid_idx, refined_grid in enumerate(ordered_grids):
        candidate_and_finer = ordered_grids[grid_idx:]
        if all(
            bool(by_grid[str(grid)]["passes_one_percent_change"])
            for by_grid in refinement_summary.values()
            for grid in candidate_and_finer
        ):
            return int(refined_grid)
    return None


def build_report(
    *,
    physical_payload: Dict[str, object],
    projection_payload: Dict[str, object],
) -> Tuple[Dict[str, object], str]:
    physical_recommendation = dict(physical_payload["recommendation"])
    projection_summary = dict(
        projection_payload["successive_refinement_summary"]
    )
    certified_projection_nv = _first_projection_grid_passing_all_cases(
        projection_summary
    )
    projection_grids = tuple(
        int(value) for value in projection_payload["projection_quadrature_Nv"]
    )
    followup_projection_nv = (
        certified_projection_nv
        if certified_projection_nv is not None
        else max(projection_grids)
    )
    physical_certified = bool(
        physical_recommendation[
            "finest_pair_passes_tolerance_for_all_cases"
        ]
    )
    physical_followup_nv = int(
        physical_recommendation["physical_Nv_for_followup"]
    )
    projection_teacher = dict(projection_payload["teacher"])
    projection_source_physical_nv = int(projection_teacher["Nv"])
    if projection_source_physical_nv != physical_followup_nv:
        raise ValueError(
            "Projection quadrature evidence must use snapshots from the "
            "recommended physical teacher grid: "
            f"projection source Nv={projection_source_physical_nv}, "
            f"recommended physical Nv={physical_followup_nv}"
        )
    teacher = dict(physical_payload["teacher"])
    combined = {
        "diagnostic": "landau_resolution_selection",
        "T_final": float(teacher["T_final"]),
        "recommended_training_parameters": {
            "TEACHER_NV": physical_followup_nv,
            "TEACHER_PROJECTION_NV": int(followup_projection_nv),
        },
        "physical_velocity_grid": {
            "successive_change_gate_passes": physical_certified,
            "finest_tested_Nv": int(
                physical_recommendation["finest_physical_Nv_tested"]
            ),
            "qualification": physical_recommendation["qualification"],
        },
        "projection_quadrature": {
            "source_physical_Nv": projection_source_physical_nv,
            "matches_recommended_physical_Nv": True,
            "one_percent_successive_change_gate_passes": (
                certified_projection_nv is not None
            ),
            "first_certified_Nv": certified_projection_nv,
            "finest_tested_Nv": max(projection_grids),
        },
        "interpretation": (
            "Projection quadrature and physical teacher resolution are separate "
            "error sources. A certified projection grid does not certify the "
            "physical velocity discretization."
        ),
    }
    physical_status = (
        "the finest tested grid and passes the successive-change gate"
        if physical_certified
        else "the finest tested grid, but does not pass the successive-change gate"
    )
    projection_status = (
        "the first projection grid passing the one-percent successive-change "
        "gate for every representative case"
        if certified_projection_nv is not None
        else "the finest projection grid tested; no grid passed the gate for every case"
    )
    markdown = f"""# Landau T={float(teacher['T_final']):g} resolution diagnostic

Recommended follow-up parameters:

```bash
TEACHER_NV={physical_followup_nv}
TEACHER_PROJECTION_NV={int(followup_projection_nv)}
```

- `TEACHER_NV={physical_followup_nv}` is {physical_status}.
- `TEACHER_PROJECTION_NV={int(followup_projection_nv)}` is {projection_status}
  on fixed `TEACHER_NV={projection_source_physical_nv}` snapshots.
- These are successive-grid self-convergence checks, not discretization-error estimates or comparisons against an analytic solution.
- The projection result cannot compensate for an under-resolved physical teacher.
"""
    return combined, markdown


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine Landau physical-grid and projection diagnostics."
    )
    parser.add_argument("--physical-json", type=Path, required=True)
    parser.add_argument("--projection-json", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    physical_payload = json.loads(args.physical_json.read_text())
    projection_payload = json.loads(args.projection_json.read_text())
    combined, markdown = build_report(
        physical_payload=physical_payload,
        projection_payload=projection_payload,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "landau_resolution_recommendation.json"
    markdown_path = args.outdir / "README.md"
    json_path.write_text(json.dumps(combined, indent=2) + "\n")
    markdown_path.write_text(markdown)
    print(f"Saved combined Landau resolution recommendation to {json_path}")
    print(f"Saved human-readable Landau resolution report to {markdown_path}")


if __name__ == "__main__":
    main()
