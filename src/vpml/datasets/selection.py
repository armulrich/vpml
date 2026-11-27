"""Pure case selection shared by preflight validation and dataset execution."""
from __future__ import annotations

from collections.abc import Sequence


def select_cases(manifest: dict, case_ids: Sequence[str] | None = None) -> list[dict]:
    """Select unique known IDs from a validated manifest, retaining its order.

    This function performs no I/O and does not import or initialize a numerical
    backend. An explicit empty selection is an error rather than an all-case run.
    """
    cases = manifest["cases"]
    if case_ids is None:
        return list(cases)
    if (not isinstance(case_ids, Sequence) or isinstance(case_ids, (str, bytes))
            or not case_ids or any(not isinstance(value, str) for value in case_ids)):
        raise ValueError("Case selection must contain unique known IDs")
    selected = set(case_ids)
    known = {case["case_id"] for case in cases}
    if len(selected) != len(case_ids) or not selected <= known:
        raise ValueError("Case selection must contain unique known IDs")
    return [case for case in cases if case["case_id"] in selected]
