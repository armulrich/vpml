import pytest

from vpml.training.gaussian_validation import validation_windows


def test_panel_pairs_shapes_and_times_across_strengths_and_excludes_test_rows():
    cases = [
        dict(split=split, k0=k0, mode_count=3, decay=1.2, shape_id=shape, strength=strength)
        for split in ("train", "validation", "test")
        for k0 in (0.3, 0.5)
        for shape in ("second", "first")
        for strength in (0.1, 1.0)
    ]
    rows = validation_windows({"cases": cases}, 526)
    assert [(cases[i]["split"], cases[i]["shape_id"], strength, anchor)
            for strength, i, anchor in rows] == [
        ("validation", "first", 0.1, 0), ("validation", "first", 1.0, 0),
        ("validation", "first", 0.1, 525), ("validation", "first", 1.0, 525),
    ]
    cases.pop(rows[-1][1])
    with pytest.raises(ValueError, match="Incomplete strength"):
        validation_windows({"cases": cases}, 526)
