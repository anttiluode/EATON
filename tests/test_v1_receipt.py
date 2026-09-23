import json
import math
from pathlib import Path

from eaton.v1_experiment import V1Config, run_v1_canonical


def _assert_receipts_close(actual, expected, *, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_receipts_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_receipts_close(a, e, path=f"{path}[{i}]")
        return
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (
            path,
            actual,
            expected,
        )
        return
    assert actual == expected, path


def test_committed_v1_receipt_matches_fresh_canonical_run():
    expected = json.loads(Path("results/v1_coordinates.json").read_text(encoding="utf-8"))
    fresh = run_v1_canonical(V1Config())
    _assert_receipts_close(fresh, expected)


def test_readme_reports_frozen_v1_result_and_claim_boundary():
    receipt = json.loads(Path("results/v1_coordinates.json").read_text(encoding="utf-8"))
    text = Path("README.md").read_text(encoding="utf-8")
    assert receipt["classification"] in text
    assert "computation coordinate" in text.lower()
    assert "temporal grammar" in text.lower()
    assert "0.445310" in text
    assert "0.675727" in text
    assert "1.697674" in text
    assert "0.098152" in text
    assert "0.900933" in text
    assert "0.726521" in text
    assert "1.809863e-06" in text
    assert "does not establish" in text.lower()
