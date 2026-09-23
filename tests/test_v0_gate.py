import json
import math
from pathlib import Path

from eaton.experiment import run_canonical

RECEIPT = Path("results/v0_handoff.json")
README = Path("README.md")


def test_committed_receipt_matches_fresh_scientific_summary():
    committed = json.loads(RECEIPT.read_text(encoding="utf-8"))
    fresh = run_canonical()
    cs = committed["summary"]
    fs = fresh["summary"]
    assert committed["config"] == fresh["config"]
    assert committed["classification"] == cs["classification"] == fs["classification"]
    for key in (
        "median_early_accuracy",
        "median_final_accuracy",
        "median_transient_minus_tonic",
        "median_transient_minus_reset",
        "transient_seed_wins_vs_tonic",
        "transient_seed_wins_vs_reset",
        "matched_invariants_valid",
        "all_finite",
    ):
        assert cs[key] == fs[key]
    assert math.isclose(cs["max_abs_state"], fs["max_abs_state"], abs_tol=1e-12)
    for arm in ("transient", "tonic", "reset"):
        assert math.isclose(
            cs["median_handoff_memory_correlation"][arm],
            fs["median_handoff_memory_correlation"][arm],
            abs_tol=1e-12,
        )


def test_readme_is_grounded_in_frozen_result_and_claim_boundary():
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    assert "Event-Addressed Transient Operator Networks" in text
    assert receipt["classification"] in text
    assert "operator lifetime" in text.lower()
    assert "synthetic" in text.lower()
    assert "1.0000" in text
    assert "0.5000" in text
    assert "0.5020" in text
    assert "64 / 64" in text
