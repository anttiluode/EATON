from eaton.core import FrozenConfig
from eaton.experiment import classify, run_seed


def test_seed_keeps_matched_arm_invariants_separate_from_stability():
    r = run_seed(0, FrozenConfig())
    assert r["matched_invariants"]["stage1_transient_tonic_equal"] is True
    assert r["matched_invariants"]["stage1_transient_reset_equal"] is True
    assert r["matched_invariants"]["event_counts_equal"] is True
    assert r["matched_invariants"]["event_amplitude_sums_equal"] is True
    assert isinstance(r["numerical_checks"]["all_finite"], bool)
    assert isinstance(r["numerical_checks"]["all_bounded"], bool)


def test_seed_records_all_paired_episode_outcomes():
    r = run_seed(0, FrozenConfig())
    for key in ("transient_vs_tonic", "transient_vs_reset"):
        counts = r["paired_episode_disagreements"][key]
        assert sum(counts.values()) == 256
        assert set(counts) == {
            "transient_only_correct",
            "attacker_only_correct",
            "both_correct",
            "both_wrong",
        }


def passing_summary():
    return {
        "matched_invariants_valid": True,
        "median_early_accuracy": {"transient": 1.0, "tonic": 1.0, "reset": 1.0},
        "median_final_accuracy": {"transient": 0.98, "tonic": 0.50, "reset": 0.50},
        "median_transient_minus_tonic": 0.48,
        "median_transient_minus_reset": 0.48,
        "transient_seed_wins_vs_tonic": 64,
        "transient_seed_wins_vs_reset": 64,
        "all_finite": True,
        "max_abs_state": 0.99,
    }


def test_classifier_matched_failure_has_highest_precedence():
    s = passing_summary()
    s["matched_invariants_valid"] = False
    s["max_abs_state"] = 99.0
    assert classify(s, FrozenConfig()) == "INVALID_MATCHED_ARMS"


def test_classifier_tonic_failure_precedes_reset_failure():
    s = passing_summary()
    s["median_transient_minus_tonic"] = 0.19
    s["median_transient_minus_reset"] = 0.0
    assert classify(s, FrozenConfig()) == "NO_HANDOFF_ADVANTAGE"


def test_classifier_reset_failure():
    s = passing_summary()
    s["median_transient_minus_reset"] = 0.29
    assert classify(s, FrozenConfig()) == "HANDOFF_WITHOUT_RESIDENT_MEMORY"


def test_classifier_instability_is_not_misreported_as_matched_failure():
    s = passing_summary()
    s["max_abs_state"] = 1.1
    assert classify(s, FrozenConfig()) == "UNSTABLE_OR_WEAK_TRANSIENT"


def test_classifier_pass():
    assert classify(passing_summary(), FrozenConfig()) == "PASS_TRANSIENT_HANDOFF"
