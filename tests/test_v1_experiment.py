import inspect
from dataclasses import replace

from eaton.coordinates import CoordinateConfig
from eaton.grammar_world import WorldConfig
from eaton.rnn import RNNConfig
from eaton.v1_experiment import V1Config, classify_v1, select_coordinate_model


def test_v1_config_is_preregistered():
    c = V1Config()
    assert c.seed_count == 8
    assert c.world.sequence_length == 48
    assert c.world.structured_switch_probability == 0.05
    assert c.rnn.hidden_size == 16
    assert c.rnn.epochs == 50
    assert c.coordinates.probe_count == 8
    assert c.coordinates.probe_epsilon == 1e-5
    assert c.coordinates.k_candidates == tuple(range(1, 9))
    assert c.pass_thresholds["response_vs_activation_nrmse_ratio_median_max"] == 0.90


def passing_summary():
    return {
        "valid_structured_seed_count": 6,
        "median_response_activation_nrmse_ratio": 0.90,
        "response_nrmse_seed_win_fraction": 0.75,
        "median_shuffled_grammar_gain": 0.10,
        "median_timebin_grammar_gain": 0.05,
        "grammar_seed_win_fraction": 0.75,
        "median_response_world_accuracy": 0.75,
        "median_response_world_accuracy_gain_over_current_symbol": 0.25,
        "median_structured_minus_volatile_grammar_gain": 0.05,
        "median_finite_difference_relative_error": 1e-3,
        "median_activation_world_accuracy": 0.75,
        "median_activation_world_accuracy_gain_over_current_symbol": 0.25,
    }


def test_training_failure_precedes_scientific_classification():
    summary = {"valid_structured_seed_count": 5}
    assert classify_v1(summary, V1Config()) == "TRAINING_FAILURE"


def test_exact_pass_boundaries_count_as_pass():
    assert classify_v1(passing_summary(), V1Config()) == "PASS_COMPUTATION_COORDINATES"


def test_each_operator_threshold_fails_just_beyond_boundary():
    cases = {
        "median_response_activation_nrmse_ratio": 0.900001,
        "response_nrmse_seed_win_fraction": 0.749999,
        "median_shuffled_grammar_gain": 0.099999,
        "median_timebin_grammar_gain": 0.049999,
        "grammar_seed_win_fraction": 0.749999,
        "median_response_world_accuracy": 0.749999,
        "median_response_world_accuracy_gain_over_current_symbol": 0.249999,
        "median_structured_minus_volatile_grammar_gain": 0.049999,
        "median_finite_difference_relative_error": 0.001001,
    }
    for key, value in cases.items():
        s = passing_summary()
        s[key] = value
        assert classify_v1(s, V1Config()) == "PASS_STATE_NOT_OPERATOR", key


def test_state_not_operator_requires_activation_readout_to_survive():
    s = passing_summary()
    s["median_response_activation_nrmse_ratio"] = 1.1
    assert classify_v1(s, V1Config()) == "PASS_STATE_NOT_OPERATOR"
    s["median_activation_world_accuracy"] = 0.74
    assert classify_v1(s, V1Config()) == "FAIL_GRAMMAR_RECOVERY"


def test_select_coordinate_model_has_no_test_argument():
    names = tuple(inspect.signature(select_coordinate_model).parameters)
    assert "test_features" not in names
    assert "test_symbols" not in names
    assert names[:4] == ("train_features", "validation_features", "train_symbols", "validation_symbols")


def test_small_config_can_be_constructed_without_mutating_frozen_defaults():
    base = V1Config()
    small = replace(
        base,
        seed_count=1,
        world=WorldConfig(train_sequences=16, validation_sequences=8, test_sequences=8, sequence_length=8),
        rnn=RNNConfig(hidden_size=6, epochs=1, batch_size=8),
        coordinates=CoordinateConfig(probe_count=3, k_candidates=(1, 2), kmeans_restarts=1, kmeans_max_iter=5),
    )
    assert small.seed_count == 1
    assert V1Config().seed_count == 8


def test_small_canonical_result_is_json_roundtrip_stable():
    import json
    from eaton.v1_experiment import run_v1_canonical
    base = V1Config()
    small = replace(
        base,
        seed_count=1,
        world=WorldConfig(train_sequences=12, validation_sequences=6, test_sequences=6, sequence_length=6),
        rnn=RNNConfig(hidden_size=5, epochs=1, batch_size=6),
        coordinates=CoordinateConfig(probe_count=2, k_candidates=(1, 2), kmeans_restarts=1, kmeans_max_iter=3),
    )
    result = run_v1_canonical(small)
    assert json.loads(json.dumps(result)) == result
