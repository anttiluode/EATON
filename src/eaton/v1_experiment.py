from __future__ import annotations

from collections import Counter
import json
from dataclasses import asdict, dataclass, field
from itertools import permutations
from statistics import median

import numpy as np

from .coordinates import (
    CoordinateConfig,
    KMeansModel,
    analytic_response_signatures,
    assign_clusters,
    fit_kmeans,
    fit_response_reconstruction,
    make_probe_bank,
    response_nrmse,
    response_signatures,
)
from .grammar import (
    GaussianClusterModel,
    TransitionModel,
    fit_gaussian_cluster_model,
    fit_transition_model,
    fit_world_readout,
    gaussian_nll,
    mdl_score,
    time_bin_labels,
    transition_nll,
    world_readout_metrics,
)
from .grammar_world import WorldBatch, WorldConfig, current_symbol_baseline, generate_world_split
from .rnn import RNNConfig, RNNParams, rollout_states, sequence_metrics, train_rnn

PASS_THRESHOLDS = {
    "minimum_valid_structured_seeds": 6,
    "structured_accuracy_min": 0.80,
    "structured_accuracy_gain_over_current_symbol_min": 0.25,
    "response_vs_activation_nrmse_ratio_median_max": 0.90,
    "response_vs_activation_seed_win_fraction_min": 0.75,
    "grammar_gain_over_shuffled_nats_median_min": 0.10,
    "grammar_gain_over_timebin_nats_median_min": 0.05,
    "grammar_seed_win_fraction_min": 0.75,
    "collapsed_world_accuracy_median_min": 0.75,
    "collapsed_world_accuracy_gain_over_current_symbol_min": 0.25,
    "structured_minus_volatile_grammar_gain_median_min": 0.05,
    "finite_difference_relative_error_median_max": 1e-3,
}


@dataclass(frozen=True)
class V1Config:
    seed_count: int = 8
    world: WorldConfig = field(default_factory=WorldConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    coordinates: CoordinateConfig = field(default_factory=CoordinateConfig)
    pass_thresholds: dict[str, float | int] = field(default_factory=lambda: dict(PASS_THRESHOLDS))


@dataclass(frozen=True)
class CoordinateSelection:
    selected_k: int
    model: KMeansModel
    train_labels: np.ndarray
    validation_labels: np.ndarray
    gaussian_model: GaussianClusterModel
    transition_model: TransitionModel
    mdl_scores: dict[int, float]


def _check_feature_grid(features: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    s = np.asarray(symbols, dtype=np.int64)
    if x.ndim != 3 or s.ndim != 2 or x.shape[:2] != s.shape:
        raise ValueError("features must be (N,T,D) aligned to symbols (N,T)")
    return x, s


def select_coordinate_model(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_symbols: np.ndarray,
    validation_symbols: np.ndarray,
    coordinate_config: CoordinateConfig,
    seed: int,
) -> CoordinateSelection:
    train_features, train_symbols = _check_feature_grid(train_features, train_symbols)
    validation_features, validation_symbols = _check_feature_grid(validation_features, validation_symbols)
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    val_flat = validation_features.reshape(-1, validation_features.shape[-1])
    best: tuple[float, int, CoordinateSelection] | None = None
    scores: dict[int, float] = {}

    for k in coordinate_config.k_candidates:
        if k > train_flat.shape[0]:
            continue
        model = fit_kmeans(train_flat, k, seed=seed + 911 * k, config=coordinate_config)
        train_labels = model.labels.reshape(train_features.shape[:2])
        val_labels = assign_clusters(model, val_flat).reshape(validation_features.shape[:2])
        train_scaled = model.scaler.transform(train_flat)
        val_scaled = model.scaler.transform(val_flat)
        gaussian = fit_gaussian_cluster_model(
            train_scaled,
            model.labels,
            k,
            coordinate_config.gaussian_variance_floor,
        )
        transition = fit_transition_model(
            train_labels,
            train_symbols,
            k,
            4,
            coordinate_config.transition_smoothing,
        )
        feature_sum = gaussian_nll(gaussian, val_scaled, val_labels.ravel(), reduction="sum")
        transition_sum = transition_nll(transition, val_labels, validation_symbols, reduction="sum")
        score = mdl_score(
            validation_feature_nll_sum=feature_sum,
            validation_transition_nll_sum=transition_sum,
            k=k,
            feature_dim=train_flat.shape[1],
            alphabet_size=4,
            n_train_points=train_flat.shape[0],
        )
        scores[int(k)] = float(score)
        selection = CoordinateSelection(
            selected_k=int(k),
            model=model,
            train_labels=train_labels,
            validation_labels=val_labels,
            gaussian_model=gaussian,
            transition_model=transition,
            mdl_scores={},
        )
        candidate = (float(score), int(k), selection)
        if np.isfinite(score) and (best is None or candidate[:2] < best[:2]):
            best = candidate

    if best is None:
        raise RuntimeError("no finite coordinate model candidate")
    selected = best[2]
    return CoordinateSelection(
        selected_k=selected.selected_k,
        model=selected.model,
        train_labels=selected.train_labels,
        validation_labels=selected.validation_labels,
        gaussian_model=selected.gaussian_model,
        transition_model=selected.transition_model,
        mdl_scores=scores,
    )


def _response_grid(params: RNNParams, batch: WorldBatch, probes: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    states = rollout_states(params, batch.symbols)
    n, t, h = states.shape
    current = batch.symbols[:, :-1]
    responses = response_signatures(
        params,
        states.reshape(-1, h),
        current.reshape(-1),
        probes,
        epsilon,
    ).reshape(n, t, -1)
    return states, responses


def _shuffle_grid(features: np.ndarray, seed: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    perm = np.random.default_rng(seed).permutation(flat.shape[0])
    return flat[perm].reshape(x.shape)


def _evaluate_family(
    selection: CoordinateSelection,
    test_features: np.ndarray,
    train_responses: np.ndarray,
    test_responses: np.ndarray,
    train_symbols_full: np.ndarray,
    test_symbols_full: np.ndarray,
    coordinate_config: CoordinateConfig,
) -> dict[str, float | int | np.ndarray]:
    k = selection.selected_k
    test_labels = assign_clusters(selection.model, test_features.reshape(-1, test_features.shape[-1])).reshape(test_features.shape[:2])
    response_means = fit_response_reconstruction(selection.train_labels.ravel(), train_responses.reshape(-1, train_responses.shape[-1]), k)
    recon = response_means[test_labels.ravel()]
    nrmse = response_nrmse(test_responses.reshape(-1, test_responses.shape[-1]), recon)
    transition = fit_transition_model(
        selection.train_labels,
        train_symbols_full[:, :-1],
        k,
        4,
        coordinate_config.transition_smoothing,
    )
    trans_nll = transition_nll(transition, test_labels, test_symbols_full[:, :-1])
    readout = fit_world_readout(selection.train_labels, train_symbols_full, k, 4, coordinate_config.transition_smoothing)
    world = world_readout_metrics(readout, test_labels, test_symbols_full)
    return {
        "selected_k": k,
        "test_labels": test_labels,
        "response_nrmse": float(nrmse),
        "transition_nll": float(trans_nll),
        "world_accuracy": float(world["accuracy"]),
        "world_nll": float(world["nll"]),
    }


def _posthoc_rule_metrics(labels: np.ndarray, rules: np.ndarray) -> tuple[float, float]:
    z = np.asarray(labels, dtype=np.int64).ravel()
    r = np.asarray(rules, dtype=np.int64).ravel()
    if len(z) != len(r):
        raise ValueError("labels and rules must align")
    kz = int(z.max()) + 1
    kr = int(r.max()) + 1
    joint = np.zeros((kz, kr), dtype=np.float64)
    np.add.at(joint, (z, r), 1.0)
    joint /= joint.sum()
    pz = joint.sum(axis=1, keepdims=True)
    pr = joint.sum(axis=0, keepdims=True)
    expected = pz @ pr
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    if kz == kr and kz <= 8:
        best = 0.0
        counts = joint * len(z)
        for perm in permutations(range(kr)):
            score = sum(counts[i, perm[i]] for i in range(kz)) / len(z)
            best = max(best, float(score))
    else:
        counts = joint * len(z)
        best = float(np.sum(np.max(counts, axis=1)) / len(z))
    return mi, best


def _run_response_pipeline(
    params: RNNParams,
    train: WorldBatch,
    validation: WorldBatch,
    test: WorldBatch,
    cfg: V1Config,
    seed: int,
    *,
    shuffled: bool,
) -> tuple[CoordinateSelection, dict[str, float | int | np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    probes = make_probe_bank(cfg.rnn.hidden_size, cfg.coordinates.probe_count, seed + 40_000)
    train_states, train_resp = _response_grid(params, train, probes, cfg.coordinates.probe_epsilon)
    val_states, val_resp = _response_grid(params, validation, probes, cfg.coordinates.probe_epsilon)
    test_states, test_resp = _response_grid(params, test, probes, cfg.coordinates.probe_epsilon)
    if shuffled:
        train_features = _shuffle_grid(train_resp, seed + 70_000 + 1000)
        val_features = _shuffle_grid(val_resp, seed + 70_000 + 2000)
        test_features = _shuffle_grid(test_resp, seed + 70_000 + 3000)
    else:
        train_features, val_features, test_features = train_resp, val_resp, test_resp
    selection = select_coordinate_model(
        train_features,
        val_features,
        train.symbols[:, :-1],
        validation.symbols[:, :-1],
        cfg.coordinates,
        seed + (20_000 if shuffled else 10_000),
    )
    evaluated = _evaluate_family(
        selection,
        test_features,
        train_resp,
        test_resp,
        train.symbols,
        test.symbols,
        cfg.coordinates,
    )
    return selection, evaluated, train_resp, val_resp, test_resp


def run_v1_seed(seed: int, config: V1Config | None = None) -> dict:
    cfg = V1Config() if config is None else config
    train = generate_world_split(cfg.world, seed, "train", False)
    validation = generate_world_split(cfg.world, seed, "validation", False)
    test = generate_world_split(cfg.world, seed, "test", False)
    trained = train_rnn(train, cfg.rnn, seed=seed)
    structured_metrics = sequence_metrics(trained.params, test.symbols)
    baseline = current_symbol_baseline(train, test)
    training_gain = structured_metrics["accuracy"] - baseline["accuracy"]
    training_valid = (
        structured_metrics["accuracy"] >= cfg.pass_thresholds["structured_accuracy_min"]
        and training_gain >= cfg.pass_thresholds["structured_accuracy_gain_over_current_symbol_min"]
    )

    probes = make_probe_bank(cfg.rnn.hidden_size, cfg.coordinates.probe_count, seed + 40_000)
    train_states, train_resp = _response_grid(trained.params, train, probes, cfg.coordinates.probe_epsilon)
    val_states, val_resp = _response_grid(trained.params, validation, probes, cfg.coordinates.probe_epsilon)
    test_states, test_resp = _response_grid(trained.params, test, probes, cfg.coordinates.probe_epsilon)

    response_selection = select_coordinate_model(
        train_resp,
        val_resp,
        train.symbols[:, :-1],
        validation.symbols[:, :-1],
        cfg.coordinates,
        seed + 10_000,
    )
    response_eval = _evaluate_family(
        response_selection,
        test_resp,
        train_resp,
        test_resp,
        train.symbols,
        test.symbols,
        cfg.coordinates,
    )

    activation_selection = select_coordinate_model(
        train_states,
        val_states,
        train.symbols[:, :-1],
        validation.symbols[:, :-1],
        cfg.coordinates,
        seed + 30_000,
    )
    activation_eval = _evaluate_family(
        activation_selection,
        test_states,
        train_resp,
        test_resp,
        train.symbols,
        test.symbols,
        cfg.coordinates,
    )

    shuffled_train = _shuffle_grid(train_resp, seed + 71_000)
    shuffled_val = _shuffle_grid(val_resp, seed + 72_000)
    shuffled_test = _shuffle_grid(test_resp, seed + 73_000)
    shuffled_selection = select_coordinate_model(
        shuffled_train,
        shuffled_val,
        train.symbols[:, :-1],
        validation.symbols[:, :-1],
        cfg.coordinates,
        seed + 20_000,
    )
    shuffled_eval = _evaluate_family(
        shuffled_selection,
        shuffled_test,
        train_resp,
        test_resp,
        train.symbols,
        test.symbols,
        cfg.coordinates,
    )

    k = response_selection.selected_k
    train_time = time_bin_labels(train.symbols.shape[0], cfg.world.sequence_length, k)
    test_time = time_bin_labels(test.symbols.shape[0], cfg.world.sequence_length, k)
    time_transition = fit_transition_model(
        train_time,
        train.symbols[:, :-1],
        k,
        4,
        cfg.coordinates.transition_smoothing,
    )
    timebin_nll = transition_nll(time_transition, test_time, test.symbols[:, :-1])

    exact = analytic_response_signatures(
        trained.params,
        test_states.reshape(-1, cfg.rnn.hidden_size),
        test.symbols[:, :-1].reshape(-1),
        probes,
    )
    fd = test_resp.reshape(-1, test_resp.shape[-1])
    fd_relative_error = float(np.linalg.norm(fd - exact) / max(np.linalg.norm(exact), 1e-300))
    rule_mi, rule_agreement = _posthoc_rule_metrics(response_eval["test_labels"], test.rules)

    vtrain = generate_world_split(cfg.world, seed, "train", True)
    vvalidation = generate_world_split(cfg.world, seed, "validation", True)
    vtest = generate_world_split(cfg.world, seed, "test", True)
    vtrained = train_rnn(vtrain, cfg.rnn, seed=seed)
    vprobes = make_probe_bank(cfg.rnn.hidden_size, cfg.coordinates.probe_count, seed + 40_000)
    _, vtrain_resp = _response_grid(vtrained.params, vtrain, vprobes, cfg.coordinates.probe_epsilon)
    _, vval_resp = _response_grid(vtrained.params, vvalidation, vprobes, cfg.coordinates.probe_epsilon)
    _, vtest_resp = _response_grid(vtrained.params, vtest, vprobes, cfg.coordinates.probe_epsilon)
    vsel = select_coordinate_model(
        vtrain_resp,
        vval_resp,
        vtrain.symbols[:, :-1],
        vvalidation.symbols[:, :-1],
        cfg.coordinates,
        seed + 10_000,
    )
    veval = _evaluate_family(vsel, vtest_resp, vtrain_resp, vtest_resp, vtrain.symbols, vtest.symbols, cfg.coordinates)
    vsh_train = _shuffle_grid(vtrain_resp, seed + 171_000)
    vsh_val = _shuffle_grid(vval_resp, seed + 172_000)
    vsh_test = _shuffle_grid(vtest_resp, seed + 173_000)
    vsh_sel = select_coordinate_model(
        vsh_train,
        vsh_val,
        vtrain.symbols[:, :-1],
        vvalidation.symbols[:, :-1],
        cfg.coordinates,
        seed + 120_000,
    )
    vsh_eval = _evaluate_family(vsh_sel, vsh_test, vtrain_resp, vtest_resp, vtrain.symbols, vtest.symbols, cfg.coordinates)

    return {
        "seed": int(seed),
        "training_valid": bool(training_valid),
        "structured_rnn_accuracy": float(structured_metrics["accuracy"]),
        "structured_rnn_nll": float(structured_metrics["nll"]),
        "current_symbol_world_accuracy": float(baseline["accuracy"]),
        "current_symbol_world_nll": float(baseline["nll"]),
        "structured_accuracy_gain_over_current_symbol": float(training_gain),
        "response_selected_k": int(response_selection.selected_k),
        "activation_selected_k": int(activation_selection.selected_k),
        "shuffled_selected_k": int(shuffled_selection.selected_k),
        "response_nrmse": float(response_eval["response_nrmse"]),
        "activation_nrmse": float(activation_eval["response_nrmse"]),
        "response_transition_nll": float(response_eval["transition_nll"]),
        "shuffled_transition_nll": float(shuffled_eval["transition_nll"]),
        "timebin_transition_nll": float(timebin_nll),
        "response_world_accuracy": float(response_eval["world_accuracy"]),
        "activation_world_accuracy": float(activation_eval["world_accuracy"]),
        "volatile_response_transition_nll": float(veval["transition_nll"]),
        "volatile_shuffled_transition_nll": float(vsh_eval["transition_nll"]),
        "volatile_selected_k": int(vsel.selected_k),
        "finite_difference_relative_error": fd_relative_error,
        "posthoc_rule_mutual_information": float(rule_mi),
        "posthoc_best_rule_agreement": float(rule_agreement),
    }


def _med(values: list[float]) -> float:
    return float(median(values))


def aggregate_v1(seed_results: list[dict], config: V1Config | None = None) -> dict:
    cfg = V1Config() if config is None else config
    valid = [r for r in seed_results if r["training_valid"]]
    summary: dict[str, object] = {
        "valid_structured_seed_count": len(valid),
        "seed_count": len(seed_results),
        "response_selected_k_distribution": {str(k): int(v) for k, v in sorted(Counter(r["response_selected_k"] for r in valid).items())},
        "activation_selected_k_distribution": {str(k): int(v) for k, v in sorted(Counter(r["activation_selected_k"] for r in valid).items())},
        "shuffled_selected_k_distribution": {str(k): int(v) for k, v in sorted(Counter(r["shuffled_selected_k"] for r in valid).items())},
    }
    if valid:
        ratios = [r["response_nrmse"] / max(r["activation_nrmse"], 1e-300) for r in valid]
        shuf_gain = [r["shuffled_transition_nll"] - r["response_transition_nll"] for r in valid]
        time_gain = [r["timebin_transition_nll"] - r["response_transition_nll"] for r in valid]
        specificity = [
            (r["shuffled_transition_nll"] - r["response_transition_nll"])
            - (r["volatile_shuffled_transition_nll"] - r["volatile_response_transition_nll"])
            for r in valid
        ]
        summary.update(
            {
                "median_response_activation_nrmse_ratio": _med(ratios),
                "response_nrmse_seed_win_fraction": float(np.mean([r["response_nrmse"] < r["activation_nrmse"] for r in valid])),
                "median_shuffled_grammar_gain": _med(shuf_gain),
                "median_timebin_grammar_gain": _med(time_gain),
                "grammar_seed_win_fraction": float(np.mean([r["response_transition_nll"] < r["shuffled_transition_nll"] and r["response_transition_nll"] < r["timebin_transition_nll"] for r in valid])),
                "median_response_world_accuracy": _med([r["response_world_accuracy"] for r in valid]),
                "median_response_world_accuracy_gain_over_current_symbol": _med([r["response_world_accuracy"] - r["current_symbol_world_accuracy"] for r in valid]),
                "median_activation_world_accuracy": _med([r["activation_world_accuracy"] for r in valid]),
                "median_activation_world_accuracy_gain_over_current_symbol": _med([r["activation_world_accuracy"] - r["current_symbol_world_accuracy"] for r in valid]),
                "median_structured_minus_volatile_grammar_gain": _med(specificity),
                "median_finite_difference_relative_error": _med([r["finite_difference_relative_error"] for r in valid]),
                "median_response_nrmse": _med([r["response_nrmse"] for r in valid]),
                "median_activation_nrmse": _med([r["activation_nrmse"] for r in valid]),
                "median_response_transition_nll": _med([r["response_transition_nll"] for r in valid]),
                "median_shuffled_transition_nll": _med([r["shuffled_transition_nll"] for r in valid]),
                "median_timebin_transition_nll": _med([r["timebin_transition_nll"] for r in valid]),
                "median_posthoc_rule_mutual_information": _med([r["posthoc_rule_mutual_information"] for r in valid]),
                "median_posthoc_best_rule_agreement": _med([r["posthoc_best_rule_agreement"] for r in valid]),
            }
        )
    summary["classification"] = classify_v1(summary, cfg)
    return summary


def classify_v1(summary: dict, config: V1Config | None = None) -> str:
    cfg = V1Config() if config is None else config
    th = cfg.pass_thresholds
    if int(summary.get("valid_structured_seed_count", 0)) < int(th["minimum_valid_structured_seeds"]):
        return "TRAINING_FAILURE"
    pass_operator = (
        summary["median_response_activation_nrmse_ratio"] <= th["response_vs_activation_nrmse_ratio_median_max"]
        and summary["response_nrmse_seed_win_fraction"] >= th["response_vs_activation_seed_win_fraction_min"]
        and summary["median_shuffled_grammar_gain"] >= th["grammar_gain_over_shuffled_nats_median_min"]
        and summary["median_timebin_grammar_gain"] >= th["grammar_gain_over_timebin_nats_median_min"]
        and summary["grammar_seed_win_fraction"] >= th["grammar_seed_win_fraction_min"]
        and summary["median_response_world_accuracy"] >= th["collapsed_world_accuracy_median_min"]
        and summary["median_response_world_accuracy_gain_over_current_symbol"] >= th["collapsed_world_accuracy_gain_over_current_symbol_min"]
        and summary["median_structured_minus_volatile_grammar_gain"] >= th["structured_minus_volatile_grammar_gain_median_min"]
        and summary["median_finite_difference_relative_error"] <= th["finite_difference_relative_error_median_max"]
    )
    if pass_operator:
        return "PASS_COMPUTATION_COORDINATES"
    if (
        summary.get("median_activation_world_accuracy", 0.0) >= th["collapsed_world_accuracy_median_min"]
        and summary.get("median_activation_world_accuracy_gain_over_current_symbol", 0.0) >= th["collapsed_world_accuracy_gain_over_current_symbol_min"]
    ):
        return "PASS_STATE_NOT_OPERATOR"
    return "FAIL_GRAMMAR_RECOVERY"



def run_v1_canonical(config: V1Config | None = None) -> dict:
    cfg = V1Config() if config is None else config
    results = [run_v1_seed(seed, cfg) for seed in range(cfg.seed_count)]
    summary = aggregate_v1(results, cfg)
    return {
        "experiment": "EATON v1 computation coordinates",
        "config": json.loads(json.dumps(asdict(cfg))),
        "seed_results": results,
        "summary": summary,
        "classification": summary["classification"],
    }
