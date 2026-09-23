from __future__ import annotations

from statistics import median

import numpy as np

from .core import FrozenConfig, run_episode
from .world import build_seed_tape

ARMS = ("transient", "tonic", "reset")


def _corr(a, b) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _metrics(results, tape) -> dict:
    return {
        "early_accuracy": float(np.mean([r.early_correct for r in results])),
        "final_accuracy": float(np.mean([r.final_correct for r in results])),
        "handoff_memory_correlation": _corr(
            [r.memory_before_compute for r in results],
            [e.context for e in tape],
        ),
        "operator_overlap_event_ticks": int(sum(r.overlap_event_ticks for r in results)),
        "event_count": int(sum(r.event_count for r in results)),
        "event_abs_sum": float(sum(r.event_abs_sum for r in results)),
        "max_abs_state": float(max(r.max_abs_state for r in results)),
    }


def _paired_counts(transient, attacker) -> dict:
    out = {
        "transient_only_correct": 0,
        "attacker_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for t, a in zip(transient, attacker):
        if t.final_correct and not a.final_correct:
            out["transient_only_correct"] += 1
        elif a.final_correct and not t.final_correct:
            out["attacker_only_correct"] += 1
        elif t.final_correct:
            out["both_correct"] += 1
        else:
            out["both_wrong"] += 1
    return out


def run_seed(seed: int, config: FrozenConfig) -> dict:
    tape = build_seed_tape(seed, config)
    raw = {arm: [run_episode(ep, arm, config) for ep in tape] for arm in ARMS}
    metrics = {arm: _metrics(raw[arm], tape) for arm in ARMS}

    sums = [metrics[arm]["event_abs_sum"] for arm in ARMS]
    matched = {
        "stage1_transient_tonic_equal": all(
            a.early_state == b.early_state and a.early_decision == b.early_decision
            for a, b in zip(raw["transient"], raw["tonic"])
        ),
        "stage1_transient_reset_equal": all(
            a.early_state == b.early_state and a.early_decision == b.early_decision
            for a, b in zip(raw["transient"], raw["reset"])
        ),
        "event_counts_equal": len({metrics[a]["event_count"] for a in ARMS}) == 1,
        "event_amplitude_sums_equal": max(sums) - min(sums) <= 1e-12,
    }
    numerical = {
        "all_finite": all(np.isfinite(metrics[a]["max_abs_state"]) for a in ARMS),
        "all_bounded": all(metrics[a]["max_abs_state"] <= config.state_bound for a in ARMS),
    }
    return {
        "seed": seed,
        "arms": metrics,
        "matched_invariants": matched,
        "numerical_checks": numerical,
        "paired_episode_disagreements": {
            "transient_vs_tonic": _paired_counts(raw["transient"], raw["tonic"]),
            "transient_vs_reset": _paired_counts(raw["transient"], raw["reset"]),
        },
    }


def classify(summary: dict, config: FrozenConfig) -> str:
    if not summary["matched_invariants_valid"]:
        return "INVALID_MATCHED_ARMS"
    if (
        summary["median_transient_minus_tonic"] < 0.20
        or summary["transient_seed_wins_vs_tonic"] < 48
    ):
        return "NO_HANDOFF_ADVANTAGE"
    if (
        summary["median_transient_minus_reset"] < 0.30
        or summary["transient_seed_wins_vs_reset"] < 56
    ):
        return "HANDOFF_WITHOUT_RESIDENT_MEMORY"
    if (
        min(summary["median_early_accuracy"].values()) < 0.95
        or summary["median_final_accuracy"]["transient"] < 0.90
        or not summary["all_finite"]
        or summary["max_abs_state"] > config.state_bound
    ):
        return "UNSTABLE_OR_WEAK_TRANSIENT"
    return "PASS_TRANSIENT_HANDOFF"


def aggregate(seed_results: list[dict], config: FrozenConfig) -> dict:
    early = {
        arm: float(median(r["arms"][arm]["early_accuracy"] for r in seed_results))
        for arm in ARMS
    }
    final = {
        arm: float(median(r["arms"][arm]["final_accuracy"] for r in seed_results))
        for arm in ARMS
    }
    memory_corr = {
        arm: float(median(r["arms"][arm]["handoff_memory_correlation"] for r in seed_results))
        for arm in ARMS
    }
    dt = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["tonic"]["final_accuracy"]
        for r in seed_results
    ]
    dr = [
        r["arms"]["transient"]["final_accuracy"] - r["arms"]["reset"]["final_accuracy"]
        for r in seed_results
    ]
    summary = {
        "matched_invariants_valid": all(
            all(r["matched_invariants"].values()) for r in seed_results
        ),
        "median_early_accuracy": early,
        "median_final_accuracy": final,
        "median_handoff_memory_correlation": memory_corr,
        "median_transient_minus_tonic": float(median(dt)),
        "median_transient_minus_reset": float(median(dr)),
        "transient_seed_wins_vs_tonic": int(sum(x > 0.0 for x in dt)),
        "transient_seed_wins_vs_reset": int(sum(x > 0.0 for x in dr)),
        "all_finite": all(r["numerical_checks"]["all_finite"] for r in seed_results),
        "max_abs_state": float(
            max(r["arms"][a]["max_abs_state"] for r in seed_results for a in ARMS)
        ),
    }
    summary["classification"] = classify(summary, config)
    return summary


def run_canonical(config: FrozenConfig | None = None) -> dict:
    cfg = FrozenConfig() if config is None else config
    seeds = [run_seed(seed, cfg) for seed in range(cfg.seed_count)]
    return {"config": cfg.__dict__, "seed_results": seeds, "summary": aggregate(seeds, cfg)}
