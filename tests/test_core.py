import math
import numpy as np

from eaton.core import EpisodeInput, FrozenConfig, run_episode


def episode(context=1, payload=-1, eta_context=0.0, eta_payload=0.0):
    return EpisodeInput(
        context=context,
        payload=payload,
        x0=(0.01, -0.02, 0.03, -0.04),
        eta_context=eta_context,
        eta_payload=eta_payload,
    )


def test_frozen_config_matches_spec():
    cfg = FrozenConfig()
    assert cfg.retention == 0.98
    assert cfg.memory_self == 0.40
    assert cfg.acquire_gain == 1.00
    assert cfg.output_gain == 2.00
    assert cfg.init_sigma == 0.05
    assert cfg.event_noise_sigma == 0.08
    assert cfg.seed_count == 64
    assert cfg.episodes_per_seed == 256
    assert cfg.repetitions_per_pair == 64


def test_all_arms_are_identical_through_stage1():
    ep = episode()
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert rs[0].early_state == rs[1].early_state == rs[2].early_state
    assert rs[0].early_decision == rs[1].early_decision == rs[2].early_decision == 1


def test_reset_changes_only_memory_at_handoff():
    ep = episode()
    t = run_episode(ep, "transient", FrozenConfig())
    r = run_episode(ep, "reset", FrozenConfig())
    diffs = [i for i, (a, b) in enumerate(zip(t.handoff_state, r.handoff_state)) if a != b]
    assert diffs == [0]
    assert r.handoff_state[0] == ep.x0[0]


def test_tonic_payload_tick_is_order_independent():
    ep = episode(context=-1, payload=1)
    ab = run_episode(ep, "tonic", FrozenConfig(), operator_order=("A", "B"))
    ba = run_episode(ep, "tonic", FrozenConfig(), operator_order=("B", "A"))
    assert ab == ba


def test_external_event_budget_matches_all_arms():
    ep = episode(eta_context=0.1, eta_payload=-0.2)
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    assert {r.event_count for r in rs} == {2}
    expected = abs(1.1) + abs(-1.2)
    assert all(math.isclose(r.event_abs_sum, expected, abs_tol=1e-15) for r in rs)


def test_noise_can_reverse_context_without_clipping():
    ep = episode(context=1, payload=1, eta_context=-1.5)
    r = run_episode(ep, "transient", FrozenConfig())
    assert ep.context_event == -0.5
    assert r.early_decision == -1
    assert r.early_correct is False


def test_large_events_still_produce_finite_bounded_written_state():
    ep = EpisodeInput(1, -1, (0.04, -0.04, 0.04, -0.04), 1000.0, -1000.0)
    r = run_episode(ep, "tonic", FrozenConfig())
    assert np.isfinite(r.max_abs_state)
    assert r.max_abs_state <= FrozenConfig().state_bound


def exact_balanced_tape():
    pairs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] * 64
    return [EpisodeInput(c, p, (0.01, -0.02, 0.03, -0.04), 0.0, 0.0) for c, p in pairs]


def test_disabling_a_leaves_early_decision_at_chance_on_balanced_tape():
    rs = [run_episode(ep, "transient", FrozenConfig(), disable_a=True) for ep in exact_balanced_tape()]
    assert sum(r.early_correct for r in rs) / len(rs) == 0.5


def test_disabling_b_leaves_final_parity_at_chance_on_balanced_tape():
    rs = [run_episode(ep, "transient", FrozenConfig(), disable_b=True) for ep in exact_balanced_tape()]
    assert sum(r.final_correct for r in rs) / len(rs) == 0.5


def test_correct_memory_clamp_allows_b_to_solve():
    rs = [
        run_episode(ep, "reset", FrozenConfig(), clamp_memory_to_context=True)
        for ep in exact_balanced_tape()
    ]
    assert all(r.final_correct for r in rs)
