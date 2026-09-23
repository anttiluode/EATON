from collections import Counter
from dataclasses import asdict

from eaton.core import FrozenConfig, run_episode
from eaton.world import build_seed_tape


def test_seed_tape_has_exact_balance():
    tape = build_seed_tape(0, FrozenConfig())
    assert len(tape) == 256
    assert Counter((e.context, e.payload) for e in tape) == {
        (-1, -1): 64,
        (-1, 1): 64,
        (1, -1): 64,
        (1, 1): 64,
    }


def test_seed_tape_is_repeatable_and_seed_specific():
    cfg = FrozenConfig()
    a = [asdict(e) for e in build_seed_tape(17, cfg)]
    b = [asdict(e) for e in build_seed_tape(17, cfg)]
    c = [asdict(e) for e in build_seed_tape(18, cfg)]
    assert a == b
    assert a != c


def test_one_episode_object_is_replayed_across_all_arms():
    ep = build_seed_tape(3, FrozenConfig())[0]
    rs = [run_episode(ep, arm, FrozenConfig()) for arm in ("transient", "tonic", "reset")]
    expected_sum = abs(ep.context_event) + abs(ep.payload_event)
    assert all(r.event_count == 2 for r in rs)
    assert all(r.event_abs_sum == expected_sum for r in rs)
