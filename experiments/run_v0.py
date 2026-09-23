from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaton.core import FrozenConfig
from eaton.experiment import aggregate, run_seed

CLAIM_BOUNDARY = (
    "Synthetic existence proof only: v0 can establish operator lifetime as a "
    "computational variable in this matched resident-state machine. It does not "
    "establish a biological mechanism, general superiority of phasic control, "
    "or equivalence to transformer attention."
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/v0_handoff.json")
    args = parser.parse_args()
    out = Path(args.out)
    cfg = FrozenConfig()
    payload = {
        "experiment": "EATON v0 transient operator handoff",
        "config": cfg.__dict__,
        "frozen_parameters": dict(cfg.__dict__),
        "seed_results": [],
        "summary": None,
        "classification": "INCOMPLETE",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write(out, payload)
    for seed in range(cfg.seed_count):
        payload["seed_results"].append(run_seed(seed, cfg))
        write(out, payload)
    payload["summary"] = aggregate(payload["seed_results"], cfg)
    payload["classification"] = payload["summary"]["classification"]
    write(out, payload)

    s = payload["summary"]
    print("classification:", payload["classification"])
    print("median early:", s["median_early_accuracy"])
    print("median final:", s["median_final_accuracy"])
    print("transient-tonic margin:", s["median_transient_minus_tonic"])
    print("transient-reset margin:", s["median_transient_minus_reset"])
    print(
        "seed wins tonic/reset:",
        s["transient_seed_wins_vs_tonic"],
        s["transient_seed_wins_vs_reset"],
    )
    print("receipt:", out)


if __name__ == "__main__":
    main()
