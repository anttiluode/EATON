from __future__ import annotations

import argparse
import json
from pathlib import Path

from eaton.v1_experiment import V1Config, run_v1_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen EATON v1 computation-coordinate experiment")
    parser.add_argument("--out", default="results/v1_coordinates.json")
    args = parser.parse_args()
    result = run_v1_canonical(V1Config())
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(result["classification"])
    for key in (
        "valid_structured_seed_count",
        "median_response_nrmse",
        "median_activation_nrmse",
        "median_shuffled_grammar_gain",
        "median_timebin_grammar_gain",
        "median_response_world_accuracy",
        "median_structured_minus_volatile_grammar_gain",
        "median_finite_difference_relative_error",
    ):
        if key in summary:
            print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
