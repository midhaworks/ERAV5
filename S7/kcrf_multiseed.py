"""Three-seed stability wrapper for the conditional K-CRF pilot."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from kcrf_conditional_benchmark import run


OUT = Path(__file__).resolve().parent / "artifacts" / "kcrf_multiseed"
SEEDS = (2308, 2309, 2310)


def main():
    started = time.perf_counter()
    runs = [run(seed=seed) for seed in SEEDS]
    improvements = [row["nll_improvement"] for row in runs]
    result = {
        "protocol": "three-seed conditional multilingual K-CRF pilot",
        "seeds": list(SEEDS), "runs": runs,
        "nll_improvement_mean": statistics.mean(improvements),
        "nll_improvement_stdev": statistics.stdev(improvements),
        "nll_improvement_min": min(improvements), "nll_improvement_max": max(improvements),
        "all_seeds_improve": all(value > 0 for value in improvements),
        "seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
