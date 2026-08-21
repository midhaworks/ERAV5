"""Matched tied K-code versus tied K-code plus low-rank residual."""
from __future__ import annotations
import json
from pathlib import Path
import torch_continuation_lm as lm
from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256

OUT = Path(__file__).resolve().parent / "artifacts" / "residual_quality_benchmark"

def run(steps: int = 1500):
    OUT.mkdir(parents=True, exist_ok=True); data, audit = build_long_dataset()
    codec = ContinuationByteCodec(24)
    tensors = {s: lm.continuation_examples(codec, rows) for s, rows in data.items()}
    results = {}
    for name, rank in (("tied", 0), ("residual_rank_8", 8)):
        model, training = lm.train_model(*tensors["train"], True, 3180, steps, residual_rank=rank)
        temperature, sweep = lm.select_temperature(model, *tensors["validation"])
        test = lm.evaluate(model, codec, data["test"], tensors["test"][0])
        nll = lm.teacher_forced_nll_report(model, data["test"], *tensors["test"], temperature)
        results[name] = {"rank": rank, "parameters": model.parameter_report(),
                         "training": training, "temperature": temperature,
                         "validation_sweep": sweep,
                         "test": {k:v for k,v in test.items() if k != "predictions"},
                         "test_nll": nll}
    base, new = results["tied"]["test_nll"]["micro_average"], results["residual_rank_8"]["test_nll"]["micro_average"]
    result = {"experiment": "low-rank residual tied K-code benchmark", "steps": steps,
              "dataset_hash": sha256(data), "dataset_audit": audit, "results": results,
              "comparison": {"tied_nll": base, "residual_nll": new,
                             "relative_nll_change": new / base - 1.0,
                             "residual_improves_nll": new < base}}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
