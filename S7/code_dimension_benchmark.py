"""Matched K-code width benchmark: 64 versus 128 dimensions."""
from __future__ import annotations
import json
from pathlib import Path
import torch_continuation_lm as lm
from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256

OUT = Path(__file__).resolve().parent / "artifacts" / "code_dimension_benchmark"

def run(steps: int = 1500):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = build_long_dataset(); codec = ContinuationByteCodec(24)
    tensors = {s: lm.continuation_examples(codec, rows) for s, rows in data.items()}
    results = {}
    for width in (64, 128):
        lm.D_CODE = width
        model, training = lm.train_model(*tensors["train"], True, 3180, steps)
        val_temp, sweep = lm.select_temperature(model, *tensors["validation"])
        test = lm.evaluate(model, codec, data["test"], tensors["test"][0])
        nll = lm.teacher_forced_nll_report(model, data["test"], *tensors["test"], val_temp)
        results[str(width)] = {"parameters": model.parameter_report(), "training": training,
                               "temperature": val_temp, "validation_sweep": sweep,
                               "test": {k:v for k,v in test.items() if k != "predictions"},
                               "test_nll": nll}
    base = results["64"]["test_nll"]["micro_average"]
    wider = results["128"]["test_nll"]["micro_average"]
    result = {"experiment": "matched tied K-code dimension benchmark", "steps": steps,
              "dataset_hash": sha256(data), "dataset_audit": audit, "results": results,
              "comparison": {"nll_64": base, "nll_128": wider,
                             "relative_nll_change": wider / base - 1.0,
                             "wider_improves_nll": wider < base}}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
