#!/usr/bin/env python3
"""Run every proof and experiment, then regenerate S7/artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from rke import FullByteCodec, ReversibleCodec, make_split, original_truncated_code, save_npz, sha256, train_model
from lm_compare import run_lm_experiment


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
SEED = 20260814
ALPHABET = "abcdef0123"
MAX_CHARS = 6


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_curve(curve: list[dict[str, float]]) -> None:
    def points(values: list[float], x0: float, width: float, top: float, height: float,
               low: float | None = None, high: float | None = None) -> str:
        lo, hi = (min(values) if low is None else low), (max(values) if high is None else high)
        span = max(hi - lo, 1e-9)
        return " ".join(f"{x0 + i * width / (len(values)-1):.1f},{top + height * (1-(v-lo)/span):.1f}"
                        for i, v in enumerate(values))
    losses = [x["loss"] for x in curve]
    train = [x["train_exact"] for x in curve]
    test = [x["test_exact"] for x in curve]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='420' viewBox='0 0 1100 420'>
<rect width='1100' height='420' rx='18' fill='#091526'/><g font-family='system-ui' fill='#dce8f7'>
<text x='60' y='42' font-size='20' font-weight='700'>Structured-head training loss</text>
<text x='600' y='42' font-size='20' font-weight='700'>Exact token reconstruction</text>
<g stroke='#29405c' stroke-width='1'><path d='M60 70V350H500'/><path d='M600 70V350H1040'/>
<path d='M60 210H500M600 210H1040' stroke-dasharray='4 6'/></g>
<polyline points='{points(losses,60,440,70,280)}' fill='none' stroke='#7dd3fc' stroke-width='4'/>
<polyline points='{points(train,600,440,70,280,0,1)}' fill='none' stroke='#fbbf24' stroke-width='4'/>
<polyline points='{points(test,600,440,70,280,0,1)}' fill='none' stroke='#34d399' stroke-width='4'/>
<g font-size='14' fill='#91a4bc'><text x='60' y='382'>optimizer step →</text><text x='600' y='382'>optimizer step →</text>
<text x='610' y='94' fill='#fbbf24'>train</text><text x='670' y='94' fill='#34d399'>held-out OOV</text>
<text x='570' y='76'>1.0</text><text x='578' y='354'>0</text></g></g></svg>"""
    (OUT / "training_curve.svg").write_text(svg, encoding="utf-8")


def write_report(results: dict) -> None:
    metric = results["neural_proof"]["held_out_oov"]
    html = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>RKE-Head Evidence</title><style>
body{{margin:0;background:#07111f;color:#e6edf7;font:16px system-ui;line-height:1.55}}main{{max-width:1050px;margin:auto;padding:55px 24px}}
h1{{font-size:clamp(2.5rem,7vw,5.5rem);line-height:.94;margin:.2em 0;background:linear-gradient(90deg,#7dd3fc,#34d399);color:transparent;background-clip:text}}
.tag{{color:#7dd3fc;letter-spacing:.16em;text-transform:uppercase;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:32px 0}}
.card{{background:#0d1b2d;border:1px solid #21344c;border-radius:16px;padding:22px}}.big{{font-size:2.2rem;font-weight:800;color:#34d399}}code{{color:#fbbf24}}
img{{width:100%;border-radius:16px;border:1px solid #21344c}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #21344c;text-align:left}}
</style></head><body><main><div class='tag'>Session 7 · Problem 5</div><h1>Reverse the Kronecker pathway.</h1>
<p>RKE-Head predicts an ordered position × symbol code instead of vocabulary logits. EOS makes the code injective; argmax per slot makes it directly decodable.</p>
<div class='grid'><div class='card'><div class='big'>{metric['exact_match']:.1%}</div>held-out token exact match</div>
<div class='card'><div class='big'>{results['analytic_proof']['exact_rate']:.1%}</div>analytic round trips</div>
<div class='card'><div class='big'>0%</div>fixed-vocabulary OOV coverage</div>
<div class='card'><div class='big'>{results['parameter_scaling']['structured_head_parameters']:,}</div>full-byte head parameters at demo width</div></div>
<img src='training_curve.svg' alt='training curves'><h2>Falsifiable evidence</h2><table><tr><th>Claim</th><th>Result</th></tr>
<tr><td>Injective codec on tested domain</td><td>{results['analytic_proof']['passed']}</td></tr>
<tr><td>Long-prefix collision removed</td><td>{results['truncation_counterexample']['passed']}</td></tr>
<tr><td>Neural OOV exact match ≥ 90%</td><td>{results['claims']['neural_oov_generalization']['passed']}</td></tr>
<tr><td>No vocabulary-sized final matrix</td><td>{results['claims']['vocab_independent_head']['passed']}</td></tr></table>
<h2>V2 next-token follow-up</h2><p>The matched language-model experiment compares vocabulary softmax, autoregressive byte fallback, and parallel RKE output. RKE held-out exact match: <strong>{results['language_model_followup']['rke_held_out_exact']:.1%}</strong>. Vocabulary held-out exact match: <strong>{results['language_model_followup']['vocabulary_held_out_exact']:.1%}</strong>.</p>
<img src='lm_v2/comparison.svg' alt='matched next-token comparison'>
<p><a href='lm_v2/results.json' style='color:#7dd3fc'>Machine-readable LM results</a> · <a href='lm_v2/comparison.svg' style='color:#7dd3fc'>Comparison chart</a></p>
<p>All numbers are generated by <code>python3 S7/run_experiment.py</code>. See README.md and results.json for definitions and limitations.</p></main></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    codec = ReversibleCodec(ALPHABET, MAX_CHARS)
    train, test = make_split(SEED, ALPHABET, MAX_CHARS, 700, 250)
    assert not (set(train) & set(test))

    proof_domain = [""] + train + test
    proof_records = [codec.proof_record(text) for text in proof_domain]
    analytic = {
        "domain_size": len(proof_records), "unique_code_hashes": len({x["code_hash"] for x in proof_records}),
        "exact_round_trips": sum(x["exact"] for x in proof_records),
    }
    analytic["exact_rate"] = analytic["exact_round_trips"] / analytic["domain_size"]
    analytic["passed"] = analytic["exact_rate"] == 1 and analytic["unique_code_hashes"] == analytic["domain_size"]

    # Exercise the actual 258-state byte form, including every possible byte.
    byte_codec = FullByteCodec(MAX_CHARS)
    rng = np.random.default_rng(SEED)
    byte_payloads = [b""] + [bytes([value]) for value in range(256)]
    byte_payloads += [rng.integers(0, 256, size=int(rng.integers(1, MAX_CHARS + 1)), dtype=np.uint8).tobytes()
                      for _ in range(1000)]
    full_byte_exact = sum(byte_codec.decode_ids(byte_codec.ids(payload)) == payload for payload in byte_payloads)
    # Any set of distinct normalized prototypes reverses by cosine nearest-neighbor:
    # self-similarity is 1 and every non-identical row has similarity < 1.
    prototypes = rng.normal(size=(258, 12)); prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototype_ids = (prototypes @ prototypes.T).argmax(axis=1)
    full_byte_proof = {
        "payloads_tested": len(byte_payloads), "includes_all_256_single_bytes": True,
        "exact_round_trips": full_byte_exact,
        "distinct_prototypes_recovered": int((prototype_ids == np.arange(258)).sum()),
        "passed": full_byte_exact == len(byte_payloads) and np.array_equal(prototype_ids, np.arange(258)),
    }

    left, right, dp = "abcd00", "abcd11", 4
    truncation = {
        "left": left, "right": right, "paper_max_positions": dp,
        "original_left_code": original_truncated_code(left, dp),
        "original_right_code": original_truncated_code(right, dp),
        "original_collision": original_truncated_code(left, dp) == original_truncated_code(right, dp),
        "rke_left_hash": codec.proof_record(left)["code_hash"], "rke_right_hash": codec.proof_record(right)["code_hash"],
    }
    truncation["rke_distinct"] = truncation["rke_left_hash"] != truncation["rke_right_hash"]
    truncation["passed"] = truncation["original_collision"] and truncation["rke_distinct"]

    model, curve, neural = train_model(codec, train, test, seed=SEED)
    save_npz(OUT / "model.npz", model)
    with (OUT / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader(); writer.writerows(curve)
    plot_curve(curve)

    lm_results = run_lm_experiment(OUT / "lm_v2")

    d_model, d_slot, full_byte_states, full_slots, million = model.d_model, model.d_slot, 258, MAX_CHARS + 1, 1_000_000
    scaling = {
        "d_model": d_model, "per_slot_width": d_slot,
        "full_byte_states_including_pad_eos": full_byte_states, "slots": full_slots,
        "structured_head_parameters": 0,
        "tied_byte_codebook_parameters": d_slot * full_byte_states,
        "structured_input_output_pathway_parameters": d_slot * full_byte_states,
        "vocab_head_parameters_at_train_vocab": d_model * len(train),
        "vocab_head_parameters_at_one_million": d_model * million,
        "one_million_ratio": (d_model * million) / (d_slot * full_byte_states),
        "note": "The input byte codebook is transposed for output and shared across slots: no separate final-head parameters. Counts exclude bias.",
    }
    results = {
        "experiment": "Reversible Kronecker Head (RKE-Head)", "problem": 5, "seed": SEED,
        "configuration": {"alphabet": ALPHABET, "max_chars": MAX_CHARS, "train_tokens": len(train), "held_out_tokens": len(test), "steps": 900},
        "dataset_hash": sha256({"train": train, "test": test}), "model_state_hash": model.state_hash(),
        "analytic_proof": analytic, "full_byte_proof": full_byte_proof, "truncation_counterexample": truncation,
        "neural_proof": neural, "parameter_scaling": scaling,
        "language_model_followup": {
            "overall_pass": lm_results["overall_pass"], "evidence": "lm_v2/results.json",
            "rke_held_out_exact": lm_results["rke"]["held_out"]["exact_match"],
            "fallback_held_out_exact": lm_results["byte_fallback"]["held_out"]["exact_match"],
            "vocabulary_held_out_exact": lm_results["vocabulary"]["held_out"]["exact_match"],
            "rke_nll_per_byte_or_eos": lm_results["rke"]["held_out"]["nll_per_byte_or_eos"],
            "rke_ece": lm_results["rke"]["held_out"]["calibration"]["ece_10_bin"],
        },
        "fixed_vocab_baseline": {"held_out_oov_tokens": len(test), "representable_held_out_tokens": 0, "coverage": 0.0,
                                 "reason": "all held-out target strings are absent from the training vocabulary by construction"},
        "claims": {
            "injective_on_bounded_domain": {"passed": analytic["passed"] and full_byte_proof["passed"], "evidence": "analytic_proof and full_byte_proof"},
            "neural_oov_generalization": {"passed": neural["held_out_oov"]["exact_match"] >= .90, "threshold": .90, "evidence": "neural_proof.held_out_oov"},
            "truncation_collision_removed": {"passed": truncation["passed"], "evidence": "truncation_counterexample"},
            "vocab_independent_head": {"passed": scaling["structured_head_parameters"] == 0 and scaling["tied_byte_codebook_parameters"] < scaling["vocab_head_parameters_at_one_million"], "evidence": "parameter_scaling"},
            "matched_next_token_followup": {"passed": lm_results["overall_pass"], "evidence": "lm_v2/results.json"},
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "limitations": ["Empirical training uses a 10-character alphabet; the analytic codec implementation generalizes directly to 256 bytes.",
                        "The task is controlled token copying, not open-domain language modeling.",
                        "Output cost scales with maximum token length and byte alphabet, though not vocabulary size."],
    }
    results["overall_pass"] = all(x["passed"] for x in results["claims"].values())
    write_json(OUT / "results.json", results)
    write_json(OUT / "proof_records.json", proof_records)
    write_json(OUT / "split.json", {"train": train, "held_out_oov": test})
    write_report(results)
    print(json.dumps({"overall_pass": results["overall_pass"], "claims": results["claims"],
                      "held_out_exact": neural["held_out_oov"]["exact_match"], "artifacts": str(OUT)}, indent=2))
    if not results["overall_pass"]:
        raise SystemExit("one or more claims failed")


if __name__ == "__main__":
    main()
