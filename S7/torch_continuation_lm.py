"""Causal PyTorch continuation model with a tied RKE codebook and matched fallback."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cross_block_lm import (MAX_BLOCKS, PRODUCTION_CONTINUATION_EXACT_THRESHOLD,
                            build_long_dataset, exact_prefix_retrieval_baseline,
                            record_languages)
from rke import ContinuationByteCodec, sha256, utf8_is_complete, utf8_prefix_is_valid


DEVICE = torch.device("cpu")
D_CODE = 64
D_HIDDEN = 192
TRAIN_STEPS = 3_000
BATCH_SIZE = 64
CALIBRATION_SCALES = (.25, .4, .5, .6, .7, .8, 1.0, 1.2)
NLL_NONINFERIORITY_RATIO = 1.01
EXACT_NONINFERIORITY_RATIO = .99
MINIMUM_PARAMETER_REDUCTION = .10
MINIMUM_THROUGHPUT_RATIO = .90


def continuation_examples(codec: ContinuationByteCodec, records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    sources, targets = [], []
    for record in records:
        blocks = codec.ids(record["target"].encode("utf-8"))
        if len(blocks) < 2:
            raise ValueError("continuation model requires a multi-block target")
        sources.append(blocks[0])
        sequence = []
        for block in blocks[1:]:
            end = int(np.flatnonzero((block == 1) | (block == 2))[0])
            sequence.extend(block[:end + 1].tolist())
        targets.append(sequence)
    maximum = max(map(len, targets))
    padded = np.zeros((len(targets), maximum), dtype=np.int64)
    for index, sequence in enumerate(targets):
        padded[index, :len(sequence)] = sequence
    return torch.tensor(np.stack(sources), dtype=torch.long), torch.tensor(padded, dtype=torch.long)


class CausalContinuationModel(nn.Module):
    """Encode block zero, then causally emit all remaining block states."""

    def __init__(self, states: int, tied_rke: bool, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.states, self.tied_rke = states, tied_rke
        self.codebook = nn.Embedding(states, D_CODE)
        self.encoder = nn.GRU(D_CODE, D_HIDDEN, batch_first=True)
        self.decoder = nn.GRU(D_CODE, D_HIDDEN, batch_first=True)
        self.to_code = nn.Linear(D_HIDDEN, D_CODE, bias=False) if tied_rke else None
        self.classifier = None if tied_rke else nn.Linear(D_HIDDEN, states, bias=False)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(self.codebook(source))
        return hidden

    def output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tied_rke:
            assert self.to_code is not None
            return self.to_code(hidden) @ self.codebook.weight.T / math.sqrt(D_CODE)
        assert self.classifier is not None
        return self.classifier(hidden)

    def forward(self, source: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(source)
        output, _ = self.decoder(self.codebook(decoder_input), hidden)
        return self.output_logits(output)

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        classifier = 0 if self.classifier is None else self.classifier.weight.numel()
        structured_adapter = 0 if self.to_code is None else self.to_code.weight.numel()
        return {"total": total, "shared_codebook": self.codebook.weight.numel(),
                "structured_output_adapter": structured_adapter,
                "separate_vocab_classifier": classifier,
                "output_specific_parameters": structured_adapter + classifier}

    def shared_state_hash(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in self.named_parameters():
            if name.startswith("classifier.") or name.startswith("to_code."):
                continue
            digest.update(name.encode()); digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


def train_model(source: torch.Tensor, target: torch.Tensor, tied_rke: bool, seed: int,
                steps: int = TRAIN_STEPS) -> tuple[CausalContinuationModel, dict[str, Any]]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = CausalContinuationModel(259, tied_rke, seed).to(DEVICE)
    initial_shared_state_hash = model.shared_state_hash()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    curve, started, stream_hash = [], time.perf_counter(), hashlib.sha256()
    source, target = source.to(DEVICE), target.to(DEVICE)
    for step in range(1, steps + 1):
        indices = torch.randint(len(source), (BATCH_SIZE,), generator=generator)
        stream_hash.update(indices.numpy().tobytes())
        batch_source, batch_target = source[indices], target[indices]
        start = torch.full((len(indices), 1), 2, dtype=torch.long, device=DEVICE)
        decoder_input = torch.cat([start, batch_target[:, :-1]], dim=1)
        logits = model(batch_source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, model.states), batch_target.reshape(-1), ignore_index=0)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": float(loss.detach())})
    return model, {"seconds": time.perf_counter() - started, "curve": curve,
                   "steps": steps, "batch_size": BATCH_SIZE, "learning_rate": 2e-3,
                   "initial_shared_state_hash": initial_shared_state_hash,
                   "batch_stream_hash": stream_hash.hexdigest()}


@torch.no_grad()
def teacher_forced_nll(model: CausalContinuationModel, source: torch.Tensor, target: torch.Tensor,
                       temperature_scale: float = 1.0) -> float:
    model.eval(); total_loss = total_states = 0
    for start_index in range(0, len(source), 128):
        batch_source = source[start_index:start_index + 128].to(DEVICE)
        batch_target = target[start_index:start_index + 128].to(DEVICE)
        beginning = torch.full((len(batch_source), 1), 2, dtype=torch.long, device=DEVICE)
        decoder_input = torch.cat([beginning, batch_target[:, :-1]], dim=1)
        logits = model(batch_source, decoder_input) * temperature_scale
        total_loss += float(F.cross_entropy(logits.reshape(-1, model.states), batch_target.reshape(-1),
                                            ignore_index=0, reduction="sum"))
        total_states += int((batch_target != 0).sum())
    return total_loss / total_states


def teacher_forced_nll_report(model: CausalContinuationModel, records: list[dict[str, Any]],
                              source: torch.Tensor, target: torch.Tensor,
                              temperature_scale: float = 1.0) -> dict[str, Any]:
    per_language = {}
    for language in record_languages(records):
        indices = torch.tensor([index for index, record in enumerate(records)
                                if record["language"] == language], dtype=torch.long)
        per_language[language] = teacher_forced_nll(
            model, source[indices], target[indices], temperature_scale)
    return {"normalization": "loss-bearing suffix byte/CONT/EOS state",
            "micro_average": teacher_forced_nll(model, source, target, temperature_scale),
            "macro_average": statistics.mean(per_language.values()),
            "per_language": per_language}


def select_temperature(model: CausalContinuationModel, source: torch.Tensor,
                       target: torch.Tensor) -> tuple[float, dict[str, float]]:
    """Select a scalar on validation only; scaling cannot change argmax output."""
    scores = {str(scale): teacher_forced_nll(model, source, target, scale)
              for scale in CALIBRATION_SCALES}
    selected = min(CALIBRATION_SCALES, key=lambda scale: (scores[str(scale)], scale))
    return selected, scores


def _valid_state(scores: np.ndarray, complete: bytes, prefix: bytes, position: int) -> int:
    for raw in np.argsort(scores)[::-1]:
        state = int(raw)
        if state == 0:
            continue
        if state == 1 and utf8_is_complete(complete + prefix):
            return state
        if state == 2 and position == 24:
            return state
        if state >= 3 and position < 24 and utf8_prefix_is_valid(complete + prefix + bytes([state - 3])):
            return state
    return -1


@torch.no_grad()
def evaluate(model: CausalContinuationModel, codec: ContinuationByteCodec, records: list[dict[str, Any]],
             source: torch.Tensor, include_breakdown: bool = True) -> dict[str, Any]:
    model.eval(); exact = valid = missing = matching = compared = 0; predictions = []
    started = time.perf_counter()
    for record, first_block in zip(records, source):
        source_item = first_block[None].to(DEVICE)
        hidden = model.encode(source_item)
        previous = torch.tensor([[2]], dtype=torch.long, device=DEVICE)
        complete = bytes((first_block[:codec.block_bytes].numpy() - 3).tolist())
        prefix, status, generated_blocks = b"", "MAX_BLOCKS", 1
        for _ in range((MAX_BLOCKS - 1) * codec.slots):
            output, hidden = model.decoder(model.codebook(previous), hidden)
            scores = model.output_logits(output)[0, 0].cpu().numpy()
            state = _valid_state(scores, complete, prefix, len(prefix))
            if state == -1:
                status = "NO_VALID_STATE"; break
            previous = torch.tensor([[state]], dtype=torch.long, device=DEVICE)
            if state == 1:
                status = "EOS"; break
            if state == 2:
                complete += prefix; prefix = b""; generated_blocks += 1
                if generated_blocks >= MAX_BLOCKS:
                    status = "MAX_BLOCKS"; break
            else:
                prefix += bytes([state - 3])
        payload = complete + prefix
        target = record["target"].encode("utf-8")
        target_suffix, predicted_suffix = target[codec.block_bytes:], payload[codec.block_bytes:]
        match = payload == target and status == "EOS"
        exact += match; valid += status == "EOS"; missing += status != "EOS"
        compared += max(len(target_suffix), len(predicted_suffix), 1)
        matching += sum(left == right for left, right in zip(target_suffix, predicted_suffix))
        predictions.append({"target": record["target"], "target_suffix_hex": target_suffix.hex(),
                            "prediction_suffix_hex": predicted_suffix.hex(), "status": status, "exact": match})
    seconds = time.perf_counter() - started
    result = {"examples": len(records), "exact_count": exact, "exact_match": exact / len(records),
              "suffix_byte_accuracy": matching / compared, "valid_chain_rate": valid / len(records),
              "missing_eos_rate": missing / len(records), "examples_per_second": len(records) / seconds,
              "predictions": predictions}
    if include_breakdown:
        metric_names = ("exact_match", "suffix_byte_accuracy", "valid_chain_rate", "missing_eos_rate")
        per_language = {}
        for language in record_languages(records):
            indices = [index for index, record in enumerate(records) if record["language"] == language]
            rows = [records[index] for index in indices]
            value = evaluate(model, codec, rows, source[indices], include_breakdown=False)
            value.pop("predictions"); value.pop("examples_per_second")
            per_language[language] = value
        result["per_language"] = per_language
        result["micro_average"] = {name: result[name] for name in metric_names}
        result["macro_average"] = {
            name: statistics.mean(value[name] for value in per_language.values())
            for name in metric_names}
    return result


def run_torch_continuation(output: Path, steps: int = TRAIN_STEPS) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    data, audit = build_long_dataset(); codec = ContinuationByteCodec(24)
    tensors = {split: continuation_examples(codec, records) for split, records in data.items()}
    results: dict[str, Any] = {}
    prediction_bundle = {}
    models = {}
    for name, tied, seed in (("rke", True, 1810), ("fallback", False, 1810)):
        model, training = train_model(*tensors["train"], tied, seed, steps)
        validation_generated = evaluate(model, codec, data["validation"], tensors["validation"][0])
        generated = evaluate(model, codec, data["test"], tensors["test"][0])
        temperature, validation_temperature_sweep = select_temperature(model, *tensors["validation"])
        raw_validation_nll_report = teacher_forced_nll_report(
            model, data["validation"], *tensors["validation"])
        validation_nll_report = teacher_forced_nll_report(
            model, data["validation"], *tensors["validation"], temperature)
        raw_test_nll_report = teacher_forced_nll_report(model, data["test"], *tensors["test"])
        test_nll_report = teacher_forced_nll_report(
            model, data["test"], *tensors["test"], temperature)
        prediction_bundle[f"{name}_validation"] = validation_generated.pop("predictions")
        prediction_bundle[name] = generated.pop("predictions")
        results[name] = {"parameters": model.parameter_report(), "training": training,
                         "calibration": {"selected_on": "validation", "temperature_scale": temperature,
                                         "validation_sweep": validation_temperature_sweep,
                                         "argmax_invariant": True},
                         "raw_validation_nll": raw_validation_nll_report["micro_average"],
                         "validation_nll": validation_nll_report["micro_average"],
                         "validation_generated": validation_generated,
                         "raw_test_nll": raw_test_nll_report["micro_average"],
                         "test_nll": test_nll_report["micro_average"],
                         "nll_reports": {"raw_validation": raw_validation_nll_report,
                                         "calibrated_validation": validation_nll_report,
                                         "raw_test": raw_test_nll_report,
                                         "calibrated_test": test_nll_report},
                         "generated": generated}
        models[name] = model
    retrieval = exact_prefix_retrieval_baseline(data["train"], data["test"], codec.block_bytes)
    throughput_samples: dict[str, list[float]] = {"rke": [], "fallback": []}
    # Alternate arm order to reduce systematic thermal/order bias. The earlier
    # validation/test evaluations serve as warmups; these are measured repeats.
    for repeat in range(5):
        order = ("rke", "fallback") if repeat % 2 == 0 else ("fallback", "rke")
        for name in order:
            trial = evaluate(models[name], codec, data["test"], tensors["test"][0],
                             include_breakdown=False)
            throughput_samples[name].append(trial["examples_per_second"])
    decode_benchmark = {
        name: {"samples_examples_per_second": samples,
               "median_examples_per_second": statistics.median(samples),
               "minimum_examples_per_second": min(samples),
               "maximum_examples_per_second": max(samples), "repeats": len(samples)}
        for name, samples in throughput_samples.items()
    }
    parameter_reduction = 1 - (results["rke"]["parameters"]["total"] /
                               results["fallback"]["parameters"]["total"])
    throughput_ratio = (decode_benchmark["rke"]["median_examples_per_second"] /
                        decode_benchmark["fallback"]["median_examples_per_second"])
    checks = {
        "at_least_500_test_examples": len(data["test"]) >= 500,
        "rke_exact_at_least_20_percent": results["rke"]["generated"]["exact_match"] >=
                                          PRODUCTION_CONTINUATION_EXACT_THRESHOLD,
        "rke_beats_retrieval": results["rke"]["generated"]["exact_match"] > retrieval["exact_match"],
        "rke_exact_at_least_99_percent_of_fallback": results["rke"]["generated"]["exact_match"] >=
                                                     results["fallback"]["generated"]["exact_match"] *
                                                     EXACT_NONINFERIORITY_RATIO,
        "rke_raw_nll_within_1_percent_of_fallback": results["rke"]["raw_test_nll"] <=
                                                    results["fallback"]["raw_test_nll"] *
                                                    NLL_NONINFERIORITY_RATIO,
        "rke_calibrated_nll_within_1_percent_of_fallback": results["rke"]["test_nll"] <=
                                                           results["fallback"]["test_nll"] *
                                                           NLL_NONINFERIORITY_RATIO,
        "rke_macro_raw_nll_within_1_percent_of_fallback":
            results["rke"]["nll_reports"]["raw_test"]["macro_average"] <=
            results["fallback"]["nll_reports"]["raw_test"]["macro_average"] *
            NLL_NONINFERIORITY_RATIO,
        "rke_macro_calibrated_nll_within_1_percent_of_fallback":
            results["rke"]["nll_reports"]["calibrated_test"]["macro_average"] <=
            results["fallback"]["nll_reports"]["calibrated_test"]["macro_average"] *
            NLL_NONINFERIORITY_RATIO,
        "rke_valid_chains_100_percent": results["rke"]["generated"]["valid_chain_rate"] == 1.0,
        "rke_no_separate_vocab_classifier":
            results["rke"]["parameters"]["separate_vocab_classifier"] == 0,
        "parameter_reduction_at_least_10_percent": parameter_reduction >= MINIMUM_PARAMETER_REDUCTION,
        "throughput_at_least_90_percent_of_fallback": throughput_ratio >= MINIMUM_THROUGHPUT_RATIO,
        "identical_shared_initialization": results["rke"]["training"]["initial_shared_state_hash"] ==
                                           results["fallback"]["training"]["initial_shared_state_hash"],
        "identical_batch_stream": results["rke"]["training"]["batch_stream_hash"] ==
                                  results["fallback"]["training"]["batch_stream_hash"],
    }
    functional_keys = ("at_least_500_test_examples", "rke_beats_retrieval",
                       "rke_valid_chains_100_percent", "rke_no_separate_vocab_classifier",
                       "identical_shared_initialization", "identical_batch_stream")
    functional_passed = all(checks[key] for key in functional_keys)
    result = {"experiment": "causal full-sequence learned continuation", "seed": 1810,
              "configuration": {"d_code": D_CODE, "d_hidden": D_HIDDEN, "steps": steps,
                                "block_bytes": codec.block_bytes, "states": codec.width,
                                "production_exact_threshold": PRODUCTION_CONTINUATION_EXACT_THRESHOLD,
                                "calibration_scales": CALIBRATION_SCALES,
                                "nll_noninferiority_ratio": NLL_NONINFERIORITY_RATIO,
                                "exact_noninferiority_ratio": EXACT_NONINFERIORITY_RATIO,
                                "minimum_parameter_reduction": MINIMUM_PARAMETER_REDUCTION,
                                "minimum_throughput_ratio": MINIMUM_THROUGHPUT_RATIO,
                                "throughput_repeats": 5,
                                "throughput_order": "alternating arms; median ratio"},
              "dataset": audit, "dataset_hash": sha256(data), "retrieval": retrieval,
              **results, "decode_benchmark": decode_benchmark,
              "efficiency": {"parameter_reduction": parameter_reduction,
                                         "throughput_ratio": throughput_ratio},
              "checks": checks, "functional_check_names": functional_keys,
              "functional_passed": functional_passed, "passed": all(checks.values())}
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(prediction_bundle, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    for name, model in models.items():
        torch.save(model.state_dict(), output / f"{name}_model.pt")
    return result


if __name__ == "__main__":
    destination = Path(__file__).resolve().parent / "artifacts" / "torch_continuation_lm"
    value = run_torch_continuation(destination)
    print(json.dumps(value, indent=2))
    if not value["passed"]:
        raise SystemExit(1)
