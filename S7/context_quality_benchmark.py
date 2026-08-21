"""Matched context-ablation benchmark for the exact K-code continuation model.

The codec and target stream are unchanged.  The context arm appends the two
preceding words (as UTF-8 byte states) after the first target block, allowing
the encoder to use linguistic context while retaining an explicit target
boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256
from torch_continuation_lm import (
    continuation_examples, train_model, evaluate, teacher_forced_nll_report,
    select_temperature, _valid_state, MAX_BLOCKS, CausalContinuationModel,
)

OUT = Path(__file__).resolve().parent / "artifacts" / "context_quality_benchmark"


def context_sources(codec: ContinuationByteCodec, records: list[dict], first: torch.Tensor,
                    enabled: bool) -> torch.Tensor:
    rows = []
    for record, block in zip(records, first):
        values = block.tolist()
        if enabled:
            context = " ".join(record["context"]).encode("utf-8")
            values.extend((np.frombuffer(context, dtype=np.uint8).astype(np.int64) + 3).tolist())
        rows.append(values)
    width = max(map(len, rows))
    padded = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        padded[i, :len(row)] = row
    return torch.tensor(padded, dtype=torch.long)


@torch.no_grad()
def beam_exact(model, codec, records, source, width: int) -> dict:
    """Constrained beam decoder; scores are accumulated model log-probabilities."""
    import time
    model.eval(); exact = valid = 0; started = time.perf_counter()
    for record, first in zip(records, source):
        complete0 = bytes((first[:codec.block_bytes].numpy() - 3).tolist())
        hidden0 = model.encode(first[None])
        beams = [(0.0, hidden0, torch.tensor([[2]]), b"", complete0, 1, False)]
        for _ in range((MAX_BLOCKS - 1) * codec.slots):
            expanded = []
            for score, hidden, previous, prefix, complete, blocks, finished in beams:
                if finished:
                    expanded.append((score, hidden, previous, prefix, complete, blocks, finished)); continue
                output, next_hidden = model.decoder(model.codebook(previous), hidden)
                logits = model.output_logits(output)[0, 0]
                logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                candidates = []
                for raw in np.argsort(logp)[::-1][:max(width * 3, 8)]:
                    state = _valid_state(logits.detach().cpu().numpy(), complete, prefix, len(prefix)) if int(raw) == int(np.argmax(logits.detach().cpu().numpy())) else int(raw)
                    if state == 0 or state < 0: continue
                    if state == 1 and not codec: continue
                    if state == 1 and not __import__('rke').utf8_is_complete(complete + prefix): continue
                    if state == 2 and len(prefix) != codec.block_bytes: continue
                    if state >= 3 and len(prefix) >= codec.block_bytes: continue
                    candidates.append((float(logp[int(raw)]), state))
                for add, state in candidates[:width]:
                    if state == 1:
                        expanded.append((score + add, next_hidden, torch.tensor([[state]]), prefix, complete, blocks, True))
                    elif state == 2:
                        expanded.append((score + add, next_hidden, torch.tensor([[state]]), b"", complete + prefix, blocks + 1, False))
                    else:
                        expanded.append((score + add, next_hidden, torch.tensor([[state]]), prefix + bytes([state - 3]), complete, blocks, False))
            if not expanded: break
            beams = sorted(expanded, key=lambda x: x[0], reverse=True)[:width]
            if all(item[-1] for item in beams): break
        best = beams[0]
        payload = best[4] + best[3]
        target = record["target"].encode("utf-8")
        exact += payload == target and best[-1]
        valid += best[-1]
    return {"width": width, "exact_match": exact / len(records),
            "valid_chain_rate": valid / len(records),
            "examples_per_second": len(records) / (time.perf_counter() - started)}


def train_scheduled(source: torch.Tensor, target: torch.Tensor, seed: int,
                    steps: int = 1500) -> tuple[torch.nn.Module, dict]:
    """Train with a linearly decayed teacher-forcing probability."""
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = CausalContinuationModel(259, True, seed)
    initial = model.shared_state_hash()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    curve = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(source), (64,), generator=generator)
        src, gold = source[idx], target[idx]
        hidden = model.encode(src)
        previous = torch.full((len(idx), 1), 2, dtype=torch.long)
        outputs = []
        teacher_probability = max(0.5, 1.0 - 0.5 * (step - 1) / max(1, steps - 1))
        for t in range(gold.shape[1]):
            out, hidden = model.decoder(model.codebook(previous), hidden)
            logits = model.output_logits(out)
            outputs.append(logits)
            prediction = logits[:, 0].argmax(dim=-1, keepdim=True)
            use_gold = torch.rand((len(idx), 1), generator=generator) < teacher_probability
            previous = torch.where(use_gold, gold[:, t:t + 1], prediction)
        logits = torch.cat(outputs, dim=1)
        loss = F.cross_entropy(logits.reshape(-1, 259), gold.reshape(-1), ignore_index=0)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": float(loss.detach()),
                          "teacher_probability": teacher_probability})
    return model, {"steps": steps, "batch_size": 64, "learning_rate": 2e-3,
                   "initial_shared_state_hash": initial, "curve": curve}


def run(steps: int = 1500) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = build_long_dataset()
    codec = ContinuationByteCodec(24)
    tensors = {split: continuation_examples(codec, records) for split, records in data.items()}
    results = {}
    for name, enabled in (("no_context", False), ("two_word_context", True)):
        sources = {split: context_sources(codec, data[split], tensors[split][0], enabled)
                  for split in data}
        model, training = train_model(sources["train"], tensors["train"][1], True, 3180, steps)
        temperature, sweep = select_temperature(model, sources["validation"], tensors["validation"][1])
        val = evaluate(model, codec, data["validation"], tensors["validation"][0])
        test = evaluate(model, codec, data["test"], tensors["test"][0])
        test_nll = teacher_forced_nll_report(model, data["test"], sources["test"], tensors["test"][1], temperature)
        beam = {str(width): beam_exact(model, codec, data["test"], sources["test"], width)
                for width in (2, 4)}
        results[name] = {
            "context_enabled": enabled,
            "parameters": model.parameter_report(),
            "training": training,
            "validation_temperature": temperature,
            "validation_sweep": sweep,
            "test": {k: v for k, v in test.items() if k != "predictions"},
            "test_nll": test_nll,
            "beam": beam,
        }
    # Isolate the training change: same two-word context, but free-running
    # scheduled-sampling updates instead of pure teacher forcing.
    context_sources_by_split = {
        split: context_sources(codec, data[split], tensors[split][0], True)
        for split in data
    }
    scheduled_model, scheduled_training = train_scheduled(
        context_sources_by_split["train"], tensors["train"][1], 3180, steps)
    scheduled_temperature, scheduled_sweep = select_temperature(
        scheduled_model, context_sources_by_split["validation"], tensors["validation"][1])
    scheduled_test = evaluate(scheduled_model, codec, data["test"], tensors["test"][0])
    scheduled_nll = teacher_forced_nll_report(
        scheduled_model, data["test"], context_sources_by_split["test"], tensors["test"][1],
        scheduled_temperature)
    results["scheduled_two_word_context"] = {
        "context_enabled": True, "training": scheduled_training,
        "validation_temperature": scheduled_temperature,
        "validation_sweep": scheduled_sweep,
        "test": {k: v for k, v in scheduled_test.items() if k != "predictions"},
        "test_nll": scheduled_nll,
    }
    baseline = results["no_context"]["test_nll"]["micro_average"]
    contextual = results["two_word_context"]["test_nll"]["micro_average"]
    result = {
        "experiment": "matched context ablation for exact K-code causal model",
        "steps": steps, "codec": {"block_bytes": codec.block_bytes, "states": codec.width},
        "dataset_hash": sha256(data), "dataset_audit": audit,
        "results": results,
        "quality": {"baseline_nll": baseline, "context_nll": contextual,
                    "relative_nll_change": contextual / baseline - 1.0,
                    "context_improves_nll": contextual < baseline},
        "note": "This is a quality experiment; exact codec audits remain separate.",
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
