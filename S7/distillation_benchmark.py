"""Fallback-logit distillation into the exact tied K-code output head."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import torch_continuation_lm as lm
from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256

OUT = Path(__file__).resolve().parent / "artifacts" / "distillation_benchmark"

def train_student(source, target, teacher, seed=3180, steps=1500, alpha=.3, temperature=1.0):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    student = lm.CausalContinuationModel(259, True, seed)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 1); curve=[]
    teacher.eval()
    for step in range(1, steps + 1):
        idx = torch.randint(len(source), (64,), generator=gen)
        src, gold = source[idx], target[idx]
        beginning = torch.full((len(idx), 1), 2, dtype=torch.long)
        decoder_input = torch.cat([beginning, gold[:, :-1]], dim=1)
        with torch.no_grad():
            teacher_logits = teacher(src, decoder_input)
        student_logits = student(src, decoder_input)
        ce = F.cross_entropy(student_logits.reshape(-1, 259), gold.reshape(-1), ignore_index=0)
        mask = gold != 0
        student_logp = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        kl_rows = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=-1)
        kl = kl_rows[mask].mean() * temperature * temperature
        loss = (1 - alpha) * ce + alpha * kl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        if step == 1 or step % 250 == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach())})
    return student, {"steps": steps, "alpha": alpha, "temperature": temperature, "curve": curve}

def run(steps=1500):
    OUT.mkdir(parents=True, exist_ok=True); data, audit = build_long_dataset(); codec=ContinuationByteCodec(24)
    tensors={s:lm.continuation_examples(codec, rows) for s,rows in data.items()}
    teacher, teacher_training = lm.train_model(*tensors["train"], False, 3180, steps)
    student, student_training = train_student(*tensors["train"], teacher, steps=steps)
    results={}
    for name, model in (("fallback_teacher", teacher), ("distilled_rke", student)):
        temperature, sweep=lm.select_temperature(model, *tensors["validation"])
        test=lm.evaluate(model, codec, data["test"], tensors["test"][0])
        nll=lm.teacher_forced_nll_report(model, data["test"], *tensors["test"], temperature)
        results[name]={"parameters":model.parameter_report(), "temperature":temperature,
                       "validation_sweep":sweep, "test":{k:v for k,v in test.items() if k!="predictions"}, "test_nll":nll}
    result={"experiment":"fallback-logit distillation into tied K-code head", "steps":steps,
            "dataset_hash":sha256(data), "dataset_audit":audit,
            "training":{"teacher":teacher_training,"student":student_training}, "results":results}
    base=results["fallback_teacher"]["test_nll"]["micro_average"]; new=results["distilled_rke"]["test_nll"]["micro_average"]
    result["comparison"]={"teacher_nll":base,"student_nll":new,"relative_nll_change":new/base-1.0,
                           "student_improves_nll":new<base,
                           "student_exact_noninferior":results["distilled_rke"]["test"]["exact_match"] >= results["fallback_teacher"]["test"]["exact_match"]*.99}
    (OUT/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

if __name__=="__main__": print(json.dumps(run(),indent=2))
