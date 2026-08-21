"""Small deterministic Qwen continued-pretraining smoke run on the shared manifest."""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL='Qwen/Qwen2.5-0.5B'; OUT=Path(__file__).resolve().parent/'artifacts'/'qwen_baseline_finetune'; STEPS=10; MAX_LEN=128

def run(steps=STEPS):
    torch.set_num_threads(1); torch.manual_seed(3180)
    manifest=json.loads((Path(__file__).resolve().parent/'artifacts'/'qwen_data_manifest'/'results.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    texts=[" ".join([r['target']]) for r in manifest['rows']['train'][:64]]
    batches=[]
    for text in texts:
        ids=tok(text,add_special_tokens=True,truncation=True,max_length=MAX_LEN,return_tensors='pt')['input_ids']
        batches.append(ids)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5); losses=[]; start=time.perf_counter()
    model.train()
    for step in range(steps):
        ids=batches[step%len(batches)]; out=model(input_ids=ids,labels=ids); loss=out.loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
    OUT.mkdir(parents=True,exist_ok=True); result={'model':MODEL,'steps':steps,'max_length':MAX_LEN,'records':len(batches),'losses':losses,'initial_seed':3180,'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'model_parameters':sum(p.numel() for p in model.parameters()),'status':'smoke_run_not_quality_claim'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result
if __name__=='__main__': print(json.dumps(run(),indent=2))
