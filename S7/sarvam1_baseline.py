"""Sarvam-1 multilingual baseline on the shared validation slice."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

MODEL='sarvamai/sarvam-1'
def run(limit=4):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True); tok.pad_token=tok.eos_token
    start=time.perf_counter(); model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.float16); model.to(device).eval(); load_seconds=time.perf_counter()-start
    vals=[]; rows=manifest['rows']['validation'][:limit]
    with torch.no_grad():
        for i in range(0,len(rows),2):
            e=tok([r['target'] for r in rows[i:i+2]],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); ids=e['input_ids'].to(device); att=e['attention_mask'].to(device); vals.append(float(model(input_ids=ids,attention_mask=att,labels=ids).loss))
    result={'model':MODEL,'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'hidden_size':model.config.hidden_size,'layers':model.config.num_hidden_layers,'vocab_size':model.config.vocab_size,'validation_mean':sum(vals)/len(vals),'validation_losses':vals,'examples':len(rows),'load_seconds':load_seconds,'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'sarvam1_multilingual_smoke' if limit<32 else 'sarvam1_multilingual_baseline'}
    out=root/'artifacts/sarvam1_baseline'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
