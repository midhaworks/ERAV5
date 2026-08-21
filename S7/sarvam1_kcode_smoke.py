"""Bounded Sarvam-1 exact K-code input smoke test."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding

MODEL='sarvamai/sarvam-1'
def run(regression_steps=100):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True); tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.float16); original=model.get_input_embeddings(); target=original.weight.detach().clone().to(device)
    k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(device=device); opt=torch.optim.AdamW([k.projection],lr=2e-3); reg=[]
    for s in range(regression_steps):
        ids=(torch.arange(512,device=device)+s*512).remainder(target.shape[0]); loss=(k(ids).float()-target[ids].float()).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); reg.append(float(loss.detach()))
    k=k.to(dtype=model.dtype); model.set_input_embeddings(k); model.to(device).eval(); rows=manifest['rows']['validation'][:4]; vals=[]; start=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(rows),2):
            e=tok([r['target'] for r in rows[i:i+2]],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); vals.append(float(model(input_ids=e['input_ids'].to(device),attention_mask=e['attention_mask'].to(device),labels=e['input_ids'].to(device)).loss))
    result={'model':MODEL,'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'original_input_parameters':original.weight.numel(),'kcode_input_parameters':k.projection.numel(),'input_parameter_reduction':1-k.projection.numel()/original.weight.numel(),'regression_steps':regression_steps,'embedding_mse_first_last':[reg[0],reg[-1]],'validation_mean':sum(vals)/len(vals),'validation_losses':vals,'examples':len(rows),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'sarvam1_exact_kcode_input_smoke'}
    out=root/'artifacts/sarvam1_kcode_smoke'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
