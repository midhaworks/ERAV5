"""Ablation: exact K-code input fixed, separate trainable structured output projection."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

def run(steps=500):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    state=torch.load(root/'artifacts/qwen_exact_kcode_regression/projection.pt',map_location='cpu',weights_only=True)
    k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); out_proj=torch.nn.Parameter(k.projection.detach().clone())
    model.set_input_embeddings(k); model.to(device); out_proj.data = out_proj.data.to(device)
    for p in model.parameters(): p.requires_grad=False
    k.projection.requires_grad=False; out_proj.requires_grad=True
    def make(items):
        out=[]
        for start in range(0,len(items),4):
            enc=tok(items[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); out.append((enc['input_ids'].to(device),enc['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); validation=make([r['target'] for r in manifest['rows']['validation'][:32]])
    def loss_fn(ids,attention):
        hidden=model.model(input_ids=ids,attention_mask=attention).last_hidden_state[:,:-1]; states=k.state_table[ids[:,1:]]; mask=attention[:,1:].bool(); logits=torch.matmul(hidden,out_proj.t()).view(*hidden.shape[:2],k.positions,k.states); per=F.cross_entropy(logits.reshape(-1,k.states),states.reshape(-1),reduction='none').view_as(states); valid=(states!=0)&mask[...,None]; return per.masked_select(valid).mean()
    opt=torch.optim.AdamW([out_proj],lr=1e-4); losses=[]; start=time.perf_counter(); model.train()
    for step in range(steps+1):
        ids,attention=batches[step%len(batches)]; loss=loss_fn(ids,attention); losses.append(float(loss.detach()))
        if step<steps: opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]
    with torch.no_grad():
        for ids,attention in validation: vals.append(float(loss_fn(ids,attention)))
    result={'model':MODEL,'device':str(device),'steps':steps,'trainable_parameters':out_proj.numel(),'positions':k.positions,'states':k.states,'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_losses':vals,'validation_mean':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'exact_kcode_decoupled_output_gate','input_projection_frozen':True,'output_projection_separate':True}
    out=root/'artifacts/qwen_exact_kcode_decoupled_output'; out.mkdir(parents=True,exist_ok=True); torch.save({'input_projection':k.projection.detach().cpu(),'output_projection':out_proj.detach().cpu(),'model':MODEL,'dataset_hash':manifest['dataset_hash']},out/'projections.pt'); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
