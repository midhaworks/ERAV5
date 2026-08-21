"""Joint adaptation with separate rates for K-code projection and final Qwen block."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

def run(steps=500):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL); k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype); state=torch.load(root/'artifacts/qwen_exact_kcode_regression/projection.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); model.set_input_embeddings(k); model.to(device)
    for p in model.parameters(): p.requires_grad=False
    k.projection.requires_grad=True
    for p in model.model.layers[-1].parameters(): p.requires_grad=True
    def make(items):
        out=[]
        for start in range(0,len(items),4):
            e=tok(items[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); validation=make([r['target'] for r in manifest['rows']['validation'][:32]])
    def loss_fn(ids,att):
        hidden=model.model(input_ids=ids,attention_mask=att).last_hidden_state[:,:-1]; states=k.state_table[ids[:,1:]]; logits=torch.matmul(hidden,k.projection.t()).view(*hidden.shape[:2],k.positions,k.states); per=F.cross_entropy(logits.reshape(-1,k.states),states.reshape(-1),reduction='none').view_as(states); return per.masked_select((states!=0)&att[:,1:].bool()[...,None]).mean()
    body=[p for p in model.model.layers[-1].parameters() if p.requires_grad]; opt=torch.optim.AdamW([{'params':[k.projection],'lr':1e-4},{'params':body,'lr':1e-5}]); losses=[]; start=time.perf_counter(); model.train()
    for i in range(steps+1):
        ids,att=batches[i%len(batches)]; loss=loss_fn(ids,att); losses.append(float(loss.detach()))
        if i<steps: opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]
    with torch.no_grad():
        for ids,att in validation: vals.append(float(loss_fn(ids,att)))
    result={'model':MODEL,'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in [k.projection]+body),'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_losses':vals,'validation_mean':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'exact_kcode_joint_last_gate','projection_lr':1e-4,'body_lr':1e-5}
    out=root/'artifacts/qwen_exact_kcode_joint_last'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
