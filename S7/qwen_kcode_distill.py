"""Distill Qwen vocabulary probabilities into an exact K-code output head."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

def run(steps=100,temperature=1.5):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL); original=model.get_input_embeddings(); original.weight.requires_grad=False
    state=torch.load(root/'artifacts/qwen_exact_kcode_regression/projection.pt',map_location='cpu',weights_only=True); k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); out_proj=nn.Parameter(k.projection.detach().clone())
    model.to(device); original=model.get_input_embeddings(); model.set_input_embeddings(original)
    for p in model.parameters(): p.requires_grad=False
    out_proj=nn.Parameter(out_proj.data.to(device)); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); k=k.to(device); k.projection.requires_grad=False
    vocab=model.lm_head.out_features; groups=torch.zeros((vocab,k.positions),dtype=torch.long,device=device); groups[:k.state_table.shape[0]]=k.state_table.to(device)
    def make(items):
        out=[]
        for start in range(0,len(items),4):
            e=tok(items[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); validation=make([r['target'] for r in manifest['rows']['validation'][:32]])
    def step_loss(ids,att,train=True):
        model.set_input_embeddings(original)
        with torch.no_grad(): teacher=model(input_ids=ids,attention_mask=att).logits[:,:-1].float()/temperature
        model.set_input_embeddings(k); hidden=model.model(input_ids=ids,attention_mask=att).last_hidden_state[:,:-1]; student=torch.matmul(hidden,out_proj.t()).view(*hidden.shape[:2],k.positions,k.states).float()/temperature
        probs=teacher.softmax(-1); b,l,v=probs.shape; target=torch.zeros((b,l,k.positions,k.states),device=device)
        for p in range(k.positions): target[:,:,p,:].scatter_add_(2,groups[:,p][None,None,:].expand(b,l,v),probs)
        target=target.clamp_min(1e-8); slog=student.log_softmax(-1); kl=(target*(target.log()-slog)).sum(-1); states=k.state_table[ids[:,1:]]; hard=F.cross_entropy(student.reshape(-1,k.states),states.reshape(-1),reduction='none').view_as(states); valid=(states!=0)&att[:,1:].bool()[...,None]
        return (kl.mean()+0.2*hard.masked_select(valid).mean())*(temperature**2)
    opt=torch.optim.AdamW([out_proj],lr=1e-4); losses=[]; start=time.perf_counter()
    for i in range(steps):
        ids,att=batches[i%len(batches)]; loss=step_loss(ids,att); losses.append(float(loss.detach())); opt.zero_grad(); loss.backward(); opt.step()
    vals=[]
    with torch.no_grad():
        for ids,att in validation: vals.append(float(step_loss(ids,att)))
    result={'model':MODEL,'device':str(device),'steps':steps,'temperature':temperature,'trainable_parameters':out_proj.numel(),'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_distill_loss':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'exact_kcode_teacher_distillation_gate'}
    out=root/'artifacts/qwen_kcode_distill'; out.mkdir(parents=True,exist_ok=True); torch.save({'input_projection':k.projection.detach().cpu(),'output_projection':out_proj.detach().cpu(),'model':MODEL,'dataset_hash':manifest['dataset_hash']},out/'projections.pt'); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
