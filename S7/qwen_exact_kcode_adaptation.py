"""Full-corpus Qwen adaptation for the separate EOS-bearing exact K-code."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

def run(steps=500):
    torch.set_num_threads(1); torch.manual_seed(3180)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype)
    state=torch.load(root/'artifacts/qwen_exact_kcode_regression/projection.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['projection'].to(k.projection.dtype))
    model.set_input_embeddings(k); model.to(device)
    for p in model.parameters(): p.requires_grad=False
    k.projection.requires_grad=True
    def make(items):
        out=[]
        for start in range(0,len(items),4):
            enc=tok(items[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); labels=enc['input_ids'].clone(); labels[enc['attention_mask']==0]=-100
            out.append((enc['input_ids'].to(device),labels.to(device),enc['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); validation=make([r['target'] for r in manifest['rows']['validation'][:32]])
    trainable=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(trainable,lr=1e-4); losses=[]; start=time.perf_counter(); model.train()
    for step in range(steps+1):
        ids,labels,attention=batches[step%len(batches)]; loss=model(input_ids=ids,attention_mask=attention,labels=labels).loss; losses.append(float(loss.detach()))
        if step<steps: opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]
    with torch.no_grad():
        for ids,labels,attention in validation: vals.append(float(model(input_ids=ids,attention_mask=attention,labels=labels).loss))
    result={'model':MODEL,'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in trainable),'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_losses':vals,'validation_mean':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'exact_kcode_adaptation_gate'}
    out=root/'artifacts/qwen_exact_kcode_adaptation'; out.mkdir(parents=True,exist_ok=True); torch.save({'projection':k.projection.detach().cpu(),'model':MODEL,'dataset_hash':manifest['dataset_hash']},out/'projection.pt'); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
