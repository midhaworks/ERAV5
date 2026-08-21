"""Bounded CPU gate: adapt only the fixed K-code projection."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_kcode_input import KCodeEmbedding,MODEL

def run(steps=20, unfreeze_first=False):
    torch.set_num_threads(1); torch.manual_seed(3180)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    root=Path(__file__).resolve().parent; manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    k=KCodeEmbedding(tok,model.config.hidden_size).to(model.dtype)
    regression_path=root/'artifacts/qwen_kcode_regression'/'projection.pt'
    if regression_path.exists():
        state=torch.load(regression_path,map_location='cpu',weights_only=True); k.projection.data.copy_(state['projection'].to(k.projection.dtype))
    model.set_input_embeddings(k); model.to(device)
    for p in model.parameters(): p.requires_grad=False
    k.projection.requires_grad=True
    if unfreeze_first:
        for p in model.model.layers[0].parameters(): p.requires_grad=True
    texts=[' '.join([r['target']]) for r in manifest['rows']['train']]
    validation_texts=[' '.join([r['target']]) for r in manifest['rows']['validation'][:32]]
    batches=[]
    for start in range(0,len(texts),4):
        enc=tok(texts[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt')
        ids=enc['input_ids']; labels=ids.clone(); labels[enc['attention_mask']==0]=-100
        batches.append((ids.to(device),labels.to(device),enc['attention_mask'].to(device)))
    validation=[]
    for start in range(0,len(validation_texts),4):
        enc=tok(validation_texts[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt')
        labels=enc['input_ids'].clone(); labels[enc['attention_mask']==0]=-100
        validation.append((enc['input_ids'].to(device),labels.to(device),enc['attention_mask'].to(device)))
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=1e-4); losses=[]; start=time.perf_counter(); model.train()
    for step in range(steps+1):
        ids,labels,attention=batches[step%len(batches)]
        out=model(input_ids=ids,attention_mask=attention,labels=labels); loss=out.loss
        losses.append(float(loss.detach()))
        if step<steps:
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); validation_losses=[]
    with torch.no_grad():
        for ids,labels,attention in validation:
            validation_losses.append(float(model(input_ids=ids,attention_mask=attention,labels=labels).loss))
    result={'model':MODEL,'device':str(device),'steps':steps,'unfreeze_first':unfreeze_first,'trainable_parameters':sum(p.numel() for p in trainable),
            'losses':losses,'loss_delta':losses[-1]-losses[0],
            'validation_losses':validation_losses,'validation_mean':sum(validation_losses)/len(validation_losses),
            'finite':all(torch.isfinite(torch.tensor(losses+validation_losses)).tolist()),
            'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],
            'status':'projection_only_adaptation_gate'}
    out=root/'artifacts/qwen_kcode_adaptation'; out.mkdir(parents=True,exist_ok=True)
    torch.save({'projection':k.projection.detach().cpu(),'model':MODEL,'dataset_hash':manifest['dataset_hash']},out/'projection.pt')
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
