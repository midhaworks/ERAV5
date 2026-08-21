"""Matched control: adapt a conventional Qwen input embedding on the same data."""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_kcode_input import MODEL

def run(steps=500):
    torch.set_num_threads(1); torch.manual_seed(3180)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    inp = model.get_input_embeddings()
    # Qwen ties input/output weights. Clone the output head so this control
    # measures input-side adaptation against the same frozen output head as K-code.
    out_head = model.get_output_embeddings()
    if out_head is not None and hasattr(out_head, 'weight'):
        cloned = nn.Linear(out_head.in_features, out_head.out_features, bias=out_head.bias is not None)
        cloned = cloned.to(dtype=out_head.weight.dtype)
        cloned.weight.data.copy_(out_head.weight.data)
        if out_head.bias is not None: cloned.bias.data.copy_(out_head.bias.data)
        model.set_output_embeddings(cloned)
    model.to(device)
    for p in model.parameters(): p.requires_grad = False
    inp.weight.requires_grad = True
    texts = [r['target'] for r in manifest['rows']['train']]
    val_texts = [r['target'] for r in manifest['rows']['validation'][:32]]
    def make_batches(items):
        batches=[]
        for start in range(0, len(items), 4):
            enc=tok(items[start:start+4], add_special_tokens=True, truncation=True,
                    max_length=128, padding=True, return_tensors='pt')
            labels=enc['input_ids'].clone(); labels[enc['attention_mask']==0]=-100
            batches.append((enc['input_ids'].to(device), labels.to(device), enc['attention_mask'].to(device)))
        return batches
    batches=make_batches(texts); validation=make_batches(val_texts)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=1e-4); losses=[]; start=time.perf_counter(); model.train()
    for step in range(steps+1):
        ids,labels,attention=batches[step%len(batches)]
        loss=model(input_ids=ids, attention_mask=attention, labels=labels).loss
        losses.append(float(loss.detach()))
        if step<steps:
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]
    with torch.no_grad():
        for ids,labels,attention in validation:
            vals.append(float(model(input_ids=ids, attention_mask=attention, labels=labels).loss))
    result={'model':MODEL,'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in trainable),
            'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_losses':vals,
            'validation_mean':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),
            'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'matched_normal_input_gate'}
    out=root/'artifacts/qwen_normal_input_adaptation'; out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': run()
