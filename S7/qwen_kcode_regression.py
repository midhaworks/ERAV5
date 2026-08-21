"""Initialize Qwen K-code projection by regressing to its pretrained embeddings."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_kcode_input import KCodeEmbedding,MODEL

def run(steps=200,batch=512):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    root=Path(__file__).resolve().parent; manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL); original=model.get_input_embeddings().weight.detach().clone()
    k=KCodeEmbedding(tok,model.config.hidden_size).to(model.dtype).to(device); target=original.to(device)
    opt=torch.optim.AdamW([k.projection],lr=2e-3); g=torch.Generator().manual_seed(3181); losses=[]; start=time.perf_counter()
    k.train()
    for step in range(steps):
        ids=(torch.arange(batch)+step*batch).remainder(target.shape[0]).to(device)
        pred=k(ids); gold=target[ids]
        loss=(pred.float()-gold.float()).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    # Evaluate language loss on the same validation subset after regression.
    model.set_input_embeddings(k); model.to(device); model.eval(); vals=[]
    for r in manifest['rows']['validation'][:8]:
        enc=tok(r['target'],add_special_tokens=True,truncation=True,max_length=128,return_tensors='pt'); ids=enc['input_ids'].to(device)
        with torch.no_grad(): vals.append(float(model(input_ids=ids,labels=ids).loss))
    out=root/'artifacts/qwen_kcode_regression'; out.mkdir(parents=True,exist_ok=True); torch.save({'projection':k.projection.detach().cpu(),'dataset_hash':manifest['dataset_hash']},out/'projection.pt')
    result={'model':MODEL,'device':str(device),'steps':steps,'batch':batch,'losses_first_last':[losses[0],losses[-1]],'embedding_mse':losses[-1],'validation_losses':vals,'validation_mean':sum(vals)/len(vals),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'embedding_regression_initialization'}
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
