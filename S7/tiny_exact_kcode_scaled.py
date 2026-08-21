"""Parameter-budget-matched larger from-scratch exact K-code transformer."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from qwen_kcode_input import MODEL
from qwen_exact_kcode_input import ExactKCodeEmbedding

class ScaledK(nn.Module):
    def __init__(self,tok,d=768,layers=4,heads=12,max_len=64):
        super().__init__(); self.k=ExactKCodeEmbedding(tok,d); self.pos=nn.Embedding(max_len,d); b=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); self.body=nn.TransformerEncoder(b,layers); self.norm=nn.LayerNorm(d)
    def forward(self,ids,att):
        x=self.k(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))[None,:,:]; m=torch.triu(torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool),1); return self.norm(self.body(x,mask=m,src_key_padding_mask=~att.bool()))

def run(steps=2000,batch_size=4):
    torch.manual_seed(3180); torch.set_num_threads(1); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent; manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=ScaledK(tok).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    def make(items):
        out=[]
        for s in range(0,len(items),batch_size):
            e=tok(items[s:s+batch_size],add_special_tokens=True,truncation=True,max_length=64,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); val=make([r['target'] for r in manifest['rows']['validation'][:32]]); losses=[]; start=time.perf_counter(); model.train()
    def loss_fn(ids,att):
        h=model(ids,att)[:,:-1]; st=model.k.state_table[ids[:,1:]]; lg=torch.matmul(h,model.k.projection.t()).view(*h.shape[:2],model.k.positions,model.k.states); valid=(st!=0)&att[:,1:].bool()[...,None]; return F.cross_entropy(lg.reshape(-1,model.k.states),st.reshape(-1),reduction='none').view_as(st).masked_select(valid).mean(),lg,st
    for i in range(steps):
        ids,att=batches[i%len(batches)]; loss,*_=loss_fn(ids,att); losses.append(float(loss.detach())); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]; correct=total=0
    with torch.no_grad():
        for ids,att in val:
            loss,lg,st=loss_fn(ids,att); vals.append(float(loss)); pred=lg.argmax(-1)
            for b in range(ids.shape[0]):
                for t in range(ids.shape[1]-1):
                    if not bool(att[b,t+1]): continue
                    out=[]
                    for q in pred[b,t].tolist():
                        if q==1: break
                        if q in (0,2): out=None; break
                        out.append(q-3)
                    target=tok.convert_ids_to_tokens(int(ids[b,t+1])).encode('utf-8',errors='replace')[:32]
                    correct += int(out is not None and bytes(out)==target); total+=1
    result={'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in model.parameters()),'validation_structured_loss':sum(vals)/len(vals),'decoded_token_accuracy':correct/total,'valid_tokens':total,'loss_first_last':[losses[0],losses[-1]],'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'scaled_from_scratch_exact_kcode_gate','d_model':768,'layers':4,'heads':12}
    out=root/'artifacts/tiny_exact_kcode_scaled'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
