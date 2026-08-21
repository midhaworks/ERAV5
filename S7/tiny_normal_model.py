"""Matched from-scratch conventional embedding baseline."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from qwen_kcode_input import MODEL

class TinyNormal(nn.Module):
    def __init__(self,vocab,d=256,layers=2,heads=4,max_len=64):
        super().__init__(); self.emb=nn.Embedding(vocab,d); self.pos=nn.Embedding(max_len,d); block=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True); self.body=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False); self.head.weight=self.emb.weight
    def forward(self,ids,att):
        x=self.emb(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))[None,:,:]; mask=torch.triu(torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool),1); return self.head(self.norm(self.body(x,mask=mask,src_key_padding_mask=~att.bool())))

def run(steps=2000,batch_size=8):
    torch.manual_seed(3180); torch.set_num_threads(1); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent; manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); max_len=64; vocab=max(tok.vocab_size,int(tok.pad_token_id or 0)+1); model=TinyNormal(vocab,d=256,max_len=max_len).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    def make(items):
        out=[]
        for s in range(0,len(items),batch_size):
            e=tok(items[s:s+batch_size],add_special_tokens=True,truncation=True,max_length=max_len,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); val=make([r['target'] for r in manifest['rows']['validation'][:32]]); losses=[]; start=time.perf_counter(); model.train()
    for i in range(steps):
        ids,att=batches[i%len(batches)]; logits=model(ids,att)[:,:-1]; labels=ids[:,1:]; mask=att[:,1:].bool(); loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').view_as(labels).masked_select(mask).mean(); losses.append(float(loss.detach())); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]; correct=total=0
    with torch.no_grad():
        for ids,att in val:
            logits=model(ids,att)[:,:-1]; labels=ids[:,1:]; mask=att[:,1:].bool(); vals.append(float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').view_as(labels).masked_select(mask).mean())); pred=logits.argmax(-1); correct+=int(((pred==labels)&mask).sum()); total+=int(mask.sum())
    result={'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in model.parameters()),'validation_token_nll':sum(vals)/len(vals),'token_accuracy':correct/total,'loss_first_last':[losses[0],losses[-1]],'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'tiny_from_scratch_normal_gate','model':'2-layer causal transformer','d_model':256,'layers':2,'heads':4}
    out=root/'artifacts/tiny_normal_model'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
