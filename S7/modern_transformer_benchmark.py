"""CPU-scale modern decoder-only comparison for normal versus K-code heads."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256
from torch_continuation_lm import continuation_examples

OUT=Path(__file__).resolve().parent/"artifacts"/"modern_transformer_benchmark"
STATES=259; WIDTH=256; LAYERS=4; HEADS=4; FF=768; STEPS=600; BATCH=32

class RMSNorm(nn.Module):
    def __init__(self,d): super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=1e-6
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

class SwiGLU(nn.Module):
    def __init__(self,d,ff): super().__init__(); self.w=nn.Linear(d,ff,bias=False); self.v=nn.Linear(d,ff,bias=False); self.o=nn.Linear(ff,d,bias=False)
    def forward(self,x): return self.o(F.silu(self.w(x))*self.v(x))

class Block(nn.Module):
    def __init__(self,d,h,ff):
        super().__init__(); self.n1=RMSNorm(d); self.attn=nn.MultiheadAttention(d,h,batch_first=True); self.n2=RMSNorm(d); self.ff=SwiGLU(d,ff)
    def forward(self,x,mask):
        y,_=self.attn(self.n1(x),self.n1(x),self.n1(x),attn_mask=mask,need_weights=False); x=x+y; return x+self.ff(self.n2(x))

class ModernLM(nn.Module):
    def __init__(self,kind,seed):
        super().__init__(); torch.manual_seed(seed); self.kind=kind; d=WIDTH
        emb_dim=d if kind in ("normal_untied","normal_tied") else 64
        self.embed=nn.Embedding(STATES,emb_dim); self.in_proj=nn.Identity() if emb_dim==d else nn.Linear(emb_dim,d,bias=False)
        self.blocks=nn.ModuleList([Block(d,HEADS,FF) for _ in range(LAYERS)]); self.norm=RMSNorm(d)
        self.head=nn.Linear(d,STATES,bias=False) if kind=="normal_untied" else None
        self.out_proj=nn.Identity() if kind=="normal_tied" else (nn.Linear(d,emb_dim,bias=False) if emb_dim!=d else nn.Identity())
    def logits(self,x):
        h=self.norm(x)
        if self.kind=="normal_untied": return self.head(h)
        return self.out_proj(h)@self.embed.weight.T
    def forward(self,source,target):
        start=torch.full((source.shape[0],1),2,dtype=torch.long); dec=torch.cat([start,target[:,:-1]],1)
        ids=torch.cat([source,dec],1); x=self.in_proj(self.embed(ids)); n=ids.shape[1]
        mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
        for block in self.blocks: x=block(x,mask)
        return self.logits(x)[:,source.shape[1]:]
    def parameter_report(self): return {"total":sum(p.numel() for p in self.parameters()),"kind":self.kind}

def train_eval(kind,train,val,test,steps=STEPS):
    model=ModernLM(kind,3180); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); g=torch.Generator().manual_seed(3181); started=time.perf_counter()
    model.train()
    for _ in range(steps):
        idx=torch.randint(len(train[0]),(BATCH,),generator=g); logits=model(train[0][idx],train[1][idx]); loss=F.cross_entropy(logits.reshape(-1,STATES),train[1][idx].reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        logits=model(test[0],test[1]); nll=float(F.cross_entropy(logits.reshape(-1,STATES),test[1].reshape(-1),ignore_index=0)); exact=float((logits.argmax(-1)==test[1]).all(1).float().mean())
    return {"parameters":model.parameter_report(),"test_nll":nll,"teacher_forced_exact":exact,"training_seconds":time.perf_counter()-started}

def run():
    OUT.mkdir(parents=True,exist_ok=True); data,audit=build_long_dataset(); codec=ContinuationByteCodec(24); t={s:continuation_examples(codec,r) for s,r in data.items()}
    results={k:train_eval(k,t["train"],t["validation"],t["test"]) for k in ("normal_untied","normal_tied","kcode_tied")}
    out={"experiment":"CPU-scale modern decoder-only Transformer head comparison","config":{"width":WIDTH,"layers":LAYERS,"heads":HEADS,"ff":FF,"steps":STEPS,"batch":BATCH,"architecture":"RMSNorm, causal attention, SwiGLU"},"dataset_hash":sha256(data),"dataset_audit":audit,"results":results}
    (OUT/"results.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); return out
if __name__=="__main__": print(json.dumps(run(),indent=2))
