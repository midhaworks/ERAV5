"""Small matched multi-block pilot for fixed dual-sided K-code output."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cross_block_lm import build_long_dataset
from rke import ContinuationByteCodec, sha256

OUT=Path(__file__).resolve().parent/"artifacts"/"block_dual_sided_benchmark"
P=25; S=259; D=P*S; H=128; STEPS=300; BATCH=32

def make_rows(records,codec):
    rows=[]
    for r in records:
        blocks=codec.ids(r['target'].encode());
        if len(blocks)<2: continue
        target=np.zeros((3,P),dtype=np.int64); mask=np.zeros(3,dtype=np.bool_)
        for j,b in enumerate(blocks[1:4]): target[j]=b; mask[j]=True
        rows.append((blocks[0],target,mask))
    return torch.tensor(np.stack([x[0] for x in rows])),torch.tensor(np.stack([x[1] for x in rows])),torch.tensor(np.stack([x[2] for x in rows]))

class BlockModel(nn.Module):
    def __init__(self,kind,seed):
        super().__init__(); torch.manual_seed(seed); self.kind=kind
        if kind=='fixed':
            self.proj=nn.Linear(D,H,bias=False); self.codebook=None
        else: self.embed=nn.Embedding(S,H)
        self.decoder=nn.GRU(H,H,batch_first=True); self.head=nn.Linear(H,D,bias=False) if kind=='normal' else None
    def encode(self,source):
        if self.kind=='fixed':
            code=torch.zeros(source.shape[0],D); code.scatter_(1,(torch.arange(P)[None,:]*S+source).long(),1)
            return self.proj(code).unsqueeze(1)
        return self.embed(source).mean(1).unsqueeze(1)
    def block_input(self, blocks):
        if self.kind=='fixed':
            code=torch.zeros(blocks.shape[0],D)
            code.scatter_(1,(torch.arange(P)[None,:]*S+blocks).long(),1)
            return self.proj(code)
        return self.embed(blocks).mean(1)
    def forward(self,source,target):
        hidden=self.encode(source).transpose(0,1)
        previous=torch.cat([source[:,None,:],target[:,:-1,:]],1)
        inp=torch.stack([self.block_input(previous[:,j,:]) for j in range(target.shape[1])],1)
        out,_=self.decoder(inp,hidden); h=out
        if self.kind=='fixed':
            # fixed projection is shared: decoder hidden is projected to K-code
            w=self.proj.weight
            logits=h@w
        else: logits=self.head(h)
        return logits
    def params(self): return sum(p.numel() for p in self.parameters())

def run_arm(kind,train,test):
    model=BlockModel(kind,3180); opt=torch.optim.AdamW(model.parameters(),lr=2e-3); g=torch.Generator().manual_seed(3181); start=time.perf_counter()
    for _ in range(STEPS):
        idx=torch.randint(len(train[0]),(BATCH,),generator=g); logits=model(train[0][idx],train[1][idx]); gold=train[1][idx]
        loss=F.cross_entropy(logits.reshape(-1,S),gold.reshape(-1),ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); exact=valid=0
    with torch.no_grad():
        logits=model(test[0],test[1]); pred=logits.reshape(len(test[0]),3,P,S).argmax(-1)
        nll=float(F.cross_entropy(logits.reshape(-1,S),test[1].reshape(-1),ignore_index=0))
        for i in range(len(pred)):
            for j in range(3):
                if not test[2][i,j]: continue
                valid+=1; exact+=int(torch.equal(pred[i,j],test[1][i,j]))
    return {'parameters':model.params(),'test_nll':nll,'loss_bearing_block_exact':exact/max(valid,1),'blocks':valid,'training_seconds':time.perf_counter()-start}

def run():
    OUT.mkdir(parents=True,exist_ok=True); data,audit=build_long_dataset(); c=ContinuationByteCodec(24)
    train=make_rows(data['train'],c); test=make_rows(data['test'],c)
    results={k:run_arm(k,train,test) for k in ('normal','fixed')}
    out={'experiment':'matched multi-block fixed dual-sided K-code pilot','config':{'positions':P,'states':S,'dimension':D,'hidden':H,'steps':STEPS},'dataset_hash':sha256(data),'dataset_audit':audit,'results':results,'note':'Block-level pilot; not yet a full SOTA language-model claim.'}
    (OUT/'results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': run()
