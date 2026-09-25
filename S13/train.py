"""S13: 20M-parameter language-model training and reversible comparison."""
from __future__ import annotations
import argparse, json, math, os, resource, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VOCAB, SEQ, TARGET_TOKENS, SEED = 256, 256, 50_000_000, 1313

class BaselineBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, ff=2048):
        super().__init__(); self.norm1=nn.LayerNorm(d_model); self.attn=nn.MultiheadAttention(d_model,heads,batch_first=True); self.norm2=nn.LayerNorm(d_model); self.ff=nn.Sequential(nn.Linear(d_model,ff),nn.GELU(),nn.Linear(ff,d_model))
    def forward(self,x,mask):
        h=self.norm1(x); x=x+self.attn(h,h,h,attn_mask=mask,need_weights=False)[0]; return x+self.ff(self.norm2(x))

class BaselineLM(nn.Module):
    def __init__(self,layers=6):
        super().__init__(); self.token=nn.Embedding(VOCAB,512); self.position=nn.Embedding(SEQ,512); self.blocks=nn.ModuleList([BaselineBlock() for _ in range(layers)]); self.norm=nn.LayerNorm(512); self.head=nn.Linear(512,VOCAB,bias=False)
    def forward(self,tokens):
        p=torch.arange(tokens.shape[1],device=tokens.device); x=self.token(tokens)+self.position(p)[None]; mask=torch.triu(torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool),1)
        for block in self.blocks: x=block(x,mask)
        return self.head(self.norm(x))

class EulerReversibleBlock(nn.Module):
    """Additive coupling Euler map y1=x1+dt*f(x2), y2=x2+dt*g(y1)."""
    def __init__(self,half=256,inner=2560,heads=4,dt=1.0):
        super().__init__(); self.dt=dt; self.nf=nn.LayerNorm(half); self.ng=nn.LayerNorm(half); self.f_attn=nn.MultiheadAttention(half,heads,batch_first=True); self.g_attn=nn.MultiheadAttention(half,heads,batch_first=True); self.f_ff=nn.Sequential(nn.Linear(half,inner),nn.GELU(),nn.Linear(inner,half)); self.g_ff=nn.Sequential(nn.Linear(half,inner),nn.GELU(),nn.Linear(inner,half))
    def _f(self,x,mask):
        h=self.nf(x); h=h+self.f_attn(h,h,h,attn_mask=mask,need_weights=False)[0]; return self.f_ff(h)
    def _g(self,x,mask):
        h=self.ng(x); h=h+self.g_attn(h,h,h,attn_mask=mask,need_weights=False)[0]; return self.g_ff(h)
    def forward(self,x,mask):
        x1,x2=x.chunk(2,dim=-1); y1=x1+self.dt*self._f(x2,mask); y2=x2+self.dt*self._g(y1,mask); return torch.cat((y1,y2),dim=-1)

class EulerReversibleLM(nn.Module):
    def __init__(self,layers=6):
        super().__init__(); self.token=nn.Embedding(VOCAB,512); self.position=nn.Embedding(SEQ,512); self.blocks=nn.ModuleList([EulerReversibleBlock() for _ in range(layers)]); self.norm=nn.LayerNorm(512); self.head=nn.Linear(512,VOCAB,bias=False)
    def forward(self,tokens):
        p=torch.arange(tokens.shape[1],device=tokens.device); x=self.token(tokens)+self.position(p)[None]; mask=torch.triu(torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool),1)
        for block in self.blocks: x=block(x,mask)
        return self.head(self.norm(x))

def make_batch(batch_size,device,generator,stream=None):
    if stream is None: return torch.randint(0,VOCAB,(batch_size,SEQ),generator=generator,device=device)
    starts=torch.randint(0,stream.numel()-SEQ,(batch_size,),generator=generator)
    return torch.stack([stream[int(start):int(start)+SEQ] for start in starts]).to(device)
def memory_peak(device):
    if device.type=="cuda": return torch.cuda.max_memory_allocated(device)/2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if os.uname().sysname=="Darwin" else 1024)
def build(variant): return BaselineLM() if variant=="baseline" else EulerReversibleLM()

def train(args,batch_size=None,steps_override=None):
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")); batch_size=batch_size or args.batch_size; model=build(args.variant).to(device); params=sum(p.numel() for p in model.parameters())
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr); generator=torch.Generator(device="cpu").manual_seed(args.seed); stream=None
    if args.corpus: stream=torch.tensor(list(Path(args.corpus).read_bytes()),dtype=torch.long)
    steps=steps_override or math.ceil(TARGET_TOKENS/(batch_size*SEQ)); steps=min(steps,args.max_steps) if args.max_steps else steps; losses=[]; seen=0; started=time.perf_counter(); model.train()
    for _ in range(steps):
        batch=make_batch(batch_size,device,generator,stream); logits=model(batch); loss=F.cross_entropy(logits[:,:-1].reshape(-1,VOCAB),batch[:,1:].reshape(-1)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); seen+=batch.numel()-batch.shape[0]; losses.append(float(loss.detach()))
    elapsed=time.perf_counter()-started
    return {"variant":args.variant,"device":str(device),"batch_size":batch_size,"steps":steps,"tokens_seen":seen,"target_tokens":TARGET_TOKENS,"final_loss":losses[-1],"initial_loss":losses[0],"tokens_per_second":seen/elapsed,"elapsed_seconds":elapsed,"peak_memory_mib":memory_peak(device),"parameters":params,"loss_trace":losses,"sequence_length":SEQ}

def find_max_batch(args):
    candidate=1; good=1; trials=[]
    while candidate<=args.max_probe_batch:
        try:
            result=train(args,batch_size=candidate,steps_override=1); trials.append({"batch_size":candidate,"status":"pass","peak_memory_mib":result["peak_memory_mib"]}); good=candidate; candidate*=2
        except RuntimeError as exc:
            trials.append({"batch_size":candidate,"status":"fail","error":str(exc)[:300]}); break
    return good,trials

def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",choices=["baseline","euler"],default="baseline"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--max-probe-batch",type=int,default=64); p.add_argument("--auto-max-batch",action="store_true"); p.add_argument("--max-steps",type=int,default=0); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--device",default=None); p.add_argument("--seed",type=int,default=SEED); p.add_argument("--corpus",default=None); args=p.parse_args()
    ARTIFACTS.mkdir(exist_ok=True); selected=args.batch_size; trials=None
    if args.auto_max_batch: selected,trials=find_max_batch(args)
    result=train(args,batch_size=selected); result["auto_max_batch_trials"]=trials; result["status"]="completed_target" if result["tokens_seen"]>=TARGET_TOKENS else "bounded_smoke_or_partial"; (ARTIFACTS/f"{args.variant}_batch_{selected}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","variant","batch_size","steps","tokens_seen","final_loss","tokens_per_second","peak_memory_mib","parameters")},indent=2))

if __name__=="__main__": main()
