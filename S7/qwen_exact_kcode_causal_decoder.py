"""Causal structured K-code decoder: output positions condition on prior byte states."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

class CausalCodeDecoder(nn.Module):
    def __init__(self,hidden,states,positions,width=256):
        super().__init__(); self.states=states; self.positions=positions; self.state=nn.Embedding(states,width); self.pos=nn.Embedding(positions,width); self.init=nn.Linear(hidden,width); self.gru=nn.GRU(width,width,batch_first=True); self.out=nn.Linear(width,states)
    def forward(self,context,targets):
        # targets: B,T,P states; teacher-force a PAD start then previous state.
        prev=torch.cat([torch.zeros_like(targets[...,:1]),targets[...,:-1]],dim=-1); b,t,p=prev.shape; x=self.state(prev)+self.pos(torch.arange(p,device=prev.device))[None,None,:, :]
        x=x.reshape(b*t,p,-1); h=self.init(context).reshape(1,b*t,-1); y,_=self.gru(x,h); return self.out(y).reshape(b,t,p,self.states)

def run(steps=500):
    torch.set_num_threads(1); torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL); k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype); state=torch.load(root/'artifacts/qwen_exact_kcode_regression/projection.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); model.set_input_embeddings(k); model.to(device)
    for p in model.parameters(): p.requires_grad=False
    decoder=CausalCodeDecoder(model.config.hidden_size,k.states,k.positions).to(device=device,dtype=model.dtype)
    def make(items):
        out=[]
        for start in range(0,len(items),4):
            enc=tok(items[start:start+4],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); out.append((enc['input_ids'].to(device),enc['attention_mask'].to(device)))
        return out
    batches=make([r['target'] for r in manifest['rows']['train']]); validation=make([r['target'] for r in manifest['rows']['validation'][:32]])
    def loss_fn(ids,attention):
        with torch.no_grad(): context=model.model(input_ids=ids,attention_mask=attention).last_hidden_state[:,:-1]
        targets=k.state_table[ids[:,1:]]; logits=decoder(context,targets); valid=(targets!=0)&attention[:,1:].bool()[...,None]; return F.cross_entropy(logits.reshape(-1,k.states),targets.reshape(-1),reduction='none').view_as(targets).masked_select(valid).mean()
    opt=torch.optim.AdamW(decoder.parameters(),lr=1e-4); losses=[]; start=time.perf_counter(); decoder.train()
    for step in range(steps+1):
        ids,attention=batches[step%len(batches)]; loss=loss_fn(ids,attention); losses.append(float(loss.detach()))
        if step<steps: opt.zero_grad(); loss.backward(); opt.step()
    decoder.eval(); vals=[]
    with torch.no_grad():
        for ids,attention in validation: vals.append(float(loss_fn(ids,attention)))
    result={'model':MODEL,'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in decoder.parameters()),'losses':losses,'loss_delta':losses[-1]-losses[0],'validation_losses':vals,'validation_mean':sum(vals)/len(vals),'finite':all(torch.isfinite(torch.tensor(losses+vals)).tolist()),'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'exact_kcode_causal_decoder_gate','input_projection_frozen':True,'output_positions_causal':True}
    out=root/'artifacts/qwen_exact_kcode_causal_decoder'; out.mkdir(parents=True,exist_ok=True); torch.save(decoder.state_dict(),out/'decoder.pt'); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
