"""Small from-scratch transformer with exact K-code on both input and output."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

class TinyKModel(nn.Module):
    def __init__(self, tokenizer, d=256, layers=2, heads=4, max_len=64):
        super().__init__(); self.k=ExactKCodeEmbedding(tokenizer,d); self.pos=nn.Embedding(max_len,d)
        block=nn.TransformerEncoderLayer(d_model=d,nhead=heads,dim_feedforward=4*d,batch_first=True,activation='gelu',norm_first=True)
        self.body=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d); self.max_len=max_len
    def forward(self,ids,att):
        x=self.k(ids); p=torch.arange(x.shape[1],device=x.device); x=x+self.pos(p)[None,:,:]; mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),1); x=self.body(x,mask=mask,src_key_padding_mask=~att.bool()); return self.norm(x)

def run(steps=500,batch_size=8,resume=False,lr=3e-4):
    torch.manual_seed(3180); torch.set_num_threads(1); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); max_len=64; model=TinyKModel(tok,max_len=max_len).to(device)
    checkpoint=root/'artifacts/tiny_exact_kcode_model/model.pt'
    if resume: model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    texts=[r['target'] for r in manifest['rows']['train']]; valtexts=[r['target'] for r in manifest['rows']['validation'][:32]]
    def make(items):
        out=[]
        for s in range(0,len(items),batch_size):
            e=tok(items[s:s+batch_size],add_special_tokens=True,truncation=True,max_length=max_len,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    batches=make(texts); val=make(valtexts)
    def loss_fn(ids,att):
        h=model(ids,att)[:,:-1]; targets=ids[:,1:]; states=model.k.state_table[targets]; logits=torch.matmul(h,model.k.projection.t()).view(*h.shape[:2],model.k.positions,model.k.states); valid=(states!=0)&att[:,1:].bool()[...,None]; return F.cross_entropy(logits.reshape(-1,model.k.states),states.reshape(-1),reduction='none').view_as(states).masked_select(valid).mean(),logits,states,valid
    losses=[]; start=time.perf_counter(); model.train()
    for i in range(steps):
        ids,att=batches[i%len(batches)]; loss,*_=loss_fn(ids,att); losses.append(float(loss.detach())); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]; exact=valid_tokens=0
    decoded_correct=0
    with torch.no_grad():
        for ids,att in val:
            loss,logits,states,valid=loss_fn(ids,att); vals.append(float(loss)); pred=logits.argmax(-1)
            for b in range(ids.shape[0]):
                for t in range(ids.shape[1]-1):
                    if not bool(att[b,t+1]): continue
                    ps=pred[b,t].tolist(); ts=states[b,t].tolist();
                    if ps==ts: exact+=1
                    def decode(seq):
                        out=[]
                        for q in seq:
                            if q==1: return bytes(out)
                            if q in (0,2): return None
                            out.append(q-3)
                        return None
                    target_bytes=tok.convert_ids_to_tokens(int(ids[b,t+1])).encode('utf-8',errors='replace')[:32]
                    if decode(ps)==target_bytes: decoded_correct+=1
                    valid_tokens+=1
    result={'device':str(device),'steps':steps,'resumed':resume,'learning_rate':lr,'trainable_parameters':sum(p.numel() for p in model.parameters()),'validation_structured_loss':sum(vals)/len(vals),'exact_code_token_accuracy':exact/valid_tokens,'decoded_token_accuracy':decoded_correct/valid_tokens,'valid_tokens':valid_tokens,'loss_first_last':[losses[0],losses[-1]],'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'tiny_from_scratch_exact_kcode_gate','model':'2-layer causal transformer','d_model':256,'layers':2,'heads':4}
    out=root/'artifacts/tiny_exact_kcode_model'; out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/'model.pt'); (out/('continued_results.json' if resume else 'results.json')).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
