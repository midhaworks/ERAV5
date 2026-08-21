"""From-scratch transformer with a causal, fully shared exact K-code decoder."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from qwen_kcode_input import MODEL
from qwen_exact_kcode_input import ExactKCodeEmbedding

class CausalSharedK(nn.Module):
    def __init__(self,tok,d=256,layers=2,heads=4,max_len=64):
        super().__init__(); self.k=ExactKCodeEmbedding(tok,d); self.pos=nn.Embedding(max_len,d)
        block=nn.TransformerEncoderLayer(d,heads,4*d,batch_first=True,activation='gelu',norm_first=True)
        self.body=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d)
        self.slot_pos=nn.Embedding(self.k.positions,d); self.bos=nn.Parameter(torch.zeros(d)); self.cell=nn.GRUCell(d,d)
    def context(self,ids,att):
        x=self.k(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))[None,:,:]
        mask=torch.triu(torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool),1)
        return self.norm(self.body(x,mask=mask,src_key_padding_mask=~att.bool()))
    def decode_teacher(self,context,targets,sample_probability=0.0):
        b,t,p=targets.shape; h=context.reshape(b*t,-1); logits=[]
        predicted_previous=None
        for slot in range(p):
            if slot==0: previous=self.bos.expand(b*t,-1)
            else:
                teacher_previous=targets[...,slot-1].reshape(-1)
                if sample_probability>0 and predicted_previous is not None:
                    choose=torch.rand(teacher_previous.shape,device=teacher_previous.device)<sample_probability
                    previous_state=torch.where(choose,predicted_previous,teacher_previous)
                else: previous_state=teacher_previous
                rows=(slot-1)*self.k.states+previous_state
                previous=self.k.projection[rows]
            h=self.cell(previous+self.slot_pos.weight[slot],h)
            codebook=self.k.projection[slot*self.k.states:(slot+1)*self.k.states]
            score=h@codebook.t(); illegal=torch.zeros(self.k.states,device=score.device,dtype=score.dtype); illegal[0]=illegal[2]=-1e4
            score=score+illegal; logits.append(score); predicted_previous=score.detach().argmax(-1)
        return torch.stack(logits,2).reshape(b,t,p,self.k.states)
    def decode_greedy(self,context):
        b,t,d=context.shape; h=context.reshape(b*t,d); states=[]
        for slot in range(self.k.positions):
            if slot==0: previous=self.bos.expand(b*t,-1)
            else:
                rows=(slot-1)*self.k.states+states[-1]
                previous=self.k.projection[rows]
            h=self.cell(previous+self.slot_pos.weight[slot],h)
            codebook=self.k.projection[slot*self.k.states:(slot+1)*self.k.states]
            score=h@codebook.t(); illegal=torch.zeros(self.k.states,device=score.device,dtype=score.dtype); illegal[0]=illegal[2]=-1e4
            states.append((score+illegal).argmax(-1))
        return torch.stack(states,-1).reshape(b,t,self.k.positions)

def run(steps=10000,batch_size=8):
    torch.manual_seed(3180); torch.set_num_threads(1); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=CausalSharedK(tok).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    def make(items):
        out=[]
        for s in range(0,len(items),batch_size):
            e=tok(items[s:s+batch_size],add_special_tokens=True,truncation=True,max_length=64,padding=True,return_tensors='pt'); out.append((e['input_ids'].to(device),e['attention_mask'].to(device)))
        return out
    train=make([r['target'] for r in manifest['rows']['train']]); val=make([r['target'] for r in manifest['rows']['validation'][:32]])
    def loss_fn(ids,att,sample_probability=0.0):
        context=model.context(ids,att)[:,:-1]; states=model.k.state_table[ids[:,1:]]; logits=model.decode_teacher(context,states,sample_probability); valid=(states!=0)&att[:,1:].bool()[...,None]
        loss=F.cross_entropy(logits.reshape(-1,model.k.states),states.reshape(-1),reduction='none').view_as(states).masked_select(valid).mean(); return loss,context,states
    losses=[]; start=time.perf_counter(); model.train()
    for i in range(steps):
        ids,att=train[i%len(train)]; probability=0.5*min(1.0,i/max(1,steps//2)); loss,*_=loss_fn(ids,att,probability); losses.append(float(loss.detach())); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); vals=[]; correct=total=valid_utf8=0
    with torch.no_grad():
        for ids,att in val:
            loss,context,targets=loss_fn(ids,att); vals.append(float(loss)); pred=model.decode_greedy(context)
            for b in range(ids.shape[0]):
                for t in range(ids.shape[1]-1):
                    if not bool(att[b,t+1]): continue
                    out=[]
                    for q in pred[b,t].tolist():
                        if q==1: break
                        if q in (0,2): out=None; break
                        out.append(q-3)
                    target=tok.convert_ids_to_tokens(int(ids[b,t+1])).encode('utf-8',errors='replace')[:32]
                    if out is not None:
                        try: bytes(out).decode('utf-8'); valid_utf8+=1
                        except UnicodeDecodeError: pass
                    correct+=int(out is not None and bytes(out)==target); total+=1
    result={'device':str(device),'steps':steps,'trainable_parameters':sum(p.numel() for p in model.parameters()),'validation_structured_loss':sum(vals)/len(vals),'decoded_token_accuracy':correct/total,'valid_utf8_rate':valid_utf8/total,'valid_tokens':total,'loss_first_last':[losses[0],losses[-1]],'seconds':time.perf_counter()-start,'dataset_hash':manifest['dataset_hash'],'status':'causal_shared_exact_kcode_gate','d_model':256,'layers':2,'heads':4,'shared_projection':True,'scheduled_sampling_max':0.5}
    out=root/'artifacts/tiny_exact_kcode_autoregressive'; out.mkdir(parents=True,exist_ok=True); (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
