"""Token-level evaluation for the shared structured K-code output head."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from qwen_exact_kcode_input import ExactKCodeEmbedding,MODEL

def run(mode='shared'):
    root=Path(__file__).resolve().parent; device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    k=ExactKCodeEmbedding(tok,model.config.hidden_size).to(model.dtype)
    if mode=='distill':
        state=torch.load(root/'artifacts/qwen_kcode_distill/projections.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['input_projection'].to(k.projection.dtype)); out_proj=state['output_projection'].to(device=device,dtype=k.projection.dtype)
    elif mode=='decoupled':
        state=torch.load(root/'artifacts/qwen_exact_kcode_decoupled_output/projections.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['input_projection'].to(k.projection.dtype)); out_proj=state['output_projection'].to(device=device,dtype=k.projection.dtype)
    else:
        state=torch.load(root/'artifacts/qwen_exact_kcode_dual_sided/projection.pt',map_location='cpu',weights_only=True); k.projection.data.copy_(state['projection'].to(k.projection.dtype)); out_proj=k.projection
    model.set_input_embeddings(k); model.to(device).eval()
    total_nll=total_tokens=correct=valid=0; byte_nll=byte_count=0
    rows=manifest['rows']['validation'][:32]
    with torch.no_grad():
      for start in range(0,len(rows),4):
        enc=tok([r['target'] for r in rows[start:start+4]],add_special_tokens=True,truncation=True,max_length=128,padding=True,return_tensors='pt'); ids=enc['input_ids'].to(device); att=enc['attention_mask'].to(device)
        hidden=model.model(input_ids=ids,attention_mask=att).last_hidden_state[:,:-1]; targets=ids[:,1:]; mask=att[:,1:].bool(); states=k.state_table[targets]
        logits=torch.matmul(hidden,out_proj.t()).view(*hidden.shape[:2],k.positions,k.states); lp=F.log_softmax(logits.float(),dim=-1)
        gathered=lp.gather(-1,states.unsqueeze(-1)).squeeze(-1); valid_states=(states!=0)&mask[...,None]; token_lp=(gathered*valid_states).sum(-1); token_n=(valid_states.sum(-1)>0)&mask
        total_nll += float((-token_lp[token_n]).sum()); total_tokens += int(token_n.sum()); byte_nll += float((-gathered[valid_states]).sum()); byte_count += int(valid_states.sum())
        pred=logits.argmax(-1)
        for b in range(ids.shape[0]):
          for t in range(ids.shape[1]-1):
            if not bool(mask[b,t]): continue
            p=pred[b,t].tolist(); out=[]
            for s in p:
              if s==1: break
              if s in (0,2): out=None; break
              out.append(s-3)
            target=bytes(tok.convert_ids_to_tokens(int(targets[b,t])).encode('utf-8',errors='replace')[:32])
            if out is not None and bytes(out)==target: correct+=1
            valid+=1
    result={'model':MODEL,'device':str(device),'examples':len(rows),'token_count':total_tokens,'token_nll':total_nll/total_tokens,'byte_state_nll':byte_nll/byte_count,'exact_token_accuracy':correct/valid,'valid_decodes':valid,'status':'dual_sided_token_level_eval','mode':mode,'dataset_hash':manifest['dataset_hash']}
    out=root/('artifacts/qwen_kcode_distill' if mode=='distill' else ('artifacts/qwen_exact_kcode_decoupled_output' if mode=='decoupled' else 'artifacts/qwen_exact_kcode_dual_sided')); (out/('token_eval_distill.json' if mode=='distill' else ('token_eval_decoupled.json' if mode=='decoupled' else 'token_eval.json'))).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
