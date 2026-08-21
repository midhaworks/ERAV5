"""Parameter-free vocabulary-constrained decoding for the trained parallel K-code model."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from qwen_kcode_input import MODEL
from tiny_exact_kcode_model import TinyKModel

def run(limit=32,alpha=1.0,split='validation',checkpoint_path=None,output_tag=None):
    torch.set_num_threads(1); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifacts/qwen_data_manifest/results.json').read_text()); tok=AutoTokenizer.from_pretrained(MODEL); model=TinyKModel(tok,max_len=64).to(device); checkpoint=Path(checkpoint_path) if checkpoint_path else root/'artifacts/tiny_exact_kcode_model/model.pt'; model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
    candidates=model.k.state_table[:tok.vocab_size].to(device); lengths=(candidates!=0).sum(-1).clamp_min(1).float(); rows=manifest['rows'][split][:limit]; correct=unconstrained_correct=total=0; start=time.perf_counter()
    with torch.no_grad():
        for s in range(0,len(rows),8):
            e=tok([r['target'] for r in rows[s:s+8]],add_special_tokens=True,truncation=True,max_length=64,padding=True,return_tensors='pt'); ids=e['input_ids'].to(device); att=e['attention_mask'].to(device); h=model(ids,att)[:,:-1]; logits=torch.matmul(h,model.k.projection.t()).view(*h.shape[:2],model.k.positions,model.k.states); lp=F.log_softmax(logits.float(),-1); active=att[:,1:].bool(); flat=lp[active]; targets=ids[:,1:][active]; target_codes=candidates[targets]; target_active=target_codes!=0; unconstrained_correct+=int(((flat.argmax(-1)==target_codes)|~target_active).all(-1).sum()); best_score=torch.full((flat.shape[0],),-torch.inf,device=device); best_id=torch.zeros(flat.shape[0],dtype=torch.long,device=device)
            for c0 in range(0,candidates.shape[0],2048):
                code=candidates[c0:c0+2048]; score=torch.zeros((flat.shape[0],code.shape[0]),device=device)
                for p in range(model.k.positions):
                    state=code[:,p]; contribution=flat[:,p,:][:,state]; score += contribution*(state!=0)[None,:]
                score=score/(lengths[c0:c0+code.shape[0]][None,:]**alpha); value,index=score.max(-1); improve=value>best_score; best_score=torch.where(improve,value,best_score); best_id=torch.where(improve,index+c0,best_id)
            correct+=int((best_id==targets).sum()); total+=int(targets.numel())
    result={'device':str(device),'split':split,'examples':len(rows),'tokens':total,'length_normalization_alpha':alpha,'vocabulary_candidates':tok.vocab_size,'token_accuracy':correct/total,'constrained_token_accuracy':correct/total,'constrained_correct':correct,'unconstrained_exact_code_accuracy':unconstrained_correct/total,'unconstrained_correct':unconstrained_correct,'absolute_accuracy_gain':(correct-unconstrained_correct)/total,'seconds':time.perf_counter()-start,'tokens_per_second':total/(time.perf_counter()-start),'trainable_decoder_parameters':0,'dataset_hash':manifest['dataset_hash'],'checkpoint':str(checkpoint),'status':'fixed_vocab_constrained_kcode_decode'}
    out=root/'artifacts/tiny_kcode_vocab_decode'; out.mkdir(parents=True,exist_ok=True); tag=f'_{output_tag}' if output_tag else ''; (out/f'results_{split}_alpha_{alpha}{tag}.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
