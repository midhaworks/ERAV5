"""Fixed UTF-8 byte-position K-code input replacement for Qwen."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL='Qwen/Qwen2.5-0.5B'; MAX_BYTES=32; BYTE_STATES=256

class KCodeEmbedding(nn.Module):
    def __init__(self, tokenizer, hidden_size, max_bytes=MAX_BYTES):
        super().__init__(); self.max_bytes=max_bytes
        self.projection=nn.Parameter(torch.empty((BYTE_STATES*max_bytes,hidden_size)))
        nn.init.normal_(self.projection, mean=0.0, std=1.0/(BYTE_STATES*max_bytes)**0.5)
        vocab_size=max(tokenizer.vocab_size, int(tokenizer.pad_token_id or 0)+1)
        table=torch.zeros((vocab_size,max_bytes),dtype=torch.long)
        for token_id in range(vocab_size):
            token=tokenizer.convert_ids_to_tokens(token_id)
            raw=token.encode('utf-8',errors='replace')[:max_bytes]
            if raw: table[token_id,:len(raw)]=torch.tensor(list(raw),dtype=torch.long)+1
        self.register_buffer('byte_table',table,persistent=True)
    def forward(self, input_ids):
        ids=self.byte_table[input_ids]
        valid=ids>0; rows=ids.clamp_min(1)-1
        offsets=torch.arange(self.max_bytes,device=ids.device)[None,None,:]*BYTE_STATES
        rows=rows+offsets
        vectors=self.projection[rows]
        vectors=vectors*valid[...,None]
        lengths=valid.sum(-1,keepdim=True).clamp_min(1).to(vectors.dtype)
        return vectors.sum(-2)/lengths.sqrt()

def smoke():
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    original=model.get_input_embeddings(); k=KCodeEmbedding(tok,model.config.hidden_size).to(model.dtype)
    model.set_input_embeddings(k)
    text='भारत की राजधानी नई दिल्ली है।'
    ids=tok(text,return_tensors='pt')['input_ids']
    with torch.no_grad(): out=model(input_ids=ids,labels=ids)
    original_params=original.weight.numel(); k_params=k.projection.numel()
    result={'model':MODEL,'vocab_size':tok.vocab_size,'hidden_size':model.config.hidden_size,
            'original_input_parameters':original_params,'kcode_projection_parameters':k_params,
            'input_parameter_reduction':1-k_params/original_params,'loss':float(out.loss),
            'tokens':int(ids.numel()),'max_bytes':MAX_BYTES,'status':'input_path_smoke'}
    outdir=Path(__file__).resolve().parent/'artifacts'/'qwen_kcode_input'; outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': smoke()
