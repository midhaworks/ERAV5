"""Benchmark an inference-time materialized K-code lookup table."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer
from qwen_kcode_input import KCodeEmbedding, MODEL

def run(iterations=100,batch=64,seq=32):
    torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(MODEL); k=KCodeEmbedding(tok,896).to(device=device,dtype=torch.float16).eval()
    with torch.no_grad():
        chunks=[]
        for start in range(0,k.byte_table.shape[0],4096):
            ids=torch.arange(start,min(start+4096,k.byte_table.shape[0]),device=device)[:,None]
            chunks.append(k(ids).squeeze(1))
        table=torch.cat(chunks,dim=0)
    cached=nn.Embedding.from_pretrained(table,freeze=True).to(device).eval()
    ids=torch.randint(0,table.shape[0],(batch,seq),device=device)
    with torch.no_grad():
        for _ in range(20): cached(ids)
        if device.type=='mps': torch.mps.synchronize()
        t=time.perf_counter()
        for _ in range(iterations): cached(ids)
        if device.type=='mps': torch.mps.synchronize()
        elapsed=time.perf_counter()-t
    result={'model':MODEL,'device':str(device),'batch':batch,'sequence':seq,'iterations':iterations,
      'cached_table_parameters':table.numel(),'cached_table_bytes':table.numel()*table.element_size(),
      'tokens_per_second':batch*seq*iterations/elapsed,'ms_per_batch':1000*elapsed/iterations,
      'status':'materialized_kcode_inference_benchmark'}
    out=Path(__file__).resolve().parent/'artifacts'/'embedding_perf'; out.mkdir(parents=True,exist_ok=True)
    (out/'cached_results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
