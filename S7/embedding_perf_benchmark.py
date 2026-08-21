"""Like-for-like embedding lookup benchmark for normal and fixed K-code paths."""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from qwen_kcode_input import KCodeEmbedding, MODEL

def run(iterations=200, batch=128, seq=32):
    torch.manual_seed(3180); device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tok=AutoTokenizer.from_pretrained(MODEL); model=AutoModelForCausalLM.from_pretrained(MODEL)
    normal=model.get_input_embeddings().to(device).eval(); k=KCodeEmbedding(tok,model.config.hidden_size).to(device=device,dtype=normal.weight.dtype).eval()
    ids=torch.randint(0, min(tok.vocab_size, normal.num_embeddings), (batch,seq), device=device)
    def measure(layer):
        with torch.no_grad():
            for _ in range(20): layer(ids)
            if device.type=='mps': torch.mps.synchronize()
            t=time.perf_counter()
            for _ in range(iterations): layer(ids)
            if device.type=='mps': torch.mps.synchronize()
            elapsed=time.perf_counter()-t
        return {'seconds':elapsed,'tokens_per_second':batch*seq*iterations/elapsed,'ms_per_batch':1000*elapsed/iterations}
    n=measure(normal); kc=measure(k)
    result={'model':MODEL,'device':str(device),'batch':batch,'sequence':seq,'iterations':iterations,
      'normal_parameters':normal.weight.numel(),'kcode_parameters':k.projection.numel(),
      'normal_parameter_bytes':normal.weight.numel()*normal.weight.element_size(),
      'kcode_parameter_bytes':k.projection.numel()*k.projection.element_size(),
      'parameter_reduction':1-k.projection.numel()/normal.weight.numel(),'normal':n,'kcode':kc,
      'kcode_speed_ratio':kc['tokens_per_second']/n['tokens_per_second'],'status':'embedding_lookup_benchmark'}
    out=Path(__file__).resolve().parent/'artifacts'/'embedding_perf'; out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': run()
