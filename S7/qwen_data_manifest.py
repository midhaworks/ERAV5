"""Build the immutable shared-data manifest for Qwen/K-code retraining."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from cross_block_lm import build_long_dataset
from rke import sha256

OUT=Path(__file__).resolve().parent/"artifacts"/"qwen_data_manifest"
MODEL='Qwen/Qwen2.5-0.5B'

def run():
    data,audit=build_long_dataset(); tokenizer=AutoTokenizer.from_pretrained(MODEL)
    rows={}; totals=defaultdict(lambda:{'records':0,'utf8_bytes':0,'qwen_tokens':0})
    for split,records in data.items():
        rows[split]=[]
        for r in records:
            text=' '.join(r['context']+[r['target']])
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            item={'language':r['language'],'document':r['document'],'target':r['target'],
                  'text_sha256':sha256(text),'utf8_bytes':len(text.encode()),'qwen_tokens':len(ids),
                  'qwen_token_sha256':sha256(ids)}
            rows[split].append(item); t=totals[(split,r['language'])]; t['records']+=1; t['utf8_bytes']+=item['utf8_bytes']; t['qwen_tokens']+=item['qwen_tokens']
    result={'model':MODEL,'tokenizer_vocab_size':tokenizer.vocab_size,'dataset_audit':audit,
            'dataset_hash':sha256(data),'rows':rows,'totals':{f'{s}/{l}':v for (s,l),v in totals.items()},
            'protocol':'same text, document splits and hashes for normal and K-code arms'}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n'); return result
if __name__=='__main__':
    r=run(); print(json.dumps({'model':r['model'],'vocab':r['tokenizer_vocab_size'],'dataset_hash':r['dataset_hash'],'totals':r['totals']},indent=2))
