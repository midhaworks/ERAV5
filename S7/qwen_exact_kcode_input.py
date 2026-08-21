"""Separate exact K-code input variant: UTF-8 bytes plus explicit EOS."""
from __future__ import annotations
import torch
from torch import nn
from transformers import AutoTokenizer
from qwen_kcode_input import MODEL

PAD, EOS, BYTE_OFFSET = 0, 1, 3

class ExactKCodeEmbedding(nn.Module):
    """Injective bounded token code; position 32 is reserved for EOS."""
    def __init__(self, tokenizer, hidden_size, payload_positions=32):
        super().__init__(); self.payload_positions=payload_positions; self.positions=payload_positions+1; self.states=259
        self.projection=nn.Parameter(torch.empty((self.states*self.positions,hidden_size)))
        nn.init.normal_(self.projection,mean=0.0,std=1.0/(self.states*self.positions)**0.5)
        vocab_size=max(tokenizer.vocab_size,int(tokenizer.pad_token_id or 0)+1)
        table=torch.zeros((vocab_size,self.positions),dtype=torch.long)
        for token_id in range(vocab_size):
            raw=tokenizer.convert_ids_to_tokens(token_id).encode('utf-8',errors='replace')[:payload_positions]
            if raw: table[token_id,:len(raw)]=torch.tensor(list(raw),dtype=torch.long)+BYTE_OFFSET
            table[token_id,len(raw)]=EOS
        self.register_buffer('state_table',table,persistent=True)
    def forward(self,input_ids):
        states=self.state_table[input_ids]; valid=states!=PAD
        rows=states + torch.arange(self.positions,device=states.device)[None,None,:]*self.states
        vectors=self.projection[rows]*valid[...,None]
        lengths=valid.sum(-1,keepdim=True).clamp_min(1).to(vectors.dtype)
        return vectors.sum(-2)/lengths.sqrt()

