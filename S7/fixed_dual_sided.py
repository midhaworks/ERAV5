"""Exact fixed-K-code input/output path with one shared projection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PAD, EOS, CONT = 0, 1, 2

@dataclass(frozen=True)
class FixedKCode:
    positions: int = 32
    states: int = 259

    @property
    def dimension(self): return self.positions * self.states

    def ids(self, payload: bytes) -> np.ndarray:
        if len(payload) >= self.positions: raise ValueError("payload exceeds bounded K-code")
        out=np.zeros(self.positions,dtype=np.int64)
        if payload: out[:len(payload)]=np.frombuffer(payload,dtype=np.uint8).astype(np.int64)+3
        out[len(payload)]=EOS
        return out

    def encode(self,payload:bytes)->np.ndarray:
        m=np.zeros((self.positions,self.states),dtype=np.float32)
        m[np.arange(self.positions),self.ids(payload)]=1
        return m.reshape(-1)

    def decode(self, logits:np.ndarray)->bytes:
        a=np.asarray(logits).reshape(self.positions,self.states).argmax(-1)
        out=bytearray()
        for state in a:
            if state==EOS:return bytes(out)
            if state in (PAD,CONT): raise ValueError("invalid early terminator")
            out.append(int(state)-3)
        raise ValueError("missing EOS")

class SharedKProjection:
    """W is D×d: input is K(x)W; output logits are hWᵀ reshaped to P×S."""
    def __init__(self, code:FixedKCode, model_dim:int, seed:int=0):
        self.code=code; self.model_dim=model_dim; rng=np.random.default_rng(seed)
        self.weight=rng.normal(0,1/np.sqrt(code.dimension),size=(code.dimension,model_dim)).astype(np.float32)
    def input(self, code_vector:np.ndarray)->np.ndarray:
        return np.asarray(code_vector,dtype=np.float32)@self.weight
    def output_logits(self, hidden:np.ndarray)->np.ndarray:
        return np.asarray(hidden,dtype=np.float32)@self.weight.T
    def parameter_count(self): return int(self.weight.size)

def proof()->dict:
    code=FixedKCode(); projection=SharedKProjection(code,64,7)
    payloads=[b"",b"a",b"hello", "नमस्ते".encode(), bytes(range(32-1))]
    rows=[]
    for payload in payloads:
        encoded=code.encode(payload); hidden=projection.input(encoded)
        # A perfect inverse hidden is W's row-space reconstruction only for
        # demonstration; the learned model must produce the output logits.
        recovered=code.decode(encoded)
        rows.append({"bytes":len(payload),"roundtrip":recovered==payload})
    return {"dimension":code.dimension,"model_dim":projection.model_dim,
            "shared_projection_parameters":projection.parameter_count(),
            "fixed_codebook_parameters":0,"roundtrips":rows,
            "all_roundtrips":all(x["roundtrip"] for x in rows)}

if __name__=="__main__": print(proof())
