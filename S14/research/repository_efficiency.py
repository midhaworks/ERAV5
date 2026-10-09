"""Opt-in optimizations; original frozen serialized benchmark is unchanged."""
import copy
import re

from repository_environment import TOOLS


def parse_calls(raw, parser):
    blocks=list(re.finditer(r"<tool_call>(.*?)</tool_call>",raw,re.DOTALL))
    if not 1<=len(blocks)<=8:
        raise ValueError("Expected 1..8 complete calls")
    # Natural-language reasoning before the first call is permitted by the native template.
    if any(raw[a.end():b.start()].strip() for a,b in zip(blocks,blocks[1:])) or raw[blocks[-1].end():].strip():
        raise ValueError("Unexpected content between or after calls")
    calls=[parser(b.group(1).strip(),tools=TOOLS) for b in blocks]
    if len(calls)>1 and any(c.get("name")!="read_file" for c in calls):
        raise ValueError("Only read_file calls may be bundled")
    return calls


def execute_calls(env,calls):
    if len(calls)>1:
        # Validate the WHOLE bundle before execution; edits/tests/finish cannot be mixed in.
        if len(calls)>8 or any(c.get("name")!="read_file" or
            not isinstance(c.get("arguments"),dict) or set(c["arguments"])!={"path"} or
            not isinstance(c["arguments"]["path"],str) or c["arguments"]["path"] not in env.task["files"]
            for c in calls):
            raise ValueError("Invalid read-only bundle")
    if not calls: raise ValueError("Empty bundle")
    return [env.execute(c["name"],c["arguments"]) for c in calls]


def common_prefix(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n


def prefix_plan(prompt,history,cached,cache_exists):
    if not prompt: raise ValueError("Empty generation prompt")
    length=min(common_prefix(prompt,history),len(prompt)-1)
    target=prompt[:length]
    reuse=len(cached) if cache_exists and target[:len(cached)]==cached else 0
    return target,reuse,not cache_exists or reuse!=len(cached)


class PrefixCache:
    """Cache canonical history ONLY; branch generation into a deep copy.

    Recurrent hybrid caches cannot safely rewind after XML normalization. Store the
    history before the generation header instead. Reuse only exact token prefixes;
    reset on any mismatch, and never trim a recurrent state.
    """
    def __init__(self,model):
        self.model=model
        self.tokens=[]
        self.cache=None

    def prepare(self,prompt_tokens,history_tokens):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.generate import generation_stream
        prompt_tokens=list(prompt_tokens)
        history_tokens=list(history_tokens)
        # Tokenization near the boundary can differ. Only cache their common prefix,
        # retaining at least one prompt token for generation to compute logits.
        target,reuse,reset=prefix_plan(prompt_tokens,history_tokens,self.tokens,self.cache is not None)
        length=len(target)
        if reset:
            self.cache=make_prompt_cache(self.model)
            self.tokens=[]
        with mx.stream(generation_stream):
            for offset in range(reuse,len(target),512):
                self.model(mx.array(target[offset:offset+512])[None],cache=self.cache)
                mx.eval([c.state for c in self.cache])
            branch=copy.deepcopy(self.cache)
        self.tokens=target
        return prompt_tokens[length:],branch,dict(reused_tokens=reuse,
            newly_prefilled_history_tokens=length-reuse,generation_input_tokens=len(prompt_tokens)-length,
            full_prompt_tokens=len(prompt_tokens),prefix_reset=reset,
            cache_types=sorted({type(c).__name__ for c in self.cache}))
