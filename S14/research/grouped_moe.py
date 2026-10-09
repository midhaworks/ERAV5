"""Reference sparse layer and physical export; no task-quality claims.

Group labels specify permitted experts, not capabilities learned by those experts.
"""
import copy

import torch
from torch import nn


class GroupedMoE(nn.Module):
    def __init__(self, experts, router, groups, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.groups = {name: tuple(ids) for name, ids in groups.items()}
        self.top_k = top_k
        if router.out_features != len(experts):
            raise ValueError("Router must have one score per expert")
        if not 1 <= top_k <= len(experts):
            raise ValueError("Invalid top_k")
        for ids in self.groups.values():
            if len(set(ids)) != len(ids) or len(ids) < top_k:
                raise ValueError("Groups require unique experts and at least top_k entries")
            if any(i < 0 or i >= len(experts) for i in ids):
                raise ValueError("Invalid expert index")

    @classmethod
    def from_linear(cls, linear, experts_per_group=2, top_k=2):
        if experts_per_group < top_k:
            raise ValueError("Group smaller than top_k")
        n = 2 * experts_per_group
        router = nn.Linear(linear.in_features, n).to(
            device=linear.weight.device, dtype=linear.weight.dtype)
        # Exact clones preserve the dense function at conversion. Specialization
        # must subsequently arise from training, not injected target information.
        return cls([copy.deepcopy(linear) for _ in range(n)], router,
                   {"coding": range(experts_per_group),
                    "agentic": range(experts_per_group, n)}, top_k)

    def forward(self, x, group=None, return_routes=False):
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        permitted = tuple(range(len(self.experts))) if group is None else self.groups[group]
        ids = torch.tensor(permitted, device=x.device)
        logits = self.router(flat).index_select(-1, ids)
        scores, local_ids = logits.topk(self.top_k, dim=-1)
        weights = scores.softmax(-1)
        selected = ids[local_ids]
        # Sparse dispatch: each expert executes only on its selected rows.
        pieces = []
        for expert_id, expert in enumerate(self.experts):
            rows, slots = (selected == expert_id).nonzero(as_tuple=True)
            if rows.numel():
                pieces.append((rows, expert(flat[rows]) * weights[rows, slots, None]))
        out_dim = pieces[0][1].shape[-1] if pieces else self.experts[0].out_features
        output = flat.new_zeros((flat.shape[0], out_dim))
        for rows, values in pieces:
            output = output.index_add(0, rows, values)
        output = output.reshape(*original_shape, out_dim)
        return (output, selected.reshape(*original_shape, self.top_k)) if return_routes else output

    def export(self, group):
        """Independent model retaining only this group's experts and router rows."""
        ids = self.groups[group]
        router = nn.Linear(self.router.in_features, len(ids)).to(
            device=self.router.weight.device, dtype=self.router.weight.dtype)
        with torch.no_grad():
            router.weight.copy_(self.router.weight[list(ids)])
            router.bias.copy_(self.router.bias[list(ids)])
        return GroupedMoE([copy.deepcopy(self.experts[i]) for i in ids], router,
                          {group: range(len(ids))}, self.top_k)


def parameters(model):
    return sum(p.numel() for p in model.parameters())
