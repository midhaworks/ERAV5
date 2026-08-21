"""Exact constrained K-CRF output head prototype.

The head uses the RKE codebook for unary scores and a low-rank transition
parameterisation.  The forward/Viterbi routines are deliberately exact and
use a small UTF-8 prefix automaton.  Low rank reduces learned transition
parameters, but does not magically reduce exact log-sum-exp complexity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from rke import utf8_is_complete, utf8_prefix_is_valid


PAD, EOS, CONT = 0, 1, 2


def utf8_dfa() -> torch.Tensor:
    """Compact 8-state UTF-8 prefix automaton; -1 denotes invalid."""
    table = torch.full((8, 256), -1, dtype=torch.long)
    for byte in range(0x00, 0x80): table[0, byte] = 0
    for byte in range(0xC2, 0xE0): table[0, byte] = 1
    for byte in list(range(0xE1, 0xED)) + list(range(0xEE, 0xF0)): table[0, byte] = 2
    table[0, 0xE0] = 3; table[0, 0xED] = 4
    for byte in range(0xF1, 0xF4): table[0, byte] = 5
    table[0, 0xF0] = 6; table[0, 0xF4] = 7
    for byte in range(0x80, 0xC0):
        table[1, byte] = 0
        table[2, byte] = table[3, byte] = table[4, byte] = 1
        table[5, byte] = table[6, byte] = table[7, byte] = 2
    table[3, :0xA0] = -1; table[4, 0xA0:] = -1
    table[6, :0x90] = -1; table[7, 0x90:] = -1
    return table


UTF8_DFA = utf8_dfa()


def _logadd(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return a + math.log1p(math.exp(-abs(a - b))) if a >= b else b + math.log1p(math.exp(-abs(a - b)))


def _valid_next(pending: bytes, state: int, position: int, last_position: int) -> tuple[bytes, bool]:
    """Return (new UTF-8 pending prefix, allowed) for one state."""
    if state == PAD:
        return pending, False
    if state == EOS:
        return pending, position == last_position and utf8_is_complete(pending)
    if state == CONT:
        # A continuation block may end in the middle of a code point.
        return pending, position == last_position
    if position == last_position:
        return pending, False
    byte = bytes([state - 3])
    candidate = pending + byte
    if not utf8_prefix_is_valid(candidate):
        return pending, False
    return (b"" if utf8_is_complete(candidate) else candidate), True


@dataclass(frozen=True)
class KCRFDecode:
    states: int
    length: int
    score: float
    ids: tuple[int, ...]


class KCRFHead(nn.Module):
    """Tied Kronecker unary scores plus low-rank structured transitions."""

    def __init__(self, hidden: int, codebook: nn.Embedding, rank: int = 8,
                 code_dim: int | None = None, active_states: tuple[int, ...] | None = None,
                 positive_transitions: bool = False):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.codebook = codebook
        self.states = int(codebook.num_embeddings)
        self.code_dim = int(code_dim or codebook.embedding_dim)
        self.to_code = nn.Linear(hidden, self.code_dim, bias=False)
        # A non-tiny scale avoids the bilinear zero-gradient regime at startup.
        self.transition_left = nn.Parameter(torch.randn(self.states, rank) * 0.20)
        self.transition_right = nn.Parameter(torch.randn(self.states, rank) * 0.20)
        self.rank = rank
        self.positive_transitions = positive_transitions
        self.active_states = tuple(active_states) if active_states is not None else tuple(range(self.states))

    def unary(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return [batch, length, states] tied codebook scores."""
        return self.to_code(hidden) @ self.codebook.weight.T / math.sqrt(self.code_dim)

    def transition(self) -> torch.Tensor:
        left, right = self.transition_left, self.transition_right
        if self.positive_transitions:
            left, right = torch.nn.functional.softplus(left), torch.nn.functional.softplus(right)
        return left @ right.T

    def parameter_report(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "transition_parameters": self.transition_left.numel() + self.transition_right.numel(),
            "dense_transition_parameters_equivalent": self.states * self.states,
            "rank": self.rank,
            "states": self.states,
            "positive_transitions": self.positive_transitions,
        }

    def _score(self, unary: torch.Tensor, target: list[int] | tuple[int, ...]) -> float:
        trans = self.transition().detach().cpu()
        values = unary.detach().cpu()
        pending = b""
        total = 0.0
        previous = CONT
        last = len(target) - 1
        for position, state in enumerate(target):
            pending, allowed = _valid_next(pending, int(state), position, last)
            if not allowed:
                return -math.inf
            total += float(values[position, state])
            if position:
                total += float(trans[previous, state])
            previous = int(state)
        return total

    def _paths(self, unary: torch.Tensor, viterbi: bool) -> KCRFDecode:
        """Exact forward or Viterbi over (last state, pending UTF-8 prefix)."""
        trans = self.transition().detach().cpu()
        values = unary.detach().cpu()
        last_position = values.shape[0] - 1
        # key=(previous emitted state, pending UTF-8 bytes)
        chart: dict[tuple[int, bytes], tuple[float, tuple[int, ...]]] = {(CONT, b""): (0.0, ())}
        for position in range(values.shape[0]):
            next_chart: dict[tuple[int, bytes], tuple[float, tuple[int, ...]]] = {}
            candidates = self.active_states
            for (previous, pending), (old_score, path) in chart.items():
                for state in candidates:
                    new_pending, allowed = _valid_next(pending, state, position, last_position)
                    if not allowed:
                        continue
                    score = old_score + float(values[position, state])
                    if position:
                        score += float(trans[previous, state])
                    key = (state, new_pending)
                    if key not in next_chart:
                        next_chart[key] = (score, path + (state,))
                    elif viterbi:
                        if score > next_chart[key][0]:
                            next_chart[key] = (score, path + (state,))
                    else:
                        current, current_path = next_chart[key]
                        next_chart[key] = (_logadd(current, score), current_path)
            chart = next_chart
            if not chart:
                return KCRFDecode(self.states, values.shape[0], -math.inf, ())
        if viterbi:
            score, path = max(chart.values(), key=lambda item: item[0])
        else:
            score = -math.inf
            path = ()
            for candidate, candidate_path in chart.values():
                score = _logadd(score, candidate)
        return KCRFDecode(self.states, values.shape[0], score, path)

    def nll(self, unary: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Exact sequence NLL for one [length, states] unary tensor."""
        if unary.ndim != 2 or unary.shape[1] != self.states:
            raise ValueError("unary must have shape [length, states]")
        target_ids = target.detach().cpu().tolist()
        # Keep this path differentiable.  The reference chart is small but
        # exact; each merge is a torch logaddexp over valid automaton paths.
        pending = b""
        gold_terms = []
        previous = CONT
        last = len(target_ids) - 1
        for position, state in enumerate(target_ids):
            pending, allowed = _valid_next(pending, int(state), position, last)
            if not allowed:
                raise ValueError("target is not a valid constrained sequence")
            term = unary[position, int(state)]
            if position:
                term = term + self.transition()[previous, int(state)]
            gold_terms.append(term)
            previous = int(state)
        gold = torch.stack(gold_terms).sum()
        partition = self._differentiable_partition(unary)
        if not math.isfinite(float(partition.detach())):
            raise ValueError("target or constrained partition is invalid")
        return partition - gold

    def _differentiable_partition(self, unary: torch.Tensor) -> torch.Tensor:
        def safe_logsumexp(values: torch.Tensor, dim: int) -> torch.Tensor:
            finite = torch.isfinite(values)
            clipped = torch.where(finite, values, values.new_full((), -1e9))
            result = torch.logsumexp(clipped, dim=dim)
            return torch.where(finite.any(dim=dim), result, result.new_full((), -math.inf))

        # Exact fast path for ASCII-only active alphabets: the UTF-8 automaton
        # has one state (complete), so the chart is an ordinary masked chain.
        ascii_only = all(state in (PAD, EOS, CONT) or (3 <= state <= 3 + 127)
                         for state in self.active_states)
        if ascii_only:
            trans = self.transition()
            alpha = unary.new_full((self.states,), -math.inf)
            alpha[CONT] = 0.0
            for position in range(unary.shape[0]):
                scores = unary[position] if position == 0 else unary[position] + safe_logsumexp(
                    alpha[:, None] + trans, dim=0)
                mask = torch.zeros(self.states, dtype=torch.bool, device=unary.device)
                if position == unary.shape[0] - 1:
                    for state in (EOS, CONT):
                        if state in self.active_states:
                            mask[state] = True
                else:
                    for state in self.active_states:
                        if state >= 3:
                            mask[state] = True
                alpha = scores.masked_fill(~mask, -math.inf)
            return torch.logsumexp(alpha, dim=0)
        # Vectorized finite-state path for arbitrary Unicode. The chart keeps
        # only (previous symbol, compact UTF-8 DFA state) tensors.
        dfa = UTF8_DFA.to(unary.device)
        trans = self.transition()
        initial = {(state, int(dfa[0, state - 3])): unary[0, state]
                   for state in self.active_states if state >= 3 and int(dfa[0, state - 3]) >= 0}
        alpha = torch.stack([torch.stack([
            initial[(state, d)] if (state, d) in initial else unary.new_full((), -math.inf)
            for d in range(8)]) for state in range(self.states)])
        route = torch.zeros((self.states, 8, 8), dtype=torch.bool, device=unary.device)
        for state in self.active_states:
            if state >= 3:
                byte = state - 3
                for old_d in range(8):
                    new_d = int(dfa[old_d, byte])
                    if new_d >= 0:
                        route[state, old_d, new_d] = True
        for position in range(1, unary.shape[0]):
            # scores[old_d, next_state] sums over the previous emitted state.
            scores = torch.stack([
                safe_logsumexp(alpha[:, old_d, None] + trans, dim=0)
                for old_d in range(8)
            ])
            byte_values = safe_logsumexp(
                scores.T[:, :, None].expand(self.states, 8, 8).masked_fill(~route, -math.inf), dim=1
            ) + unary[position, :, None]
            if position == unary.shape[0] - 1:
                next_alpha = unary.new_full((self.states, 8), -math.inf)
                if EOS in self.active_states:
                    next_alpha[EOS, 0] = scores[0, EOS] + unary[position, EOS]
                if CONT in self.active_states:
                    next_alpha[CONT, 0] = safe_logsumexp(scores[:, CONT], dim=0) + unary[position, CONT]
                alpha = next_alpha
            else:
                alpha = byte_values.masked_fill(~route.any(dim=1), -math.inf)
        return torch.logsumexp(alpha.reshape(-1), dim=0)

    @torch.no_grad()
    def viterbi(self, unary: torch.Tensor) -> KCRFDecode:
        if unary.ndim != 2 or unary.shape[1] != self.states:
            raise ValueError("unary must have shape [length, states]")
        dfa = UTF8_DFA.to(unary.device)
        trans = self.transition().detach()
        length = unary.shape[0]
        route = torch.zeros((self.states, 8, 8), dtype=torch.bool, device=unary.device)
        for state in self.active_states:
            if state >= 3:
                for old_d in range(8):
                    new_d = int(dfa[old_d, state - 3])
                    if new_d >= 0:
                        route[state, old_d, new_d] = True
        alpha = unary.new_full((self.states, 8), -math.inf)
        for state in self.active_states:
            if state >= 3:
                new_d = int(dfa[0, state - 3])
                if new_d >= 0:
                    alpha[state, new_d] = unary[0, state]
        previous_state = torch.full((length, self.states, 8), -1, dtype=torch.long, device=unary.device)
        previous_dfa = torch.full_like(previous_state, -1)
        for position in range(1, length):
            combined = alpha[:, :, None] + trans[:, None, :]
            best_score, best_state = combined.max(dim=0)  # [old_dfa, next_state]
            next_alpha = unary.new_full((self.states, 8), -math.inf)
            if position == length - 1:
                if EOS in self.active_states:
                    next_alpha[EOS, 0] = best_score[0, EOS] + unary[position, EOS]
                    previous_state[position, EOS, 0] = best_state[0, EOS]
                    previous_dfa[position, EOS, 0] = 0
                if CONT in self.active_states:
                    old_d, value = max(enumerate(best_score[:, CONT].tolist()), key=lambda item: item[1])
                    next_alpha[CONT, 0] = best_score[old_d, CONT] + unary[position, CONT]
                    previous_state[position, CONT, 0] = best_state[old_d, CONT]
                    previous_dfa[position, CONT, 0] = old_d
            else:
                for state in self.active_states:
                    if state < 3:
                        continue
                    for new_d in range(8):
                        old_ds = [old_d for old_d in range(8) if route[state, old_d, new_d]]
                        if not old_ds:
                            continue
                        old_d = max(old_ds, key=lambda item: float(best_score[item, state]))
                        next_alpha[state, new_d] = best_score[old_d, state] + unary[position, state]
                        previous_state[position, state, new_d] = best_state[old_d, state]
                        previous_dfa[position, state, new_d] = old_d
            alpha = next_alpha
        final_state, final_dfa = max(
            ((state, d) for state in (EOS, CONT) if state in self.active_states for d in range(8)),
            key=lambda item: float(alpha[item[0], item[1]])
        )
        score = float(alpha[final_state, final_dfa])
        if not math.isfinite(score):
            return KCRFDecode(self.states, length, -math.inf, ())
        ids = [0] * length
        state, dfa_state = final_state, final_dfa
        for position in range(length - 1, 0, -1):
            ids[position] = state
            old_state = int(previous_state[position, state, dfa_state])
            old_dfa = int(previous_dfa[position, state, dfa_state])
            state, dfa_state = old_state, old_dfa
        ids[0] = state
        return KCRFDecode(self.states, length, score, tuple(ids))
