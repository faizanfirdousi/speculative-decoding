"""
Pure (CPU) helpers for greedy speculative decoding accept/reject semantics.
Used by the engine and by unit tests without GPU models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class AcceptRejectResult:
    n_accepted: int
    append_authoritative: bool
    emitted_draft_tokens: int
    emitted_authoritative: int


def greedy_accept_count(
    draft_ids: Sequence[int],
    target_pred_ids: Sequence[int],
    eos_token_id: Optional[int] = None,
) -> int:
    """
    Count leading draft tokens that match target greedy predictions at the same step.

    ``target_pred_ids[i]`` is the target argmax after the prefix ending at draft position ``i``
    (same indexing as the engine's verification forward).
    """
    n = 0
    for i, d in enumerate(draft_ids):
        if i >= len(target_pred_ids):
            break
        if d != target_pred_ids[i]:
            break
        n += 1
        if eos_token_id is not None and d == eos_token_id:
            break
    return n


def plan_greedy_spec_step(
    draft_ids: Sequence[int],
    target_pred_ids: Sequence[int],
    *,
    eos_token_id: Optional[int],
    output_tokens_so_far: int,
    max_new_tokens: int,
) -> AcceptRejectResult:
    """
    Decide how many tokens to emit for one speculative verification cycle.

    Matches ``SpeculativeEngine.custom_speculative_generate`` append semantics.
    """
    n_accepted = greedy_accept_count(draft_ids, target_pred_ids, eos_token_id)
    has_eos = n_accepted > 0 and eos_token_id is not None and draft_ids[n_accepted - 1] == eos_token_id

    emitted_draft = n_accepted
    append_auth = False
    emitted_auth = 0

    if not has_eos and output_tokens_so_far + emitted_draft < max_new_tokens:
        append_auth = True
        emitted_auth = 1

    return AcceptRejectResult(
        n_accepted=n_accepted,
        append_authoritative=append_auth,
        emitted_draft_tokens=emitted_draft,
        emitted_authoritative=emitted_auth,
    )


def expected_tokens_per_cycle(accept_probs: Sequence[float], k: int) -> float:
    """
    Expected number of tokens emitted per verification cycle for lookahead ``k``.

    ``accept_probs[i]`` is the probability that the draft token at position ``i + 1``
    is accepted, *conditioned on all earlier drafts having been accepted* (i.e. the
    empirical match rate at that position). The cycle always emits one authoritative
    (correction or bonus) token in addition to the accepted drafts, hence the leading 1.

    For a stationary per-position rate this is ``(1 - a**(k+1)) / (1 - a)``.
    """
    total = 1.0
    cumulative = 1.0
    for i in range(min(k, len(accept_probs))):
        cumulative *= accept_probs[i]
        total += cumulative
    return total


def optimal_lookahead(
    accept_probs: Sequence[float],
    target_cost: float,
    draft_cost: float,
    *,
    min_k: int = 1,
    max_k: int = 8,
    current_k: Optional[int] = None,
    min_gain: float = 0.0,
) -> int:
    """
    Pick the lookahead ``K`` that maximizes predicted decode throughput.

    ``target_cost`` is the latency of one verification forward, ``draft_cost`` the
    latency of one draft token. Throughput is modelled as

        throughput(K) = expected_tokens_per_cycle(accept_probs, K) / (target_cost + K * draft_cost)

    ``min_gain`` applies hysteresis: when ``current_k`` is valid, it is retained unless
    some other ``K`` beats it by more than this relative margin, which prevents the
    lookahead from oscillating between adjacent values.
    """
    target_cost = max(target_cost, 1e-9)
    draft_cost = max(draft_cost, 1e-9)
    lo = max(1, min_k)
    hi = max(lo, max_k)

    def throughput(k: int) -> float:
        return expected_tokens_per_cycle(accept_probs, k) / (target_cost + k * draft_cost)

    best_k = lo
    best_value = throughput(lo)
    for k in range(lo + 1, hi + 1):
        value = throughput(k)
        if value > best_value:
            best_k, best_value = k, value

    if current_k is not None and lo <= current_k <= hi:
        if best_value <= throughput(current_k) * (1.0 + min_gain):
            return current_k
    return best_k
