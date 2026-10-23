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
