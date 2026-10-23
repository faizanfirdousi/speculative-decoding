import pytest
from spec_decoding_4060.spec_logic import (
    AcceptRejectResult,
    greedy_accept_count,
    plan_greedy_spec_step,
)


def test_greedy_accept_count_all_accepted():
    draft = [10, 20, 30, 40, 50]
    target_preds = [10, 20, 30, 40, 50, 60]  # target_preds[5] is bonus
    assert greedy_accept_count(draft, target_preds) == 5


def test_greedy_accept_count_partial_rejection():
    draft = [10, 20, 30, 40, 50]
    target_preds = [10, 20, 99, 40, 50, 60]  # mismatch at index 2
    assert greedy_accept_count(draft, target_preds) == 2


def test_greedy_accept_count_immediate_rejection():
    draft = [10, 20, 30]
    target_preds = [99, 20, 30, 40]
    assert greedy_accept_count(draft, target_preds) == 0


def test_greedy_accept_count_with_eos():
    eos_id = 999
    draft = [10, 20, eos_id, 30]
    target_preds = [10, 20, eos_id, 30, 40]
    # Stops accepting at EOS
    assert greedy_accept_count(draft, target_preds, eos_token_id=eos_id) == 3


def test_plan_greedy_spec_step_all_accepted_with_bonus():
    """All K tokens accepted -> append K draft tokens + 1 bonus token."""
    draft = [1, 2, 3]
    target = [1, 2, 3, 4]  # 4 is bonus
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=999,
        output_tokens_so_far=0,
        max_new_tokens=10,
    )
    assert step.n_accepted == 3
    assert step.append_authoritative is True
    assert step.emitted_draft_tokens == 3
    assert step.emitted_authoritative == 1
    # Total tokens emitted this cycle = 3 + 1 = 4


def test_plan_greedy_spec_step_rejection_with_correction():
    """Mismatch at index 1 -> append 1 accepted draft token + 1 correction token."""
    draft = [1, 2, 3]
    target = [1, 99, 3, 4]  # mismatch at index 1 -> target[1] is correction
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=999,
        output_tokens_so_far=0,
        max_new_tokens=10,
    )
    assert step.n_accepted == 1
    assert step.append_authoritative is True
    assert step.emitted_draft_tokens == 1
    assert step.emitted_authoritative == 1
    # Total emitted this cycle = 1 + 1 = 2


def test_plan_greedy_spec_step_immediate_rejection():
    """Mismatch at index 0 -> append 0 draft tokens + 1 correction token."""
    draft = [1, 2, 3]
    target = [99, 2, 3, 4]  # target[0] is correction
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=999,
        output_tokens_so_far=0,
        max_new_tokens=10,
    )
    assert step.n_accepted == 0
    assert step.append_authoritative is True
    assert step.emitted_draft_tokens == 0
    assert step.emitted_authoritative == 1


def test_plan_greedy_spec_step_eos_suppresses_bonus():
    """Draft emits EOS -> accepted up to EOS, but bonus token is suppressed."""
    eos_id = 999
    draft = [1, eos_id, 3]
    target = [1, eos_id, 3, 4]
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=eos_id,
        output_tokens_so_far=0,
        max_new_tokens=10,
    )
    assert step.n_accepted == 2
    assert step.append_authoritative is False
    assert step.emitted_draft_tokens == 2
    assert step.emitted_authoritative == 0


def test_plan_greedy_spec_step_budget_limit():
    """When budget is reached exactly by accepted draft tokens, authoritative token is suppressed."""
    draft = [1, 2, 3]
    target = [1, 2, 3, 4]
    # output_tokens_so_far = 7, max_new_tokens = 10 -> accepting 3 reaches exactly 10
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=999,
        output_tokens_so_far=7,
        max_new_tokens=10,
    )
    assert step.n_accepted == 3
    assert step.append_authoritative is False
    assert step.emitted_draft_tokens == 3
    assert step.emitted_authoritative == 0
    assert 7 + step.emitted_draft_tokens + step.emitted_authoritative == 10


def test_plan_greedy_spec_step_incomplete_final_block():
    """When remaining < gamma, draft length is smaller than normal gamma."""
    draft = [1, 2]  # only 2 tokens drafted because only 2 were remaining
    target = [1, 2, 3]
    step = plan_greedy_spec_step(
        draft,
        target,
        eos_token_id=999,
        output_tokens_so_far=8,
        max_new_tokens=10,
    )
    assert step.n_accepted == 2
    assert step.append_authoritative is False
    assert step.emitted_draft_tokens == 2
    assert step.emitted_authoritative == 0
    assert 8 + step.emitted_draft_tokens + step.emitted_authoritative == 10
