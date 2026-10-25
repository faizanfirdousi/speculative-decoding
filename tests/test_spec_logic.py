import pytest
from spec_decoding_4060.spec_logic import (
    AcceptRejectResult,
    expected_tokens_per_cycle,
    greedy_accept_count,
    optimal_lookahead,
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


def test_expected_tokens_per_cycle_perfect_acceptance():
    """All-accept: every lookahead yields K accepted tokens plus the bonus token."""
    probs = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert expected_tokens_per_cycle(probs, 5) == pytest.approx(6.0)
    assert expected_tokens_per_cycle(probs, 2) == pytest.approx(3.0)


def test_expected_tokens_per_cycle_matches_closed_form():
    """Stationary rate a: E = (1 - a**(k+1)) / (1 - a)."""
    a = 0.841
    probs = [a] * 5
    closed_form = (1 - a ** 6) / (1 - a)
    assert expected_tokens_per_cycle(probs, 5) == pytest.approx(closed_form, rel=1e-6)


def test_expected_tokens_per_cycle_zero_acceptance():
    """No drafts ever accepted: only the authoritative token is emitted."""
    assert expected_tokens_per_cycle([0.0, 0.0, 0.0], 3) == pytest.approx(1.0)


def test_optimal_lookahead_high_acceptance_prefers_large_k():
    """High acceptance and cheap drafts: larger K wins."""
    probs = [0.95] * 8
    k = optimal_lookahead(probs, target_cost=1.0, draft_cost=0.05, min_k=1, max_k=8)
    assert k == 8


def test_optimal_lookahead_low_acceptance_prefers_small_k():
    """Low acceptance: extra drafts are wasted, so a small K wins."""
    probs = [0.4] * 8
    k = optimal_lookahead(probs, target_cost=1.0, draft_cost=0.2, min_k=1, max_k=8)
    assert k < 5


def test_optimal_lookahead_hysteresis_keeps_current_k():
    """A marginally better K must not displace the current one when min_gain applies."""
    probs = [0.4] * 8
    plain = optimal_lookahead(probs, target_cost=1.0, draft_cost=0.2, min_k=1, max_k=8)
    held = optimal_lookahead(
        probs, target_cost=1.0, draft_cost=0.2,
        min_k=1, max_k=8, current_k=5, min_gain=0.5,
    )
    assert plain != 5
    assert held == 5


def test_optimal_lookahead_respects_bounds():
    probs = [1.0] * 8
    assert optimal_lookahead(probs, 1.0, 0.0, min_k=2, max_k=6) == 6
    assert 2 <= optimal_lookahead([0.1] * 8, 1.0, 0.9, min_k=2, max_k=6) <= 6
