import math
import pytest
import torch

from spec_decoding_4060.metrics import (
    acceptance_rate_pct,
    accepted_tokens_per_verification_cycle,
    compare_greedy_outputs,
    output_tokens_per_target_forward,
    summarize_timing_runs,
    validate_speculative_counters,
)


def test_acceptance_rate_reported_examples():
    """Verify acceptance rate calculations for all benchmark tasks."""
    assert math.isclose(acceptance_rate_pct(207, 242), 85.537, abs_tol=0.01)
    assert math.isclose(acceptance_rate_pct(197, 289), 68.166, abs_tol=0.01)
    assert math.isclose(acceptance_rate_pct(172, 419), 41.050, abs_tol=0.01)


def test_acceptance_rate_zero_drafted():
    """Verify zero drafted proposals returns 0.0% without ZeroDivisionError."""
    assert acceptance_rate_pct(0, 0) == 0.0
    assert acceptance_rate_pct(0, -5) == 0.0


def test_output_tokens_per_target_forward_example():
    """Verify generated_tokens / target_forward_passes."""
    assert output_tokens_per_target_forward(256, 50) == pytest.approx(5.12)
    assert output_tokens_per_target_forward(256, 60) == pytest.approx(4.266666, rel=1e-4)
    assert output_tokens_per_target_forward(256, 85) == pytest.approx(3.011764, rel=1e-4)
    assert output_tokens_per_target_forward(10, 0) == 0.0


def test_accepted_tokens_per_verification_cycle():
    """Verify accepted tokens per verification cycle."""
    # 50 target forward passes = 1 prefill + 49 verification cycles
    assert accepted_tokens_per_verification_cycle(207, 49) == pytest.approx(4.224489, rel=1e-4)
    assert accepted_tokens_per_verification_cycle(197, 59) == pytest.approx(3.338983, rel=1e-4)
    assert accepted_tokens_per_verification_cycle(172, 84) == pytest.approx(2.047619, rel=1e-4)
    assert accepted_tokens_per_verification_cycle(10, 0) == 0.0


def test_validate_speculative_counters_ok():
    """Valid counter accounting must not raise."""
    validate_speculative_counters(
        output_tokens=256,
        max_new_tokens=256,
        total_drafted=242,
        total_accepted=207,
        target_forward_passes=50,
        verification_cycles=49,
    )


def test_validate_speculative_counters_rejects_overdraft():
    """Accepted tokens exceeding drafted tokens is physically impossible."""
    with pytest.raises(AssertionError, match="total_accepted"):
        validate_speculative_counters(
            output_tokens=256,
            max_new_tokens=256,
            total_drafted=10,
            total_accepted=11,
            target_forward_passes=5,
            verification_cycles=4,
        )


def test_validate_speculative_counters_rejects_budget_overflow():
    """Output tokens exceeding max_new_tokens must raise."""
    with pytest.raises(AssertionError, match="exceeds max_new_tokens"):
        validate_speculative_counters(
            output_tokens=257,
            max_new_tokens=256,
            total_drafted=200,
            total_accepted=150,
            target_forward_passes=50,
            verification_cycles=49,
        )


def test_validate_speculative_counters_rejects_invalid_forward_counts():
    """target_forward_passes must equal 1 (prefill) + verification_cycles."""
    with pytest.raises(AssertionError, match="prefill"):
        validate_speculative_counters(
            output_tokens=10,
            max_new_tokens=256,
            total_drafted=20,
            total_accepted=10,
            target_forward_passes=0,
            verification_cycles=0,
        )

    with pytest.raises(AssertionError, match="target_forward_passes must equal"):
        validate_speculative_counters(
            output_tokens=10,
            max_new_tokens=256,
            total_drafted=20,
            total_accepted=10,
            target_forward_passes=10,
            verification_cycles=5,
        )


def test_compare_greedy_outputs_match():
    """Identical sequences produce exact_match: True."""
    prompt_len = 4
    ids = torch.tensor([[1, 2, 3, 4, 10, 11, 12]])
    out = compare_greedy_outputs(ids, ids.clone(), prompt_len)
    assert out["exact_match"] is True
    assert out["baseline_output_tokens"] == 3
    assert out["speculative_output_tokens"] == 3
    assert "first_divergence" not in out


def test_compare_greedy_outputs_divergence():
    """Mismatched sequence identifies correct 0-indexed output token index and tokens."""
    prompt_len = 2
    # Prompt is [1, 2]. Output token 0 is 10. Output token 1 is 11 vs 99.
    a = torch.tensor([[1, 2, 10, 11]])
    b = torch.tensor([[1, 2, 10, 99]])
    out = compare_greedy_outputs(a, b, prompt_len)
    assert out["exact_match"] is False
    assert out["first_divergence"]["output_token_index"] == 1
    assert out["first_divergence"]["global_index"] == 3
    assert out["first_divergence"]["baseline_token_id"] == 11
    assert out["first_divergence"]["speculative_token_id"] == 99


def test_compare_greedy_outputs_length_divergence():
    """Mismatched length identifies divergence at the truncation point."""
    prompt_len = 2
    a = torch.tensor([[1, 2, 10, 11, 12]])
    b = torch.tensor([[1, 2, 10, 11]])
    out = compare_greedy_outputs(a, b, prompt_len)
    assert out["exact_match"] is False
    assert out["first_divergence"]["output_token_index"] == 2
    assert out["first_divergence"]["baseline_token_id"] == 12
    assert out["first_divergence"]["speculative_token_id"] is None


def test_summarize_timing_runs():
    """Verify median, std dev, and raw run preservation."""
    assert summarize_timing_runs([]) == {"tps": 0.0, "time_sec": 0.0, "tps_std": 0.0, "runs": []}

    runs = [
        {"tps": 40.0, "time_sec": 2.5},
        {"tps": 50.0, "time_sec": 2.0},
        {"tps": 60.0, "time_sec": 1.67},
    ]
    summary = summarize_timing_runs(runs)
    assert summary["tps"] == 50.0
    assert summary["time_sec"] == 2.0
    assert summary["tps_std"] == pytest.approx(8.164965, rel=1e-4)
    assert len(summary["runs"]) == 3
