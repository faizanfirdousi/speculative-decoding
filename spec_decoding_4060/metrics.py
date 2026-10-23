"""
Benchmark metric definitions and consistency checks for speculative decoding runs.

All rates/counts refer to the custom speculative path unless noted otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def acceptance_rate_pct(total_accepted: int, total_drafted: int) -> float:
    """
    Fraction of draft proposals that were accepted by the target model, as a percentage.

    Numerator: sum over verification cycles of accepted draft tokens (0..gamma per cycle).
    Denominator: sum over verification cycles of proposed draft tokens (gamma per cycle,
    reduced on the final cycle when fewer than gamma tokens remain under max_new_tokens).

    Rejected drafts, correction tokens, bonus tokens, and emitted output tokens are not
    included in this ratio.
    """
    if total_drafted <= 0:
        return 0.0
    return (total_accepted / total_drafted) * 100.0


def output_tokens_per_target_forward(
    output_tokens: int, target_forward_passes: int
) -> float:
    """
    Average number of emitted output tokens per target-model forward pass.

    ``target_forward_passes`` counts every ``target_model(...)`` invocation during
    generation, including the initial prompt prefill and each parallel verification
    forward. This is *not* the average number of accepted draft tokens per cycle.
    """
    if target_forward_passes <= 0:
        return 0.0
    return output_tokens / target_forward_passes


def accepted_tokens_per_verification_cycle(
    total_accepted: int, verification_cycles: int
) -> float:
    """
    Average number of draft tokens accepted per verification cycle.

    ``verification_cycles`` counts target-model verification forward passes
    after the initial prompt prefill (i.e. target_forward_passes - 1).
    Does not include the prompt prefill or authoritative/correction/bonus tokens.
    """
    if verification_cycles <= 0:
        return 0.0
    return total_accepted / verification_cycles


def compare_greedy_outputs(
    baseline_ids: torch.Tensor,
    speculative_ids: torch.Tensor,
    prompt_len: int,
) -> Dict[str, Any]:
    """
    Compare full sequences (prompt + completion) from baseline vs speculative greedy runs.

    ``exact_match`` is True iff token IDs are identical for the entire tensor.
    If a mismatch occurs, identifies the first divergent output token position,
    the conflicting token IDs, and surrounding context tokens.
    """
    match = torch.equal(baseline_ids, speculative_ids)
    result: Dict[str, Any] = {
        "exact_match": match,
        "baseline_output_tokens": int(baseline_ids.shape[1] - prompt_len),
        "speculative_output_tokens": int(speculative_ids.shape[1] - prompt_len),
    }
    if match:
        return result

    b = baseline_ids[0]
    s = speculative_ids[0]
    min_len = min(b.shape[0], s.shape[0])
    first_global: Optional[int] = None
    for idx in range(prompt_len, min_len):
        if b[idx].item() != s[idx].item():
            first_global = idx
            break

    if first_global is None and b.shape[0] != s.shape[0]:
        first_global = min_len

    if first_global is not None:
        rel = first_global - prompt_len
        ctx_start = max(prompt_len, first_global - 3)
        result["first_divergence"] = {
            "output_token_index": rel,
            "global_index": first_global,
            "baseline_token_id": int(b[first_global].item()) if first_global < b.shape[0] else None,
            "speculative_token_id": int(s[first_global].item()) if first_global < s.shape[0] else None,
            "baseline_context_ids": b[ctx_start : first_global + 2].tolist(),
            "speculative_context_ids": s[ctx_start : first_global + 2].tolist(),
        }
    return result


def validate_speculative_counters(
    *,
    output_tokens: int,
    max_new_tokens: int,
    total_drafted: int,
    total_accepted: int,
    target_forward_passes: int,
    verification_cycles: int,
) -> None:
    """
    Raise AssertionError if speculative decoding counters violate basic mathematical invariants.
    
    Invariants:
    1. 0 <= output_tokens <= max_new_tokens
    2. 0 <= total_accepted <= total_drafted
    3. target_forward_passes >= 1 (includes initial prompt prefill)
    4. verification_cycles >= 0
    5. target_forward_passes == 1 + verification_cycles
    """
    if output_tokens < 0:
        raise AssertionError(f"output_tokens must be non-negative, got {output_tokens}")
    if output_tokens > max_new_tokens:
        raise AssertionError(
            f"output_tokens ({output_tokens}) exceeds max_new_tokens ({max_new_tokens})"
        )
    if total_drafted < 0:
        raise AssertionError(f"total_drafted must be non-negative, got {total_drafted}")
    if total_accepted < 0:
        raise AssertionError(f"total_accepted must be non-negative, got {total_accepted}")
    if total_accepted > total_drafted:
        raise AssertionError(
            f"total_accepted ({total_accepted}) exceeds total_drafted ({total_drafted})"
        )
    if target_forward_passes < 1:
        raise AssertionError(f"target_forward_passes must include at least prefill (>= 1), got {target_forward_passes}")
    if verification_cycles < 0:
        raise AssertionError(f"verification_cycles must be non-negative, got {verification_cycles}")
    # Each verification cycle performs exactly one target forward pass after prefill.
    if verification_cycles + 1 != target_forward_passes:
        raise AssertionError(
            "target_forward_passes must equal 1 (prefill) + verification_cycles "
            f"({target_forward_passes} != 1 + {verification_cycles})"
        )


def summarize_timing_runs(runs: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Return median tps and time from a list of run dicts with 'tps' and 'time_sec'.
    Also calculates standard deviation and preserves all raw runs.
    """
    if not runs:
        return {"tps": 0.0, "time_sec": 0.0, "tps_std": 0.0, "runs": []}
    tps_vals = sorted(r["tps"] for r in runs)
    time_vals = sorted(r["time_sec"] for r in runs)
    n = len(tps_vals)
    mid = n // 2
    tps_med = tps_vals[mid] if n % 2 else (tps_vals[mid - 1] + tps_vals[mid]) / 2
    time_med = time_vals[mid] if n % 2 else (time_vals[mid - 1] + time_vals[mid]) / 2
    mean_tps = sum(tps_vals) / n
    var = sum((x - mean_tps) ** 2 for x in tps_vals) / n
    return {
        "tps": tps_med,
        "time_sec": time_med,
        "tps_std": var ** 0.5,
        "runs": runs,
    }
