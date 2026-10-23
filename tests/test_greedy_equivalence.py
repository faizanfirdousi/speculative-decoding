from dataclasses import dataclass
from typing import List, Optional
import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from kv_cache_utils import crop_past_key_values


class DeterministicMockModel(nn.Module):
    """
    Mock causal LM that produces deterministic next-token predictions based
    on a fixed pseudo-random hash of the input sequence context.
    
    Guarantees that given any prefix [x_0, ..., x_t], the output logits
    are strictly deterministic and causal (independent of future tokens).
    """
    def __init__(self, vocab_size: int = 100, eos_token_id: int = 99):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_len = past_len + seq_len

        # Deterministic logits based on prefix sum and position
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        
        # Build deterministic next token predictions
        for b in range(batch_size):
            for s in range(seq_len):
                abs_pos = past_len + s
                val = input_ids[b, s].item()
                # Deterministic function of current token and position
                favored_token = (int(val) * 7 + abs_pos * 13 + 3) % (self.vocab_size - 1)
                logits[b, s, favored_token] = 10.0

        if past_key_values is not None:
            # Simulate caching by storing dummy key/value tensors of length total_len
            dummy_k = torch.zeros(batch_size, 1, seq_len, 4)
            dummy_v = torch.zeros(batch_size, 1, seq_len, 4)
            past_key_values.update(dummy_k, dummy_v, layer_idx=0)

        @dataclass
        class MockOutput:
            logits: torch.Tensor
            past_key_values: Optional[DynamicCache]

        return MockOutput(logits=logits, past_key_values=past_key_values)


class DeterministicMockDraftModel(nn.Module):
    """Mock draft model that occasionally matches the target model and occasionally diverges."""
    def __init__(self, target_model: DeterministicMockModel, agreement_rate: float = 0.7):
        super().__init__()
        self.target_model = target_model
        self.agreement_rate = agreement_rate

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: bool = True,
    ):
        out = self.target_model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        logits = out.logits.clone()
        # With probability (1 - agreement_rate), alter the top token
        for b in range(input_ids.shape[0]):
            for s in range(input_ids.shape[1]):
                abs_pos = (past_key_values.get_seq_length() if past_key_values else 0) - input_ids.shape[1] + s
                if (abs_pos % 3) == 0:  # deterministic divergence every 3 positions
                    top_idx = torch.argmax(logits[b, s]).item()
                    alt_idx = (top_idx + 1) % self.target_model.vocab_size
                    logits[b, s, alt_idx] = 20.0
        return out.__class__(logits=logits, past_key_values=out.past_key_values)


def run_mock_autoregressive(target_model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int):
    curr_ids = input_ids.clone()
    past_kv = DynamicCache()
    next_input = curr_ids

    for _ in range(max_new_tokens):
        out = target_model(input_ids=next_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
        next_input = next_tok
        if next_tok.item() == eos_id:
            break
    return curr_ids


def run_mock_speculative(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    eos_id: int,
):
    target_kv = DynamicCache()
    draft_kv = DynamicCache()
    curr_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]

    # Prefill
    t_out = target_model(input_ids=curr_ids, past_key_values=target_kv, use_cache=True)
    target_kv = t_out.past_key_values
    d_out = draft_model(input_ids=curr_ids[:, :-1], past_key_values=draft_kv, use_cache=True)
    draft_kv = d_out.past_key_values
    first_token = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
    curr_ids = torch.cat([curr_ids, first_token], dim=-1)

    while (curr_ids.shape[1] - prompt_len) < max_new_tokens:
        if curr_ids[0, -1].item() == eos_id:
            break

        remaining = max_new_tokens - (curr_ids.shape[1] - prompt_len)
        current_gamma = min(gamma, remaining)
        if current_gamma <= 0:
            break

        # Draft K tokens
        draft_tokens = []
        draft_input = curr_ids[:, draft_kv.get_seq_length():]
        for _ in range(current_gamma):
            d_out = draft_model(input_ids=draft_input, past_key_values=draft_kv, use_cache=True)
            draft_kv = d_out.past_key_values
            next_d = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_d)
            draft_input = next_d

        draft_seq = torch.cat(draft_tokens, dim=-1)

        # Target verification
        verify_input = torch.cat([curr_ids[:, -1:], draft_seq], dim=-1)
        t_out = target_model(input_ids=verify_input, past_key_values=target_kv, use_cache=True)
        target_kv = t_out.past_key_values
        target_preds = torch.argmax(t_out.logits, dim=-1)

        # Accept / Reject
        n_accepted = 0
        for i in range(current_gamma):
            if draft_seq[0, i].item() == target_preds[0, i].item():
                n_accepted += 1
                if draft_seq[0, i].item() == eos_id:
                    break
            else:
                break

        if n_accepted > 0:
            curr_ids = torch.cat([curr_ids, draft_seq[:, :n_accepted]], dim=-1)

        has_eos = curr_ids[0, -1].item() == eos_id
        if not has_eos and (curr_ids.shape[1] - prompt_len) < max_new_tokens:
            authoritative_token = target_preds[:, n_accepted : n_accepted + 1]
            curr_ids = torch.cat([curr_ids, authoritative_token], dim=-1)

        valid_cache_len = curr_ids.shape[1] - 1
        crop_past_key_values(target_kv, valid_cache_len)
        crop_past_key_values(draft_kv, min(draft_kv.get_seq_length(), valid_cache_len))

    return curr_ids


@pytest.mark.parametrize("gamma", [1, 3, 5, 8])
def test_mock_greedy_exact_match(gamma):
    """
    Greedy speculative decoding with a causal, deterministic model MUST produce
    the 100% exact same token sequence as baseline autoregressive greedy decoding.
    """
    target = DeterministicMockModel(vocab_size=50, eos_token_id=99)
    draft = DeterministicMockDraftModel(target, agreement_rate=0.7)
    
    prompt = torch.tensor([[1, 5, 12, 23]])
    max_tokens = 30
    
    ar_out = run_mock_autoregressive(target, prompt, max_tokens, eos_id=99)
    spec_out = run_mock_speculative(target, draft, prompt, max_tokens, gamma=gamma, eos_id=99)
    
    assert torch.equal(ar_out, spec_out), (
        f"Mismatch with gamma={gamma}:\nAR  : {ar_out.tolist()}\nSpec: {spec_out.tolist()}"
    )


def test_mock_greedy_exact_match_perfect_draft():
    """When draft model is 100% accurate, all tokens are accepted with bonus tokens."""
    target = DeterministicMockModel(vocab_size=50, eos_token_id=99)
    # Draft identical to target
    prompt = torch.tensor([[3, 7, 19]])
    ar_out = run_mock_autoregressive(target, prompt, 25, eos_id=99)
    spec_out = run_mock_speculative(target, target, prompt, 25, gamma=4, eos_id=99)
    assert torch.equal(ar_out, spec_out)


def test_mock_greedy_exact_match_zero_acceptance():
    """When draft model is 0% accurate, every draft token is rejected and correction tokens emitted."""
    target = DeterministicMockModel(vocab_size=50, eos_token_id=99)
    
    class InvertedDraftModel(nn.Module):
        def forward(self, input_ids, past_key_values=None, use_cache=True):
            out = target(input_ids, past_key_values=past_key_values, use_cache=use_cache)
            # Guarantee mismatch by picking lowest logit
            argmin = torch.argmin(out.logits, dim=-1, keepdim=True)
            new_logits = torch.zeros_like(out.logits)
            new_logits.scatter_(-1, argmin, 100.0)
            return out.__class__(logits=new_logits, past_key_values=out.past_key_values)
    
    bad_draft = InvertedDraftModel()
    prompt = torch.tensor([[2, 4, 8]])
    ar_out = run_mock_autoregressive(target, prompt, 20, eos_id=99)
    spec_out = run_mock_speculative(target, bad_draft, prompt, 20, gamma=3, eos_id=99)
    assert torch.equal(ar_out, spec_out)
