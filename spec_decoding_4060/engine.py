from typing import Any, Optional
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kv_cache_utils import crop_past_key_values
from spec_decoding_4060.metrics import (
    acceptance_rate_pct,
    accepted_tokens_per_verification_cycle,
    output_tokens_per_target_forward,
    validate_speculative_counters,
)
try:
    from spec_decoding_4060.streamer import LiveTokenStreamer
except ImportError:
    try:
        from streamer import LiveTokenStreamer
    except ImportError:
        LiveTokenStreamer = None


# Patch upstream HF AssistedCandidateGenerator bug in transformers >= 5.x
# where assistant KV cache is not properly rolled back when draft tokens are rejected
try:
    from transformers.generation.candidate_generator import AssistedCandidateGenerator
    _orig_update_past = AssistedCandidateGenerator._update_past_and_masks

    def _patched_update_past(self, input_ids, remove_from_pkv=0, num_added_tokens=1):
        res = _orig_update_past(self, input_ids, remove_from_pkv, num_added_tokens)
        pkv = self.assistant_kwargs.get("past_key_values", None)
        if pkv is not None:
            cur_pkv_len = pkv.get_seq_length()
            target_len = input_ids.shape[-1] - 1
            if cur_pkv_len > target_len:
                pkv.crop(-(cur_pkv_len - target_len))
        return res

    AssistedCandidateGenerator._update_past_and_masks = _patched_update_past
except (ImportError, AttributeError):
    pass

class SpeculativeEngine:
    def __init__(self, config: Any):
        self.cfg = config
        self.target_dtype = getattr(self.cfg, "TARGET_DTYPE", self.cfg.DTYPE)

        # Explicitly activate selected CUDA device
        if torch.cuda.is_available() and hasattr(self.cfg, "DEVICE") and "cuda" in str(self.cfg.DEVICE):
            device_str = str(self.cfg.DEVICE)
            device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
            torch.cuda.set_device(device_idx)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if getattr(self.cfg, "CUDNN_BENCHMARK", False):
                torch.backends.cudnn.benchmark = True
            if getattr(self.cfg, "ALLOW_TF32", True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            # An fp32 target exists solely to guarantee AR == spec token-for-token. TF32
            # truncates fp32 matmul inputs to a 10-bit mantissa (~5e-4 relative error),
            # which reintroduces the near-tie argmax flips and breaks exactness. Force
            # true IEEE fp32 for the target whenever it is loaded in float32.
            if self.target_dtype == torch.float32:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
                print("[Init] fp32 target detected -> TF32 disabled (IEEE fp32 matmul) for exactness.")
            if getattr(self.cfg, "TORCH_DETERMINISTIC", False):
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)

        print(f"[Init] Loading Tokenizer from {self.cfg.TARGET_MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TARGET_MODEL_ID)

        print(f"[Init] Loading Target Model ({self.cfg.TARGET_MODEL_ID}) in {self.target_dtype}...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.TARGET_MODEL_ID,
            dtype=self.target_dtype,
            attn_implementation=self.cfg.ATTN_IMPLEMENTATION,
            low_cpu_mem_usage=True
        ).to(self.cfg.DEVICE).eval()

        print(f"[Init] Loading Draft Model ({self.cfg.DRAFT_MODEL_ID}) in {self.cfg.DTYPE}...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.DRAFT_MODEL_ID,
            dtype=self.cfg.DTYPE,
            attn_implementation=self.cfg.ATTN_IMPLEMENTATION,
            low_cpu_mem_usage=True
        ).to(self.cfg.DEVICE).eval()

        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        total_vram_gb = getattr(self.cfg, "VRAM_TOTAL_GB", None)
        if total_vram_gb is None and torch.cuda.is_available():
            try:
                device_str = str(getattr(self.cfg, "DEVICE", "cuda:0"))
                device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                total_vram_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            except Exception:
                total_vram_gb = 8.00
        total_vram_str = f"{total_vram_gb:.2f} GB" if total_vram_gb is not None else "N/A"
        print(f"[Init] Models loaded successfully! Static VRAM Usage: {vram_gb:.2f} GB / {total_vram_str}\n")

    def format_prompt(self, prompt_text: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": "You are a helpful, concise technical assistant."},
            {"role": "user", "content": prompt_text}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.cfg.DEVICE)

    @torch.inference_mode()
    def autoregressive_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stream: bool = False,
        streamer: Any = None,
        colorize: bool = False,
    ):
        """Standard token-by-token greedy decoding on the Target Model."""
        if stream and streamer is None and LiveTokenStreamer is not None:
            streamer = LiveTokenStreamer(
                self.tokenizer,
                label="Baseline Autoregressive (3B)",
                is_speculative=False,
                colorize=colorize,
            )

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        if streamer is not None and hasattr(streamer, "start"):
            streamer.start()

        past_kv = DynamicCache()
        curr_ids = input_ids.clone()
        next_input = curr_ids

        for _ in range(max_new_tokens):
            outputs = self.target_model(
                input_ids=next_input,
                past_key_values=past_kv,
                use_cache=True
            )
            past_kv = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)
            next_input = next_token

            if streamer is not None and hasattr(streamer, "put_token"):
                streamer.put_token(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        generated_tokens = curr_ids.shape[1] - input_ids.shape[1]

        if streamer is not None and hasattr(streamer, "end"):
            streamer.end()

        display_time = streamer.display_time if (streamer is not None and hasattr(streamer, "display_time")) else 0.0
        pure_inf_time = max(1e-6, elapsed - display_time)

        return {
            "output_ids": curr_ids,
            "tokens_generated": generated_tokens,
            "time_sec": elapsed,
            "tps": generated_tokens / elapsed,
            "inference_time_sec": pure_inf_time,
            "display_time_sec": display_time,
            "inference_tps": generated_tokens / pure_inf_time,
        }

    @torch.inference_mode()
    def custom_speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        gamma: int = 4,
        adaptive: bool = False,
        min_gamma: int = 1,
        max_gamma: int = 8,
        baseline_tps: Optional[float] = None,
        stream: bool = False,
        streamer: Any = None,
        colorize: bool = False,
    ):
        """
        Custom Greedy Speculative Decoding with KV-Cache Rollback.
        Guarantees zero prompt recomputation and 100% mathematical equivalence to Target.
        Supports fixed lookahead (gamma) or cost-aware dynamic adaptive lookahead (adaptive=True).
        """
        if gamma <= 0:
            adaptive = True
            gamma = 4

        if stream and streamer is None and LiveTokenStreamer is not None:
            label_str = (
                f"Custom Speculative (Adaptive K={min_gamma}..{max_gamma})"
                if adaptive
                else f"Custom Speculative (0.5B->3B, K={gamma})"
            )
            streamer = LiveTokenStreamer(
                self.tokenizer,
                label=label_str,
                is_speculative=True,
                colorize=colorize,
            )

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        if streamer is not None and hasattr(streamer, "start"):
            streamer.start()

        target_kv = DynamicCache()
        draft_kv = DynamicCache()

        curr_ids = input_ids.clone()
        prompt_len = input_ids.shape[1]

        # 1. Prefill both models with the initial prompt
        target_out = self.target_model(input_ids=curr_ids, past_key_values=target_kv, use_cache=True)
        target_kv = target_out.past_key_values
        
        draft_out = self.draft_model(input_ids=curr_ids, past_key_values=draft_kv, use_cache=True)
        draft_kv = draft_out.past_key_values

        # First token comes directly from the target model's prefill pass
        first_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)
        curr_ids = torch.cat([curr_ids, first_token], dim=-1)

        if streamer is not None and hasattr(streamer, "put_token"):
            streamer.put_token(first_token.item())

        total_drafted = 0
        total_accepted = 0
        target_forward_passes = 1  # initial prompt prefill
        verification_cycles = 0

        current_k = gamma
        recent_cycles: List[Dict[str, Any]] = []
        k_history: List[int] = [current_k]

        # 2. Main Speculative Loop
        while (curr_ids.shape[1] - prompt_len) < max_new_tokens:
            if curr_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break

            # Determine how many tokens to draft without exceeding max_new_tokens
            remaining = max_new_tokens - (curr_ids.shape[1] - prompt_len)
            current_gamma = min(current_k, remaining)
            if current_gamma <= 0:
                break

            # --- STEP A: DRAFTING K TOKENS ---
            t_d0 = time.perf_counter()
            draft_tokens = []
            # Draft model needs to process from its current KV cache state up to curr_ids
            draft_input = curr_ids[:, draft_kv.get_seq_length():]

            for _ in range(current_gamma):
                d_out = self.draft_model(input_ids=draft_input, past_key_values=draft_kv, use_cache=True)
                draft_kv = d_out.past_key_values
                next_draft_token = torch.argmax(d_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_draft_token)
                draft_input = next_draft_token

            draft_seq = torch.cat(draft_tokens, dim=-1)  # Shape: (1, current_gamma)
            total_drafted += current_gamma
            t_draft = time.perf_counter() - t_d0

            # --- STEP B: PARALLEL TARGET VERIFICATION ---
            # Target model currently has KV cache up to `curr_ids.shape[1] - 1`.
            # We feed the last confirmed token + all K draft tokens in ONE forward pass.
            t_v0 = time.perf_counter()
            verify_input = torch.cat([curr_ids[:, -1:], draft_seq], dim=-1)  # Shape: (1, current_gamma + 1)
            t_out = self.target_model(input_ids=verify_input, past_key_values=target_kv, use_cache=True)
            target_kv = t_out.past_key_values
            target_forward_passes += 1
            verification_cycles += 1

            # Target predictions for each of the drafted positions + 1 bonus position
            target_preds = torch.argmax(t_out.logits, dim=-1)  # Shape: (1, current_gamma + 1)
            t_verify = time.perf_counter() - t_v0

            # --- STEP C: ACCEPT / REJECT & KV ROLLBACK ---
            # Vectorized on-device comparison. The previous per-token loop issued up to
            # K+1 `.item()` calls, each forcing a full CUDA sync; this performs a single
            # sync for the first-mismatch index (plus one rare sync when EOS is present).
            matches = draft_seq[0, :current_gamma] == target_preds[0, :current_gamma]
            mismatches = (~matches).nonzero(as_tuple=True)[0]
            n_accepted = int(mismatches[0].item()) if mismatches.numel() > 0 else current_gamma

            # EOS counts as an accepted token, but generation must stop right after it.
            if n_accepted > 0 and self.tokenizer.eos_token_id is not None:
                eos_positions = (
                    draft_seq[0, :n_accepted] == self.tokenizer.eos_token_id
                ).nonzero(as_tuple=True)[0]
                if eos_positions.numel() > 0:
                    n_accepted = int(eos_positions[0].item()) + 1

            total_accepted += n_accepted

            accepted_chunk = None
            if n_accepted > 0:
                accepted_chunk = draft_seq[:, :n_accepted]
                curr_ids = torch.cat([curr_ids, accepted_chunk], dim=-1)

            # Always append the target model's authoritative token at position `n_accepted`
            # (Either the correction token if rejected early, or the bonus (K+1)-th token if all K accepted!)
            authoritative_token = None
            has_eos_in_accepted = (curr_ids[0, -1].item() == self.tokenizer.eos_token_id)
            if not has_eos_in_accepted and (curr_ids.shape[1] - prompt_len) < max_new_tokens:
                authoritative_token = target_preds[:, n_accepted : n_accepted + 1]
                curr_ids = torch.cat([curr_ids, authoritative_token], dim=-1)

            if streamer is not None and hasattr(streamer, "put_speculative_chunk"):
                streamer.put_speculative_chunk(
                    accepted_tokens=accepted_chunk,
                    authoritative_token=authoritative_token,
                    drafted_count=current_gamma,
                )

            # --- STEP D: CROP KV CACHES TO VALID LENGTH ---
            # Target KV currently holds up to the end of `verify_input`.
            # Valid processed cache length is `curr_ids.shape[1] - 1` (since the last token in curr_ids hasn't been passed through yet)
            valid_cache_len = curr_ids.shape[1] - 1
            crop_past_key_values(target_kv, valid_cache_len)
            crop_past_key_values(draft_kv, min(draft_kv.get_seq_length(), valid_cache_len))

            # --- STEP E: COST-AWARE ADAPTIVE LOOKAHEAD HEURISTIC ---
            if adaptive:
                cycle_time = t_draft + t_verify
                tokens_emitted = n_accepted + (1 if authoritative_token is not None else 0)
                cycle_tps = tokens_emitted / max(1e-6, cycle_time)

                recent_cycles.append({
                    "k": current_gamma,
                    "accepted": n_accepted,
                    "t_draft": t_draft,
                    "t_verify": t_verify,
                    "cycle_time": cycle_time,
                    "tokens": tokens_emitted,
                    "cycle_tps": cycle_tps,
                })
                if len(recent_cycles) > 4:
                    recent_cycles.pop(0)

                avg_accepted = sum(c["accepted"] for c in recent_cycles) / len(recent_cycles)
                tot_tok = sum(c["tokens"] for c in recent_cycles)
                tot_time = sum(c["cycle_time"] for c in recent_cycles)
                rolling_tps = tot_tok / max(1e-6, tot_time)
                b_tps = (
                    baseline_tps
                    if (baseline_tps is not None and baseline_tps > 0)
                    else (1.0 / max(1e-6, t_verify * 0.85))
                )

                if n_accepted == current_gamma:
                    # 100% acceptance in this cycle: under-speculating, increase lookahead
                    current_k = min(max_gamma, current_k + 1)
                elif n_accepted == 0:
                    # Complete rejection on the first draft token: immediately shrink lookahead
                    current_k = max(min_gamma, current_k - 2)
                elif rolling_tps < b_tps:
                    # Speculative throughput is failing to beat baseline AR: reduce draft load
                    current_k = max(min_gamma, current_k - 1)
                elif avg_accepted < 1.4 and current_k > 2:
                    # Average yield is less than 1.4 tokens: large K wastes drafting time
                    current_k = max(min_gamma, current_k - 1)
                elif avg_accepted >= current_k - 0.5 and rolling_tps >= b_tps * 1.15:
                    # High yield and healthy speedup: expand lookahead
                    current_k = min(max_gamma, current_k + 1)

                k_history.append(current_k)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        generated_tokens = curr_ids.shape[1] - prompt_len

        if streamer is not None and hasattr(streamer, "end"):
            streamer.end()

        acceptance_rate = acceptance_rate_pct(total_accepted, total_drafted)
        otptf = output_tokens_per_target_forward(generated_tokens, target_forward_passes)
        atpvc = accepted_tokens_per_verification_cycle(total_accepted, verification_cycles)

        validate_speculative_counters(
            output_tokens=generated_tokens,
            max_new_tokens=max_new_tokens,
            total_drafted=total_drafted,
            total_accepted=total_accepted,
            target_forward_passes=target_forward_passes,
            verification_cycles=verification_cycles,
        )

        display_time = streamer.display_time if (streamer is not None and hasattr(streamer, "display_time")) else 0.0
        pure_inf_time = max(1e-6, elapsed - display_time)

        return {
            "output_ids": curr_ids,
            "tokens_generated": generated_tokens,
            "time_sec": elapsed,
            "tps": generated_tokens / elapsed,
            "inference_time_sec": pure_inf_time,
            "display_time_sec": display_time,
            "inference_tps": generated_tokens / pure_inf_time,
            "acceptance_rate": acceptance_rate,
            "tokens_per_target_step": otptf,
            "output_tokens_per_target_forward": otptf,
            "accepted_tokens_per_verification_cycle": atpvc,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "target_forward_passes": target_forward_passes,
            "verification_cycles": verification_cycles,
            "adaptive": adaptive,
            "k_history": k_history,
            "final_k": current_k if adaptive else gamma,
        }

    @torch.inference_mode()
    def hf_assisted_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stream: bool = False,
        streamer: Any = None,
    ):
        """Native Hugging Face Assisted Decoding for baseline reference."""
        if stream and streamer is None:
            from transformers import TextStreamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        gen_kwargs = {
            "assistant_model": self.draft_model,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        out_ids = self.target_model.generate(
            input_ids,
            **gen_kwargs
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        generated_tokens = out_ids.shape[1] - input_ids.shape[1]
        return {
            "output_ids": out_ids,
            "tokens_generated": generated_tokens,
            "time_sec": elapsed,
            "tps": generated_tokens / elapsed
        }
