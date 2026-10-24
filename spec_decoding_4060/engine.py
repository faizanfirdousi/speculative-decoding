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
            if getattr(self.cfg, "TORCH_DETERMINISTIC", False):
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)

        print(f"[Init] Loading Tokenizer from {self.cfg.TARGET_MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.TARGET_MODEL_ID)

        print(f"[Init] Loading Target Model ({self.cfg.TARGET_MODEL_ID}) in {self.cfg.DTYPE}...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.TARGET_MODEL_ID,
            dtype=self.cfg.DTYPE,
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

        return {
            "output_ids": curr_ids,
            "tokens_generated": generated_tokens,
            "time_sec": elapsed,
            "tps": generated_tokens / elapsed
        }

    @torch.inference_mode()
    def custom_speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        gamma: int,
        stream: bool = False,
        streamer: Any = None,
        colorize: bool = False,
    ):
        """
        Custom Greedy Speculative Decoding with KV-Cache Rollback.
        Guarantees zero prompt recomputation and 100% mathematical equivalence to Target.
        """
        if stream and streamer is None and LiveTokenStreamer is not None:
            streamer = LiveTokenStreamer(
                self.tokenizer,
                label=f"Custom Speculative (0.5B->3B, K={gamma})",
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
        
        draft_out = self.draft_model(input_ids=curr_ids[:, :-1], past_key_values=draft_kv, use_cache=True)
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

        # 2. Main Speculative Loop
        while (curr_ids.shape[1] - prompt_len) < max_new_tokens:
            if curr_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break

            # Determine how many tokens to draft without exceeding max_new_tokens
            remaining = max_new_tokens - (curr_ids.shape[1] - prompt_len)
            current_gamma = min(gamma, remaining)
            if current_gamma <= 0:
                break

            # --- STEP A: DRAFTING K TOKENS ---
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

            # --- STEP B: PARALLEL TARGET VERIFICATION ---
            # Target model currently has KV cache up to `curr_ids.shape[1] - 1`.
            # We feed the last confirmed token + all K draft tokens in ONE forward pass.
            verify_input = torch.cat([curr_ids[:, -1:], draft_seq], dim=-1)  # Shape: (1, current_gamma + 1)
            t_out = self.target_model(input_ids=verify_input, past_key_values=target_kv, use_cache=True)
            target_kv = t_out.past_key_values
            target_forward_passes += 1
            verification_cycles += 1

            # Target predictions for each of the drafted positions + 1 bonus position
            target_preds = torch.argmax(t_out.logits, dim=-1)  # Shape: (1, current_gamma + 1)

            # --- STEP C: ACCEPT / REJECT & KV ROLLBACK ---
            n_accepted = 0
            for i in range(current_gamma):
                if draft_seq[0, i].item() == target_preds[0, i].item():
                    n_accepted += 1
                    if draft_seq[0, i].item() == self.tokenizer.eos_token_id:
                        break
                else:
                    break

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

        return {
            "output_ids": curr_ids,
            "tokens_generated": generated_tokens,
            "time_sec": elapsed,
            "tps": generated_tokens / elapsed,
            "acceptance_rate": acceptance_rate,
            "tokens_per_target_step": otptf,
            "output_tokens_per_target_forward": otptf,
            "accepted_tokens_per_verification_cycle": atpvc,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "target_forward_passes": target_forward_passes,
            "verification_cycles": verification_cycles,
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
