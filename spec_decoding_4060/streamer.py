"""
Live Token Streamer and Speed Monitor for Speculative Decoding.

Provides zero-latency streaming of generated tokens to stdout with real-time
throughput calculation (tokens/sec), ANSI colorization of draft vs target tokens,
and terminal window/tab title updates.
"""
import sys
import time
from typing import Any, Dict, List, Optional
import torch


class LiveTokenStreamer:
    """
    Real-time terminal streamer for Autoregressive and Speculative Decoding.
    
    Streams decoded tokens to stdout as they are generated, handles UTF-8 multibyte
    boundaries cleanly, tracks real-time throughput (tok/s), and displays
    visual telemetry upon completion.
    """

    def __init__(
        self,
        tokenizer: Any,
        label: str = "Generation",
        is_speculative: bool = False,
        colorize: bool = False,
        show_stats: bool = True,
        update_title: bool = True,
    ):
        self.tokenizer = tokenizer
        self.label = label
        self.is_speculative = is_speculative
        self.colorize = colorize
        self.show_stats = show_stats
        self.update_title = update_title

        self.start_time: Optional[float] = None
        self.total_tokens: int = 0
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.verification_cycles: int = 0

        # Token cache for multi-byte UTF-8 boundary safety
        self.token_cache: List[int] = []
        self.print_len: int = 0

    def start(self):
        """Synchronize CUDA device and mark generation start time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self.total_tokens = 0
        self.total_drafted = 0
        self.total_accepted = 0
        self.verification_cycles = 0
        self.token_cache = []
        self.print_len = 0

    def _decode_and_print(self, token_ids: List[int], color_code: Optional[str] = None):
        """Safely decode and stream new tokens to stdout without breaking multi-byte UTF-8 sequences."""
        if not token_ids:
            return

        self.token_cache.extend(token_ids)
        self.total_tokens += len(token_ids)

        full_text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)

        # Do not print if ending on an incomplete UTF-8 sequence replacement character
        if full_text.endswith("\ufffd"):
            return

        printable = full_text[self.print_len :]
        if printable:
            if self.colorize and color_code:
                sys.stdout.write(f"{color_code}{printable}\033[0m")
            else:
                sys.stdout.write(printable)
            sys.stdout.flush()
            self.print_len = len(full_text)

        # Update terminal tab/window title with live speed if terminal supports it
        if self.update_title and self.start_time and sys.stdout.isatty():
            elapsed = time.perf_counter() - self.start_time
            if elapsed > 0:
                cur_tps = self.total_tokens / elapsed
                sys.stdout.write(
                    f"\033]0;⚡ {self.label}: {cur_tps:.1f} tok/s ({self.total_tokens} tok)\007"
                )
                sys.stdout.flush()

    def put_token(self, token_id: int):
        """Put a single token (e.g. from Autoregressive decode or initial prefill)."""
        if self.start_time is None:
            self.start()
        # White/standard for target authoritative token
        self._decode_and_print([token_id], color_code="\033[97m")

    def put_speculative_chunk(
        self,
        accepted_tokens: Optional[torch.Tensor],
        authoritative_token: Optional[torch.Tensor],
        drafted_count: int,
    ):
        """
        Put a verified speculative decoding chunk.
        - Accepted draft tokens are streamed first (e.g. in cyan).
        - Authoritative target token is streamed next (e.g. in green).
        """
        if self.start_time is None:
            self.start()

        self.verification_cycles += 1
        self.total_drafted += drafted_count

        if accepted_tokens is not None and accepted_tokens.numel() > 0:
            acc_list = accepted_tokens.view(-1).tolist()
            self.total_accepted += len(acc_list)
            # Cyan color for draft accepted tokens
            self._decode_and_print(acc_list, color_code="\033[96m")

        if authoritative_token is not None and authoritative_token.numel() > 0:
            auth_list = authoritative_token.view(-1).tolist()
            # Green color for target model authoritative / bonus token
            self._decode_and_print(auth_list, color_code="\033[92m")

    def end(self) -> Dict[str, Any]:
        """Finalize the stream, flush any remaining text, and display telemetry stats."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0.0

        # Flush any remaining tokens in cache
        full_text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        remaining = full_text[self.print_len :]
        if remaining:
            sys.stdout.write(remaining)
            sys.stdout.flush()
            self.print_len = len(full_text)

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Reset terminal title if changed
        if self.update_title and sys.stdout.isatty():
            sys.stdout.write("\033]0;\007")
            sys.stdout.flush()

        tps = self.total_tokens / elapsed if elapsed > 0 else 0.0
        stats = {
            "tokens": self.total_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "is_speculative": self.is_speculative,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "verification_cycles": self.verification_cycles,
        }

        if self.show_stats and elapsed > 0:
            print("\n" + "─" * 68)
            if self.is_speculative and self.total_drafted > 0:
                acc_rate = 100.0 * self.total_accepted / self.total_drafted
                fwd_passes = self.verification_cycles + 1  # includes prefill
                t_fwd_eff = self.total_tokens / fwd_passes if fwd_passes > 0 else 0.0
                print(
                    f"⚡ {self.label} Telemetry:\n"
                    f"   • Throughput Speed   : \033[1;32m{tps:.2f} tokens/sec\033[0m\n"
                    f"   • Generation Time    : {elapsed:.2f} seconds\n"
                    f"   • Tokens Generated   : {self.total_tokens} tokens\n"
                    f"   • Draft Acceptance   : \033[1;36m{acc_rate:.1f}%\033[0m ({self.total_accepted}/{self.total_drafted} draft tokens accepted)\n"
                    f"   • Verification Passes: {fwd_passes} passes (~{t_fwd_eff:.2f} tokens / target forward)"
                )
                if self.colorize:
                    print(
                        f"   • Visual Legend      : \033[96m■ Draft Accepted\033[0m | "
                        f"\033[92m■ Target Authoritative / Bonus\033[0m"
                    )
            else:
                print(
                    f"⚡ {self.label} Telemetry:\n"
                    f"   • Throughput Speed : \033[1;32m{tps:.2f} tokens/sec\033[0m\n"
                    f"   • Generation Time  : {elapsed:.2f} seconds\n"
                    f"   • Tokens Generated : {self.total_tokens} tokens"
                )
            print("─" * 68 + "\n")

        return stats
