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
        self.display_time: float = 0.0

        # Token cache for multi-byte UTF-8 boundary safety
        self.token_cache: List[int] = []
        self.print_len: int = 0
        self._last_title_update: float = 0.0

    def start(self):
        """Synchronize CUDA device and mark generation start time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self.total_tokens = 0
        self.total_drafted = 0
        self.total_accepted = 0
        self.verification_cycles = 0
        self.display_time = 0.0
        self.token_cache = []
        self.print_len = 0
        self._last_title_update = 0.0

    def _maybe_update_title(self, now: float):
        """Throttle terminal title updates to at most once per 150ms to prevent terminal redraw stalls."""
        if self.update_title and sys.stdout.isatty() and (now - self._last_title_update >= 0.15):
            elapsed = now - self.start_time if self.start_time else 0.0
            if elapsed > 0:
                cur_tps = self.total_tokens / elapsed
                sys.stdout.write(
                    f"\033]0;⚡ {self.label}: {cur_tps:.1f} tok/s ({self.total_tokens} tok)\007"
                )
                sys.stdout.flush()
                self._last_title_update = now

    def put_token(self, token_id: int):
        """Put a single token (e.g. from Autoregressive decode or initial prefill)."""
        t0 = time.perf_counter()
        if self.start_time is None:
            self.start()

        self.token_cache.append(token_id)
        self.total_tokens += 1

        full_text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
        if not full_text.endswith("\ufffd"):
            printable = full_text[self.print_len :]
            if printable:
                out_chunk = f"\033[97m{printable}\033[0m" if self.colorize else printable
                sys.stdout.write(out_chunk)
                sys.stdout.flush()
                self.print_len = len(full_text)
                self._maybe_update_title(t0)

        self.display_time += (time.perf_counter() - t0)

    def put_speculative_chunk(
        self,
        accepted_tokens: Optional[torch.Tensor],
        authoritative_token: Optional[torch.Tensor],
        drafted_count: int,
    ):
        """
        Put a verified speculative decoding chunk with consolidated I/O.
        - Formats accepted tokens (cyan) and authoritative token (green) in a single buffer.
        - Issues exactly ONE sys.stdout.write and ONE sys.stdout.flush per cycle.
        """
        t0 = time.perf_counter()
        if self.start_time is None:
            self.start()

        self.verification_cycles += 1
        self.total_drafted += drafted_count

        acc_list = accepted_tokens.view(-1).tolist() if (accepted_tokens is not None and accepted_tokens.numel() > 0) else []
        auth_list = authoritative_token.view(-1).tolist() if (authoritative_token is not None and authoritative_token.numel() > 0) else []

        self.total_accepted += len(acc_list)
        chunk_tokens = acc_list + auth_list
        if not chunk_tokens:
            self.display_time += (time.perf_counter() - t0)
            return

        self.total_tokens += len(chunk_tokens)

        if self.colorize:
            out_parts = []
            if acc_list:
                self.token_cache.extend(acc_list)
                text_acc = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
                if not text_acc.endswith("\ufffd"):
                    printable_acc = text_acc[self.print_len :]
                    if printable_acc:
                        out_parts.append(f"\033[96m{printable_acc}\033[0m")
                        self.print_len = len(text_acc)

            if auth_list:
                self.token_cache.extend(auth_list)
                text_auth = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
                if not text_auth.endswith("\ufffd"):
                    printable_auth = text_auth[self.print_len :]
                    if printable_auth:
                        out_parts.append(f"\033[92m{printable_auth}\033[0m")
                        self.print_len = len(text_auth)

            if out_parts:
                sys.stdout.write("".join(out_parts))
                sys.stdout.flush()
                self._maybe_update_title(t0)
        else:
            self.token_cache.extend(chunk_tokens)
            full_text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True)
            if not full_text.endswith("\ufffd"):
                printable = full_text[self.print_len :]
                if printable:
                    sys.stdout.write(printable)
                    sys.stdout.flush()
                    self.print_len = len(full_text)
                    self._maybe_update_title(t0)

        self.display_time += (time.perf_counter() - t0)

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
        pure_inf_elapsed = max(1e-6, elapsed - self.display_time)
        pure_inf_tps = self.total_tokens / pure_inf_elapsed if pure_inf_elapsed > 0 else 0.0
        overhead_pct = (self.display_time / elapsed * 100.0) if elapsed > 0 else 0.0

        stats = {
            "tokens": self.total_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "inference_time_sec": pure_inf_elapsed,
            "display_time_sec": self.display_time,
            "inference_tps": pure_inf_tps,
            "overhead_pct": overhead_pct,
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
                    f"   • Throughput Speed   : \033[1;32m{tps:.2f} tokens/sec\033[0m ({elapsed:.2f}s total wall-clock)\n"
                    f"   • Inference Compute  : \033[1;36m{pure_inf_tps:.2f} tokens/sec\033[0m ({pure_inf_elapsed:.2f}s pure model compute)\n"
                    f"   • Display Overhead   : {self.display_time*1000:.1f}ms ({overhead_pct:.1f}% UI/streaming latency)\n"
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
                    f"   • Throughput Speed : \033[1;32m{tps:.2f} tokens/sec\033[0m ({elapsed:.2f}s total wall-clock)\n"
                    f"   • Inference Compute: \033[1;36m{pure_inf_tps:.2f} tokens/sec\033[0m ({pure_inf_elapsed:.2f}s pure model compute)\n"
                    f"   • Display Overhead : {self.display_time*1000:.1f}ms ({overhead_pct:.1f}% UI/streaming latency)\n"
                    f"   • Tokens Generated : {self.total_tokens} tokens"
                )
            print("─" * 68 + "\n")

        return stats
