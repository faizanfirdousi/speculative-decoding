#!/usr/bin/env python3
"""
Interactive Real-Time Speculative Decoding Streamer & Speed Monitor.

Enables live terminal streaming of generated tokens with real-time throughput
(tokens/sec), draft acceptance statistics, and side-by-side comparison between
standard autoregressive baseline and speculative decoding on NVIDIA Ada GPUs.

Usage:
    # Run interactive prompt session:
    python stream_demo.py

    # Run speculative decoding with live streaming tokens and speed:
    python stream_demo.py --prompt "Write a quicksort in Python"

    # Compare Baseline AR vs Speculative Decoding side-by-side live:
    python stream_demo.py --prompt "Explain quantum computing" --mode compare

    # Colorize draft-accepted tokens vs target bonus/correction tokens:
    python stream_demo.py --prompt "Write a BST in Python" --color
"""
import argparse
import os
from pathlib import Path
import sys
import time
from typing import Optional

# WSL2 memory configuration cleanup if present
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    clean_parts = [
        p for p in os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",")
        if "expandable_segments" not in p.lower()
    ]
    if clean_parts:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(clean_parts)
    else:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from spec_decoding_4060.config import SpecConfig
from spec_decoding_4060.engine import SpeculativeEngine
from spec_decoding_4060.streamer import LiveTokenStreamer


PRESET_PROMPTS = [
    {
        "title": "Code Generation (Python Binary Search Tree)",
        "prompt": "Write a complete, well-documented Python implementation of a Binary Search Tree with insert, search, and in-order traversal methods.",
    },
    {
        "title": "Structured JSON (5 Software Engineers)",
        "prompt": "Generate a JSON array containing 5 fictional software engineers with fields: id, name, role, skills (array), and years_of_experience.",
    },
    {
        "title": "Technical Explanation (CPU Caches & MESI)",
        "prompt": "Explain how CPU L1, L2, and L3 caches work, including cache lines, associativity, and the MESI coherence protocol.",
    },
]


def print_banner(cfg: SpecConfig, gpu_name: str, vram_gb: float):
    print("\n" + "═" * 72)
    print("  🚀 Speculative Decoding Live Streamer & Speed Telemetry")
    print(f"  • GPU Hardware : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"  • Target Model : {cfg.TARGET_MODEL_ID} ({cfg.DTYPE})")
    print(f"  • Draft Model  : {cfg.DRAFT_MODEL_ID} ({cfg.DTYPE})")
    print(f"  • Lookahead (K): {cfg.GAMMA} draft tokens per verification step")
    print("═" * 72 + "\n")


def run_stream_session(
    engine: SpeculativeEngine,
    prompt_text: str,
    mode: str = "spec",
    gamma: int = 5,
    max_new_tokens: int = 200,
    colorize: bool = False,
):
    print(f"\n\033[1;34m[Prompt]\033[0m {prompt_text}\n")
    input_ids = engine.format_prompt(prompt_text)

    if mode in ("compare", "baseline", "ar"):
        print("─" * 72)
        print(f"\033[1;33m[Mode 1/2: Standard Autoregressive Baseline (3B Model)]\033[0m")
        print("Streaming tokens one-by-one from target model...")
        print("─" * 72)
        streamer_ar = LiveTokenStreamer(
            engine.tokenizer,
            label="Baseline Autoregressive (3B)",
            is_speculative=False,
            colorize=False,
            show_stats=True,
        )
        torch.cuda.reset_peak_memory_stats()
        ar_res = engine.autoregressive_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            stream=True,
            streamer=streamer_ar,
        )

    if mode in ("compare", "spec", "speculative"):
        print("─" * 72)
        print(
            f"\033[1;36m[Mode 2/2: Custom Speculative Decoding (0.5B -> 3B, Lookahead K={gamma})]\033[0m"
        )
        print("Streaming tokens in parallel verification chunks...")
        if colorize:
            print("Visual Color Code: \033[96m■ Cyan = Draft Accepted\033[0m | \033[92m■ Green = Target Authoritative\033[0m")
        print("─" * 72)
        streamer_spec = LiveTokenStreamer(
            engine.tokenizer,
            label=f"Custom Speculative (K={gamma})",
            is_speculative=True,
            colorize=colorize,
            show_stats=True,
        )
        torch.cuda.reset_peak_memory_stats()
        spec_res = engine.custom_speculative_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            stream=True,
            streamer=streamer_spec,
            colorize=colorize,
        )

    if mode == "compare":
        ar_tps = ar_res["tps"]
        spec_tps = spec_res["tps"]
        speedup = spec_tps / ar_tps if ar_tps > 0 else 0.0
        acc_rate = spec_res["acceptance_rate"]
        otptf = spec_res.get("output_tokens_per_target_forward", 0.0)

        print("═" * 72)
        print("  📊 COMPARISON SUMMARY (Live Speed & Efficiency)")
        print("═" * 72)
        print(f"  • Baseline Autoregressive (3B) : \033[1m{ar_tps:6.2f} tokens/sec\033[0m ({ar_res['time_sec']:.2f}s)")
        print(f"  • Speculative Decoding (K={gamma})   : \033[1;32m{spec_tps:6.2f} tokens/sec\033[0m ({spec_res['time_sec']:.2f}s)")
        print(f"  • Effective Acceleration       : \033[1;32m{speedup:6.2f}x FASTER\033[0m ⚡")
        print(f"  • Draft Model Acceptance Rate  : \033[1;36m{acc_rate:6.1f}%\033[0m")
        print(f"  • Output Tok / Target Forward  : \033[1m{otptf:6.2f} tokens/fwd\033[0m")
        print("═" * 72 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live Speculative Decoding Streamer & Real-Time Speed Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        "--profile",
        dest="gpu",
        default="auto",
        choices=["rtx4060", "rtx4000", "auto"],
        help="Hardware profile to execute ('rtx4060', 'rtx4000', or 'auto').",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow running a profile even if the detected GPU does not match the expected profile hardware.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Input text prompt to generate from.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="compare",
        choices=["spec", "speculative", "ar", "baseline", "compare"],
        help="Generation mode: 'spec' (fast speculative), 'ar' (standard baseline), or 'compare' (both).",
    )
    parser.add_argument(
        "--gamma",
        "-k",
        type=int,
        default=None,
        help="Lookahead draft tokens count K per verification cycle (default: from profile config).",
    )
    parser.add_argument(
        "--max-new-tokens",
        "-n",
        type=int,
        default=180,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        default=True,
        help="Colorize draft-accepted tokens in cyan and target-authoritative tokens in green.",
    )
    parser.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable ANSI token colorization.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch continuous interactive prompt session.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    from spec_decoding_4060.benchmark import (
        auto_detect_profile,
        detect_gpu_hardware,
        load_profile_config,
        validate_profile_gpu,
    )

    hardware = detect_gpu_hardware()
    profile_name = args.gpu
    if profile_name == "auto":
        profile_name = auto_detect_profile(hardware)
        print(f"[Auto-Detect] Detected Hardware: '{hardware['gpu_name']}' -> Profile: '{profile_name}'")

    overrides = {}
    if args.gamma is not None:
        overrides["GAMMA"] = args.gamma
    if args.max_new_tokens is not None:
        overrides["MAX_NEW_TOKENS"] = args.max_new_tokens

    cfg = load_profile_config(profile_name, overrides)
    validate_profile_gpu(cfg, profile_name, hardware, allow_mismatch=args.allow_mismatch)

    effective_gamma = getattr(cfg, "GAMMA", 5)
    gpu_name = hardware["gpu_name"]
    vram_gb = hardware["total_vram_gb"]
    print_banner(cfg, gpu_name, vram_gb)

    print("[Init] Loading Speculative Decoding Engine...")
    engine = SpeculativeEngine(cfg)

    # CUDA warm-up to ensure stable execution speed
    print("[Warmup] Warming up CUDA kernels...")
    warmup_ids = engine.format_prompt("Hello")
    _ = engine.autoregressive_generate(warmup_ids, max_new_tokens=8)
    _ = engine.custom_speculative_generate(warmup_ids, max_new_tokens=8, gamma=effective_gamma)
    torch.cuda.synchronize()
    print("[Warmup] Ready!\n")

    if args.prompt:
        run_stream_session(
            engine,
            args.prompt,
            mode=args.mode,
            gamma=effective_gamma,
            max_new_tokens=args.max_new_tokens,
            colorize=args.color,
        )
        if not args.interactive:
            return

    # Interactive Menu Loop
    while True:
        print("\n" + "─" * 60)
        print("Choose an option:")
        for idx, item in enumerate(PRESET_PROMPTS, 1):
            print(f"  [{idx}] {item['title']}")
        print("  [c] Enter Custom Prompt")
        print("  [q] Quit")
        print("─" * 60)

        try:
            choice = input("Enter choice (1-3, c, q) [default=1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not choice:
            choice = "1"

        if choice.lower() == "q":
            print("Goodbye!")
            break
        elif choice in ("1", "2", "3"):
            prompt_text = PRESET_PROMPTS[int(choice) - 1]["prompt"]
        elif choice.lower() == "c":
            try:
                prompt_text = input("\nEnter your custom prompt: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt_text:
                print("Empty prompt, skipping.")
                continue
        else:
            print("Invalid selection.")
            continue

        run_stream_session(
            engine,
            prompt_text,
            mode=args.mode,
            gamma=effective_gamma,
            max_new_tokens=args.max_new_tokens,
            colorize=args.color,
        )


if __name__ == "__main__":
    main()
