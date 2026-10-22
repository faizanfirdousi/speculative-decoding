import torch
from tabulate import tabulate
from config import SpecConfig
from engine import SpeculativeEngine

def run_benchmarks():
    cfg = SpecConfig()
    engine = SpeculativeEngine(cfg)

    # 1. CUDA Warm-up Pass (Crucial for accurate RTX 4060 clock speeds & kernel caching)
    print("[Warm-up] Running CUDA kernel warm-up pass...")
    warmup_ids = engine.format_prompt("Say hello in one word.")
    _ = engine.custom_speculative_generate(warmup_ids, max_new_tokens=16, gamma=cfg.GAMMA)
    print("[Warm-up] Complete.\n")

    summary_rows = []

    for idx, prompt_item in enumerate(cfg.PROMPTS, 1):
        name = prompt_item["name"]
        input_ids = engine.format_prompt(prompt_item["text"])
        print(f"======================================================================")
        print(f"Benchmark {idx}/{len(cfg.PROMPTS)}: {name}")
        print(f"======================================================================")

        # Run 1: Standard Autoregressive Baseline
        ar_res = engine.autoregressive_generate(input_ids, cfg.MAX_NEW_TOKENS)
        print(f"  -> Baseline Autoregressive : {ar_res['tps']:.2f} tok/s ({ar_res['time_sec']:.2f}s)")

        # Run 2: Custom Speculative Decoding
        spec_res = engine.custom_speculative_generate(input_ids, cfg.MAX_NEW_TOKENS, cfg.GAMMA)
        print(f"  -> Custom Speculative (K={cfg.GAMMA}): {spec_res['tps']:.2f} tok/s ({spec_res['time_sec']:.2f}s)")

        # Run 3: Hugging Face Native Assisted Generation
        hf_res = engine.hf_assisted_generate(input_ids, cfg.MAX_NEW_TOKENS)
        print(f"  -> HF Native Assisted      : {hf_res['tps']:.2f} tok/s ({hf_res['time_sec']:.2f}s)")

        # Verify Exact Output Equivalence
        exact_match = torch.equal(ar_res["output_ids"], spec_res["output_ids"])
        speedup = spec_res["tps"] / ar_res["tps"]

        summary_rows.append([
            name,
            f"{ar_res['tps']:.2f} t/s",
            f"{spec_res['tps']:.2f} t/s",
            f"{hf_res['tps']:.2f} t/s",
            f"{speedup:.2f}x",
            f"{spec_res['acceptance_rate']:.1f}%",
            f"{spec_res['tokens_per_target_step']:.2f}",
            "PASS" if exact_match else "MISMATCH"
        ])
        print()

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("\n================ FINAL RTX 4060 BENCHMARK RESULTS ================")
    headers = [
        "Task Type",
        "Baseline (3B)",
        "Custom Spec (0.5B->3B)",
        "HF Assisted",
        "Speedup",
        "Accept Rate",
        "Tok/Target Step",
        "Exact Match"
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))
    print(f"\n[Telemetry] Peak GPU VRAM Allocated: {peak_vram:.2f} GB / 8.00 GB")

if __name__ == "__main__":
    run_benchmarks()
