"""
Multi-GPU Speculative Decoding Benchmark Suite.

Supports switching between hardware profiles (e.g. RTX 4060 8GB and RTX 4000 Ada ~20GB),
runtime GPU validation, fair benchmark comparison, telemetry collection,
and result serialization to JSON and CSV.
"""
import argparse
import csv
import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path so modules in spec_decoding_* can be imported
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent if _CURRENT_DIR.name.startswith("spec_decoding_") else _CURRENT_DIR
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Profile registry: profile_name -> (module_path, class_name, description)
SUPPORTED_PROFILES: Dict[str, Dict[str, str]] = {
    "rtx4060": {
        "module": "spec_decoding_4060.config",
        "class_name": "SpecConfig",
        "description": "NVIDIA GeForce RTX 4060 (8GB VRAM profile)",
    },
    "rtx4000": {
        "module": "spec_decoding_4000.config",
        "class_name": "SpecConfig4000",
        "description": "NVIDIA RTX 4000 Ada Generation (~20GB VRAM profile)",
    },
}


def get_git_commit_hash() -> str:
    """Retrieve current Git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def detect_gpu_hardware(device_str: str = "cuda:0") -> Dict[str, Any]:
    """
    Validate PyTorch and CUDA availability, querying hardware properties.
    Fails explicitly if CUDA is not available without falling back silently.
    """
    try:
        import torch
    except ImportError as exc:
        print(f"[FATAL] PyTorch is not installed or importable: {exc}", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "\n[FATAL] CUDA is not available to PyTorch!\n"
            "  Speculative decoding benchmarks require an active NVIDIA CUDA GPU.\n"
            "  - On native Linux: check 'nvidia-smi' and PyTorch CUDA build.\n"
            "  - On WSL2: verify NVIDIA Windows drivers >= 535 and WSL2 GPU passthrough.\n"
            "  Benchmark cannot fall back to CPU.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
    if device_idx >= torch.cuda.device_count():
        print(
            f"[FATAL] Configured device '{device_str}' does not exist. "
            f"Available CUDA devices: {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(device_idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    cap = torch.cuda.get_device_capability(device_idx)
    driver_version = "unknown"

    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        driver_version = smi_out.strip().split("\n")[0]
    except Exception:
        pass

    return {
        "device_idx": device_idx,
        "device_str": f"cuda:{device_idx}",
        "gpu_name": gpu_name,
        "total_vram_bytes": total_bytes,
        "total_vram_gb": total_bytes / (1024 ** 3),
        "free_vram_bytes": free_bytes,
        "free_vram_gb": free_bytes / (1024 ** 3),
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda or "none",
        "driver_version": driver_version,
    }


def load_profile_config(profile_name: str, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Load configuration class for the given profile and apply optional overrides."""
    if profile_name not in SUPPORTED_PROFILES:
        valid_keys = ", ".join(SUPPORTED_PROFILES.keys())
        raise ValueError(f"Unknown GPU profile '{profile_name}'. Supported profiles: {valid_keys}")

    meta = SUPPORTED_PROFILES[profile_name]
    mod = importlib.import_module(meta["module"])
    cls = getattr(mod, meta["class_name"])
    cfg = cls()

    if overrides:
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


def validate_profile_gpu(
    cfg: Any,
    profile_name: str,
    hardware: Dict[str, Any],
    allow_mismatch: bool = False,
) -> bool:
    """
    Compare requested profile against detected GPU.
    Displays clear error on mismatch unless allow_mismatch is explicitly True.
    """
    expected_substr = getattr(cfg, "EXPECTED_GPU_SUBSTR", "").lower()
    detected_gpu = hardware["gpu_name"].lower()
    is_match = expected_substr in detected_gpu

    if not is_match:
        if not allow_mismatch:
            print(
                f"\n======================================================================\n"
                f"[ERROR] GPU Profile Mismatch Detected!\n"
                f"======================================================================\n"
                f"  Requested Profile : '{profile_name}'\n"
                f"  Expected GPU Substring: '{getattr(cfg, 'EXPECTED_GPU_SUBSTR', '')}'\n"
                f"  Detected Hardware : '{hardware['gpu_name']}'\n"
                f"  Total VRAM        : {hardware['total_vram_gb']:.2f} GB "
                f"(Free: {hardware['free_vram_gb']:.2f} GB)\n"
                f"----------------------------------------------------------------------\n"
                f"To run this profile anyway on the detected GPU (e.g. for testing),\n"
                f"pass '--allow-mismatch':\n"
                f"  python benchmark.py --gpu {profile_name} --allow-mismatch\n"
                f"======================================================================\n",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                f"\n[WARNING] GPU Profile Mismatch Overridden (--allow-mismatch)!\n"
                f"  Running profile '{profile_name}' on detected GPU '{hardware['gpu_name']}'.\n"
                f"  Note: Benchmark results will reflect detected '{hardware['gpu_name']}' hardware.\n"
            )
    return is_match


def auto_detect_profile(hardware: Dict[str, Any]) -> str:
    """Detect matching profile from GPU name, or default to rtx4060."""
    gpu_name = hardware["gpu_name"].lower()
    if "rtx 4000" in gpu_name:
        return "rtx4000"
    if "rtx 4060" in gpu_name:
        return "rtx4060"
    return "rtx4060"


def save_results(
    results_payload: Dict[str, Any],
    table_rows: List[List[Any]],
    output_dir_path: str,
    profile_name: str,
) -> Tuple[Path, Path]:
    """Serialize benchmark results to both JSON and CSV files."""
    out_dir = Path(output_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{profile_name}_{timestamp_str}.json"
    latest_json = out_dir / "latest_results.json"
    csv_path = out_dir / f"benchmark_{profile_name}_{timestamp_str}.csv"
    summary_csv = out_dir / "summary.csv"

    # Save timestamped JSON and latest JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Save CSV
    csv_headers = [
        "task_name",
        "baseline_tps",
        "custom_spec_tps",
        "hf_assisted_tps",
        "speedup",
        "acceptance_rate_pct",
        "tokens_per_target_step",
        "exact_match",
        "prompt_tokens",
        "generated_tokens",
        "peak_vram_gb",
        "gpu_model",
        "profile",
        "git_commit",
        "timestamp",
    ]

    csv_data = []
    for item in results_payload["benchmarks"]:
        csv_data.append([
            item["task_name"],
            f"{item['baseline_ar']['tps']:.2f}",
            f"{item['custom_speculative']['tps']:.2f}",
            f"{item['hf_assisted']['tps']:.2f}",
            f"{item['custom_speculative']['speedup']:.2f}",
            f"{item['custom_speculative']['acceptance_rate']:.1f}",
            f"{item['custom_speculative']['tokens_per_target_step']:.2f}",
            "PASS" if item["exact_match"] else "MISMATCH",
            item["prompt_tokens"],
            item["generated_tokens"],
            f"{results_payload['telemetry']['peak_vram_allocated_gb']:.2f}",
            results_payload["hardware"]["gpu_name"],
            profile_name,
            results_payload["metadata"]["git_commit"],
            results_payload["metadata"]["timestamp"],
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)

    # Append to summary CSV if exists, else create
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_headers)
        writer.writerows(csv_data)

    return json_path, csv_path


def run_benchmarks(
    gpu: str = "rtx4060",
    allow_mismatch: bool = False,
    gamma: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_output: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Main benchmark execution entry point.
    Preserves exact behavior of original RTX 4060 benchmark while adding
    profile selection, GPU validation, telemetry, and serialization.
    """
    import torch
    from tabulate import tabulate

    # 1. Hardware Detection
    hardware = detect_gpu_hardware()

    # 2. Profile Selection
    profile_name = gpu
    if profile_name == "auto":
        profile_name = auto_detect_profile(hardware)
        print(f"[Auto-Detect] Hardware: '{hardware['gpu_name']}' -> Selected Profile: '{profile_name}'")

    overrides = {}
    if gamma is not None:
        overrides["GAMMA"] = gamma
    if max_new_tokens is not None:
        overrides["MAX_NEW_TOKENS"] = max_new_tokens
    if output_dir is not None:
        overrides["OUTPUT_DIR"] = output_dir

    cfg = load_profile_config(profile_name, overrides)

    # 3. Validation
    is_match = validate_profile_gpu(cfg, profile_name, hardware, allow_mismatch=allow_mismatch)

    out_dir = getattr(cfg, "OUTPUT_DIR", f"results/{profile_name}")
    git_commit = get_git_commit_hash()
    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("\n" + "=" * 70)
    print(f" SPECULATIVE DECODING BENCHMARK — PROFILE: {profile_name.upper()}")
    print("=" * 70)
    print(f"  Target Model        : {cfg.TARGET_MODEL_ID}")
    print(f"  Draft Model         : {cfg.DRAFT_MODEL_ID}")
    print(f"  Precision & Attn    : {cfg.DTYPE} | {cfg.ATTN_IMPLEMENTATION}")
    print(f"  Speculative Gamma   : {cfg.GAMMA} lookahead tokens")
    print(f"  Max New Tokens      : {cfg.MAX_NEW_TOKENS}")
    print(f"  Active Device       : {cfg.DEVICE}")
    print(f"  Detected Hardware   : {hardware['gpu_name']} ({hardware['total_vram_gb']:.2f} GB VRAM)")
    print(f"  Available Memory    : {hardware['free_vram_gb']:.2f} GB free at idle")
    print(f"  PyTorch / CUDA      : {hardware['pytorch_version']} | CUDA {hardware['pytorch_cuda_version']}")
    print(f"  Git Commit          : {git_commit}")
    print(f"  Output Directory    : {out_dir}")
    print("=" * 70 + "\n")

    if dry_run:
        print("[Dry-Run] Configuration validated successfully. Skipping model loading and inference.")
        return {
            "status": "dry_run_success",
            "profile": profile_name,
            "hardware": hardware,
            "git_commit": git_commit,
        }

    # 4. Initialize Engine
    # Import SpeculativeEngine (shared implementation in spec_decoding_4060)
    from spec_decoding_4060.engine import SpeculativeEngine

    engine = SpeculativeEngine(cfg)

    # 5. CUDA Warm-up Pass
    warmup_tokens = getattr(cfg, "WARMUP_TOKENS", 16)
    warmup_prompt = getattr(cfg, "WARMUP_PROMPT", "Say hello in one word.")
    print(f"[Warm-up] Running CUDA kernel warm-up pass ({warmup_prompt!r}, tokens={warmup_tokens})...")
    warmup_ids = engine.format_prompt(warmup_prompt)
    _ = engine.custom_speculative_generate(warmup_ids, max_new_tokens=warmup_tokens, gamma=cfg.GAMMA)
    print("[Warm-up] Complete.\n")

    summary_rows = []
    benchmarks_data = []

    # 6. Benchmark Iterations
    for idx, prompt_item in enumerate(cfg.PROMPTS, 1):
        name = prompt_item["name"]
        input_ids = engine.format_prompt(prompt_item["text"])
        prompt_len = input_ids.shape[1]

        print("=" * 70)
        print(f"Benchmark {idx}/{len(cfg.PROMPTS)}: {name} (Prompt tokens: {prompt_len})")
        print("=" * 70)

        # Run 1: Standard Autoregressive Baseline
        ar_res = engine.autoregressive_generate(input_ids, cfg.MAX_NEW_TOKENS)
        print(f"  -> Baseline Autoregressive : {ar_res['tps']:.2f} tok/s ({ar_res['time_sec']:.2f}s)")

        # Run 2: Custom Speculative Decoding
        spec_res = engine.custom_speculative_generate(input_ids, cfg.MAX_NEW_TOKENS, cfg.GAMMA)
        print(f"  -> Custom Speculative (K={cfg.GAMMA}): {spec_res['tps']:.2f} tok/s ({spec_res['time_sec']:.2f}s)")

        # Run 3: Hugging Face Native Assisted Generation
        hf_res = engine.hf_assisted_generate(input_ids, cfg.MAX_NEW_TOKENS)
        print(f"  -> HF Native Assisted      : {hf_res['tps']:.2f} tok/s ({hf_res['time_sec']:.2f}s)")

        # Exact Output Equivalence Verification
        exact_match = torch.equal(ar_res["output_ids"], spec_res["output_ids"])
        speedup = spec_res["tps"] / ar_res["tps"] if ar_res["tps"] > 0 else 0.0
        hf_speedup = hf_res["tps"] / ar_res["tps"] if ar_res["tps"] > 0 else 0.0

        summary_rows.append([
            name,
            f"{ar_res['tps']:.2f} t/s",
            f"{spec_res['tps']:.2f} t/s",
            f"{hf_res['tps']:.2f} t/s",
            f"{speedup:.2f}x",
            f"{spec_res['acceptance_rate']:.1f}%",
            f"{spec_res['tokens_per_target_step']:.2f}",
            "PASS" if exact_match else "MISMATCH",
        ])

        benchmarks_data.append({
            "task_name": name,
            "prompt_text": prompt_item["text"],
            "prompt_tokens": prompt_len,
            "generated_tokens": spec_res["tokens_generated"],
            "baseline_ar": {
                "tps": ar_res["tps"],
                "time_sec": ar_res["time_sec"],
                "tokens_generated": ar_res["tokens_generated"],
            },
            "custom_speculative": {
                "tps": spec_res["tps"],
                "time_sec": spec_res["time_sec"],
                "speedup": speedup,
                "acceptance_rate": spec_res["acceptance_rate"],
                "tokens_per_target_step": spec_res["tokens_per_target_step"],
                "total_drafted": spec_res.get("total_drafted", 0),
                "total_accepted": spec_res.get("total_accepted", 0),
                "target_forward_passes": spec_res.get("target_forward_passes", 0),
            },
            "hf_assisted": {
                "tps": hf_res["tps"],
                "time_sec": hf_res["time_sec"],
                "speedup": hf_speedup,
                "tokens_generated": hf_res["tokens_generated"],
            },
            "exact_match": exact_match,
        })
        print()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    total_ceiling_gb = getattr(cfg, "VRAM_TOTAL_GB", hardware["total_vram_gb"])

    # 7. Print Formatted Table
    print(f"\n================ FINAL {profile_name.upper()} BENCHMARK RESULTS ================")
    headers = [
        "Task Type",
        "Baseline (3B)",
        "Custom Spec (0.5B->3B)",
        "HF Assisted",
        "Speedup",
        "Accept Rate",
        "Tok/Target Step",
        "Exact Match",
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))
    print(f"\n[Telemetry] Peak GPU VRAM Allocated: {peak_vram_gb:.2f} GB / {total_ceiling_gb:.2f} GB")

    # 8. Compile Results Payload
    results_payload = {
        "metadata": {
            "timestamp": timestamp_iso,
            "git_commit": git_commit,
            "profile": profile_name,
            "is_controlled_hardware_comparison": is_match,
        },
        "hardware": hardware,
        "configuration": {
            "target_model_id": cfg.TARGET_MODEL_ID,
            "draft_model_id": cfg.DRAFT_MODEL_ID,
            "device": cfg.DEVICE,
            "dtype": str(cfg.DTYPE),
            "attn_implementation": cfg.ATTN_IMPLEMENTATION,
            "gamma": cfg.GAMMA,
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
            "warmup_iterations": getattr(cfg, "WARMUP_ITERATIONS", 1),
            "warmup_tokens": warmup_tokens,
        },
        "benchmarks": benchmarks_data,
        "telemetry": {
            "peak_vram_allocated_gb": peak_vram_gb,
            "total_vram_gb": total_ceiling_gb,
        },
    }

    # 9. Save Results
    if save_output:
        json_file, csv_file = save_results(results_payload, summary_rows, out_dir, profile_name)
        print(f"[Results] JSON report saved: {json_file}")
        print(f"[Results] CSV report saved : {csv_file}")

    return results_payload


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Multi-GPU Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpu",
        "--profile",
        dest="gpu",
        default="rtx4060",
        choices=["rtx4060", "rtx4000", "auto"],
        help="GPU benchmark profile to execute ('rtx4060', 'rtx4000', or 'auto' detection).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all supported GPU profiles and exit.",
    )
    parser.add_argument(
        "--show-gpu",
        action="store_true",
        help="Detect and display active GPU hardware telemetry, then exit.",
    )
    parser.add_argument(
        "--allow-mismatch",
        "--force",
        action="store_true",
        help="Allow running a profile even if the detected GPU does not match the expected profile hardware.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=None,
        help="Override lookahead draft token count K (default: from profile config).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override maximum generated tokens per prompt (default: from profile config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override directory where benchmark results are saved.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving benchmark JSON and CSV results to disk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment and profile settings without loading model weights.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_profiles:
        from tabulate import tabulate

        profile_rows = []
        for name, meta in SUPPORTED_PROFILES.items():
            mod = importlib.import_module(meta["module"])
            cls = getattr(mod, meta["class_name"])
            cfg = cls()
            profile_rows.append([
                name,
                meta["description"],
                getattr(cfg, "TARGET_MODEL_ID", ""),
                getattr(cfg, "DRAFT_MODEL_ID", ""),
                str(getattr(cfg, "GAMMA", "")),
                f"{getattr(cfg, 'VRAM_TOTAL_GB', 0):.1f} GB",
            ])
        print("\nAvailable GPU Profiles:")
        print(tabulate(
            profile_rows,
            headers=["Profile", "Description", "Target Model", "Draft Model", "Gamma", "Nominal VRAM"],
            tablefmt="fancy_grid",
        ))
        print("\nUsage example:")
        print("  python benchmark.py --gpu rtx4060")
        print("  python benchmark.py --gpu rtx4000\n")
        return

    if args.show_gpu:
        hardware = detect_gpu_hardware()
        print("\nDetected GPU Hardware:")
        print(f"  GPU Name           : {hardware['gpu_name']}")
        print(f"  Compute Capability : {hardware['compute_capability']}")
        print(f"  Total VRAM         : {hardware['total_vram_gb']:.2f} GB ({hardware['total_vram_bytes']:,} bytes)")
        print(f"  Free VRAM (Idle)   : {hardware['free_vram_gb']:.2f} GB ({hardware['free_vram_bytes']:,} bytes)")
        print(f"  PyTorch Version    : {hardware['pytorch_version']}")
        print(f"  CUDA Runtime       : {hardware['pytorch_cuda_version']}")
        print(f"  NVIDIA Driver      : {hardware['driver_version']}")
        matched = auto_detect_profile(hardware)
        print(f"  Recommended Profile: --gpu {matched}\n")
        return

    run_benchmarks(
        gpu=args.gpu,
        allow_mismatch=args.allow_mismatch,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        save_output=not args.no_save,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
