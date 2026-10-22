"""
Hardware configuration for NVIDIA RTX 4000 Ada Generation (~20 GB VRAM).

Design intent
-------------
All algorithm parameters (models, dtype, gamma, prompts, token budget) are
intentionally identical to SpecConfig4060 so that the two GPUs can be
compared fairly on the **same** workload.

Only hardware-specific fields differ:
  - GPU_PROFILE          : profile name used by the CLI and result files
  - EXPECTED_GPU_SUBSTR  : substring matched against torch.cuda.get_device_name()
  - DEVICE               : CUDA device string (change if the Ada is not cuda:0)
  - VRAM_TOTAL_GB        : informational only; detected at runtime
  - OUTPUT_DIR           : output directory for benchmark results

To run with larger context, bigger batch size, or higher gamma on the Ada,
pass explicit overrides via the CLI (--max-new-tokens, --gamma …) or subclass
this config. Do NOT change these defaults; doing so would break the
apples-to-apples comparison.
"""
import torch


class SpecConfig4000:
    # ---- Profile identity -----------------------------------------------
    GPU_PROFILE: str = "rtx4000"
    EXPECTED_GPU_SUBSTR: str = "RTX 4000"   # matched case-insensitively
    VRAM_TOTAL_GB: float = 20.0             # informational; actual value queried at runtime

    # ---- Model selection (identical to RTX 4060 profile for fair comparison) --
    TARGET_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
    DRAFT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- Hardware settings -----------------------------------------------
    DEVICE: str = "cuda:0"
    DTYPE: torch.dtype = torch.bfloat16          # Ada Lovelace Tensor Core native
    ATTN_IMPLEMENTATION: str = "sdpa"            # PyTorch SDPA

    # ---- Speculative decoding hyperparameters (identical to 4060) -------
    GAMMA: int = 5                               # draft lookahead K
    MAX_NEW_TOKENS: int = 256

    # ---- Benchmark Execution Settings ------------------------------------
    WARMUP_ITERATIONS: int = 1
    WARMUP_TOKENS: int = 16
    WARMUP_PROMPT: str = "Say hello in one word."
    OUTPUT_DIR: str = "results/rtx4000"

    # ---- Benchmark prompts (identical to 4060 for fair comparison) ------
    PROMPTS = [
        {
            "name": "Code Generation (High Acceptance)",
            "text": (
                "Write a complete, well-documented Python implementation of a "
                "Binary Search Tree with insert, search, and in-order traversal methods."
            ),
        },
        {
            "name": "Structured JSON/Data (High Acceptance)",
            "text": (
                "Generate a JSON array containing 5 fictional software engineers "
                "with fields: id, name, role, skills (array), and years_of_experience."
            ),
        },
        {
            "name": "Technical Explanation (Medium Acceptance)",
            "text": (
                "Explain how CPU L1, L2, and L3 caches work, including cache lines, "
                "associativity, and the MESI coherence protocol."
            ),
        },
    ]


# Alias for backward-compatible imports
SpecConfig = SpecConfig4000
