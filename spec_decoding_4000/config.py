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
    DTYPE: torch.dtype = torch.bfloat16          # Draft model dtype (Ada Tensor Core native)
    # Target-model compute dtype. The 20 GB Ada can hold the 3B target in fp32, which
    # removes the bfloat16 near-tie argmax flips (2 ULP at ~35.25) that made speculative
    # output diverge from greedy AR. Greedy spec MUST reproduce AR token-for-token, so the
    # target (both the AR baseline and the verify pass) runs in fp32 here. The 0.5B draft
    # stays bfloat16. This is the ONE deliberate deviation from the "identical parameters"
    # rule above, and it is required for exactness; set TARGET_DTYPE = DTYPE to disable.
    TARGET_DTYPE: torch.dtype = torch.float32
    ATTN_IMPLEMENTATION: str = "sdpa"            # PyTorch SDPA

    # ---- Speculative decoding (tuned for ~20 GB VRAM / higher bandwidth) -
    # K=5, matching the 4060 profile. K=8 was measured on the Ada to *hurt*
    # low-acceptance prompts (MESI 0.82x, custom-prompt 0.83x vs AR baseline):
    # extra drafting latency is not repaid when acceptance is low.
    GAMMA: int = 5
    MAX_NEW_TOKENS: int = 256

    # ---- Benchmark Execution Settings ------------------------------------
    WARMUP_ITERATIONS: int = 1
    WARMUP_TOKENS: int = 16
    WARMUP_PROMPT: str = "Say hello in one word."
    MEASURED_REPETITIONS: int = 3                # median over repetitions for stable tok/s
    BENCHMARK_SEED: int = 42
    CUDNN_BENCHMARK: bool = True                 # fixed-shape decode; improves SDPA matmul selection
    ALLOW_TF32: bool = True                      # Draft/bf16 path only: TF32 is force-disabled for
                                                 # the fp32 target (its ~5e-4 truncation breaks exactness)
    TORCH_DETERMINISTIC: bool = False
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
