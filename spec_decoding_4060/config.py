import torch

class SpecConfig:
    # Profile Identity & Hardware Verification
    GPU_PROFILE: str = "rtx4060"
    EXPECTED_GPU_SUBSTR: str = "RTX 4060"
    VRAM_TOTAL_GB: float = 8.0  # Informational; detected at runtime

    # Model Selection (Fits in 8GB VRAM in pure bfloat16)
    TARGET_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
    DRAFT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Hardware Settings
    DEVICE: str = "cuda:0"
    DTYPE: torch.dtype = torch.bfloat16  # Native Tensor Core support on RTX 4060 (Ada)
    ATTN_IMPLEMENTATION: str = "sdpa"    # PyTorch Native Scaled Dot Product Attention
    
    # Speculative Decoding Hyperparameters
    GAMMA: int = 5                       # Lookahead K (5 is optimal for 0.5B -> 3B)
    MAX_NEW_TOKENS: int = 512

    # Benchmark Execution Settings
    WARMUP_ITERATIONS: int = 1
    WARMUP_TOKENS: int = 64
    WARMUP_PROMPT: str = "Say hello in one word."
    MEASURED_REPETITIONS: int = 5
    BENCHMARK_SEED: int = 42
    CUDNN_BENCHMARK: bool = False
    ALLOW_TF32: bool = True
    TORCH_DETERMINISTIC: bool = False
    OUTPUT_DIR: str = "results/rtx4060"
    
    # Benchmark Prompts (High acceptance rate & standard reasoning)
    PROMPTS = [
        {
            "name": "Code Generation (High Acceptance)",
            "text": "Write a complete, well-documented Python implementation of a Binary Search Tree with insert, search, and in-order traversal methods."
        },
        {
            "name": "Structured JSON/Data (High Acceptance)",
            "text": "Generate a JSON array containing 5 fictional software engineers with fields: id, name, role, skills (array), and years_of_experience."
        },
        {
            "name": "Technical Explanation (Medium Acceptance)",
            "text": "Explain how CPU L1, L2, and L3 caches work, including cache lines, associativity, and the MESI coherence protocol."
        }
    ]


# Alias for explicit profile-based imports
SpecConfig4060 = SpecConfig
