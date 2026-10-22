import torch

class SpecConfig:
    # Model Selection (Fits in 8GB VRAM in pure bfloat16)
    TARGET_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
    DRAFT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Hardware Settings
    DEVICE: str = "cuda:0"
    DTYPE: torch.dtype = torch.bfloat16  # Native Tensor Core support on RTX 4060 (Ada)
    ATTN_IMPLEMENTATION: str = "sdpa"    # PyTorch Native Scaled Dot Product Attention
    
    # Speculative Decoding Hyperparameters
    GAMMA: int = 5                       # Lookahead K (5 is optimal for 0.5B -> 3B)
    MAX_NEW_TOKENS: int = 256
    
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
