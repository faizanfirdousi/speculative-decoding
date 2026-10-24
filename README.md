# Speculative Decoding Benchmark

A PyTorch implementation and benchmarking harness for speculative decoding with LLMs. Compares custom greedy speculative decoding, standard autoregressive baseline generation, and Hugging Face native assisted generation on NVIDIA Ada Lovelace GPUs (RTX 4060 8GB and RTX 4000 Ada 20GB).

## Overview

Speculative decoding accelerates inference on memory-bandwidth-bound target models by using a small draft model to propose blocks of candidate tokens. The larger target model verifies the candidate block in parallel in a single forward pass, accepting matching tokens and correcting divergences.

This repository implements:
- Target model: `Qwen/Qwen2.5-3B-Instruct` (BF16, SDPA attention)
- Draft model: `Qwen/Qwen2.5-0.5B-Instruct` (BF16, SDPA attention)
- In-place KV cache rollback via Hugging Face `DynamicCache` slicing, preventing prompt recomputation on rejection
- Hardware profiles for consumer laptops (RTX 4060 8GB) and workstation setups (RTX 4000 Ada 20GB)
- Strict metric accounting, divergence inspection, and unit tests

## How It Works

1. **Prefill**: Both the target model and draft model ingest the prompt. The target model outputs the first authoritative completion token.
2. **Drafting ($K$ tokens)**: The draft model autoregressively generates $K$ lookahead tokens from its current cache state.
3. **Parallel Verification**: The target model receives `[last_confirmed_token, draft_0, ..., draft_{K-1}]` and evaluates all positions simultaneously in one forward pass.
4. **Accept / Reject**:
   - Compares draft tokens to target greedy predictions sequentially.
   - If draft tokens match up to index $n$, those $n$ tokens are accepted.
   - If a mismatch occurs at index $n < K$, the target model's prediction at position $n$ is appended as the correction token.
   - If all $K$ tokens match ($n = K$), the target model's prediction at position $K$ is appended as a bonus token.
   - If EOS is encountered in the accepted sequence, generation terminates and bonus token emission is suppressed.
5. **KV Cache Rollback**: The target and draft KV caches are cropped in-place to the valid prefix length, discarding keys and values for rejected tokens without re-running past tokens.

## Hardware Profiles

| Parameter | RTX 4060 Profile (`rtx4060`) | RTX 4000 Ada Profile (`rtx4000`) |
| :--- | :--- | :--- |
| Target GPU | NVIDIA GeForce RTX 4060 Laptop | NVIDIA RTX 4000 Ada Generation |
| VRAM | 8 GB GDDR6 (128-bit bus, ~272 GB/s) | 20 GB GDDR6 ECC (160-bit bus, ~360 GB/s) |
| Lookahead ($K$) | 5 | 8 |
| Default Repetitions | 1 | 3 (reports median tok/s) |
| TF32 Acceleration | Enabled | Enabled |
| cuDNN Benchmark | Disabled | Enabled |
| Target Model | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| Draft Model | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` |
| Precision | `torch.bfloat16` | `torch.bfloat16` |

On the RTX 4000 Ada, higher memory bandwidth and 20 GB of VRAM allow a larger speculative lookahead ($\gamma = 8$) and multi-run median measurements for benchmark stability.

## Repository Layout

```text
spec-decoding-STC/
├── benchmark.py -> spec_decoding_4060/benchmark.py  # Unified benchmark entry point
├── config.py -> spec_decoding_4060/config.py        # Default RTX 4060 profile config
├── engine.py -> spec_decoding_4060/engine.py        # Speculative engine implementation
├── kv_cache_utils.py -> ...                         # DynamicCache cropping helpers
├── metrics.py -> ...                                # Benchmark metrics and invariants
├── spec_logic.py -> ...                             # Pure step decision planning logic
├── requirements.txt                                 # Dependencies
├── spec_decoding_4060/                              # RTX 4060 profile package
│   ├── benchmark.py
│   ├── config.py
│   ├── engine.py
│   ├── kv_cache_utils.py
│   ├── metrics.py
│   └── spec_logic.py
├── spec_decoding_4000/                              # RTX 4000 profile package
│   ├── config.py
│   └── __init__.py
├── tests/                                           # Unit and regression test suite
│   ├── test_benchmark_config.py
│   ├── test_greedy_equivalence.py
│   ├── test_metrics.py
│   ├── test_serialization.py
│   └── test_spec_logic.py
└── results/                                         # Output logs, JSON, and CSVs (git-ignored)
```

## Setup

Requires Python 3.10+, PyTorch with CUDA support, and an NVIDIA Ada Lovelace GPU.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

## Running Benchmarks

### Basic Execution

By default, the benchmark detects your GPU and runs the matching profile:

```bash
# Auto-detect profile from detected GPU
python benchmark.py

# Explicit profile selection
python benchmark.py --gpu rtx4060
python benchmark.py --gpu rtx4000
```

### Command Line Options

```bash
# Run a dry-run check without loading models
python benchmark.py --dry-run

# Run benchmarks with live token streaming to terminal
python benchmark.py --stream

# Run RTX 4000 configuration on a different GPU for testing
python benchmark.py --gpu rtx4000 --allow-mismatch

# Override speculative lookahead K
python benchmark.py --gamma 6

# Override token generation budget
python benchmark.py --max-new-tokens 128

# Set measured timing repetitions per prompt (median reported)
python benchmark.py --repetitions 3

# Show detected hardware properties
python benchmark.py --show-gpu

# List available profiles
python benchmark.py --list-profiles
```

## Live Interactive Streaming & Speed Monitor

Watch tokens generate live directly on your terminal in real-time with continuous speed telemetry (tokens/sec) and draft acceptance stats:

```bash
# 1. Launch interactive session (preset prompts or custom text input)
python stream_demo.py

# 2. Side-by-side comparison (Baseline AR at ~35 tok/s vs Speculative at ~78 tok/s)
python stream_demo.py --prompt "Explain quantum computing in simple terms" --mode compare

# 3. Speculative mode with visual color coding (Cyan = Draft accepted, Green = Target authoritative)
python stream_demo.py --prompt "Write a Binary Search Tree in Python" --mode spec --color

# 4. Custom lookahead K and token limits
python stream_demo.py --prompt "Explain CPU caches" --gamma 5 --max-new-tokens 200
```

## Metric Definitions

- **`acceptance_rate`**:
  $$\text{acceptance\_rate} = 100 \times \frac{\text{total\_accepted}}{\text{total\_drafted}}$$
  Calculated over draft proposals only. Draft proposals can exceed total output tokens because rejected proposals in earlier cycles are discarded.
- **`output_tokens_per_target_forward`**:
  $$\text{output\_tokens\_per\_target\_forward} = \frac{\text{generated\_tokens}}{\text{target\_forward\_passes}}$$
  Average generated output tokens per target model forward invocation, including the initial prompt prefill.
- **`accepted_tokens_per_verification_cycle`**:
  $$\text{accepted\_tokens\_per\_verification\_cycle} = \frac{\text{total\_accepted}}{\text{verification\_cycles}}$$
  Average draft tokens accepted per verification forward pass (excluding prefill).
- **`target_forward_passes`**:
  Total forward passes executed by the target model ($1 \text{ prefill} + \text{verification\_cycles}$).
- **`exact_match`**:
  Boolean flag. True if and only if the baseline greedy output token ID sequence matches the custom speculative sequence identically across all positions.
  When a mismatch occurs, the benchmark reports the first divergent output token index, conflicting token IDs, and decoded context snippets.
- **`throughput_tps`**:
  Tokens generated divided by wall-clock time in seconds, enclosed by `torch.cuda.synchronize()` barriers before and after generation.

### Mathematical Invariants

For any run completing without encountering EOS:
$$\text{output\_tokens} = 1 \; (\text{prefill}) + \text{total\_accepted} + \text{emitted\_authoritative}$$

Where $\text{emitted\_authoritative}$ is either $\text{verification\_cycles}$ (when a correction or bonus token was appended on the final cycle) or $\text{verification\_cycles} - 1$ (when accepted draft tokens filled the exact budget).

## Running Tests

The test suite runs on CPU using deterministic mock models and pure logic fixtures without downloading model weights:

```bash
PYTHONPATH=. pytest tests/ -v
```

Tests cover:
- Metric calculations, edge cases (zero division), and counter invariant validation
- Divergence index and context extraction
- Speculative decision planning (bonus tokens, correction tokens, EOS handling, and budget clamping)
- Deterministic mock model proving 100% exact greedy sequence equivalence between baseline and speculative decoding
- JSON and CSV result serialization
- Profile configuration loading, hardware validation, and CLI overrides
