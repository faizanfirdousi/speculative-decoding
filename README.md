# Multi-GPU Speculative Decoding Engine & Benchmark

A high-performance PyTorch & Hugging Face pipeline implementing Greedy Speculative Decoding with dynamic KV-cache rollback. Built to support multiple NVIDIA Ada Lovelace GPUs—from consumer laptop cards (**RTX 4060 8GB**) to workstation cards (**RTX 4000 Ada ~20GB**) in native Linux and Windows WSL2 environments—while providing a fair, reproducible hardware comparison framework.

Based on the original specification: [`speculative_decoding_prd_sop_rtx_4060_8gb.md`](./speculative_decoding_prd_sop_rtx_4060_8gb.md)

---

## 🎯 Architecture & Comparison Strategy

| Role | Model | Precision | VRAM Footprint | Attention |
| :--- | :--- | :--- | :--- | :--- |
| **Target Model ($M_p$)** | `Qwen/Qwen2.5-3B-Instruct` | `torch.bfloat16` | ~6.18 GB | Native SDPA |
| **Draft Model ($M_q$)** | `Qwen/Qwen2.5-0.5B-Instruct` | `torch.bfloat16` | ~0.98 GB | Native SDPA |
| **Total Models** | Shared Vocab (`151,936`) | Native Tensor Cores | **~7.16 GB** | Direct Ada FP8/BF16 |

### Hardware Profiles

1. **`rtx4060` (Laptop Profile)**
   - GPU: NVIDIA GeForce RTX 4060 Laptop GPU
   - VRAM: ~8.0 GB GDDR6 (128-bit bus, ~272 GB/s bandwidth)
   - Configuration: [`spec_decoding_4060/config.py`](file:///home/faizan/projects/spec-decoding-STC/spec_decoding_4060/config.py)
   - Target Environment: Host Linux

2. **`rtx4000` (Workstation Profile)**
   - GPU: NVIDIA RTX 4000 Ada Generation
   - VRAM: ~20.0 GB GDDR6 with ECC (160-bit bus, ~360 GB/s bandwidth, 130W TDP)
   - Configuration: [`spec_decoding_4000/config.py`](file:///home/faizan/projects/spec-decoding-STC/spec_decoding_4000/config.py)
   - Target Environment: Windows Host with Ubuntu WSL2

### Key Design Tenets
1. **Shared Engine & Utilities:** A single speculative decoding engine (`engine.py`) and DynamicCache rollback mechanism (`kv_cache_utils.py`) are reused across all GPUs. Only hardware configurations differ.
2. **Controlled Hardware Comparison:** Both profiles use identical models (`Qwen2.5-3B` & `0.5B`), identical precision (`bfloat16`), identical speculative lookahead ($K=5$), and identical benchmark prompts.
3. **Dynamic KV-Cache Slicing:** Avoids costly re-computation on rejection by rolling back Hugging Face `DynamicCache` states in-place.
4. **Lossless Equivalence:** Output token sequences under Greedy Speculative Decoding ($T=0$) preserve mathematical equivalence to pure autoregressive decoding on the target model.

---

## 📁 Repository Structure

```text
spec-decoding-STC/
├── benchmark.py -> spec_decoding_4060/benchmark.py  # Unified multi-GPU entry point
├── config.py -> spec_decoding_4060/config.py        # Default RTX 4060 config symlink
├── engine.py -> spec_decoding_4060/engine.py        # Shared speculative engine symlink
├── kv_cache_utils.py -> spec_decoding_4060/...      # DynamicCache cropping symlink
├── requirements.txt -> spec_decoding_4060/...      # Common dependencies symlink
├── spec_decoding_4060/                              # RTX 4060 profile module
│   ├── benchmark.py                                 # Shared benchmark runner & CLI
│   ├── config.py                                    # SpecConfig / SpecConfig4060
│   ├── engine.py                                    # SpeculativeEngine implementation
│   ├── kv_cache_utils.py                            # Dynamic KV-cache rollback logic
│   ├── requirements.txt                             # Common requirements
│   └── __init__.py
├── spec_decoding_4000/                              # RTX 4000 Ada profile module
│   ├── config.py                                    # SpecConfig4000 configuration
│   ├── requirements.txt -> ../spec_decoding_4060/requirements.txt
│   └── __init__.py
├── results/                                         # Benchmark outputs (git-ignored)
│   ├── rtx4060/                                     # JSON & CSV results from RTX 4060
│   └── rtx4000/                                     # JSON & CSV results from RTX 4000
└── speculative_decoding_prd_sop_rtx_4060_8gb.md     # PRD & SOP documentation
```

---

## 🛠️ Environment Setup

Both the RTX 4060 and RTX 4000 Ada belong to NVIDIA's **Ada Lovelace** generation (Compute Capability 8.9) and utilize the same PyTorch and CUDA runtime dependencies.

### 1. Verification of GPU Access

Verify your NVIDIA driver and CUDA availability:

```bash
nvidia-smi
```

Expected output:
- On laptop: `NVIDIA GeForce RTX 4060 ... | Memory: ~8188MiB`
- On workstation (WSL2): `NVIDIA RTX 4000 Ada Generation | Memory: ~20475MiB`

### 2. WSL2 Setup (for Windows Host / College PC)

If running on Windows host via WSL2:
1. Ensure the latest NVIDIA Game Ready or Studio/Enterprise driver is installed on the **Windows host** (CUDA in WSL2 passes through directly without installing separate Linux display drivers).
2. Inside Ubuntu WSL2, verify GPU passthrough:
   ```bash
   nvidia-smi
   ```
3. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Verify PyTorch sees CUDA:
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
   ```

---

## 🚀 Running Benchmarks

### Selecting GPU Profiles

The benchmark suite detects the environment, validates the requested profile against active hardware, and prints telemetry.

```bash
# 1. Run default RTX 4060 profile (laptop)
python benchmark.py
# or explicitly:
python benchmark.py --gpu rtx4060

# 2. Run RTX 4000 Ada profile (college PC / workstation)
python benchmark.py --gpu rtx4000

# 3. Auto-detect profile based on detected GPU
python benchmark.py --gpu auto
```

### Profile & GPU Inspection

```bash
# List all supported GPU profiles
python benchmark.py --list-profiles

# Display active GPU hardware and telemetry without running models
python benchmark.py --show-gpu

# Validate profile configuration and environment without loading models
python benchmark.py --gpu rtx4060 --dry-run
python benchmark.py --gpu rtx4000 --dry-run --allow-mismatch
```

### Profile Mismatch Safety

If you request `--gpu rtx4000` while running on an RTX 4060 (or vice versa), the runner displays an explicit error and exits without silently using wrong assumptions:

```text
======================================================================
[ERROR] GPU Profile Mismatch Detected!
======================================================================
  Requested Profile : 'rtx4000'
  Expected GPU Substring: 'RTX 4000'
  Detected Hardware : 'NVIDIA GeForce RTX 4060 Laptop GPU'
  Total VRAM        : 7.65 GB (Free: 7.55 GB)
----------------------------------------------------------------------
To run this profile anyway on the detected GPU (e.g. for testing),
pass '--allow-mismatch':
  python benchmark.py --gpu rtx4000 --allow-mismatch
======================================================================
```

### CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--gpu`, `--profile` | GPU profile to execute (`rtx4060`, `rtx4000`, `auto`) | `rtx4060` |
| `--list-profiles` | Display all configured profiles and exit | `False` |
| `--show-gpu` | Detect GPU hardware, driver, VRAM, and exit | `False` |
| `--allow-mismatch`, `--force` | Allow running profile when GPU does not match expected model | `False` |
| `--gamma` | Override lookahead token budget $K$ (e.g. `--gamma 6`) | Config default (`5`) |
| `--max-new-tokens` | Override maximum generated tokens per prompt | Config default (`256`) |
| `--output-dir` | Custom directory for JSON and CSV output files | `results/<profile>` |
| `--no-save` | Skip writing result files to disk | `False` |
| `--dry-run` | Validate configuration and GPU without loading model weights | `False` |

---

## 📊 Result Serialization & Benchmarks

Each benchmark run automatically logs telemetry and outputs:
- **Timestamped JSON Report**: `results/<profile>/benchmark_<profile>_<timestamp>.json`
- **Latest JSON Report**: `results/<profile>/latest_results.json`
- **Timestamped CSV**: `results/<profile>/benchmark_<profile>_<timestamp>.csv`
- **Cumulative Summary CSV**: `results/<profile>/summary.csv`

### Recorded Metrics:
- **Environment**: GPU model, Compute Capability, Total VRAM, Free VRAM at idle, PyTorch version, CUDA runtime version, Driver version.
- **Git State**: Active commit hash.
- **Performance**:
  - Baseline Autoregressive Throughput (Tokens/s)
  - Custom Speculative Decoding Throughput (Tokens/s)
  - Hugging Face Assisted Generation Throughput (Tokens/s)
  - Speedup factor ($TPS_{spec} / TPS_{ar}$)
  - Acceptance Rate ($\%$)
  - Average accepted tokens per verification iteration (`tokens_per_target_step`)
  - Exact token match verification (`PASS` / `MISMATCH`)
  - Peak GPU memory allocation (`torch.cuda.max_memory_allocated()`)

---

## ⚖️ Guidelines for a Fair Hardware Comparison

To ensure performance comparisons between the RTX 4060 and RTX 4000 Ada isolate GPU hardware differences (memory bandwidth, clock speeds, Tensor Core count):

1. **Keep Models Identical:** Both profiles must run `Qwen/Qwen2.5-3B-Instruct` as target and `Qwen/Qwen2.5-0.5B-Instruct` as draft. Do not increase model size on the 4000 Ada when comparing hardware.
2. **Keep Precision & Attention Identical:** Both profiles must use `torch.bfloat16` and `sdpa` attention.
3. **Keep Decoding Hyperparameters Identical:** Speculative lookahead $\gamma = 5$ and generation length $\text{max\_new\_tokens} = 256$ must remain constant.
4. **Controlled Workloads:** Both use the exact same 3 test prompts (Code Generation, Structured JSON, Technical Explanation).
5. **Warm-Up:** Keep the 1-iteration CUDA kernel warm-up pass active so GPU clocks and memory controllers reach steady-state.

---

## ➕ Adding a New GPU Profile

To add support for another GPU (e.g., RTX 4090, RTX 3060):

1. **Create profile directory:**
   ```bash
   mkdir spec_decoding_<name>
   touch spec_decoding_<name>/__init__.py
   ```
2. **Define configuration (`spec_decoding_<name>/config.py`):**
   ```python
   import torch

   class SpecConfig<Name>:
       GPU_PROFILE: str = "<name>"
       EXPECTED_GPU_SUBSTR: str = "<GPU substring to match>"
       VRAM_TOTAL_GB: float = <nominal_vram>

       # Retain identical model and algorithm settings for fair comparison
       TARGET_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"
       DRAFT_MODEL_ID: str = "Qwen/Qwen2.5-0.5B-Instruct"
       DEVICE: str = "cuda:0"
       DTYPE: torch.dtype = torch.bfloat16
       ATTN_IMPLEMENTATION: str = "sdpa"
       GAMMA: int = 5
       MAX_NEW_TOKENS: int = 256
       OUTPUT_DIR: str = "results/<name>"
       PROMPTS = [ ... ]

   SpecConfig = SpecConfig<Name>
   ```
3. **Register profile in `spec_decoding_4060/benchmark.py`:**
   Add entry to `SUPPORTED_PROFILES`:
   ```python
   "<name>": {
       "module": "spec_decoding_<name>.config",
       "class_name": "SpecConfig<Name>",
       "description": "<Description of GPU>",
   }
   ```
4. **Execute benchmark:**
   ```bash
   python benchmark.py --gpu <name>
   ```
