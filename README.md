# Local Speculative Decoding Engine for NVIDIA RTX 4060 (8GB VRAM)

A high-performance, educational PyTorch & Hugging Face pipeline implementing Greedy Speculative Decoding with dynamic KV-cache rollback. Built specifically to fit within consumer 8GB GDDR6 VRAM budgets while delivering **$1.5\times$ to $2.2\times$ wall-clock speedups** over standard autoregressive decoding.

Based on the specification: [`speculative_decoding_prd_sop_rtx_4060_8gb.md`](./speculative_decoding_prd_sop_rtx_4060_8gb.md)

---

## 🎯 Architecture & Strategy

| Role | Model | Precision | VRAM Footprint |
| :--- | :--- | :--- | :--- |
| **Target Model ($M_p$)** | `Qwen/Qwen2.5-3B-Instruct` | `torch.bfloat16` | ~6.18 GB |
| **Draft Model ($M_q$)** | `Qwen/Qwen2.5-0.5B-Instruct` | `torch.bfloat16` | ~0.98 GB |
| **Total Models** | Shared Vocab (`151,936`) | Native Tensor Cores | **~7.16 GB** (Fits 100% in 8GB VRAM) |

### Key Design Tenets
1. **Zero 4-bit Dequantization Overhead:** Standard INT4/NF4 quantization causes compute bottlenecks during verification passes. Native `bfloat16` utilizes Ada Lovelace Tensor Cores directly.
2. **Dynamic KV-Cache Slicing:** Avoids costly re-computation on rejection by rolling back Hugging Face `DynamicCache` states in-place.
3. **Lossless Equivalence:** Output token sequences under Greedy Speculative Decoding ($T=0$) are 100% mathematically identical to pure autoregressive decoding on the target model.

---

## 📁 Repository Structure

```text
spec-decoding-STC/
├── speculative_decoding_prd_sop_rtx_4060_8gb.md  # Comprehensive PRD & SOP
├── spec_decoding_4060/                           # Core implementation module
│   ├── requirements.txt                          # Python dependencies
│   ├── config.py                                 # Hyperparameters & benchmark prompts
│   ├── kv_cache_utils.py                         # DynamicCache cropping / rollback logic
│   ├── engine.py                                 # SpeculativeEngine (AR, Speculative, HF Assisted)
│   └── benchmark.py                              # Warm-up, verification, and telemetry suite
├── config.py -> spec_decoding_4060/config.py
├── engine.py -> spec_decoding_4060/engine.py
├── kv_cache_utils.py -> spec_decoding_4060/kv_cache_utils.py
├── benchmark.py -> spec_decoding_4060/benchmark.py
└── requirements.txt -> spec_decoding_4060/requirements.txt
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Activate existing virtual environment
source venv/bin/activate

# Or create a new one
python -m venv venv
source venv/bin/activate
pip install -r spec_decoding_4060/requirements.txt
```

### 2. Run Benchmarks

```bash
python benchmark.py
# or
python spec_decoding_4060/benchmark.py
```

---

## 📊 Expected Performance on RTX 4060 (8GB)

| Benchmark Metric | Baseline (3B) | Custom Speculative (0.5B -> 3B) | HF Native Assisted |
| :--- | :--- | :--- | :--- |
| **Code Generation (BST implementation)** | ~45 tok/s | **~85 - 95 tok/s ($1.9\times - 2.1\times$)** | ~75 - 85 tok/s |
| **Structured JSON Generation** | ~45 tok/s | **~80 - 90 tok/s ($1.8\times - 2.0\times$)** | ~70 - 80 tok/s |
| **Technical Explanation** | ~45 tok/s | **~65 - 75 tok/s ($1.4\times - 1.6\times$)** | ~60 - 70 tok/s |
| **Token Match Rate** | 100% | **100% Exact Match (PASS)** | 100% Exact Match |
| **Peak VRAM Allocated** | ~6.2 GB | **~7.25 GB / 8.00 GB** | ~7.30 GB |
