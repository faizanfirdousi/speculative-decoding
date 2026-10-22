"""
Speculative Decoding Configuration for NVIDIA RTX 4000 Ada Generation (~20 GB VRAM).
Reuses the shared engine, benchmark, and KV-cache implementation from spec_decoding_4060/.
"""
from .config import SpecConfig, SpecConfig4000

__all__ = ["SpecConfig", "SpecConfig4000"]
