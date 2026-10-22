"""
Local Speculative Decoding Engine for NVIDIA RTX 4060 (8GB VRAM).
"""
from .config import SpecConfig, SpecConfig4060
from .engine import SpeculativeEngine
from .kv_cache_utils import crop_past_key_values

__all__ = ["SpecConfig", "SpecConfig4060", "SpeculativeEngine", "crop_past_key_values"]
