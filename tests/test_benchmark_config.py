import pytest
from spec_decoding_4060.benchmark import (
    auto_detect_profile,
    load_profile_config,
    validate_profile_gpu,
)


def test_load_profile_config_rtx4060():
    cfg = load_profile_config("rtx4060")
    assert cfg.GPU_PROFILE == "rtx4060"
    assert cfg.GAMMA == 5
    assert cfg.MAX_NEW_TOKENS == 256
    assert cfg.ALLOW_TF32 is True


def test_load_profile_config_rtx4000():
    cfg = load_profile_config("rtx4000")
    assert cfg.GPU_PROFILE == "rtx4000"
    assert cfg.GAMMA == 8
    assert cfg.MEASURED_REPETITIONS == 3
    assert cfg.CUDNN_BENCHMARK is True
    assert cfg.ALLOW_TF32 is True


def test_load_profile_config_overrides():
    overrides = {
        "GAMMA": 10,
        "MAX_NEW_TOKENS": 128,
        "MEASURED_REPETITIONS": 5,
        "OUTPUT_DIR": "/tmp/custom_results",
    }
    cfg = load_profile_config("rtx4000", overrides=overrides)
    assert cfg.GAMMA == 10
    assert cfg.MAX_NEW_TOKENS == 128
    assert cfg.MEASURED_REPETITIONS == 5
    assert cfg.OUTPUT_DIR == "/tmp/custom_results"


def test_auto_detect_profile():
    assert auto_detect_profile({"gpu_name": "NVIDIA RTX 4000 Ada Generation"}) == "rtx4000"
    assert auto_detect_profile({"gpu_name": "NVIDIA RTX 4000 SFF Ada Generation"}) == "rtx4000"
    assert auto_detect_profile({"gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU"}) == "rtx4060"
    assert auto_detect_profile({"gpu_name": "NVIDIA GeForce RTX 4060"}) == "rtx4060"
    assert auto_detect_profile({"gpu_name": "Tesla V100"}) == "rtx4060"  # default fallback


def test_validate_profile_gpu():
    cfg_4000 = load_profile_config("rtx4000")
    matching_hw = {"gpu_name": "NVIDIA RTX 4000 Ada Generation", "total_vram_gb": 20.0, "free_vram_gb": 19.0}
    mismatch_hw = {"gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "total_vram_gb": 8.0, "free_vram_gb": 7.0}

    assert validate_profile_gpu(cfg_4000, "rtx4000", matching_hw, allow_mismatch=False) is True
    assert validate_profile_gpu(cfg_4000, "rtx4000", mismatch_hw, allow_mismatch=True) is False
