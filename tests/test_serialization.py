import csv
import json
from pathlib import Path
import tempfile
import pytest

from spec_decoding_4060.benchmark import save_results


@pytest.fixture
def sample_payload():
    return {
        "metadata": {
            "timestamp": "2026-10-24T00:00:00Z",
            "git_commit": "abcdef123456",
            "profile": "rtx4000",
            "is_controlled_hardware_comparison": True,
        },
        "hardware": {
            "device_idx": 0,
            "device_str": "cuda:0",
            "gpu_name": "NVIDIA RTX 4000 Ada Generation",
            "total_vram_gb": 20.0,
            "free_vram_gb": 19.5,
        },
        "configuration": {
            "target_model_id": "Qwen/Qwen2.5-3B-Instruct",
            "draft_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
            "attn_implementation": "sdpa",
            "gamma": 8,
            "max_new_tokens": 256,
            "warmup_iterations": 1,
            "warmup_tokens": 16,
            "measured_repetitions": 3,
            "benchmark_seed": 42,
        },
        "metric_definitions": {
            "acceptance_rate": "100 * total_accepted / total_drafted",
            "output_tokens_per_target_forward": "Output tokens divided by target forward passes",
            "accepted_tokens_per_verification_cycle": "Accepted draft tokens divided by verification cycles",
            "target_forward_passes": "target_model forward calls",
            "verification_cycles": "verify forward calls after prefill",
            "exact_match": "True iff baseline greedy equals spec greedy",
            "throughput_tps": "generated tokens / wall time",
        },
        "benchmarks": [
            {
                "task_name": "Code Generation",
                "prompt_text": "Write a binary tree.",
                "prompt_tokens": 40,
                "generated_tokens": 256,
                "baseline_ar": {
                    "tps": 50.0,
                    "time_sec": 5.12,
                    "tokens_generated": 256,
                    "raw_runs": [{"tps": 50.0, "time_sec": 5.12}],
                },
                "custom_speculative": {
                    "tps": 110.0,
                    "time_sec": 2.32,
                    "speedup": 2.2,
                    "acceptance_rate": 85.54,
                    "tokens_per_target_step": 5.12,
                    "output_tokens_per_target_forward": 5.12,
                    "accepted_tokens_per_verification_cycle": 4.22,
                    "total_drafted": 242,
                    "total_accepted": 207,
                    "target_forward_passes": 50,
                    "verification_cycles": 49,
                    "raw_runs": [{"tps": 110.0, "time_sec": 2.32}],
                },
                "hf_assisted": {
                    "tps": 90.0,
                    "time_sec": 2.84,
                    "speedup": 1.8,
                    "tokens_generated": 256,
                    "raw_runs": [{"tps": 90.0, "time_sec": 2.84}],
                },
                "exact_match": True,
                "greedy_equivalence": {"exact_match": True},
            }
        ],
        "telemetry": {
            "peak_vram_allocated_gb": 6.8,
            "peak_vram_reserved_gb": 7.2,
            "system_free_vram_gb": 12.8,
            "total_vram_gb": 20.0,
        },
    }


def test_save_results_creates_files(sample_payload):
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path, csv_path = save_results(
            results_payload=sample_payload,
            table_rows=[],
            output_dir_path=tmp_dir,
            profile_name="rtx4000",
        )
        assert json_path.exists()
        assert csv_path.exists()
        
        latest_json = Path(tmp_dir) / "latest_results.json"
        summary_csv = Path(tmp_dir) / "summary.csv"
        assert latest_json.exists()
        assert summary_csv.exists()

        # Validate JSON content
        with open(latest_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["metadata"]["profile"] == "rtx4000"
        assert "metric_definitions" in data
        assert "accepted_tokens_per_verification_cycle" in data["metric_definitions"]
        assert len(data["benchmarks"]) == 1
        assert data["benchmarks"][0]["custom_speculative"]["total_accepted"] == 207

        # Validate CSV content
        with open(summary_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        headers = reader[0]
        assert "task_name" in headers
        assert "output_tokens_per_target_forward" in headers
        assert "accepted_tokens_per_verification_cycle" in headers
        assert "target_forward_passes" in headers
        assert "verification_cycles" in headers
        row = reader[1]
        assert row[0] == "Code Generation"
