from unittest.mock import MagicMock, patch
import pytest
from kubeflow_mcp.clients.trainer.monitoring import (
    estimate_resources_fn, get_training_events_fn, get_training_logs_fn,
    _LOG_TAIL_CAP
)

def test_estimate_resources_with_user_params():
    result = estimate_resources_fn(
        model="meta-llama/Llama-3.2-7B",
        peft_method="lora",
        user_provided_params={"param_count": 7_000_000_000, "num_layers": 32, "hidden_size": 4096}
    )
    assert "total_gpu_memory_gb" in result
    assert result["breakdown"]["model_gb"] > 0
    assert result["breakdown"]["activations_gb"] > 0
    assert result["confidence"] == "high"

def test_estimate_resources_full_higher_than_lora():
    lora = estimate_resources_fn("x", peft_method="lora",
                                  user_provided_params={"param_count": 7_000_000_000})
    full = estimate_resources_fn("x", peft_method="full",
                                  user_provided_params={"param_count": 7_000_000_000})
    assert full["total_gpu_memory_gb"] > lora["total_gpu_memory_gb"]

def test_estimate_resources_user_provided_params():
    result = estimate_resources_fn(
        model="private-model",
        user_provided_params={"param_count": 13_000_000_000, "num_layers": 40, "hidden_size": 5120}
    )
    assert result["confidence"] == "high"
    assert result["param_count"] == 13_000_000_000

def test_get_training_logs_tail_cap():
    with patch("kubeflow_mcp.clients.trainer.monitoring.create_trainer_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = iter([f"line {i}" for i in range(2000)])
        mock_factory.return_value = mock_client
        result = get_training_logs_fn("my-job", namespace="default")
        assert len(result["logs"]) <= _LOG_TAIL_CAP

def test_get_training_events_oom_suggestion():
    with patch("kubeflow_mcp.clients.trainer.monitoring.create_trainer_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.get_job_events.return_value = ["OOMKilled: container exceeded memory limit"]
        mock_factory.return_value = mock_client
        result = get_training_events_fn("my-job")
        assert any("batch_size" in e for e in result["events"])
