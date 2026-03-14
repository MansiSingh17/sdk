from unittest.mock import MagicMock, patch
import pytest

def test_list_training_jobs_returns_summaries():
    mock_job = MagicMock()
    mock_job.metadata.name = "ft-qwen-abc123"
    mock_job.metadata.namespace = "default"
    mock_job.metadata.creation_timestamp = "2026-03-01"
    mock_job.status.conditions = None
    with patch("kubeflow_mcp.clients.trainer.discovery.create_trainer_client") as mock_factory:
        mock_client = MagicMock()
        mock_client.list_jobs.return_value = [mock_job]
        mock_factory.return_value = mock_client
        from kubeflow_mcp.clients.trainer.discovery import list_training_jobs_fn
        result = list_training_jobs_fn(namespace="default")
        assert result["count"] == 1
        assert result["jobs"][0]["name"] == "ft-qwen-abc123"

def test_list_training_jobs_handles_error():
    with patch("kubeflow_mcp.clients.trainer.discovery.create_trainer_client") as mock_factory:
        mock_factory.return_value = MagicMock(
            list_jobs=MagicMock(side_effect=Exception("cluster unreachable"))
        )
        from kubeflow_mcp.clients.trainer.discovery import list_training_jobs_fn
        result = list_training_jobs_fn()
        assert "error" in result

def test_get_runtime_fn_returns_name():
    mock_runtime = MagicMock()
    mock_runtime.metadata.name = "torch-distributed"
    mock_runtime.spec = MagicMock()
    with patch("kubeflow_mcp.clients.trainer.discovery.create_trainer_client") as mock_factory:
        mock_factory.return_value = MagicMock(
            get_runtime=MagicMock(return_value=mock_runtime)
        )
        from kubeflow_mcp.clients.trainer.discovery import get_runtime_fn
        result = get_runtime_fn("torch-distributed")
        assert result["name"] == "torch-distributed"
