from unittest.mock import MagicMock, patch
import pytest
from kubeflow_mcp.clients.trainer.lifecycle import (
    delete_training_job_fn, suspend_training_job_fn, resume_training_job_fn
)

def test_delete_training_job_success():
    with patch("kubeflow_mcp.clients.trainer.lifecycle.create_trainer_client") as mock_factory:
        mock_client = MagicMock()
        mock_factory.return_value = mock_client
        result = delete_training_job_fn("my-job", namespace="default")
        assert result["success"] == True
        assert result["deleted"] == "my-job"
        mock_client.delete_job.assert_called_once_with(name="my-job")

def test_suspend_training_job_patches_crd():
    with patch("kubeflow_mcp.clients.trainer.lifecycle._load_k8s_config"), \
         patch("kubernetes.client.CustomObjectsApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        result = suspend_training_job_fn("my-job", namespace="default")
        assert result["success"] == True
        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["body"] == {"spec": {"suspend": True}}

def test_resume_training_job_patches_crd():
    with patch("kubeflow_mcp.clients.trainer.lifecycle._load_k8s_config"), \
         patch("kubernetes.client.CustomObjectsApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        result = resume_training_job_fn("my-job", namespace="default")
        assert result["success"] == True
        call_kwargs = mock_api.patch_namespaced_custom_object.call_args[1]
        assert call_kwargs["body"] == {"spec": {"suspend": False}}

def test_delete_training_job_handles_error():
    with patch("kubeflow_mcp.clients.trainer.lifecycle.create_trainer_client") as mock_factory:
        mock_factory.return_value = MagicMock(
            delete_job=MagicMock(side_effect=Exception("not found"))
        )
        result = delete_training_job_fn("missing-job")
        assert result["success"] == False
        assert "not found" in result["error"]
