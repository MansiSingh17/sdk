from unittest.mock import MagicMock, patch
import pytest
from kubeflow_mcp.clients.trainer.training import (
    fine_tune_fn, run_custom_training_fn, run_container_training_fn
)

def test_fine_tune_preview_when_not_confirmed():
    result = fine_tune_fn(model="Qwen/Qwen2.5-7B", dataset="tatsu-lab/alpaca", confirmed=False)
    assert result["status"] == "preview"
    assert "confirmed=True" in result["message"]

def test_fine_tune_preview_includes_resources_per_node():
    result = fine_tune_fn(model="x", dataset="y",
                          resources_per_node={"nvidia.com/gpu": "4"}, confirmed=False)
    assert result["config"]["resources_per_node"] == {"nvidia.com/gpu": "4"}

def test_fine_tune_preview_includes_runtime():
    result = fine_tune_fn(model="x", dataset="y", runtime="torch-distributed", confirmed=False)
    assert result["config"]["runtime"] == "torch-distributed"

def test_run_custom_training_ast_blocks_os_import():
    result = run_custom_training_fn(func_code="import os\ndef train(**kwargs): pass", confirmed=True)
    assert result["success"] == False
    assert "os" in result["error"]

def test_run_custom_training_ast_blocks_subprocess():
    result = run_custom_training_fn(func_code="import subprocess\ndef train(**kwargs): pass", confirmed=True)
    assert result["success"] == False

def test_run_custom_training_ast_blocks_eval():
    result = run_custom_training_fn(func_code="def train(**kwargs): eval('x')", confirmed=True)
    assert result["success"] == False
    assert "eval" in result["error"]

def test_run_custom_training_requires_train_function():
    result = run_custom_training_fn(func_code="def not_train(): pass", confirmed=True)
    assert result["success"] == False
    assert "train" in result["error"]

def test_run_custom_training_preview():
    result = run_custom_training_fn(func_code="def train(**kwargs): pass", confirmed=False)
    assert result["status"] == "preview"

def test_run_container_training_preview():
    result = run_container_training_fn(image="ghcr.io/myorg/trainer:v1", confirmed=False)
    assert result["status"] == "preview"
    assert result["config"]["image"] == "ghcr.io/myorg/trainer:v1"
