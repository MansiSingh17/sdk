# src/kubeflow_mcp/clients/trainer/monitoring.py
import time
from typing import Optional
from mcp.server.fastmcp import FastMCP
from kubeflow_mcp.core.auth import create_trainer_client
from kubeflow.trainer.constants.constants import TRAINJOB_COMPLETE

_last_log_call: dict = {}
_LOG_RATE_LIMIT_SEC = 1.0
_LOG_TAIL_CAP = 1000
_LOG_CACHE: dict = {}
_LOG_CACHE_TTL = 5.0


def get_training_logs_fn(name, step="node-0", namespace=None):
    now = time.time()
    cache_key = f"{name}:{step}"

    # Return cached result if within TTL
    if cache_key in _LOG_CACHE:
        cached_at, cached_result = _LOG_CACHE[cache_key]
        if now - cached_at < _LOG_CACHE_TTL:
            return cached_result

    # Rate limit: 1 req/sec per job
    last_call = _last_log_call.get(cache_key, 0)
    elapsed = now - last_call
    if elapsed < _LOG_RATE_LIMIT_SEC:
        time.sleep(_LOG_RATE_LIMIT_SEC - elapsed)

    _last_log_call[cache_key] = time.time()

    try:
        client = create_trainer_client(namespace)
        logs = list(client.get_job_logs(name=name, step=step, follow=False))
        if len(logs) > _LOG_TAIL_CAP:
            logs = logs[-_LOG_TAIL_CAP:]
        result = {"logs": logs, "job": name, "step": step, "lines": len(logs)}
        _LOG_CACHE[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        return {"error": str(e)}


def wait_for_training_fn(name, timeout=3600, namespace=None):
    try:
        client = create_trainer_client(namespace)
        client.wait_for_job_status(
            name=name,
            status={TRAINJOB_COMPLETE},
            timeout=timeout,
            polling_interval=30,
        )
        return {"status": "completed", "job": name}
    except Exception as e:
        return {"status": "timeout_or_error", "error": str(e)}


def get_training_events_fn(name, namespace=None):
    try:
        client = create_trainer_client(namespace)
        events = client.get_job_events(name=name)
        formatted = []
        for e in events:
            msg = str(e)
            if "OOMKilled" in msg:
                msg += " | SUGGESTION: reduce batch_size or use QLoRA (peft_method='qlora')"
            elif "BackoffLimitExceeded" in msg:
                msg += " | SUGGESTION: check logs with get_training_logs(), job likely crashed on startup"
            formatted.append(msg)
        return {"events": formatted, "job": name}
    except Exception as e:
        return {"error": str(e)}


def estimate_resources_fn(
    model: str,
    peft_method: str = "lora",
    batch_size: int = 4,
    sequence_length: int = 2048,
    quantization: str = "bf16",
    num_nodes: int = 1,
    user_provided_params: Optional[dict] = None,
) -> dict:
    """
    4-step memory estimation from DESIGN.md:
    1. HF Hub lookup (or user_provided_params for private models)
    2. Base model memory
    3. Activation memory (batch_size x sequence_length x layers x hidden_size)
    4. Optimizer + gradient memory
    """
    try:
        param_count = None
        num_layers = 32
        hidden_size = 4096
        confidence = "low"

        if user_provided_params:
            param_count = user_provided_params.get("param_count")
            num_layers = user_provided_params.get("num_layers", num_layers)
            hidden_size = user_provided_params.get("hidden_size", hidden_size)
            confidence = "high"
        else:
            try:
                from huggingface_hub import model_info as hf_model_info
                info = hf_model_info(model)
                card = info.cardData or {}
                param_count = card.get("num_parameters")
                confidence = "high" if param_count else "low"
            except Exception:
                confidence = "low"

        if not param_count:
            import re
            match = re.search(r'(\d+)B', model, re.IGNORECASE)
            param_count = int(match.group(1)) * 1_000_000_000 if match else 7_000_000_000

        quant_bytes = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1, "int4": 0.5}
        bytes_per_param = quant_bytes.get(quantization, 2)
        base_memory_gb = (param_count * bytes_per_param) / (1024 ** 3)

        activation_memory_gb = (batch_size * sequence_length * num_layers * hidden_size * 2) / (1024 ** 3)

        if peft_method == "full":
            optimizer_memory_gb = base_memory_gb * 2
            gradient_memory_gb = base_memory_gb
        else:
            optimizer_memory_gb = base_memory_gb * 0.1
            gradient_memory_gb = base_memory_gb * 0.05

        total_gb = base_memory_gb + activation_memory_gb + optimizer_memory_gb + gradient_memory_gb
        total_per_node = total_gb / max(num_nodes, 1)

        recommended_gpu = (
            "A100 40GB" if total_per_node <= 40 else
            "A100 80GB" if total_per_node <= 80 else
            f"Multi-GPU required ({total_per_node:.0f}GB per node)"
        )

        return {
            "model": model,
            "param_count": param_count,
            "breakdown": {
                "model_gb": round(base_memory_gb, 1),
                "activations_gb": round(activation_memory_gb, 1),
                "optimizer_gb": round(optimizer_memory_gb, 1),
                "gradients_gb": round(gradient_memory_gb, 1),
            },
            "total_gpu_memory_gb": round(total_gb, 1),
            "per_node_gb": round(total_per_node, 1),
            "recommended_gpu": recommended_gpu,
            "confidence": confidence,
        }
    except Exception as e:
        return {"error": str(e)}


def register_monitoring_tools(mcp: FastMCP):

    @mcp.tool()
    def get_training_logs(
        name: str,
        step: str = "node-0",
        namespace: Optional[str] = None,
    ) -> dict:
        """Get logs for a training job.
        Rate-limited to 1 req/sec per job, cached 5s TTL, tail capped at 1000 lines.
        """
        return get_training_logs_fn(name, step, namespace)

    @mcp.tool()
    def wait_for_training(
        name: str,
        timeout: int = 3600,
        namespace: Optional[str] = None,
    ) -> dict:
        """Wait for training job to complete. Returns status on timeout without canceling the job."""
        return wait_for_training_fn(name, timeout, namespace)

    @mcp.tool()
    def get_training_events(name: str, namespace: Optional[str] = None) -> dict:
        """Get Kubernetes events for a training job with LLM-friendly error summaries."""
        return get_training_events_fn(name, namespace)

    @mcp.tool()
    def estimate_resources(
        model: str,
        peft_method: str = "lora",
        batch_size: int = 4,
        sequence_length: int = 2048,
        quantization: str = "bf16",
        num_nodes: int = 1,
        user_provided_params: Optional[dict] = None,
    ) -> dict:
        """Estimate GPU memory requirements before submitting a training job.
        Uses HF Hub for model metadata, or user_provided_params for private/custom models.
        Returns breakdown, total_gpu_memory_gb, and recommended_gpu.
        """
        return estimate_resources_fn(model, peft_method, batch_size,
                                     sequence_length, quantization,
                                     num_nodes, user_provided_params)
