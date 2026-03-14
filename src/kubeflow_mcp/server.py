# src/kubeflow_mcp/server.py
from mcp.server.fastmcp import FastMCP
from kubeflow_mcp.clients.trainer.discovery import register_discovery_tools
from kubeflow_mcp.clients.trainer.training import register_training_tools
from kubeflow_mcp.clients.trainer.monitoring import register_monitoring_tools
from kubeflow_mcp.clients.trainer.lifecycle import register_lifecycle_tools

PERSONA_TOOLS = {
    "readonly": {
        "get_cluster_resources", "list_training_jobs", "get_training_job",
        "get_training_logs", "get_training_events", "list_runtimes", "get_runtime",
    },
    "data-scientist": {
        "get_cluster_resources", "list_training_jobs", "get_training_job",
        "get_training_logs", "get_training_events", "list_runtimes", "get_runtime",
        "estimate_resources", "fine_tune", "run_custom_training",
        "wait_for_training", "delete_training_job",
    },
    "ml-engineer": {
        "get_cluster_resources", "list_training_jobs", "get_training_job",
        "get_training_logs", "get_training_events", "list_runtimes", "get_runtime",
        "estimate_resources", "fine_tune", "run_custom_training",
        "wait_for_training", "delete_training_job",
        "run_container_training", "get_runtime_packages",
        "suspend_training_job", "resume_training_job",
    },
    "platform-admin": None,  # None = all tools
}


def create_server(clients: list = None, persona: str = "ml-engineer") -> FastMCP:
    if clients is None:
        clients = ["trainer"]

    allowed_tools = PERSONA_TOOLS.get(persona)

    mcp = FastMCP(
        "kubeflow-mcp",
        instructions="""
Kubeflow MCP Server - AI Model Training on Kubernetes

WORKFLOW: Fine-Tuning LLMs
1. get_cluster_resources() - check GPU availability
2. estimate_resources(model, peft_method) - memory requirements
3. fine_tune(model, dataset, confirmed=False) - preview config
4. fine_tune(model, dataset, confirmed=True) - submit after user approval
5. get_training_logs(job_id) / wait_for_training(job_id) - monitor

TOOL SELECTION:
- Fine-tune HuggingFace models: fine_tune()
- Run custom Python training function: run_custom_training()
- Run pre-built container: run_container_training()
        """
    )

    if "trainer" in clients:
        _register_filtered(mcp, register_discovery_tools, allowed_tools)
        _register_filtered(mcp, register_training_tools, allowed_tools)
        _register_filtered(mcp, register_monitoring_tools, allowed_tools)
        _register_filtered(mcp, register_lifecycle_tools, allowed_tools)

    return mcp


def _register_filtered(mcp, register_fn, allowed_tools):
    register_fn(mcp)
    if allowed_tools is None:
        return
    if hasattr(mcp, '_tools'):
        to_remove = [name for name in list(mcp._tools) if name not in allowed_tools]
        for name in to_remove:
            del mcp._tools[name]
