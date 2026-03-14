# src/kubeflow_mcp/clients/trainer/lifecycle.py
from typing import Optional
from mcp.server.fastmcp import FastMCP
from kubeflow_mcp.core.auth import create_trainer_client


def _load_k8s_config():
    """Try in-cluster config first, fall back to kubeconfig for local dev."""
    import kubernetes
    try:
        kubernetes.config.load_incluster_config()
    except kubernetes.config.ConfigException:
        kubernetes.config.load_kube_config()


def delete_training_job_fn(name, namespace=None):
    try:
        client = create_trainer_client(namespace)
        client.delete_job(name=name)
        return {"success": True, "deleted": name}
    except Exception as e:
        return {"success": False, "error": str(e)}


def suspend_training_job_fn(name, namespace=None):
    """
    Suspend a training job by patching TrainJob CRD suspend field.
    suspend_job() is not in TrainerClient — uses K8s API directly.
    Phase 4 adds checkpoint coordination on top of this stub.
    """
    try:
        import kubernetes
        _load_k8s_config()
        api = kubernetes.client.CustomObjectsApi()
        ns = namespace or "default"
        api.patch_namespaced_custom_object(
            group="kubeflow.org", version="v1",
            namespace=ns, plural="trainjobs", name=name,
            body={"spec": {"suspend": True}},
        )
        return {"success": True, "suspended": name}
    except Exception as e:
        return {"success": False, "error": str(e)}


def resume_training_job_fn(name, namespace=None):
    """
    Resume a suspended training job.
    Phase 4 adds checkpoint integrity verification on top of this stub.
    """
    try:
        import kubernetes
        _load_k8s_config()
        api = kubernetes.client.CustomObjectsApi()
        ns = namespace or "default"
        api.patch_namespaced_custom_object(
            group="kubeflow.org", version="v1",
            namespace=ns, plural="trainjobs", name=name,
            body={"spec": {"suspend": False}},
        )
        return {"success": True, "resumed": name}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_lifecycle_tools(mcp: FastMCP):

    @mcp.tool()
    def delete_training_job(name: str, namespace: Optional[str] = None) -> dict:
        """Delete a training job."""
        return delete_training_job_fn(name, namespace)

    @mcp.tool()
    def suspend_training_job(name: str, namespace: Optional[str] = None) -> dict:
        """Suspend a training job by patching TrainJob CRD suspend field.
        Phase 4 adds checkpoint coordination on top of this stub.
        """
        return suspend_training_job_fn(name, namespace)

    @mcp.tool()
    def resume_training_job(name: str, namespace: Optional[str] = None) -> dict:
        """Resume a suspended training job.
        Phase 4 adds checkpoint integrity verification on top of this stub.
        """
        return resume_training_job_fn(name, namespace)
