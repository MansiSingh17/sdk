# src/kubeflow_mcp/clients/trainer/discovery.py
from typing import Optional
from mcp.server.fastmcp import FastMCP
from kubeflow_mcp.core.auth import create_trainer_client


def _get_job_status(job) -> str:
    if job.status and job.status.conditions:
        return job.status.conditions[-1].type
    return "Unknown"


def list_training_jobs_fn(namespace=None):
    try:
        client = create_trainer_client(namespace)
        jobs = client.list_jobs()
        summaries = []
        for job in jobs:
            summaries.append({
                "name": job.metadata.name,
                "namespace": job.metadata.namespace,
                "status": _get_job_status(job),
                "created_at": str(job.metadata.creation_timestamp),
            })
        return {"jobs": summaries, "count": len(summaries)}
    except Exception as e:
        return {"error": str(e)}


def get_training_job_fn(name, namespace=None):
    try:
        client = create_trainer_client(namespace)
        job = client.get_job(name=name)
        return {
            "name": job.metadata.name,
            "namespace": job.metadata.namespace,
            "status": _get_job_status(job),
            "spec": str(job.spec),
            "conditions": [str(c) for c in (job.status.conditions or [])],
        }
    except Exception as e:
        return {"error": str(e)}


def list_runtimes_fn(namespace=None):
    try:
        client = create_trainer_client(namespace)
        runtimes = client.list_runtimes()
        return {"runtimes": [r.name for r in runtimes]}
    except Exception as e:
        return {"error": str(e)}


def get_runtime_fn(name, namespace=None):
    try:
        client = create_trainer_client(namespace)
        runtime = client.get_runtime(name=name)
        return {"name": runtime.metadata.name, "spec": str(runtime.spec)}
    except Exception as e:
        return {"error": str(e)}


def get_runtime_packages_fn(name, namespace=None):
    try:
        client = create_trainer_client(namespace)
        runtime = client.get_runtime(name=name)
        packages = []
        if runtime.spec and hasattr(runtime.spec, 'containers'):
            for container in runtime.spec.containers:
                if hasattr(container, 'env'):
                    packages.extend([e.value for e in (container.env or [])
                                     if e.name == "PIP_PACKAGES"])
        return {"runtime": name, "packages": packages}
    except Exception as e:
        return {"error": str(e)}


def register_discovery_tools(mcp: FastMCP):

    @mcp.tool()
    def get_cluster_resources() -> dict:
        """Check GPU/CPU/memory availability across cluster nodes."""
        try:
            import kubernetes
            try:
                kubernetes.config.load_incluster_config()
            except kubernetes.config.ConfigException:
                kubernetes.config.load_kube_config()
            v1 = kubernetes.client.CoreV1Api()
            nodes = v1.list_node()
            resources = []
            for node in nodes.items:
                allocatable = node.status.allocatable or {}
                resources.append({
                    "name": node.metadata.name,
                    "gpu": allocatable.get("nvidia.com/gpu", "0"),
                    "cpu": allocatable.get("cpu", "0"),
                    "memory": allocatable.get("memory", "0"),
                })
            return {"nodes": resources, "total_nodes": len(resources)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_training_jobs(namespace: Optional[str] = None) -> dict:
        """List training jobs with summary info (name, status, created_at)."""
        return list_training_jobs_fn(namespace)

    @mcp.tool()
    def get_training_job(name: str, namespace: Optional[str] = None) -> dict:
        """Get full TrainJob spec including config, resources, conditions."""
        return get_training_job_fn(name, namespace)

    @mcp.tool()
    def list_runtimes(namespace: Optional[str] = None) -> dict:
        """List available training runtimes."""
        return list_runtimes_fn(namespace)

    @mcp.tool()
    def get_runtime(name: str, namespace: Optional[str] = None) -> dict:
        """Get details of a specific training runtime."""
        return get_runtime_fn(name, namespace)

    @mcp.tool()
    def get_runtime_packages(name: str, namespace: Optional[str] = None) -> dict:
        """Get packages available in a specific training runtime."""
        return get_runtime_packages_fn(name, namespace)
