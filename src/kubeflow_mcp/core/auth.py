# src/kubeflow_mcp/core/auth.py
from typing import Optional
from kubeflow.trainer import TrainerClient
try:
    from kubeflow.trainer import KubernetesBackendConfig
except ImportError:
    KubernetesBackendConfig = None

def create_trainer_client(namespace: Optional[str] = None) -> TrainerClient:
    """
    Create a TrainerClient with optional namespace.
    Must be instantiated per-request because TrainerClient does not
    accept namespace as a per-method argument (Phase 3 multi-tenancy note).
    """
    if namespace and KubernetesBackendConfig:
        backend = KubernetesBackendConfig(namespace=namespace)
        return TrainerClient(backend_config=backend)
    return TrainerClient()