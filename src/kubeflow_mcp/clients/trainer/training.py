# src/kubeflow_mcp/clients/trainer/training.py
import ast, tempfile, importlib.util, hashlib, os
from typing import Optional
from mcp.server.fastmcp import FastMCP
from kubeflow_mcp.core.auth import create_trainer_client
from kubeflow.trainer import (
    TrainerClient, BuiltinTrainer, CustomTrainer, CustomTrainerContainer,
    TorchTuneConfig, LoraConfig, Initializer,
    HuggingFaceModelInitializer, HuggingFaceDatasetInitializer,
)

# ── standalone logic functions (MCP tools delegate to these; tests call them directly) ──

def fine_tune_fn(
    model: str,
    dataset: str,
    peft_method: str = "lora",
    epochs: int = 3,
    batch_size: int = 4,
    num_nodes: int = 1,
    resources_per_node: Optional[dict] = None,
    runtime: Optional[str] = None,
    namespace: Optional[str] = None,
    confirmed: bool = False,
) -> dict:
    if not confirmed:
        return {
            "status": "preview",
            "message": "Review config and call again with confirmed=True to submit",
            "config": {
                "model": model, "dataset": dataset,
                "peft_method": peft_method, "epochs": epochs,
                "batch_size": batch_size, "num_nodes": num_nodes,
                "resources_per_node": resources_per_node, "runtime": runtime,
            }
        }
    try:
        client = create_trainer_client(namespace)
        trainer = BuiltinTrainer(
            config=TorchTuneConfig(
                epochs=epochs, batch_size=batch_size, num_nodes=num_nodes,
                resources_per_node=resources_per_node,
                peft_config=LoraConfig() if peft_method == "lora" else None,
            )
        )
        initializer = Initializer(
            model=HuggingFaceModelInitializer(storage_uri=f"hf://{model}"),
            dataset=HuggingFaceDatasetInitializer(storage_uri=f"hf://{dataset}"),
        )
        job_name = client.train(runtime=runtime, trainer=trainer, initializer=initializer)
        return {"success": True, "job_id": job_name, "trainer_type": "BuiltinTrainer"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_custom_training_fn(
    func_code: str,
    func_args: Optional[dict] = None,
    packages_to_install: Optional[list] = None,
    num_nodes: int = 1,
    resources_per_node: Optional[dict] = None,
    runtime: Optional[str] = None,
    namespace: Optional[str] = None,
    confirmed: bool = False,
) -> dict:
    if not confirmed:
        return {
            "status": "preview",
            "message": "Set confirmed=True to submit",
            "config": {"func_args": func_args, "num_nodes": num_nodes,
                       "packages_to_install": packages_to_install},
        }
    try:
        tree = ast.parse(func_code)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error: {e}"}

    denied_imports = {"os", "subprocess", "sys", "shutil", "socket"}
    denied_calls = {"eval", "exec", "compile", "__import__", "open"}
    func_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_names.add(node.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in denied_imports:
                    return {"success": False, "error": f"Import '{alias.name}' not allowed. Use run_container_training() instead."}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in denied_calls:
                return {"success": False, "error": f"'{node.func.id}()' not allowed."}

    if "train" not in func_names:
        return {"success": False, "error": "func_code must define a train(**kwargs) function"}

    script_hash = hashlib.md5(func_code.encode()).hexdigest()[:8]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py',
            prefix=f'mcp_train_{script_hash}_', delete=False
        ) as f:
            f.write(func_code)
            temp_path = f.name

        spec = importlib.util.spec_from_file_location("training_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train_func = module.train

        client = create_trainer_client(namespace)
        trainer = CustomTrainer(
            func=train_func, func_args=func_args,
            packages_to_install=packages_to_install,
            num_nodes=num_nodes, resources_per_node=resources_per_node,
        )
        job_name = client.train(runtime=runtime, trainer=trainer)
        return {"success": True, "job_id": job_name, "trainer_type": "CustomTrainer"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def run_container_training_fn(
    image: str,
    num_nodes: int = 1,
    resources_per_node: Optional[dict] = None,
    env: Optional[dict] = None,
    runtime: Optional[str] = None,
    namespace: Optional[str] = None,
    confirmed: bool = False,
) -> dict:
    if not confirmed:
        return {
            "status": "preview",
            "message": "Set confirmed=True to submit",
            "config": {"image": image, "num_nodes": num_nodes, "env": env},
        }
    try:
        client = create_trainer_client(namespace)
        trainer = CustomTrainerContainer(
            image=image, num_nodes=num_nodes,
            resources_per_node=resources_per_node, env=env,
        )
        job_name = client.train(runtime=runtime, trainer=trainer)
        return {"success": True, "job_id": job_name, "trainer_type": "CustomTrainerContainer"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── MCP tool registration (delegates to _fn functions above) ──

def register_training_tools(mcp: FastMCP):

    @mcp.tool()
    def fine_tune(
        model: str,
        dataset: str,
        peft_method: str = "lora",
        epochs: int = 3,
        batch_size: int = 4,
        num_nodes: int = 1,
        resources_per_node: Optional[dict] = None,
        runtime: Optional[str] = None,
        namespace: Optional[str] = None,
        confirmed: bool = False,
    ) -> dict:
        """Fine-tune an LLM using BuiltinTrainer with TorchTune.
        Use confirmed=False (default) to preview config before submitting.
        Use confirmed=True only after user approves the preview.
        Internally calls: TrainerClient.train(trainer=BuiltinTrainer(...))
        """
        return fine_tune_fn(model, dataset, peft_method, epochs, batch_size,
                            num_nodes, resources_per_node, runtime, namespace, confirmed)

    @mcp.tool()
    def run_custom_training(
        func_code: str,
        func_args: Optional[dict] = None,
        packages_to_install: Optional[list] = None,
        num_nodes: int = 1,
        resources_per_node: Optional[dict] = None,
        runtime: Optional[str] = None,
        namespace: Optional[str] = None,
        confirmed: bool = False,
    ) -> dict:
        """Run distributed training with user-provided Python code.
        func_code must define a train(**kwargs) function.
        MCP Bridge: func_code (str) is converted to Callable via importlib (file-backed).
        """
        return run_custom_training_fn(func_code, func_args, packages_to_install,
                                      num_nodes, resources_per_node, runtime,
                                      namespace, confirmed)

    @mcp.tool()
    def run_container_training(
        image: str,
        num_nodes: int = 1,
        resources_per_node: Optional[dict] = None,
        env: Optional[dict] = None,
        runtime: Optional[str] = None,
        namespace: Optional[str] = None,
        confirmed: bool = False,
    ) -> dict:
        """Run training with a pre-built container image."""
        return run_container_training_fn(image, num_nodes, resources_per_node,
                                         env, runtime, namespace, confirmed)
