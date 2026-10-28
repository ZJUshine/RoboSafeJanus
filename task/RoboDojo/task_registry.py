"""Dynamic task loader for RoboDojo."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def is_group_module(module_name: str) -> bool:
    """Task-group modules (``heat_X_microwave``) hold the Common class shared by a group's tasks; they are not tasks.
    Like the group names in ``task/robosafejanus_tasks.json``, exactly one ``_``-separated segment is ``X``."""
    return module_name.split("_").count("X") == 1


def load_task_class(task_name: str):
    module_name = f"{__package__}.tasks.{task_name}"
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"Could not resolve task {task_name!r}")
    task_module = importlib.import_module(module_name)
    if not hasattr(task_module, task_name):
        raise ModuleNotFoundError(f"Could not resolve task class {task_name!r}")
    return task_name, getattr(task_module, task_name)


def task_config_path(config_dir: str | Path, task_name: str) -> Path:
    return Path(config_dir) / f"{task_name}.yml"
