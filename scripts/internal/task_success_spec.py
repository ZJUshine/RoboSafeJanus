#!/usr/bin/env python3
"""从任务代码的 ``run_reward`` 提取成功判定,写进 / 对照 ``task/robosafejanus_sim/<group>.json`` 中各任务的 ``success``。

不导入 Isaac:Isaac 相关模块换成桩,``RewardManager`` 换成记录器(判定函数直接复用
``env/reward_manager/reward_manager.py`` 里的同名包装方法,参数字典与运行时完全相同),再调用任务类的
``run_reward`` 记下 ``check`` / ``trigger_check``。

JSON 中 ``success`` 只存判定本身(``fn`` / ``args``),不含注释和说明::

    "success": {
      "check": [                       # 按顺序的阶段;阶段内全部满足(同一步)后进入下一阶段
        [ {"fn": ..., "args": {...}},
          {"any_of": [[判定, ...], [判定, ...]]} ],   # 任一组满足即可,组内全部满足
        ...
      ],
      "trigger_check": [               # 每当 trigger 满足(按 trigger_mode)时检查 check,满足一次即通过
        {"trigger": [判定, ...], "trigger_mode": "rising_edge", "check": [判定, ...]}
      ]
    }

``check`` 的所有阶段和全部 ``trigger_check`` 都通过才算成功。``args`` 里的 label 用 JSON ``assets`` 的名字
(与任务配置 label 不同时按资产对应关系换算)。各判定的含义和整体规则写在 ``docs/task_success_specs.md``。

用法::

    python scripts/internal/task_success_spec.py --task heat_egg_microwave        # 打印提取结果
    python scripts/internal/task_success_spec.py --all --write                   # 写回所有已实现任务
    python scripts/internal/task_success_spec.py --group heat_X_microwave --write
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
REWARD_MANAGER_PY = ROOT_DIR / "env" / "reward_manager" / "reward_manager.py"
TASKS_JSON = ROOT_DIR / "task" / "robosafejanus_sim"  # success 写在 <group>.json 的各任务下
LABEL_ARGS = ("label", "label_A", "label_B", "label_C")
LEVELS = ("safe_tasks", "risky_tasks")


# ---------------------------------------------------------------------------
# 记录器:复用 RewardManager 的判定包装方法
# ---------------------------------------------------------------------------


def _wrapper_methods() -> dict[str, Any]:
    """reward_manager.py 中形如 ``return ("<name>", args)`` 的判定包装方法,编译成普通函数。"""
    tree = ast.parse(REWARD_MANAGER_PY.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardManager")
    methods = {}
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        ret = node.body[-1]
        if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 2):
            continue
        first = ret.value.elts[0]
        if not (isinstance(first, ast.Constant) and first.value == node.name):
            continue
        module = ast.Module(body=[node], type_ignores=[])
        namespace: dict[str, Any] = {}
        exec(compile(module, str(REWARD_MANAGER_PY), "exec"), namespace)  # noqa: S102 - 仓库内的源码
        methods[node.name] = namespace[node.name]
    return methods


class Recorder:
    """替代 RewardManager:判定方法返回与运行时相同的 (name, args),check / trigger_check 记录调用。"""

    def __init__(self, methods):
        self.stages: list[list] = []
        self.triggers: list[tuple] = []
        for name, fn in methods.items():
            setattr(self, name, types.MethodType(fn, self))

    def check(self, check_list):
        self.stages.append(check_list)

    def trigger_check(self, condition_check_list, check_list, trigger_mode="level"):
        self.triggers.append((condition_check_list, check_list, trigger_mode))


class _Stub(types.ModuleType):
    """任意属性都返回一个可调用 / 可继承的占位对象。"""

    __all__: list[str] = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


@contextmanager
def _stubbed_imports():
    stub_names = ["pxr", "env.environment", "env.environment.task_env", "env.reward_manager",
                  "env.reward_manager.reward_manager", "utils.transformer"]
    for optional in ("numpy", "transforms3d"):
        try:
            importlib.import_module(optional)
        except ImportError:
            stub_names.append(optional)
    saved = {name: sys.modules.get(name) for name in stub_names}
    before = set(sys.modules)
    for name in stub_names:
        sys.modules[name] = _Stub(name)
    try:
        yield
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("task."):
                del sys.modules[name]
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def run_reward_calls(task: str, benchmark: str = "RoboDojo") -> tuple[Recorder, type]:
    """调用任务类的 ``run_reward``,返回记录器和任务类。"""
    methods = _wrapper_methods()
    with _stubbed_imports():
        module = importlib.import_module(f"task.{benchmark}.tasks.{task}")
        cls = getattr(module, task)
        obj = object.__new__(cls)
        obj.num_envs = 1
        recorder = Recorder(methods)
        obj.reward_manager = recorder
        cls.run_reward(obj)
    return recorder, cls


# ---------------------------------------------------------------------------
# 转成 JSON 结构
# ---------------------------------------------------------------------------


def _item(func, labels) -> dict:
    name, args = func
    args = {k: (labels.get(v, v) if k in LABEL_ARGS else v) for k, v in args.items() if v is not None}
    return {"fn": name, "args": args}


def _or_group(checks, labels) -> dict:
    """check_once 的语义:阶段里的 list 是 OR,OR 里的 list 是 AND。"""
    groups = []
    for sub in checks:
        groups.append(_and_group(sub, labels) if isinstance(sub, list) else [_item(sub, labels)])
    return {"any_of": groups}


def _and_group(checks, labels) -> list:
    return [_or_group(c, labels) if isinstance(c, list) else _item(c, labels) for c in checks]


def extract_success(task: str, labels: dict[str, str] | None = None) -> tuple[dict, type]:
    """``(success, 任务类)``;``labels`` 把代码 label 换成 JSON label。"""
    labels = labels or {}
    recorder, cls = run_reward_calls(task)
    success: dict[str, Any] = {}
    if recorder.stages:
        success["check"] = [_and_group(stage, labels) for stage in recorder.stages]
    if recorder.triggers:
        success["trigger_check"] = [
            {"trigger": _and_group(cond, labels), "trigger_mode": mode, "check": _and_group(checks, labels)}
            for cond, checks, mode in recorder.triggers
        ]
    return success, cls


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def code_to_json_labels(task: str, json_assets: dict) -> dict[str, str]:
    """任务配置 label → JSON label(同名优先,其余按资产相同配对)。"""
    sys.path.insert(0, str(Path(__file__).parent))
    from check_task_alignment import TASK_ROOT, config_assets, pair_labels, parse_asset

    cfg = config_assets(TASK_ROOT / "config" / f"{task}.yml")
    parsed = {label: parse_asset(a["asset"]) for label, a in json_assets.items()}
    pairs, _, _ = pair_labels(parsed, cfg)
    return {code: js for js, code in pairs.items()}


def spec_for(task: str, json_assets: dict) -> dict:
    labels = code_to_json_labels(task, json_assets)
    success, _ = extract_success(task, labels)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="任务名,逗号分隔")
    parser.add_argument("--group", default=None, help="组名,逗号分隔")
    parser.add_argument("--all", action="store_true", help="所有已实现(有 tasks/<task>.py)的任务")
    parser.add_argument("--write", action="store_true",
                        help=f"写回 {TASKS_JSON.name}")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT_DIR))
    from src.data_collect.benchmark_layout import load_tasks, save_success

    base = load_tasks()
    where = {t: (g, lv) for g, levels in base.items() for lv in LEVELS for t in levels.get(lv, {})}
    implemented = {t for t in where if (ROOT_DIR / "task" / "RoboDojo" / "tasks" / f"{t}.py").exists()}
    names = {t.strip() for t in (args.task or "").split(",") if t.strip()}
    groups = {g.strip() for g in (args.group or "").split(",") if g.strip()}
    selected = sorted(t for t in implemented if args.all or t in names or where[t][0] in groups)
    missing = sorted((names - implemented) | (groups - set(base)))
    if missing:
        parser.error(f"not implemented or not in {TASKS_JSON.name}: {', '.join(missing)}")
    if not selected:
        parser.error("give --task, --group or --all")

    for task in selected:
        group, level = where[task]
        spec = base[group][level][task]
        new = spec_for(task, spec["assets"])
        if args.write:
            save_success(task, new)
        else:
            print(f"== {task}")
            print(json.dumps(new, indent=2, ensure_ascii=False))
    if args.write:
        print(f"[task_success_spec] wrote success for {len(selected)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
