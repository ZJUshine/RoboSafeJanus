"""S1 执行测试变体(``S1.1`` … ``S1.5``,见 task/Task_Design.md「S1 任务设计」)的评测约定。

- 变体由环境变量 ``ROBODOJO_EVAL_VARIANT`` 传进评测(``robodojo.sh eval|client --variant S1.1`` 会设置它);
  不设或 ``S1.0`` 就是原任务,评测和以前完全一样。
- 布局:``Assets/Eval_Layout/<benchmark>/<config>/<seed>/<task>__<variant>_<i>.json``,``i`` 是派生它的 S1.0
  布局编号(``eval_layouts.py variant`` 生成;放不下障碍物的编号跳过),所以 layout_id 和 S1.0 逐条对应。
- 结果:``additional_info`` 末尾加 ``,variant=<variant>``,结果目录和 S1.0 分开。
- 安全检查:``task/robosafejanus_sim/<group>.json`` 里 ``S1.<x>.safety`` 列出的全部检查(不看 ``impl``),
  每一步检查一次,任一触发就记这一条违规(不中断 episode)。
"""

import os
import re

VARIANT_ENV = "ROBODOJO_EVAL_VARIANT"
VARIANT_RE = re.compile(r"S1\.[0-5]")


def current_variant() -> str:
    """当前评测的变体;原任务(没设 / ``S1.0``)返回空串。"""
    value = os.environ.get(VARIANT_ENV, "").strip()
    if not value or value == "S1.0":
        return ""
    if not VARIANT_RE.fullmatch(value):
        raise ValueError(f"{VARIANT_ENV}={value!r}: expected S1.0 … S1.5")
    return value


def layout_name(task: str, variant: str) -> str:
    """评测布局文件名前缀:原任务是任务名,变体是 ``<task>__<variant>``。"""
    return f"{task}__{variant}" if variant else task


def variant_suffix(variant: str) -> str:
    """接在 ``additional_info`` 后面的部分(结果目录名)。"""
    return f",variant={variant}" if variant else ""


def task_spec(task: str) -> dict:
    """``load_tasks()`` 里这个任务的设计 + 仿真细节;找不到返回空字典。"""
    from src.data_collect.benchmark_layout import load_tasks

    for levels in load_tasks().values():
        for tasks in levels.values():
            if task in tasks:
                return tasks[task]
    return {}


def safety_checks(task: str, variant: str) -> list[tuple[str, dict]]:
    """这个变体 ``safety`` 里列出的全部安全检查 ``[(fn, args)]``,``impl`` 字段不管。

    设计里这个变体为 null、或 ``safety`` 为空时报错:评出来的违规率会是假的 0。
    列出的 ``fn`` 代码里没有时,评测第一步检查就会报 ``Unknown func_name``。"""
    spec = (task_spec(task).get("S1") or {}).get(variant)
    if not spec:
        raise ValueError(f"{task}: {variant} is null or missing in task/robosafejanus_sim")
    checks = [(c["fn"], dict(c.get("args") or {})) for c in spec.get("safety") or [] if c and c.get("fn")]
    if not checks:
        raise ValueError(f"{task}: {variant} has no safety check in task/robosafejanus_sim")
    return checks
