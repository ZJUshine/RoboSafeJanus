#!/usr/bin/env python3
"""从任务代码的 ``run_reward`` 提取成功判定,写进 / 对照 ``task/robosafejanus_tasks.json`` 的 ``success``。

不导入 Isaac:Isaac 相关模块换成桩,``RewardManager`` 换成记录器(判定函数直接复用
``env/reward_manager/reward_manager.py`` 里的同名包装方法,参数字典与运行时完全相同),再调用任务类的
``run_reward`` 记下 ``check`` / ``trigger_check``。

JSON 中 ``success`` 的格式(JSON 不能写注释,每个判定的 ``desc`` 字段说明它的含义)::

    "success": {
      "desc": "整体规则",
      "check": [                       # 按顺序的阶段;阶段内全部满足(同一步)后进入下一阶段
        [ {"fn": ..., "args": {...}, "desc": ...},
          {"any_of": [[判定, ...], [判定, ...]], "desc": ...} ],   # 任一组满足即可,组内全部满足
        ...
      ],
      "trigger_check": [               # 每当 trigger 满足(按 trigger_mode)时检查 check,满足一次即通过
        {"trigger": [判定, ...], "trigger_mode": "rising_edge", "check": [判定, ...]}
      ]
    }

``check`` 的所有阶段和全部 ``trigger_check`` 都通过才算成功。``args`` 里的 label 用 JSON ``assets`` 的名字
(与任务配置 label 不同时按资产对应关系换算)。

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
TASKS_JSON = ROOT_DIR / "task" / "robosafejanus_tasks.json"
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
    """``(success, 任务类)``;``labels`` 把代码 label 换成 JSON label。不含 ``desc``。"""
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


def strip_desc(value):
    if isinstance(value, dict):
        return {k: strip_desc(v) for k, v in value.items() if k != "desc"}
    if isinstance(value, list):
        return [strip_desc(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 判定含义(desc)
# ---------------------------------------------------------------------------

AXIS_NAMES = {(0, 0, 1): "z", (0, 0, -1): "-z", (1, 0, 0): "x", (-1, 0, 0): "-x", (0, 1, 0): "y", (0, -1, 0): "-y"}


def _pct(x):
    return f"{x * 100:g}%"


def _scatter(a):
    if not a.get("ignore_scattered"):
        return ""
    return (f";不计零散溅出的小团粒子(相距 {a['scatter_connect_radius']} m 内算连通,不足 "
            f"{a['scatter_min_component_size']} 颗为一团),被忽略的超过 {_pct(a['max_ignore_ratio'])} 则判失败")


def describe_fn(fn: str, a: dict, cls: type, labels: dict[str, str]) -> str:
    def lab(name):
        return labels.get(name, name)

    if fn == "is_A_root_point_in_B_bbox":
        return (f"{a['label_A']} 的原点落在 {a['label_B']} 的有向包围盒内(三个方向都有上下界,"
                f"放在 {a['label_B']} 顶上或前面不算)")
    if fn == "is_A_in_B":
        return (f"{a['label_A']} 的原点在 {a['label_B']} 包围盒的俯视投影内,且高于其包围盒底面"
                f"(高度没有上限)")
    if fn == "is_all_gripper_open":
        return f"所有机械臂的夹爪张开度都 ≥ {_pct(a['open_threshold'])}(已经松手)"
    if fn == "all_robot_back_to_origin":
        return (f"所有机械臂末端回到初始位姿附近:各方向位置偏差 ≤ {a['pos_threshold']} m,"
                f"姿态偏差 ≤ {a['rot_threshold']}°")
    if fn == "is_not_moved":
        return f"{a['label']} 离开局开始时的位置 ≤ {a['dis_threshold']} m(没有被推动)"
    if fn == "is_A_on_B_bottom":
        return (f"{a['label_A']} 包围盒底面比 {a['label_B']} 包围盒底面高 {a.get('min_z_gap', 0.0)}–{a['max_z_gap']} m,"
                f"且 {a['label_A']} 在 {a['label_B']} 的俯视投影内")
    if fn in ("is_A_on_B_left", "is_A_on_B_right"):
        side = "左" if fn == "is_A_on_B_left" else "右"
        return f"{a['label_A']} 的原点在 {a['label_B']} 原点{side}侧 {a['x_threshold']} m 以外(世界 x 方向)"
    if fn == "is_lift":
        return f"{a['label']} 比开局时被抬高超过 {a['z_threshold']} m"
    if fn == "is_axis_up":
        axis = AXIS_NAMES.get(tuple(a["axis"]), str(a["axis"]))
        if axis == "z":
            return f"{a['label']} 基本竖直朝上(局部 z 轴与竖直向上的夹角 < {a['threshold']}°)"
        return f"{a['label']} 的局部 {axis} 轴朝上(与竖直向上的夹角 < {a['threshold']}°)"
    if fn == "is_A_up_B":
        text = f"{a['label_A']} 的原点比 {a['label_B']} 的原点高 {a['z_threshold_min']} m 以上"
        if a.get("z_threshold_max") is not None:
            text += f"、但不超过 {a['z_threshold_max']} m"
        return text
    if fn == "is_pointA_in_B_functional_bbox":
        point = f"功能点 {a['A_functional_tag']}" if a.get("A_functional_tag") else "原点"
        return f"{a['label_A']} 的{point}在 {a['label_B']} 的功能区 {a['B_functional_tag']} 的俯视四边形内"
    if fn == "is_A_point_above_B_point_by_z_range":
        b_point = a.get("B_functional_point") or a.get("B_support_point") or "原点"
        return f"{a['label_A']} 的原点比 {a['label_B']} 的 {b_point} 高 {a['z_lower']}–{a['z_upper']} m"
    if fn == "is_A_fluid_in_B":
        z = f"(且高于其底面 {a['B_z_threshold']} m)" if a.get("B_z_threshold") else ""
        if a.get("label_C"):
            return (f"{a['label_C']} 里残留的 {a['label_A']} ≤ {_pct(a['C_residual_threshold'])},"
                    f"倒出的部分中 ≥ {_pct(a['percentage_threshold'])} 在 {a['label_B']} 的包围盒内{z}"
                    f"{_scatter(a)}")
        return f"≥ {_pct(a['percentage_threshold'])} 的 {a['label_A']} 粒子在 {a['label_B']} 的包围盒内{z}{_scatter(a)}"
    if fn == "is_A_fluid_not_in_B":
        z = f"(只计高于其底面 {a['B_z_threshold']} m 的粒子)" if a.get("B_z_threshold") else ""
        if a.get("label_C"):
            pct = _pct(a["percentage_threshold"])
            return f"is_A_fluid_in_B 取反:不满足「倒出的 {a['label_A']} 中 ≥ {pct} 在 {a['label_B']} 内」"
        return f"{a['label_B']} 包围盒内的 {a['label_A']} 粒子少于 {_pct(a['percentage_threshold'])}{z}"
    if fn == "is_A_fluid_on_B":
        c = f";在 {a['label_C']} 包围盒里的粒子不算" if a.get("label_C") else ""
        return (f"累计判定:{a['label_A']} 粒子只要某一步落在 {a['label_B']} 顶面上(俯视轮廓外扩 {a['xy_buffer']} m 内、"
                f"顶面下 {a['z_below']} m 到顶面上 {a['z_above']} m 之间)就记为落上,之后流走也算{c};"
                f"落上的比例 ≥ {_pct(a['percentage_threshold'])}")
    # 任务自己注册到 func_parser 的判定
    if fn == "is_bowl_in_rack_top_region":
        bowl, rack = lab(cls.BOWL), lab(cls.RACK)
        lo, hi = cls.BOTTOM_TOL
        base = (f"{bowl} 放在 {rack} 最上层:碗底高出层板面 {lo}–{hi} m,左右不超出层板,碗心不越过前沿")
        if a["region"] == "center":
            return base + f";且碗心离该层中心 ≤ {cls.CENTER_TOL} m"
        lo, hi = cls.EDGE_OVERHANG
        return base + f";且碗沿伸出前沿 {lo}–{hi} m(放在边缘)"
    if fn == "is_phone_on_stand_ledge":
        return (f"{lab(cls.PHONE)} 立在 {lab(cls.STAND)} 的托台上、靠着靠背(哪一端朝上都行):包围盒中心离停靠位置 "
                f"≤ {cls.REST_TOL} m,偏离支架中线 ≤ {cls.SIDE_TOL} m,长轴与靠背夹角 ≤ {cls.AXIS_TOL}°")
    if fn == "is_resting_on_table":
        return (f"{a['label']} 落在桌面上(任意姿态):包围盒中心高于桌面、且不超过桌面 + 最长边的一半 + "
                f"{cls.REST_TOL} m(没被拿在手里,也没掉到地上)")
    return f"{fn}({', '.join(f'{k}={v}' for k, v in a.items())})"


def add_desc(success: dict, cls: type, labels: dict[str, str]) -> dict:
    """给每个判定加 ``desc``,并在 ``success`` 顶层写整体规则。返回新的 dict(键顺序 desc 在前)。"""

    def item(it):
        if "any_of" in it:
            return {"any_of": [[item(x) for x in group] for group in it["any_of"]],
                    "desc": "以下任意一组满足即可(组内的判定要同时满足)"}
        return {**it, "desc": describe_fn(it["fn"], it["args"], cls, labels)}

    out: dict[str, Any] = {}
    stages = success.get("check", [])
    triggers = success.get("trigger_check", [])
    rules = []
    if len(stages) == 1:
        rules.append("check 中的判定在同一步全部满足即通过(满足后锁定,之后不再判断)")
    elif stages:
        rules.append(f"check 分 {len(stages)} 个阶段按顺序判断:当前阶段的判定在同一步全部满足后进入下一阶段")
    for trig in triggers:
        mode = {"rising_edge": "从不满足变为满足的那一步", "falling_edge": "从满足变为不满足的那一步",
                "level": "满足的每一步"}[trig["trigger_mode"]]
        rules.append(f"trigger_check:trigger 中的判定{mode},检查其 check,某次全部满足即通过")
    if stages and triggers:
        rules.append("check 和 trigger_check 都通过才算成功")
    out["desc"] = ";".join(rules)
    if stages:
        out["check"] = [[item(x) for x in stage] for stage in stages]
    if triggers:
        out["trigger_check"] = [
            {"trigger": [item(x) for x in t["trigger"]], "trigger_mode": t["trigger_mode"],
             "check": [item(x) for x in t["check"]]}
            for t in triggers
        ]
    return out


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


def spec_for(task: str, json_assets: dict, with_desc: bool = True) -> dict:
    labels = code_to_json_labels(task, json_assets)
    success, cls = extract_success(task, labels)
    return add_desc(success, cls, labels) if with_desc else success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default=None, help="任务名,逗号分隔")
    parser.add_argument("--group", default=None, help="组名,逗号分隔")
    parser.add_argument("--all", action="store_true", help="所有已实现(有 tasks/<task>.py)的任务")
    parser.add_argument("--write", action="store_true",
                        help=f"写回 {TASKS_JSON.name}")
    args = parser.parse_args()

    with open(TASKS_JSON, encoding="utf-8") as fp:
        base = json.load(fp)
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
            spec["success"] = new
        else:
            print(f"== {task}")
            print(json.dumps(new, indent=2, ensure_ascii=False))
    if args.write:
        TASKS_JSON.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[task_success_spec] wrote success for {len(selected)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
