#!/usr/bin/env python3
"""检查任务代码、采集数据、上传仓库等是否与 ``task/robosafejanus_tasks.json`` 对齐。

``robosafejanus_tasks.json`` 决定 ``<group>/<safe_tasks|risky_tasks>/<task>`` 的数据布局
(``src/data_collect/benchmark_layout.py``),也记录每个任务的指令、资产和成功判定。本脚本只读,
不导入 Isaac,逐项对照::

    json        JSON 自身:任务名唯一、资产路径存在、成功判定引用的 label 都在 assets 中、expanded 不含基础版的组 / 任务
    code        task/<BENCHMARK>/{tasks,config,experts}/<task> 是否齐全;task/RoboSafeJanus 镜像是否一致;
                代码里的指令 / reward 函数与 JSON 是否一致
    config      任务 YAML 的 label 与资产(类别 / 编号)是否与 JSON 的 assets 一致
    inventory   scripts/internal/task_inventory.py 的 safety 维度、task/<BENCHMARK>/config/_task.yml
    eval_layout Assets/Eval_Layout/<BENCHMARK>/<config>/<seed>/<task>_<i>.json 是否存在、资产是否一致
    data        data/RoboSafeJanus-Benchmark:目录是否在 JSON 规定的位置、seed / hdf5 / layout 数量、
                每个 episode 实际用到的资产是否与 JSON 一致(--deep 再读 hdf5 指令,需要 h5py)
    flat        data/RoboSafeJanus-Benchmark-flat 软链接视图
    lerobot     data/RoboSafeJanus-Benchmark-Lerobot/<v21|v30>/... 导出目录
    modelscope  远端 <repo>/RoboSafeJanus/<task>/ 与 JSON、本地数据逐文件(大小)是否一致(需要 modelscope_hub)
    hf          同上,Hugging Face(需要 huggingface_hub)

用法::

    python scripts/internal/check_task_alignment.py                       # 本地检查
    python scripts/internal/check_task_alignment.py --deep                # 另读 hdf5 指令(需 h5py)
    python scripts/internal/check_task_alignment.py --modelscope zjushine/RoboSafeJanus-Benchmark
    python scripts/internal/check_task_alignment.py --group heat_X_microwave --deep   # 只看确认过的组
    python scripts/internal/check_task_alignment.py --only data,modelscope --format json

有 ERROR 时退出码为 1;加 ``--strict`` 时有 WARN 也返回 1。
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import filecmp
import glob
import json
import os
import re
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from src.data_collect.benchmark_layout import (  # noqa: E402
    BENCHMARK_NAME,
    LEVELS,
    REMOTE_PREFIX,
    TASKS_JSON,
    task_dir,
    task_groups,
)

try:
    from env.global_configs import BENCHMARK  # noqa: E402
except ImportError:
    BENCHMARK = "RoboDojo"

import yaml  # noqa: E402

from task.RoboDojo.task_registry import is_group_module  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from task_success_spec import spec_for, strip_desc  # noqa: E402

EXPANDED_JSON = Path(TASKS_JSON).with_name(Path(TASKS_JSON).stem + "_expanded.json")
TASK_ROOT = ROOT_DIR / "task" / BENCHMARK
MIRROR_ROOT = ROOT_DIR / "task" / "RoboSafeJanus"
OBJECTS_DIR = ROOT_DIR / "Assets" / "Object" / BENCHMARK
EVAL_LAYOUT_DIR = ROOT_DIR / "Assets" / "Eval_Layout" / BENCHMARK
INVENTORY_PY = ROOT_DIR / "scripts" / "internal" / "task_inventory.py"
ASSET_SECTIONS = ("Rigid", "Articulation", "Geometry", "Fluid", "Garment", "Clutter", "Dynamic")
# 成功判定参数里引用 label 的键
LABEL_ARGS = ("label", "label_A", "label_B", "label_C")
LEROBOT_VERSIONS = ("v21", "v30")
FLAT_VIEW_CMD = "python src/data_collect/benchmark_layout.py --flat_view"
SECTIONS = ("json", "code", "config", "inventory", "eval_layout", "data", "flat", "lerobot", "modelscope", "hf")
LEVEL_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


class Report:
    def __init__(self):
        self.items: list[dict[str, str]] = []
        self.skipped: dict[str, str] = {}

    def add(self, level: str, section: str, msg: str, task: str = ""):
        self.items.append({"level": level, "section": section, "task": task, "msg": msg})

    def error(self, section, msg, task=""):
        self.add("ERROR", section, msg, task)

    def warn(self, section, msg, task=""):
        self.add("WARN", section, msg, task)

    def info(self, section, msg, task=""):
        self.add("INFO", section, msg, task)

    def count(self, level):
        return sum(1 for item in self.items if item["level"] == level)


# ---------------------------------------------------------------------------
# 资产路径:``Section/name[/index]``,多个候选用 `` | `` 连接
# ---------------------------------------------------------------------------


def parse_asset(value: str) -> set[tuple[str, str, int | None]]:
    paths = set()
    for part in value.split("|"):
        pieces = part.strip().split("/")
        index = int(pieces[2]) if len(pieces) > 2 and pieces[2].isdigit() else None
        paths.add((pieces[0], pieces[1] if len(pieces) > 1 else "", index))
    return paths


def format_assets(paths) -> str:
    return " | ".join(f"{s}/{n}" + ("" if i is None else f"/{i:05d}") for s, n, i in sorted(paths, key=str))


def same_assets(a, b) -> bool:
    """任一侧缺编号时只比较类别和名称。"""
    if any(i is None for *_, i in a) or any(i is None for *_, i in b):
        return {p[:2] for p in a} == {p[:2] for p in b}
    return a == b


def asset_allowed(path, allowed) -> bool:
    return any(path[:2] == p[:2] and (p[2] is None or path[2] == p[2]) for p in allowed)


def asset_exists(path) -> bool:
    section, name, index = path
    target = OBJECTS_DIR / section / name
    return (target / f"{index:05d}").is_dir() if index is not None else target.is_dir()


def iter_success_items(value, path=""):
    """遍历 ``success`` 中所有 ``{"fn", "args"}`` 判定,产出 (位置, 判定)。新旧两种写法都支持。"""
    if isinstance(value, dict):
        if "fn" in value:
            yield path, value
            return
        for key, sub in value.items():
            yield from iter_success_items(sub, f"{path}.{key}")
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            yield from iter_success_items(sub, f"{path}[{i}]")


def first_diff(actual, expected, path="success"):
    """两个 JSON 值第一处不同的位置与内容;相同返回 None。"""
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in list(expected) + [k for k in actual if k not in expected]:
            if key not in actual:
                return f"{path}.{key} 缺失(代码: {json.dumps(expected[key], ensure_ascii=False)[:120]})"
            if key not in expected:
                return f"{path}.{key} 在代码中没有"
            diff = first_diff(actual[key], expected[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path} 长度 {len(actual)},代码为 {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            diff = first_diff(a, e, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if actual != expected:
        return f"{path} 为 {json.dumps(actual, ensure_ascii=False)},代码为 {json.dumps(expected, ensure_ascii=False)}"
    return None


# ---------------------------------------------------------------------------
# 读取 JSON / YAML / 布局
# ---------------------------------------------------------------------------


def load_json_tasks(path) -> dict[str, dict[str, Any]]:
    """``{task: {"group", "level", "instruction", "assets": {label: set}, "success"}}``。"""
    with open(path, encoding="utf-8") as fp:
        groups = json.load(fp)
    tasks = {}
    for group, levels in groups.items():
        for level in LEVELS:
            for name, spec in levels.get(level, {}).items():
                tasks[name] = {
                    "group": group,
                    "level": level,
                    "instruction": spec.get("instruction"),
                    "raw_assets": spec.get("assets", {}),
                    "assets": {label: parse_asset(a["asset"]) for label, a in spec.get("assets", {}).items()},
                    "success": spec.get("success", {}),
                }
    return tasks


def config_assets(config_path: Path) -> dict[str, set]:
    """任务 YAML 中 ``{label: {(section, name, index)}}``;``index`` 省略时展开为资产目录下的全部编号。"""
    with open(config_path, encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    labels = {}
    for section, entries in cfg.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            paths = set()
            for category in entry.get("category") or []:
                name, index = category.get("name"), category.get("index")
                if index is None:
                    variants = sorted(p.name for p in (OBJECTS_DIR / section / name).glob("[0-9]*") if p.is_dir())
                    index = [int(v) for v in variants] or [None]
                paths |= {(section, name, i) for i in index}
            for label in (entry.get("select_mode") or {}).get("label") or []:
                labels.setdefault(label, set()).update(paths)
    return labels


def layout_assets(layout: dict) -> dict[str, set]:
    """采集 / 评测布局 JSON 中 ``{label: {(section, category, category_idx)}}``。"""
    labels = {}
    for section in ASSET_SECTIONS:
        for objs in (layout.get(section) or {}).values():
            if not isinstance(objs, list):
                continue
            for obj in objs:
                if isinstance(obj, dict) and obj.get("label"):
                    labels.setdefault(obj["label"], set()).add((section, obj.get("category"), obj.get("category_idx")))
    return labels


def pair_labels(json_assets: dict[str, set], other: dict[str, set]) -> tuple[dict[str, str], list[str], list[str]]:
    """把 JSON label 对到另一侧的 label:同名优先,其余按资产相同配对。返回(配对, JSON 多出, 另一侧多出)。"""
    pairs = {label: label for label in json_assets if label in other}
    rest_json = [label for label in json_assets if label not in pairs]
    rest_other = [label for label in other if label not in pairs.values()]
    for label in list(rest_json):
        match = [o for o in rest_other if same_assets(json_assets[label], other[o])]
        if len(match) == 1:
            pairs[label] = match[0]
            rest_json.remove(label)
            rest_other.remove(match[0])
    return pairs, rest_json, rest_other


# ---------------------------------------------------------------------------
# 静态解析任务代码(不导入 Isaac)
# ---------------------------------------------------------------------------


def string_constants(nodes) -> list[str]:
    return [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]


class TaskSource:
    """在任务模块及其从 ``task.<BENCHMARK>.tasks.*`` 导入的基类里查找指令与 reward 函数。"""

    def __init__(self):
        self._cache: dict[str, ast.Module | None] = {}

    def _module(self, name: str) -> ast.Module | None:
        if name not in self._cache:
            path = TASK_ROOT / "tasks" / f"{name}.py"
            self._cache[name] = ast.parse(path.read_text(encoding="utf-8")) if path.exists() else None
        return self._cache[name]

    def _classes(self, module_name: str, seen=None) -> list[ast.ClassDef]:
        """模块内的类,加上沿基类链能找到的其他任务模块里的类。"""
        seen = seen if seen is not None else set()
        if module_name in seen:
            return []
        seen.add(module_name)
        tree = self._module(module_name)
        if tree is None:
            return []
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        imported = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(f"task.{BENCHMARK}.tasks."):
                for alias in node.names:
                    imported[alias.asname or alias.name] = node.module.rsplit(".", 1)[1]
        for cls in list(classes):
            for base in cls.bases:
                if isinstance(base, ast.Name) and base.id in imported:
                    classes += self._classes(imported[base.id], seen)
        return classes

    def class_exists(self, task: str) -> bool:
        tree = self._module(task)
        return tree is not None and any(isinstance(n, ast.ClassDef) and n.name == task for n in tree.body)

    def instructions(self, task: str) -> list[str]:
        found = []
        for cls in self._classes(task):
            for node in cls.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "INSTRUCTION" for t in node.targets
                ):
                    found += string_constants(ast.walk(node.value))
                if isinstance(node, ast.FunctionDef) and node.name == "gen_instruction":
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.List):
                            found += string_constants(sub.elts)
            if found:
                break
        return found


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------


def check_json(report: Report, tasks: dict) -> None:
    sec = "json"
    try:
        task_groups()
    except ValueError as e:
        report.error(sec, str(e))
    for name, spec in tasks.items():
        # 组名是 <动作>_<交互物体>_<被交互物体>,其中恰好一段是 X(安全 / 危险任务之间变化的那一项),
        # 任务名 = 组名把 X 换成具体名字
        words = spec["group"].split("_")
        group = spec["group"]
        if words.count("X") != 1:
            report.error(sec, f"组名 {group} 中应恰好有一段 X", name)
        elif not re.fullmatch("_".join(r"[a-z0-9]+(?:_[a-z0-9]+)*" if w == "X" else re.escape(w) for w in words), name):
            report.error(sec, f"任务名与组名 {group} 不符(把 X 换成具体名字应得到任务名)", name)
        for label, paths in spec["assets"].items():
            for path in paths:
                if not asset_exists(path):
                    where = OBJECTS_DIR.relative_to(ROOT_DIR)
                    report.error(sec, f"资产 {label}: {format_assets([path])} 在 {where} 中不存在", name)
            if any(i is None for *_, i in paths):
                report.info(sec, f"资产 {label}: {format_assets(paths)} 未写编号", name)
        for path, check in iter_success_items(spec["success"]):
            for key in LABEL_ARGS:
                label = (check.get("args") or {}).get(key)
                if isinstance(label, str) and label not in spec["assets"]:
                    report.error(sec, f"success{path}.{check.get('fn')} 的 {key}={label!r} 不在 assets 中", name)
    if EXPANDED_JSON.exists():
        with open(TASKS_JSON, encoding="utf-8") as fp:
            base = json.load(fp)
        with open(EXPANDED_JSON, encoding="utf-8") as fp:
            expanded = json.load(fp)
        # expanded 只放基础版之外的扩展任务:组名和任务名都不能与基础版重复
        base_tasks = {t for levels in base.values() for lv in ("safe_tasks", "risky_tasks") for t in levels.get(lv, {})}
        for group, levels in expanded.items():
            if group in base:
                report.error(sec, f"组 {group} 已在基础版中,不应再出现在 {EXPANDED_JSON.name}", group)
            for lv in ("safe_tasks", "risky_tasks"):
                for task in levels.get(lv, {}):
                    if task in base_tasks:
                        report.error(sec, f"任务 {task} 已在基础版中,不应再出现在 {EXPANDED_JSON.name}", task)


def check_code(report: Report, tasks: dict, src: TaskSource) -> set[str]:
    """返回已实现(tasks/*.py 中有同名类且有 YAML)的 JSON 任务。"""
    sec = "code"
    implemented = set()
    not_started = []
    for name, spec in tasks.items():
        parts = {
            "tasks": TASK_ROOT / "tasks" / f"{name}.py",
            "config": TASK_ROOT / "config" / f"{name}.yml",
            "experts": TASK_ROOT / "experts" / f"{name}.py",
        }
        have = {k: p.exists() for k, p in parts.items()}
        if not any(have.values()):
            not_started.append(name)
            continue
        missing = [str(parts[k].relative_to(ROOT_DIR)) for k, ok in have.items() if not ok]
        if missing:
            report.error(sec, f"缺少 {', '.join(missing)}", name)
        if have["tasks"] and not src.class_exists(name):
            report.error(sec, f"tasks/{name}.py 中没有类 {name}", name)
        if have["tasks"] and have["config"]:
            implemented.add(name)
        # task/RoboSafeJanus 镜像
        for kind, path in parts.items():
            mirror = MIRROR_ROOT / kind / path.name
            if path.exists() and not mirror.exists():
                report.warn(sec, f"镜像缺少 {mirror.relative_to(ROOT_DIR)}", name)
            elif path.exists() and not filecmp.cmp(path, mirror, shallow=False):
                report.warn(sec, f"{mirror.relative_to(ROOT_DIR)} 与 {path.relative_to(ROOT_DIR)} 内容不同", name)
        if not have["tasks"]:
            continue
        # 指令
        found = src.instructions(name)
        if not found:
            report.info(sec, "无法从代码静态解析指令,跳过指令比对", name)
        elif spec["instruction"] not in found:
            report.warn(sec, f"指令不一致: JSON {spec['instruction']!r} vs 代码 {found}", name)
        # 成功判定:与 run_reward 实际调用逐项比较(不比 desc)
        if not have["config"]:
            continue
        try:
            expected = spec_for(name, spec["raw_assets"], with_desc=False)
        except Exception as e:  # noqa: BLE001 - 代码无法静态执行时只提示
            report.info(sec, f"无法从 run_reward 提取成功判定({type(e).__name__}: {e}),跳过比对", name)
            continue
        diff = first_diff(strip_desc(spec["success"]), expected)
        if diff:
            fix = f"python scripts/internal/task_success_spec.py --task {name} --write"
            report.warn(sec, f"success 与 run_reward 不一致: {diff};可用 {fix} 按代码重写", name)
    # 镜像里有、JSON 里没有的任务:benchmark_layout 会把它们的数据放在扁平目录
    groups = {spec["group"] for spec in tasks.values()}
    for path in sorted((MIRROR_ROOT / "tasks").glob("*.py")):
        rel = path.relative_to(ROOT_DIR)
        if is_group_module(path.stem):
            # 任务组公共模块(heat_X_microwave.py):不是任务,但组员通过 task.<BENCHMARK>.tasks.<组名> 导入它
            if path.stem not in groups:
                report.warn(sec, f"{rel} 不是 {Path(TASKS_JSON).name} 中的组名", path.stem)
            link = TASK_ROOT / "tasks" / path.name
            if not link.exists():
                report.error(sec, f"缺少 {link.relative_to(ROOT_DIR)},组员无法导入", path.stem)
            continue
        if path.stem not in tasks:
            report.error(sec, f"{rel} 不在 {Path(TASKS_JSON).name} 中,数据会落到扁平目录", path.stem)
    for path in sorted((MIRROR_ROOT / "config").glob("*.yml")):
        if path.stem not in tasks and not (MIRROR_ROOT / "tasks" / f"{path.stem}.py").exists():
            report.warn(sec, f"{path.relative_to(ROOT_DIR)} 没有对应的任务", path.stem)
    if not_started:
        report.info(sec, f"{len(not_started)} 个 JSON 任务尚无代码 / 配置 / 专家(未实现)")
    return implemented


def check_config(report: Report, tasks: dict, implemented: set[str]) -> None:
    sec = "config"
    for name in sorted(implemented):
        cfg = config_assets(TASK_ROOT / "config" / f"{name}.yml")
        json_assets = tasks[name]["assets"]
        pairs, only_json, only_cfg = pair_labels(json_assets, cfg)
        for j, c in pairs.items():
            if j != c:
                report.error(sec, f"label 名不一致: JSON {j!r} ↔ 配置 {c!r}(资产相同)", name)
            elif not same_assets(json_assets[j], cfg[c]):
                report.error(sec, f"{j}: JSON {format_assets(json_assets[j])} ↔ 配置 {format_assets(cfg[c])}", name)
        for label in only_json:
            report.error(sec, f"JSON label {label!r}({format_assets(json_assets[label])})在配置中不存在;"
                              f"配置 label: {sorted(cfg)}", name)
        for label in only_cfg:
            report.error(sec, f"配置 label {label!r}({format_assets(cfg[label])})不在 JSON assets 中", name)


def check_inventory(report: Report, tasks: dict, implemented: set[str]) -> None:
    sec = "inventory"
    tree = ast.parse(INVENTORY_PY.read_text(encoding="utf-8"))
    safety = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "DIMENSION_TASKS" and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == "safety":
                    safety = {e.value for e in value.elts}
    rel = INVENTORY_PY.relative_to(ROOT_DIR)
    if safety is None:
        report.warn(sec, f"{rel} 中找不到 DIMENSION_TASKS['safety']")
    else:
        for name in sorted(implemented - safety):
            report.warn(sec, f"已实现但不在 {rel} 的 safety 维度中", name)
        for name in sorted(safety - set(tasks)):
            report.error(sec, f"{rel} 的 safety 维度包含 JSON 中没有的任务", name)
        for name in sorted((safety & set(tasks)) - implemented):
            report.warn(sec, f"在 {rel} 的 safety 维度中,但任务代码 / 配置不全", name)
    task_yml = TASK_ROOT / "config" / "_task.yml"
    with open(task_yml, encoding="utf-8") as fp:
        overrides = (yaml.safe_load(fp) or {}).get("tasks") or {}
    for name in overrides:
        if not (TASK_ROOT / "tasks" / f"{name}.py").exists():
            report.warn(sec, f"{task_yml.relative_to(ROOT_DIR)} 中的条目没有对应的任务代码(改名后遗留?)", name)


def iter_layout_files(task: str, config_name: str):
    yield from sorted(glob.glob(str(EVAL_LAYOUT_DIR / config_name / "*" / f"{task}_[0-9]*.json")))


def compare_layouts(report: Report, sec: str, task: str, json_assets: dict, files, where: str) -> None:
    """逐个布局文件对照 JSON 资产,汇总后每个(label, 实际资产)报一条。"""
    wrong: Counter = Counter()
    missing_label: Counter = Counter()
    seen_labels: set[str] = set()
    for path in files:
        try:
            with open(path, encoding="utf-8") as fp:
                actual = layout_assets(json.load(fp))
        except (OSError, json.JSONDecodeError) as e:
            report.error(sec, f"无法读取 {path}: {e}", task)
            continue
        seen_labels |= set(actual)
        pairs, only_json, _ = pair_labels(json_assets, actual)
        for label in only_json:
            missing_label[label] += 1
        for j, a in pairs.items():
            for path_ in actual[a]:
                if not asset_allowed(path_, json_assets[j]):
                    wrong[(j, a, format_assets([path_]))] += 1
    total = len(files)
    for (j, a, got), n in sorted(wrong.items()):
        label = j if j == a else f"{j}(布局 label {a})"
        expected = format_assets(json_assets[j])
        report.error(sec, f"{where}: {n}/{total} 个 episode 的 {label} 用的是 {got},JSON 为 {expected}", task)
    for label, n in sorted(missing_label.items()):
        msg = f"{where}: {n}/{total} 个 episode 中找不到 JSON label {label!r}(布局 label: {sorted(seen_labels)})"
        report.error(sec, msg, task)


def check_eval_layout(report: Report, tasks: dict, implemented: set[str], config_name: str) -> None:
    sec = "eval_layout"
    if not EVAL_LAYOUT_DIR.is_dir():
        report.warn(sec, f"{EVAL_LAYOUT_DIR.relative_to(ROOT_DIR)} 不存在")
        return
    for name in sorted(implemented):
        files = list(iter_layout_files(name, config_name))
        if not files:
            report.warn(sec, f"{EVAL_LAYOUT_DIR.relative_to(ROOT_DIR)}/{config_name}/*/ 中没有 {name}_<i>.json", name)
            continue
        compare_layouts(report, sec, name, tasks[name]["assets"], files, "评测布局")


def task_entries(root: Path, report: Report, sec: str, tasks: dict):
    """遍历 ``<root>/<group>/<level>/<task>[_failed]``,报告不在 JSON 位置上的目录,产出 (task, failed, path)。"""
    if not root.is_dir():
        return
    groups = {spec["group"] for spec in tasks.values()}
    for top in sorted(root.iterdir()):
        if top.name.startswith(".") or not top.is_dir() or top.name == "_shards":
            continue
        base = top.name.removesuffix("_failed")
        if top.name not in groups:
            if base in tasks:
                report.error(sec, f"{top.relative_to(ROOT_DIR)} 是旧的扁平布局,应位于 "
                                  f"{tasks[base]['group']}/{tasks[base]['level']}/{top.name}", base)
            else:
                report.error(sec, f"{top.relative_to(ROOT_DIR)} 既不是 JSON 中的组也不是任务", top.name)
            continue
        for level_dir in sorted(p for p in top.iterdir() if p.is_dir()):
            if level_dir.name not in LEVELS:
                report.error(sec, f"{level_dir.relative_to(ROOT_DIR)}: 级别应为 {'/'.join(LEVELS)}")
                continue
            for tdir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                base = tdir.name.removesuffix("_failed")
                spec = tasks.get(base)
                if spec is None:
                    report.error(sec, f"{tdir.relative_to(ROOT_DIR)}: 任务不在 JSON 中(改名或删除后遗留?)", base)
                elif (spec["group"], spec["level"]) != (top.name, level_dir.name):
                    expected = f"{spec['group']}/{spec['level']}"
                    report.error(sec, f"{tdir.relative_to(ROOT_DIR)}: JSON 中应位于 {expected}", base)
                else:
                    yield base, tdir.name != base, tdir


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def check_data(
    report: Report, tasks: dict, implemented: set[str], data_dir: Path, deep: bool
) -> dict[str, dict[str, int]]:
    """返回 ``{<task>[_failed]: {任务目录内的相对路径: 字节数}}``,供远端对比。"""
    sec = "data"
    local_files: dict[str, dict[str, int]] = {}
    if not data_dir.is_dir():
        report.warn(sec, f"{data_dir} 不存在,跳过本地数据检查")
        return local_files
    h5py = None
    if deep:
        try:
            import h5py  # noqa: F401
        except ImportError:
            report.warn(sec, "--deep 需要 h5py(在 RoboDojo 环境中运行),跳过 hdf5 指令检查")
    shards = data_dir / "_shards"
    if shards.is_dir():
        for path in sorted(p for p in shards.iterdir() if p.is_dir()):
            report.warn(sec, f"{path.relative_to(ROOT_DIR)}: 未合并 / 未清理的并行采集分片", path.name)
    collected = set()
    for name, failed, tdir in task_entries(data_dir, report, sec, tasks):
        rel = str(tdir.relative_to(data_dir))
        local_files[tdir.name] = {
            str(path.relative_to(tdir)): path.stat().st_size for path in tdir.rglob("*") if path.is_file()
        }
        if failed:
            continue
        collected.add(name)
        if name not in implemented:
            report.warn(sec, f"{rel} 有数据,但任务代码 / 配置不全", name)
        for cdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            where = f"{rel}/{cdir.name}"
            n_h5 = count_files(cdir / "data", "*.hdf5")
            counts = {"hdf5": n_h5}
            if (cdir / "seed.txt").exists():
                counts["seed.txt"] = len((cdir / "seed.txt").read_text().split())
            if (cdir / "layout").is_dir():
                counts["layout"] = count_files(cdir / "layout", "*.json")
            if len(set(counts.values())) > 1:
                report.warn(sec, f"{where}: 数量不一致 {counts}", name)
            if n_h5 == 0:
                report.warn(sec, f"{where}: 没有 hdf5", name)
            layouts = sorted(glob.glob(str(cdir / "layout" / "*.json")))
            if layouts:
                compare_layouts(report, sec, name, tasks[name]["assets"], layouts, where)
            if h5py is not None and n_h5:
                check_hdf5_instruction(report, sec, name, tasks[name]["instruction"], cdir, where, h5py)
    for name in sorted(implemented - collected):
        report.info(sec, "已实现但还没有采集数据", name)
    return local_files


def check_hdf5_instruction(report, sec, task, instruction, cdir, where, h5py) -> None:
    mismatched = Counter()
    files = sorted((cdir / "data").glob("*.hdf5"))
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                value = f["instruction"][()]
        except (OSError, KeyError) as e:
            report.error(sec, f"{where}: 无法读取 {path.name} 的 instruction: {e}", task)
            continue
        value = value.decode() if isinstance(value, bytes) else str(value)
        if value != instruction:
            mismatched[value] += 1
    for value, n in mismatched.items():
        report.warn(sec, f"{where}: {n}/{len(files)} 个 hdf5 的指令为 {value!r},JSON 为 {instruction!r}", task)


def check_flat(report: Report, tasks: dict, data_dir: Path, flat_dir: Path) -> None:
    sec = "flat"
    if not flat_dir.is_dir():
        report.info(sec, f"{flat_dir} 不存在(export_lerobot.sh 会自动建)")
        return
    linked = set()
    for entry in sorted(flat_dir.iterdir()):
        base = entry.name.removesuffix("_failed")
        if not entry.is_symlink():
            report.warn(sec, f"{entry.name} 不是软链接")
            continue
        linked.add(entry.name)
        if base not in tasks:
            report.error(sec, f"{entry.name} 不在 JSON 中", base)
            continue
        expected = Path(task_dir(str(data_dir), base) + entry.name[len(base):])
        if not entry.exists():
            report.error(sec, f"{entry.name} 是断链 -> {os.readlink(entry)}", base)
        elif entry.resolve() != expected.resolve():
            report.error(sec, f"{entry.name} -> {os.readlink(entry)},应指向 {expected}", base)
    for name, _, tdir in task_entries(data_dir, Report(), sec, tasks):
        if tdir.name not in linked:
            report.warn(sec, f"{tdir.name} 有数据但没有软链接;运行 {FLAT_VIEW_CMD}", name)


def check_lerobot(report: Report, tasks: dict, data_dir: Path, lerobot_dir: Path) -> None:
    sec = "lerobot"
    if not lerobot_dir.is_dir():
        report.info(sec, f"{lerobot_dir} 不存在")
        return
    raw = {name for name, failed, _ in task_entries(data_dir, Report(), sec, tasks) if not failed}
    for top in sorted(p for p in lerobot_dir.iterdir() if p.is_dir()):
        if top.name not in LEROBOT_VERSIONS:
            report.warn(sec, f"{top.relative_to(ROOT_DIR)}: 不是 {'/'.join(LEROBOT_VERSIONS)} 版本目录(旧导出遗留?)")
            continue
        for name, _, tdir in task_entries(top, report, sec, tasks):
            if name not in raw:
                report.warn(sec, f"{tdir.relative_to(ROOT_DIR)}: 原始数据中已没有该任务", name)


# ---------------------------------------------------------------------------
# 远端仓库
# ---------------------------------------------------------------------------


def list_modelscope(repo_id: str, endpoint: str | None) -> dict[str, int]:
    """``{远端路径: 字节数}``。"""
    from modelscope_hub import HubApi

    api = HubApi(endpoint=endpoint)
    if not api.repo_exists(repo_id, "dataset"):
        raise RuntimeError(f"dataset {repo_id} 不存在或无权限")
    return {f.path: f.size for f in api.list_repo_files(repo_id, "dataset", recursive=True) if not f.is_dir}


def list_hf(repo_id: str, endpoint: str | None) -> dict[str, int]:
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    tree = HfApi(endpoint=endpoint).list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return {f.path: f.size for f in tree if isinstance(f, RepoFile)}


def check_remote(
    report: Report, sec: str, tasks: dict, remote: dict[str, int], local_files: dict[str, dict[str, int]], prefix: str
) -> None:
    """远端为扁平布局 ``<prefix>/<task>[_failed]/<config>/...``,逐文件对照本地(按大小)。"""
    remote_tasks: dict[str, dict[str, int]] = defaultdict(dict)
    outside: Counter = Counter()
    for path, size in remote.items():
        parts = path.split("/")
        if parts[0] != prefix or len(parts) < 3:
            if len(parts) > 1:
                outside[parts[0]] += 1
            continue
        remote_tasks[parts[1]]["/".join(parts[2:])] = size
    for top, n in sorted(outside.items()):
        report.warn(sec, f"远端 {top}/ 下有 {n} 个文件,不是 {prefix}/<task>/ 格式")
    for name in sorted(set(remote_tasks) - set(local_files)):
        base = name.removesuffix("_failed")
        n = len(remote_tasks[name])
        if base not in tasks:
            report.error(sec, f"远端 {prefix}/{name}/({n} 个文件)不在 JSON 中(改名或删除前的旧任务?)", base)
        else:
            report.warn(sec, f"远端 {prefix}/{name}/({n} 个文件)本地没有", base)
    for name in sorted(local_files):
        base = name.removesuffix("_failed")
        local = local_files[name]
        if name not in remote_tasks:
            report.warn(sec, f"本地 {name}({len(local)} 个文件)尚未上传到 {prefix}/{name}/", base)
            continue
        theirs = remote_tasks[name]
        missing = sorted(set(local) - set(theirs))
        extra = sorted(set(theirs) - set(local))
        changed = sorted(p for p in set(local) & set(theirs) if local[p] != theirs[p])
        for label, paths in (("尚未上传", missing), ("远端多出(本地已没有)", extra), ("大小与远端不同", changed)):
            if paths:
                report.warn(sec, f"{prefix}/{name}: {len(paths)} 个文件{label},如 {paths[0]}", base)
        if not (missing or extra or changed):
            report.info(sec, f"{prefix}/{name}: {len(local)} 个文件与本地一致", base)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def print_text(report: Report, sections: list[str]) -> None:
    by_section = defaultdict(list)
    for item in report.items:
        by_section[item["section"]].append(item)
    for sec in sections:
        items = sorted(by_section.get(sec, []), key=lambda i: (LEVEL_ORDER[i["level"]], i["task"]))
        if sec in report.skipped:
            print(f"\n== {sec}: 跳过({report.skipped[sec]})")
            continue
        levels = Counter(i["level"] for i in items)
        status = ", ".join(f"{levels[lv]} {lv}" for lv in LEVEL_ORDER if levels[lv]) or "OK"
        print(f"\n== {sec}: {status}")
        for item in items:
            task = f"[{item['task']}] " if item["task"] else ""
            print(f"  {item['level']:<5} {task}{item['msg']}")
    print(f"\n合计: {report.count('ERROR')} ERROR, {report.count('WARN')} WARN, {report.count('INFO')} INFO")


def parse_sections(value: str | None, remote: list[str]) -> list[str]:
    if not value:
        return [s for s in SECTIONS if s not in ("modelscope", "hf") or s in remote]
    chosen = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s): {', '.join(unknown)} (choose from: {', '.join(SECTIONS)})")
    return [s for s in SECTIONS if s in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(ROOT_DIR / "data" / BENCHMARK_NAME))
    parser.add_argument("--config_name", default="arx_x5", help="env_cfg 的 config_name(用于评测布局)")
    parser.add_argument("--deep", action="store_true", help="逐个读取 hdf5 的 instruction(需要 h5py)")
    parser.add_argument("--modelscope", metavar="REPO_ID", default=os.environ.get("MS_DATA_REPO_ID"),
                        help="检查 ModelScope dataset(也可用环境变量 MS_DATA_REPO_ID)")
    parser.add_argument("--hf", metavar="REPO_ID", default=os.environ.get("HF_DATA_REPO_ID"),
                        help="检查 Hugging Face dataset(也可用环境变量 HF_DATA_REPO_ID)")
    parser.add_argument("--endpoint", default=None, help="ModelScope / HF endpoint")
    parser.add_argument(
        "--remote_prefix", default=REMOTE_PREFIX, help="远端顶层目录,布局为 <prefix>/<task>/<config>/..."
    )
    parser.add_argument("--group", default=None, help="只显示这些组(逗号分隔)的结果,如 heat_X_microwave")
    parser.add_argument("--task", default=None, help="只显示这些任务(逗号分隔)的结果")
    parser.add_argument("--only", default=None, help=f"只跑这些检查,逗号分隔: {','.join(SECTIONS)}")
    parser.add_argument("--min_level", default="INFO", choices=list(LEVEL_ORDER), help="只显示该级别及以上")
    parser.add_argument("--format", default="text", choices=["text", "json"])
    parser.add_argument("--strict", action="store_true", help="有 WARN 也返回 1")
    args = parser.parse_args()

    remote = [name for name, repo in (("modelscope", args.modelscope), ("hf", args.hf)) if repo]
    sections = parse_sections(args.only, remote)
    data_dir = Path(args.data_dir).resolve()
    report = Report()
    tasks = load_json_tasks(TASKS_JSON)
    src = TaskSource()

    if "json" in sections:
        check_json(report, tasks)
    # code 决定哪些任务已实现,其他检查都依赖它;未选中时不输出它的结果
    code_report = report if "code" in sections else Report()
    implemented = check_code(code_report, tasks, src)
    if "config" in sections:
        check_config(report, tasks, implemented)
    if "inventory" in sections:
        check_inventory(report, tasks, implemented)
    if "eval_layout" in sections:
        check_eval_layout(report, tasks, implemented, args.config_name)
    data_report = report if "data" in sections else Report()
    local_files = check_data(data_report, tasks, implemented, data_dir, args.deep)
    if "flat" in sections:
        check_flat(report, tasks, data_dir, data_dir.with_name(f"{BENCHMARK_NAME}-flat"))
    if "lerobot" in sections:
        check_lerobot(report, tasks, data_dir, data_dir.with_name(f"{BENCHMARK_NAME}-Lerobot"))
    for sec, repo, lister in (("modelscope", args.modelscope, list_modelscope), ("hf", args.hf, list_hf)):
        if sec not in sections:
            continue
        if not repo:
            report.skipped[sec] = f"未指定 --{sec} REPO_ID"
            continue
        try:
            remote_files = lister(repo, args.endpoint)
        except ImportError as e:
            report.skipped[sec] = f"缺少依赖 {e.name},请在 RoboDojo 环境中运行"
            continue
        except Exception as e:  # noqa: BLE001 - 网络 / 鉴权错误都只报告,不中断其他检查
            report.error(sec, f"无法列出 {repo} 的文件: {type(e).__name__}: {e}")
            continue
        report.info(sec, f"{repo}: 远端共 {len(remote_files)} 个文件")
        check_remote(report, sec, tasks, remote_files, local_files, args.remote_prefix)

    min_rank = LEVEL_ORDER[args.min_level]
    report.items = [i for i in report.items if LEVEL_ORDER[i["level"]] <= min_rank]
    if args.group or args.task:
        # 检查照常对全部任务做,只按组 / 任务过滤输出;不属于任何任务的全局条目不显示
        groups = {g.strip() for g in (args.group or "").split(",") if g.strip()}
        unknown = groups - {spec["group"] for spec in tasks.values()}
        if unknown:
            raise SystemExit(f"unknown group(s): {', '.join(sorted(unknown))}")
        selected = groups | {name for name, spec in tasks.items() if spec["group"] in groups}
        selected |= {t.strip() for t in (args.task or "").split(",") if t.strip()}
        report.items = [i for i in report.items if i["task"] in selected]
    if args.format == "json":
        print(json.dumps({"items": report.items, "skipped": report.skipped,
                          "counts": {lv: report.count(lv) for lv in LEVEL_ORDER}}, ensure_ascii=False, indent=2))
    else:
        print_text(report, sections)
    failed = report.count("ERROR") or (args.strict and report.count("WARN"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
