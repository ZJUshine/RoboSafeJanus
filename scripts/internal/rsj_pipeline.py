#!/usr/bin/env python3
"""RoboSafeJanus 任务生产流水线:任务构造按阶段推进、每个阶段都要人工批准后才能进入下一阶段。

状态按组存在 ``data/pipeline/<group>.json``,审核包写在 ``data/pipeline/<group>/<stage>[.<task>].md``,
总览页是看板的 ``data/PIPELINE.html``(``pipeline_page.py``,每次修改状态后重新生成;
用 ``sync_status.py --serve`` 打开时可以在页面上批准 / 驳回 / 重新提交 / 选资产)。

阶段。seed 和 ingest 按组(这时还没有任务列表);其余按任务(``*``),组内每个任务各审一次,``--task`` 必填。
design 对草稿里的全部任务审(含 ``same_as`` 共用的);layout 起只审自有任务(``same_as`` 引用的任务在归属组里审)。
一个任务通过后就能单独往下走,不用等同组其他任务::

    seed     task/risk_seeds.json 中的种子            init 时自动提交
    (assets) 不审核:用户在看板 assets 格「填资产链接」写进 task/robosafejanus_assets.json[<group>],保存就算完成
    ingest   ext_assets/integrate/simready_assets.py --group 下载 / 转换 / 挂进 Assets(看板 ingest 格「⬇ 下载资产」);
             产物 = 资产 JSON 里该组 + 每个 label 的 robodojo_format 目录与 Assets 目录(自动列出),--report 附入库报告;
             改了资产链接 ingest 就失效
    design*  任务列表在这里定(set-tasks,先给用户确认),然后
             task/robosafejanus_tasks.json[<group>] 里有该任务 + robosafejanus_tasks/<group>.json 的该任务和 core_risk
             + robosafejanus_sim/<group>.json 的该任务(success 不算在内,它在 expert 阶段由 task_success_spec.py 回填)
    layout*  task/RoboSafeJanus/{config,tasks}/<task> 及其 *_X_* 组模块
    expert*  task/RoboSafeJanus/experts/<task> 及其组模块 + sim 中该任务的 success
    eval*    评测结果目录 / 报告(--artifact,每个 _result.json 的 eval_time >= 1)

批量采集、LeRobot 转换、上传 ModelScope 不是审核阶段(2026-11-13 起):expert 批准后直接用看板大表里
collect / exchange / upload 格的按钮做(``group_jobs.py``),做完不提交。上传脚本 ``--task`` 真正上传时只要求
该任务的 expert 已批准(``gate <group> expert --task <task> --approved``)。以前的 collect / upload 条目读状态时丢掉。

旧状态读的时候自动迁移(``migrate``):assets 条目丢掉(2026-11-13 起不审);按任务的 ingest 合并成按组的一条
(有一个任务仍然有效地"通过"就算整组通过,产物重新记 hash);更早按组审的 design / eval 拆成每个任务一条。

闸门:
  * 提交(submit)前,上游所有阶段必须已批准且仍然有效;提交时跑该阶段的检查(不导入 Isaac),失败则拒绝。
  * 批准(approve)时重新计算产物 hash,与提交时不同则拒绝(提交后又改过,需要重新提交)。
  * 已批准阶段的产物之后又被改动、或上游被重新批准,状态变为 stale,下游的闸门随之关闭,重新提交 / 批准即可。

用法::

    python scripts/internal/rsj_pipeline.py init pour_liquid_toaster          # 从种子建状态,提交 seed
    # 除 seed / ingest 外每个阶段都要 --task <task>
    python scripts/internal/rsj_pipeline.py status [group]
    python scripts/internal/rsj_pipeline.py set-tasks <group> <task> ...       # design 阶段先定的任务草稿
    python scripts/internal/rsj_pipeline.py resubmit <group> <stage> [--task T]   # 沿用上次的附件 / 指标
    python scripts/internal/rsj_pipeline.py submit <group> <stage> [--task T] [--artifact PATH ...]
                                               [--metric k=v ...] [--report PATH] [--note TEXT]
    python scripts/internal/rsj_pipeline.py approve <group> <stage> [--task T] [--note TEXT]
    python scripts/internal/rsj_pipeline.py reject  <group> <stage> [--task T] --note TEXT
    python scripts/internal/rsj_pipeline.py notes   <group> <stage> [--task T]   # 审核意见(skill 重跑前读)
    python scripts/internal/rsj_pipeline.py gate    <group> <stage> [--task T] [--approved]
    python scripts/internal/rsj_pipeline.py adopt   <group> --through <stage>  # 流水线之前已完成的组
    python scripts/internal/rsj_pipeline.py html

``gate`` 退出码 0 表示可以开始该阶段(加 ``--approved`` 表示该阶段本身已批准且有效,供 hook 拦截上传等操作)。

产物写法:文件 / 目录路径(相对仓库根);``path#a.b`` 只取 JSON 的某个键,``[k=v]`` 选列表中 ``k == v`` 的元素;
``state:draft_tasks`` 为状态里的任务草稿。文件按内容 hash,目录按(相对路径, 大小, mtime)hash。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_collect.benchmark_layout import BENCHMARK_NAME, load_index, load_tasks, task_dir  # noqa: E402

STATE_DIR = Path(os.environ.get("RSJ_PIPELINE_DIR", ROOT / "data" / "pipeline"))  # 测试时可改
SEEDS_JSON = ROOT / "task" / "risk_seeds.json"
ASSETS_JSON = ROOT / "task" / "robosafejanus_assets.json"
DESIGN_DIR = ROOT / "task" / "robosafejanus_tasks"
SIM_DIR = ROOT / "task" / "robosafejanus_sim"
MIRROR = ROOT / "task" / "RoboSafeJanus"
ALIGN = ROOT / "scripts" / "internal" / "check_task_alignment.py"
# 与 check_task_alignment.py 的资产来源规则一致:只允许 RoboDojo 官方和 SimReady
ROBODOJO_REPO = "RoboDojo-Benchmark/RoboDojo"
SIMREADY_FORMAT = Path("ext_assets") / "robodojo_format" / "SimReady"
OBJECTS = Path("Assets") / "Object" / "RoboDojo"
EXT_MANIFEST = ROOT / "ext_assets" / "integrate" / "manifest.json"
ALLOWED_REPOS = (
    ROBODOJO_REPO,
    "nvidia/simready-assets",
    "PalatialSim/palatial-nv-asset-library",
    "ForgeCAD/simready-assets",
    "LGElectronics/simready-assets",
)

COLLECT_TARGET = int(os.environ.get("RSJ_COLLECT_TARGET", "100"))  # 每个任务的采集条数
CONFIG_NAME = "arx_x5"
STAGES = ("seed", "ingest", "design", "layout", "expert", "eval")
DROPPED_STAGES = ("assets", "collect", "upload")  # 以前的审核阶段:assets 用户自己填,采集 / 上传是看板按钮直接做
GROUP_STAGES = ("seed", "ingest")  # 按组:这时还没有任务列表(任务列表在 design 定)
TASK_STAGES = tuple(s for s in STAGES if s not in GROUP_STAGES)
DRAFT_STAGES = ("design",)  # 对草稿里全部任务审(含 same_as 共用的),之后只审自有任务
STAGE_DESC = {
    "seed": "种子",
    "ingest": "下载资产 / 入库",
    "design": "设计 / sim JSON",
    "layout": "布局 + 指令",
    "expert": "专家脚本",
    "eval": "下载评测",
}
SEED_FIELDS = ("name", "background", "core_risk", "risk_reasoning", "risk_category", "action",
               "interact_object", "target_object", "risk_variation", "safe_variation")
SNAKE = re.compile(r"[a-z0-9]+(_[a-z0-9A-Z]+)*")
HASH_FULL_LIMIT = 64 << 20  # 更大的单个文件只用大小 + mtime


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def state_path(group: str) -> Path:
    return STATE_DIR / f"{group}.json"


def load_state(group: str) -> dict:
    path = state_path(group)
    if not path.exists():
        raise SystemExit(f"{group}: 还没有流水线状态,先运行 init {group}")
    return migrate(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(state["group"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def group_order() -> dict[str, int]:
    """组的先后:``task/robosafejanus_tasks.json`` 索引里的顺序,索引里没有的按 ``task/risk_seeds.json`` 的顺序。"""
    names: list[str] = []
    try:
        names += list(load_index())
    except Exception:  # noqa: BLE001 - 索引读不了时只按种子排
        pass
    try:
        names += [s.get("name") for s in read_json(SEEDS_JSON) if isinstance(s, dict)]
    except (OSError, ValueError):
        pass
    order: dict[str, int] = {}
    for name in names:
        if name and name not in order:
            order[name] = len(order)
    return order


def all_states() -> list[dict]:
    """全部组的状态,按 ``group_order``(JSON 里的顺序)排;两个 JSON 里都没有的排最后,按名字。"""
    if not STATE_DIR.exists():
        return []
    order = group_order()
    paths = sorted(STATE_DIR.glob("*.json"), key=lambda p: (order.get(p.stem, len(order)), p.stem))
    return [migrate(json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def key_of(stage: str, task: str | None) -> str:
    if stage in TASK_STAGES:
        if not task:
            raise SystemExit(f"{stage} 是按任务的阶段,需要 --task")
        return f"{stage}/{task}"
    if task:
        raise SystemExit(f"{stage} 是按组的阶段,不接受 --task")
    return stage


def report_path(group: str, key: str) -> Path:
    return STATE_DIR / group / (key.replace("/", ".") + ".md")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def own_tasks(state: dict) -> list[str]:
    """组内需要在本组审 layout 及之后阶段的任务:草稿列表去掉 sim 里写了 ``same_as`` 的任务。"""
    sim = SIM_DIR / f"{state['group']}.json"
    shared = set()
    if sim.exists():
        shared = {t for t, s in read_json(sim).get("tasks", {}).items() if "same_as" in s}
    return [t for t in state.get("draft_tasks", []) if t not in shared]


def stage_tasks(state: dict, stage: str) -> list[str]:
    """按任务的阶段要审哪些任务:design 是草稿全部,之后是自有任务。"""
    if stage in GROUP_STAGES:
        return []
    return list(state.get("draft_tasks", [])) if stage in DRAFT_STAGES else own_tasks(state)


# ---------------------------------------------------------------------------
# 产物 hash
# ---------------------------------------------------------------------------


def _select(data, selector: str):
    for part in selector.split("."):
        m = re.fullmatch(r"\[(\w+)=(.+)\]", part)
        if m:
            k, v = m.groups()
            data = next((x for x in data if isinstance(x, dict) and str(x.get(k)) == v), None)
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            data = None
        if data is None:
            return None
    return data


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    st = path.stat()
    if st.st_size > HASH_FULL_LIMIT:
        return _hash_bytes(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return _hash_bytes(path.read_bytes())


def _hash_dir(path: Path) -> str:
    h = hashlib.sha256()
    for top, dirs, files in os.walk(path, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            p = Path(top) / name
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            h.update(f"{p.relative_to(path)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def _drop_success(sim: dict) -> dict:
    sim = json.loads(json.dumps(sim))
    for spec in sim.get("tasks", {}).values():
        if isinstance(spec, dict):
            spec.pop("success", None)
    return sim


def artifact_hash(spec: str, state: dict) -> str:
    """产物当前的 hash;不存在为 ``MISSING``。"""
    if spec == "state:draft_tasks":
        return _hash_bytes(json.dumps(state.get("draft_tasks", [])).encode())
    if spec.startswith("state:task:"):  # 该任务在草稿里
        task = spec.split(":", 2)[2]
        return _hash_bytes(task.encode()) if task in state.get("draft_tasks", []) else "MISSING"
    path_s, _, selector = spec.partition("#")
    path = ROOT / path_s
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        return _hash_dir(path)
    if not selector:
        return _hash_file(path)
    data = read_json(path)
    if selector == "!success":
        data = _drop_success(data)
    elif selector.endswith("!success"):  # tasks.<t>!success:该任务去掉 success
        data = _select(data, selector[: -len("!success")])
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "success"}
    elif "@" in selector:  # <key>@<item>:列表里有这一项
        key, _, item = selector.partition("@")
        items = _select(data, key)
        data = item if isinstance(items, list) and item in items else None
    else:
        data = _select(data, selector)
    if data is None:
        return "MISSING"
    return _hash_bytes(json.dumps(data, sort_keys=True, ensure_ascii=False).encode())


def _group_modules(path: Path, kind: str) -> list[str]:
    """任务 / 专家文件引用的 ``*_X_*`` 组模块(``common.py`` 全部任务共用,不计入)。"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    names = set(re.findall(rf"task\.RoboDojo\.{kind}\.(\w+)", text))
    names |= set(re.findall(rf"from task\.RoboDojo\.{kind} import (\w+)", text))
    names = {n for n in names if n.split("_").count("X") == 1}
    return [f"task/RoboSafeJanus/{kind}/{n}.py" for n in sorted(names)]


def ingest_targets(group: str) -> list[dict]:
    """该组每个资产 label 入库后应有的目录:SimReady 为 robodojo_format + 挂进 Assets 的目录(类型取自
    ext_link 的 manifest,还没挂时为空),RoboDojo 官方为 ``--include`` 对应的 Assets 目录。"""
    entries = read_json(EXT_MANIFEST).get("entries", {}) if EXT_MANIFEST.exists() else {}
    out = []
    for label, cmd in (read_json(ASSETS_JSON).get(group) or {}).items():
        args = shlex.split(cmd or "")
        repo = args[2] if len(args) > 2 else ""
        opt = {k: args[i + 1] for i, k in enumerate(args[:-1]) if k.startswith("--")}
        item = {"label": label, "repo": repo, "revision": opt.get("--revision", ""),
                "dir": opt.get("--include", "").rstrip("*").rstrip("/"), "paths": []}
        if repo == ROBODOJO_REPO:
            item["paths"] = [item["dir"]]
        elif repo:
            item["paths"] = [str(SIMREADY_FORMAT / label / "00000")]
            dst = (entries.get(f"SimReady/{label}/00000") or {}).get("dst")
            if dst:
                item["paths"].append(str(OBJECTS / dst))
        out.append(item)
    return out


def default_artifacts(state: dict, stage: str, task: str | None) -> list[str]:
    g = state["group"]
    if stage == "seed":
        return [f"task/risk_seeds.json#[name={g}]"]
    if stage == "design":
        return [f"task/robosafejanus_tasks.json#{g}@{task}", f"task/robosafejanus_tasks/{g}.json#core_risk",
                f"task/robosafejanus_tasks/{g}.json#tasks.{task}",
                f"task/robosafejanus_sim/{g}.json#tasks.{task}!success"]
    if stage == "layout":
        tasks_py = f"task/RoboSafeJanus/tasks/{task}.py"
        return [f"task/RoboSafeJanus/config/{task}.yml", tasks_py, *_group_modules(ROOT / tasks_py, "tasks")]
    if stage == "expert":
        expert_py = f"task/RoboSafeJanus/experts/{task}.py"
        return [expert_py, *_group_modules(ROOT / expert_py, "experts"),
                f"task/robosafejanus_sim/{g}.json#tasks.{task}.success"]
    if stage == "ingest":
        return [f"task/robosafejanus_assets.json#{g}", *(p for item in ingest_targets(g) for p in item["paths"])]
    return []  # eval:由 --artifact 给出


def current_hashes(state: dict, entry: dict) -> dict[str, str]:
    return {spec: artifact_hash(spec, state) for spec in entry.get("artifacts", {})}


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


def direct_upstream(state: dict, stage: str, task: str | None) -> list[str]:
    i = STAGES.index(stage)
    if i == 0:
        return []
    prev = STAGES[i - 1]
    return [f"{prev}/{task}"] if prev in TASK_STAGES else [prev]


def entry_status(state: dict, key: str, _seen=None) -> tuple[str, str]:
    """``(状态, 原因)``;状态为 none / awaiting_review / changed / rejected / approved / stale。"""
    entry = state["stages"].get(key)
    if not entry:
        return "none", ""
    status = entry["status"]
    if status in ("awaiting_review", "approved"):
        changed = [s for s, h in current_hashes(state, entry).items() if entry["artifacts"][s] != h]
        if changed:
            return ("stale" if status == "approved" else "changed"), "产物已改动: " + ", ".join(changed)
    if status == "approved":
        for up, at in entry.get("upstream", {}).items():
            up_entry = state["stages"].get(up)
            if not up_entry or up_entry.get("approved_at") != at:
                return "stale", f"上游 {up} 已重新提交 / 批准"
    return status, ""


def assets_problems(group: str) -> list[str]:
    """资产链接(不审核,用户自己填)还缺什么:没填、来源不允许或命令格式不对。"""
    sources = (read_json(ASSETS_JSON).get(group) if ASSETS_JSON.exists() else None) or {}
    if not sources:
        return ["assets: 还没填资产链接(看板 assets 格「填资产链接」)"]
    bad = [label for label, ok, _ in source_checks(sources) if not ok]
    return [f"assets: {', '.join(bad)} 的来源不允许或格式不对"] if bad else []


def source_checks(sources: dict) -> list[tuple[str, bool, str]]:
    """``[(label, 是否合规, 来源库)]``:只允许 RoboDojo 官方和 SimReady,命令要带 --include。"""
    out = []
    for label, cmd in sources.items():
        try:
            args = shlex.split(cmd or "")
        except ValueError:
            args = []
        ok = args[:2] == ["hf", "download"] and len(args) > 2 and args[2] in ALLOWED_REPOS and "--include" in args
        out.append((label, ok, args[2] if len(args) > 2 else (cmd or "(空)")))
    return out


def upstream_problems(state: dict, stage: str, task: str | None) -> list[str]:
    """开始 ``stage`` 之前未满足的上游条件。"""
    problems = []
    if stage == "ingest":
        problems += assets_problems(state["group"])
    for up in STAGES[: STAGES.index(stage)]:
        keys = [f"{up}/{task}"] if up in TASK_STAGES else [up]
        for key in keys:
            status, why = entry_status(state, key)
            if status != "approved":
                problems.append(f"{key}: {status}" + (f"({why})" if why else ""))
    return problems


# ---------------------------------------------------------------------------
# 各阶段的检查(不导入 Isaac)
# ---------------------------------------------------------------------------


def run_align(*args: str) -> tuple[bool, str]:
    cmd = [sys.executable, str(ALIGN), *args, "--min_level", "WARN"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = f"$ {' '.join(shlex.quote(c) for c in cmd[1:])}\n{proc.stdout}{proc.stderr}".rstrip()
    return proc.returncode == 0, out


def find_seed(group: str) -> dict | None:
    return next((s for s in read_json(SEEDS_JSON) if s.get("name") == group), None)


def check_seed(state: dict, task: str | None) -> list[tuple[bool, str]]:
    g = state["group"]
    seed = find_seed(g)
    if seed is None:
        return [(False, f"{SEEDS_JSON.name} 中没有 name == {g} 的种子")]
    out = []
    names = [s.get("name") for s in read_json(SEEDS_JSON)]
    out.append((names.count(g) == 1, f"name 唯一({names.count(g)} 条)"))
    out.append((bool(SNAKE.fullmatch(g)), "name 为 snake_case"))
    missing = [f for f in SEED_FIELDS if f not in seed]
    out.append((not missing, "字段齐全" + (f",缺 {missing}" if missing else "")))
    risk = (seed.get("risk_variation") or {}).get("elements") or []
    safe = (seed.get("safe_variation") or {}).get("elements") or []
    out.append((bool(risk), f"risk_variation.elements 非空({len(risk)} 个)"))
    # 安全任务可以借用别的组(sim 里写 same_as),所以 safe_variation 为空不拦
    hint = "" if safe else "(为空:安全任务需从别的组 same_as 借用)"
    out.append((True, f"safe_variation.elements {len(safe)} 个{hint}"))
    return out


def check_ingest(state: dict, task: str | None) -> list[tuple[bool, str]]:
    targets = ingest_targets(state["group"])
    out = [(bool(targets), f"{ASSETS_JSON.name} 中有 {len(targets)} 个 label")]
    sources = read_json(ASSETS_JSON).get(state["group"]) or {}
    out += [(ok, f"{label}: 来源 {repo}" + ("" if ok else " ← 来源不允许或格式不对"))
            for label, ok, repo in source_checks(sources)]
    for item in targets:
        label = item["label"]
        if item["repo"] == ROBODOJO_REPO:
            ok = bool(list((ROOT / item["dir"]).glob("object.usd*")))
            out.append((ok, f"{label}: 官方资产 {item['dir']}" + ("" if ok else " 本地不存在(scripts/init_assets.sh)")))
            continue
        fmt_dir = ROOT / item["paths"][0]
        src = read_json(fmt_dir / "metadata.json").get("source", {}) if (fmt_dir / "metadata.json").exists() else {}
        same = (src.get("repo"), src.get("revision"), src.get("asset_dir")) == (item["repo"], item["revision"],
                                                                              item["dir"])
        ok = same and (fmt_dir / "object.usd").exists()
        out.append((ok, f"{label}: {item['paths'][0]}" + ("" if ok else " 缺失或与命令不一致(重跑 simready_assets)")))
        linked = len(item["paths"]) > 1 and (ROOT / item["paths"][1] / "object.usd").exists()
        out.append((linked, f"{label}: 挂进 Assets " + (item["paths"][1] if linked else "未完成(ext_link link)")))
    report = state["stages"].get(key_of("ingest", task), {}).get("report")
    out.append((bool(report), "--report 附入库报告(simready_assets.py --group ... --report)"))
    return out


def check_design(state: dict, task: str | None) -> list[tuple[bool, str]]:
    g = state["group"]
    try:
        groups = load_tasks(include_shared=True)
    except Exception as e:  # noqa: BLE001 - 报告给审核人
        return [(False, f"load_tasks() 失败: {e}")]
    if g not in groups:
        return [(False, f"{g} 不在任务定义中")]
    level = next((lv for lv, ts in groups[g].items() if task in ts), None)
    out = [(level is not None, f"{task} 在 {g} 的任务定义里" + (f"({level})" if level else ""))]
    sim = read_json(SIM_DIR / f"{g}.json").get("tasks", {}) if (SIM_DIR / f"{g}.json").exists() else {}
    shared = "(same_as)" if "same_as" in sim.get(task, {}) else ""
    out.append((task in sim, f"{task} 在 robosafejanus_sim/{g}.json 里{shared}"))
    out.append(run_align("--only", "json", "--group", g, "--task", task))
    return out


def _mirror_ok(kind: str, name: str) -> tuple[bool, str]:
    real, link = MIRROR / kind / name, ROOT / "task" / "RoboDojo" / kind / name
    ok = real.exists() and link.is_symlink() and link.resolve() == real.resolve()
    return ok, f"{kind}/{name} 存在且 task/RoboDojo 有软链接"


def check_layout(state: dict, task: str | None) -> list[tuple[bool, str]]:
    out = [_mirror_ok("config", f"{task}.yml"), _mirror_ok("tasks", f"{task}.py")]
    out.append(run_align("--only", "config", "--task", task))
    return out


def check_expert(state: dict, task: str | None) -> list[tuple[bool, str]]:
    out = [_mirror_ok("experts", f"{task}.py")]
    metrics = state["stages"].get(f"expert/{task}", {}).get("metrics", {})
    rate = metrics.get("success_rate", "未给")
    out.append(("success_rate" in metrics, f"小批量成功率 --metric success_rate=...({rate})"))
    out.append(run_align("--only", "code,config", "--task", task))
    return out


def _task_rel(t: str) -> str:
    """任务在 Benchmark 里的相对目录 ``<group>/<level>/<task>``(LeRobot 镜像同样的布局)。"""
    base = ROOT / "data" / BENCHMARK_NAME
    return Path(task_dir(str(base), t)).relative_to(base).as_posix()


def _bench_dir(t: str) -> Path:
    """任务的采集目录 ``data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/<config>``。"""
    return Path(task_dir(str(ROOT / "data" / BENCHMARK_NAME), t)) / CONFIG_NAME


def _hdf5_count(t: str) -> int:
    return len(list((_bench_dir(t) / "data").glob("*.hdf5")))


def check_eval(state: dict, task: str | None) -> list[tuple[bool, str]]:
    arts = [s for s in state["stages"].get(key_of("eval", task), {}).get("artifacts", {}) if not s.startswith("state:")]
    out = [(bool(arts), f"--artifact 列出评测结果目录或报告({len(arts)} 个)")]
    for spec in arts:
        path = ROOT / spec
        results = [path] if path.name == "_result.json" else list(path.rglob("_result.json")) if path.is_dir() else []
        if path.suffix in (".md", ".json") and path.name != "_result.json":
            out.append((path.exists(), f"{spec}: 报告" + ("" if path.exists() else "不存在")))
            continue
        bad = [r for r in results if (read_json(r).get("eval_time") or 0) < 1]
        out.append((bool(results) and not bad,
                    f"{spec}: {len(results)} 个 _result.json" + (f",{len(bad)} 个 eval_time < 1" if bad else "")))
    repo = os.environ.get("MS_DATA_REPO_ID")
    if repo:
        out.append(run_align("--only", "modelscope", "--task", task, "--modelscope", repo))
    else:
        out.append((True, "未设 MS_DATA_REPO_ID,跳过远端 HDF5 核对"))
    return out


CHECKS = {s: globals()[f"check_{s}"] for s in STAGES}


# ---------------------------------------------------------------------------
# 迁移:以前按组审的阶段拆成按任务
# ---------------------------------------------------------------------------


def _merge_task_ingest(state: dict) -> bool:
    """以前按任务审的 ``ingest/<task>`` 合并成按组的 ``ingest``:有一个仍然有效地"通过"(产物没变)就算整组通过,
    产物按现在的定义(加上资产 JSON 里该组)重新记 hash;design 记的上游 ``ingest/<task>`` 改成 ``ingest``。"""
    stages = state.get("stages", {})
    keys = [k for k in stages if k.startswith("ingest/")]
    if not keys:
        return False
    valid = [stages[k] for k in keys if stages[k].get("status") == "approved"
             and all(artifact_hash(s, state) == h for s, h in stages[k].get("artifacts", {}).items())]
    for k in keys:
        stages.pop(k)
        report_path(state["group"], k).unlink(missing_ok=True)
    at = None
    if valid:
        best = max(valid, key=lambda e: e.get("approved_at") or "")
        entry = json.loads(json.dumps(best))
        entry["artifacts"] = {s: artifact_hash(s, state) for s in default_artifacts(state, "ingest", None)}
        seed = stages.get("seed") or {}
        entry["upstream"] = {"seed": seed["approved_at"]} if seed.get("approved_at") else {}
        note(entry, "approve", f"由按任务的 ingest({len(valid)}/{len(keys)} 个有效通过)合并成按组")
        stages["ingest"] = entry
        at = entry["approved_at"]
        print(f"{state['group']}: ingest 改成按组,{len(keys)} 条合并成 1 条(通过)", file=sys.stderr)
    else:
        print(f"{state['group']}: ingest 改成按组,{len(keys)} 条旧条目都不是有效的通过,丢掉,需要重新下载 / 提交",
              file=sys.stderr)
    for key, entry in stages.items():  # design 记的上游
        ups = entry.get("upstream") or {}
        if any(u.startswith("ingest/") for u in ups):
            entry["upstream"] = {u: v for u, v in ups.items() if not u.startswith("ingest/")}
            entry["upstream"]["ingest"] = at
    return True


def migrate(state: dict) -> dict:
    """丢掉已不审的 assets / collect / upload 条目(``DROPPED_STAGES``);按任务的 ingest 合并成按组
    (``_merge_task_ingest``);把更早的组级条目(``design`` / ``eval`` 等)拆成每个任务一条。

    仍然有效的"通过"拆成各任务的"通过":产物按任务重新记 hash,批准时间不变;其他状态(待审、失效、驳回……)
    丢掉,需要按任务重新提交。下游条目记的上游(如 layout 记的 ``design``)改成同一任务的 ``design/<task>``。
    有改动时写回状态文件。
    """
    stages = state.get("stages", {})
    dropped = [k for k in stages if k.split("/")[0] in DROPPED_STAGES]
    for key in dropped:
        stages.pop(key)
        report_path(state["group"], key).unlink(missing_ok=True)
    for entry in stages.values():  # eval 以前记的上游是 upload,现在是 expert(批准后重新记)
        for up in [u for u in entry.get("upstream", {}) if u.split("/")[0] in DROPPED_STAGES]:
            entry["upstream"].pop(up)
    if dropped:
        print(f"{state['group']}: assets / collect / upload 不再审核,丢掉 {len(dropped)} 条旧条目", file=sys.stderr)
    merged = _merge_task_ingest(state)
    old = [s for s in TASK_STAGES if s in stages]
    if not old:
        if dropped or merged:
            save_state(state)
            if merged:
                write_report(state, "ingest")
        return state
    status = {s: entry_status(state, s)[0] for s in old}  # 用旧的产物判断,必须在拆之前
    for stage in TASK_STAGES:
        if stage not in old:
            continue
        entry = stages.pop(stage)
        if status[stage] != "approved":
            print(f"{state['group']}: 组级 {stage} 是 {status[stage]},按任务拆分时丢掉,需要重新提交", file=sys.stderr)
            continue
        extra = [s for s in entry.get("artifacts", {}) if stage == "eval"]
        for task in stage_tasks(state, stage):
            key = f"{stage}/{task}"
            if key in stages:
                continue
            new = json.loads(json.dumps(entry))
            specs = list(dict.fromkeys([*default_artifacts(state, stage, task), *extra]))
            new["artifacts"] = {s: artifact_hash(s, state) for s in specs}
            new["upstream"] = {u: stages[u]["approved_at"] for u in direct_upstream(state, stage, task)
                               if u in stages and stages[u].get("approved_at")}
            note(new, "approve", f"由组级 {stage} 的批准({entry.get('approved_at')})拆成按任务")
            stages[key] = new
    for key, entry in stages.items():  # 下游条目记的组级上游 -> 同一任务的按任务上游
        if "/" not in key or not entry.get("upstream"):
            continue
        task = key.split("/", 1)[1]
        entry["upstream"] = {(f"{u}/{task}" if u in old else u): at for u, at in entry["upstream"].items()}
    save_state(state)
    for key in stages:
        if key.split("/")[0] in old and "/" in key:
            write_report(state, key)
    for stage in old:  # 旧的组级审核包
        report_path(state["group"], stage).unlink(missing_ok=True)
    return state


# ---------------------------------------------------------------------------
# 审核包
# ---------------------------------------------------------------------------


def git_diff(specs) -> str:
    paths = sorted({s.partition("#")[0] for s in specs if not s.startswith("state:")})
    tracked = [p for p in paths if (ROOT / p).is_file()]
    if not tracked:
        return ""
    proc = subprocess.run(["git", "diff", "HEAD", "--", *tracked], cwd=ROOT, capture_output=True, text=True)
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", *tracked],
                               cwd=ROOT, capture_output=True, text=True).stdout.split()
    diff = proc.stdout
    if len(diff) > 20000:
        diff = diff[:20000] + f"\n... (截断,共 {len(proc.stdout)} 字符)\n"
    if untracked:
        diff += "\n新文件(未进 git): " + ", ".join(untracked) + "\n"
    return diff


def write_report(state: dict, key: str) -> Path:
    entry = state["stages"][key]
    checks = entry.get("checks", [])
    stage = key.split("/")[0]
    lines = [f"# {state['group']} · {key}({STAGE_DESC[stage]})", "",
             f"- 提交: {entry['submitted_at']}", f"- 状态: {entry['status']}"]
    if entry.get("report"):
        lines.append(f"- 附件: {entry['report']}")
    for k, v in entry.get("metrics", {}).items():
        lines.append(f"- {k}: {v}")
    if stage == "seed":
        lines += ["", "## 种子", "```json", json.dumps(find_seed(state["group"]), indent=2, ensure_ascii=False), "```"]
    lines += ["", "## 检查"]
    for ok, msg in checks:
        if "\n" in msg:
            lines += [f"- {'✅' if ok else '❌'}", "```", msg, "```"]
        else:
            lines.append(f"- {'✅' if ok else '❌'} {msg}")
    lines += ["", "## 产物", *[f"- `{s}` {h}" for s, h in entry["artifacts"].items()]]
    diff = git_diff(entry["artifacts"])
    if diff:
        lines += ["", "## 相对 HEAD 的改动", "```diff", diff.rstrip(), "```"]
    if entry.get("notes"):
        lines += ["", "## 记录", *[f"- {n['at']} [{n['kind']}] {n['text']}" for n in entry["notes"]]]
    path = report_path(state["group"], key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 总览页
# ---------------------------------------------------------------------------

STATUS_LABEL = {"none": "", "awaiting_review": "待审", "changed": "提交后已改", "rejected": "驳回",
                "approved": "通过", "stale": "失效"}


def write_html() -> Path | None:
    """重新生成看板 ``<data>/PIPELINE.html`` 的流水线大表片段(见 pipeline_page.py)。"""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import pipeline_page
        import sync_status
    except ImportError as e:  # 看板脚本不可用时只是不刷新页面
        print(f"(未刷新流水线页: {e})", file=sys.stderr)
        return None
    return pipeline_page.render(STATE_DIR.parent, sync_status.nav_html, sync_status.BASE_CSS)


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


def note(entry: dict, kind: str, text: str | None) -> None:
    if text is not None:
        entry.setdefault("notes", []).append({"at": now(), "kind": kind, "text": text})


def do_submit(state: dict, stage: str, task: str | None, artifacts=(), metrics=(), report=None,
              text=None, force_checks=False) -> int:
    key = key_of(stage, task)
    if stage in TASK_STAGES and task not in stage_tasks(state, stage):
        what = "草稿里的任务" if stage in DRAFT_STAGES else "自有任务(same_as 共用的在归属组审)"
        raise SystemExit(f"{task} 不是 {state['group']} 的{what}(草稿: {state.get('draft_tasks')})")
    problems = upstream_problems(state, stage, task)
    if problems:
        print(f"拒绝提交 {key}:上游未就绪\n  " + "\n  ".join(problems))
        return 1
    old = state["stages"].get(key, {})
    specs = list(dict.fromkeys([*default_artifacts(state, stage, task), *artifacts]))
    entry = {"status": "awaiting_review", "submitted_at": now(), "notes": old.get("notes", []),
             "metrics": dict(m.split("=", 1) for m in metrics), "report": report, "artifacts": {}}
    state["stages"][key] = entry
    entry["artifacts"] = {s: artifact_hash(s, state) for s in specs}
    missing = [s for s, h in entry["artifacts"].items() if h == "MISSING"]
    checks = CHECKS[stage](state, task)
    if missing:
        checks.insert(0, (False, "产物不存在: " + ", ".join(missing)))
    failed = [msg for ok, msg in checks if not ok]
    entry["checks"] = [list(c) for c in checks]
    note(entry, "submit", text or "")
    path = write_report(state, key)
    if failed and not force_checks:
        # 状态文件不动(审核包照样写出,方便看哪里没过)
        if old:
            state["stages"][key] = old
        else:
            state["stages"].pop(key)
        print(f"拒绝提交 {key}:检查未通过(详见 {rel(path)})")
        for msg in failed:
            print("  ❌ " + msg.splitlines()[0])
        return 1
    if failed:
        note(entry, "force", "检查未通过仍提交: " + "; ".join(m.splitlines()[0] for m in failed))
        write_report(state, key)
    save_state(state)
    print(f"{key} 已提交,等待审核: {rel(path)}")
    return 0


def cmd_resubmit(a) -> int:
    """按上次提交的附加产物 / 指标 / 附件重新提交(产物改动后、或上游重新批准后用)。"""
    state = load_state(a.group)
    key = key_of(a.stage, a.task)
    old = state["stages"].get(key)
    if not old:
        raise SystemExit(f"{key} 还没有提交过,用 submit")
    defaults = set(default_artifacts(state, a.stage, a.task))
    extra = [s for s in old.get("artifacts", {}) if s not in defaults]
    metrics = [f"{k}={v}" for k, v in old.get("metrics", {}).items()]
    return do_submit(state, a.stage, a.task, extra, metrics, old.get("report"), a.note or "重新提交")


def cmd_init(a) -> int:
    g = a.group
    if state_path(g).exists():
        raise SystemExit(f"{g} 已有状态 {rel(state_path(g))}")
    state = {"group": g, "created_at": now(), "draft_tasks": [], "stages": {}}
    return do_submit(state, "seed", None)


def cmd_set_tasks(a) -> int:
    state = load_state(a.group)
    state["draft_tasks"] = list(a.tasks)
    save_state(state)
    print(f"{a.group} 任务草稿: {', '.join(a.tasks)}")
    return 0


def cmd_submit(a) -> int:
    state = load_state(a.group)
    return do_submit(state, a.stage, a.task, a.artifact, a.metric, a.report, a.note, a.force_checks)


def cmd_review(a, approve: bool) -> int:
    state = load_state(a.group)
    key = key_of(a.stage, a.task)
    entry = state["stages"].get(key)
    if not entry:
        raise SystemExit(f"{key} 还没有提交")
    if approve:
        if entry["status"] == "rejected":
            raise SystemExit(f"{key} 已驳回,修改后重新 submit")
        changed = [s for s, h in current_hashes(state, entry).items() if entry["artifacts"][s] != h]
        if changed:
            print(f"拒绝批准 {key}:提交后产物已改动,请重新 submit\n  " + "\n  ".join(changed))
            return 1
        problems = upstream_problems(state, a.stage, a.task)
        if problems:
            print(f"拒绝批准 {key}:上游未就绪\n  " + "\n  ".join(problems))
            return 1
        entry["status"] = "approved"
        entry["approved_at"] = now()
        entry["upstream"] = {u: state["stages"][u]["approved_at"] for u in direct_upstream(state, a.stage, a.task)}
    else:
        if not a.note:
            raise SystemExit("驳回需要 --note 写明要改什么")
        entry["status"] = "rejected"
        entry.pop("approved_at", None)
    note(entry, "approve" if approve else "reject", a.note or "")
    write_report(state, key)
    save_state(state)
    print(f"{key}: {'已批准' if approve else '已驳回'}")
    return 0


def cmd_notes(a) -> int:
    state = load_state(a.group)
    entry = state["stages"].get(key_of(a.stage, a.task), {})
    for n in entry.get("notes", []):
        print(f"{n['at']} [{n['kind']}] {n['text']}")
    return 0


def cmd_gate(a) -> int:
    state = load_state(a.group)
    key = key_of(a.stage, a.task)
    if a.approved:
        status, why = entry_status(state, key)
        ok = status == "approved"
        print(f"{key}: {status}" + (f"({why})" if why else ""))
        return 0 if ok else 1
    problems = upstream_problems(state, a.stage, a.task)
    print("可以开始 " + key if not problems else f"不能开始 {key}:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


def _adopt_one(state: dict, stage: str, task: str | None) -> bool:
    metrics = ["success_rate=adopted"] if stage == "expert" else []
    if do_submit(state, stage, task, metrics=metrics, text="adopt: 流水线之前已完成",
                 force_checks=stage in ("ingest", "eval")) != 0:
        return False
    save_state(state)
    ns = argparse.Namespace(group=state["group"], stage=stage, task=task, note="adopt")
    return cmd_review(ns, approve=True) == 0


def cmd_adopt(a) -> int:
    """流水线之前已完成的组:以当前产物为准,逐阶段提交并批准到 ``--through``(检查仍然要过)。

    seed 失败即停;其余阶段每个任务各自往下走,某个任务失败不影响别的任务。
    已有状态时从未批准的地方接着做(已批准的不动)。
    """
    g = a.group
    index = load_index()
    if g not in index:
        raise SystemExit(f"{g} 不在 task/robosafejanus_tasks.json")
    if state_path(g).exists():
        state = load_state(g)
    else:
        state = {"group": g, "created_at": now(), "draft_tasks": list(index[g]), "stages": {}, "adopted": True}
    stopped = []
    for stage in STAGES[: STAGES.index(a.through) + 1]:
        tasks = stage_tasks(state, stage) if stage in TASK_STAGES else [None]
        for t in tasks:
            key = key_of(stage, t)
            if entry_status(state, key)[0] == "approved":
                continue
            if upstream_problems(state, stage, t):
                continue  # 上游已经停下,这里不再尝试
            if not _adopt_one(state, stage, t):
                stopped.append(key)
            state = load_state(g) if state_path(g).exists() else state
        if stopped and stage not in TASK_STAGES:
            break
    save_state(state)
    if stopped:
        print("adopt 未通过: " + ", ".join(stopped) + "(处理后再 adopt 一次,或用 submit / approve)")
        return 1
    return 0


def cmd_status(a) -> int:
    states = [load_state(a.group)] if a.group else all_states()
    if not states:
        print("还没有组。init <group> 从 task/risk_seeds.json 的种子开始。")
    for state in states:
        print(f"\n{state['group']}  (草稿任务 {len(state.get('draft_tasks', []))} 个)")
        nxt = None
        for stage in STAGES:
            keys = [f"{stage}/{t}" for t in stage_tasks(state, stage)] if stage in TASK_STAGES else [stage]
            for key in keys or [f"{stage}/<无任务>"]:
                status, why = entry_status(state, key)
                mark = {"approved": "✅", "awaiting_review": "⏳", "rejected": "❌", "stale": "♻️",
                        "changed": "⚠️"}.get(status, "  ")
                print(f"  {mark} {key:<40} {STATUS_LABEL.get(status, status) or '未开始'}{('  ' + why) if why else ''}")
                if nxt is None and status != "approved":
                    nxt = (key, status)
        if nxt:
            key, status = nxt
            hint = {"awaiting_review": "等你审核", "rejected": "按意见修改后重新提交", "stale": "重新提交或批准",
                    "changed": "重新提交"}.get(status, "可以开始")
            print(f"  → 下一步: {key}({hint})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, *, group=True, stage=False, **kw):
        sp = sub.add_parser(name, **kw)
        if group:
            sp.add_argument("group")
        if stage:
            sp.add_argument("stage", choices=STAGES)
            sp.add_argument("--task")
        return sp

    add("init", help="从种子新建组状态,提交 seed")
    sp = sub.add_parser("status")
    sp.add_argument("group", nargs="?")
    sp = add("set-tasks", help="设置任务草稿")
    sp.add_argument("tasks", nargs="+")
    add("resubmit", stage=True).add_argument("--note")
    sp = add("submit", stage=True)
    sp.add_argument("--artifact", action="append", default=[], help="额外产物(可重复)")
    sp.add_argument("--metric", action="append", default=[], help="k=v,如 success_rate=0.9")
    sp.add_argument("--report", help="附件(视频目录、截图页等)路径")
    sp.add_argument("--note")
    sp.add_argument("--force_checks", action="store_true", help="检查未通过仍提交(记录在审核包里)")
    for name in ("approve", "reject"):
        add(name, stage=True).add_argument("--note")
    add("notes", stage=True)
    add("gate", stage=True).add_argument("--approved", action="store_true")
    add("adopt").add_argument("--through", required=True, choices=STAGES)
    sub.add_parser("html", help="重新生成 data/PIPELINE.html")
    a = p.parse_args()

    handlers = {"init": cmd_init, "status": cmd_status, "set-tasks": cmd_set_tasks,
                "submit": cmd_submit, "resubmit": cmd_resubmit,
                "approve": lambda x: cmd_review(x, True), "reject": lambda x: cmd_review(x, False),
                "notes": cmd_notes, "gate": cmd_gate, "adopt": cmd_adopt, "html": lambda x: 0}
    rc = handlers[a.cmd](a)
    if a.cmd not in ("notes", "gate"):
        write_html()
    return rc


if __name__ == "__main__":
    sys.exit(main())
