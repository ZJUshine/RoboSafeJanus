#!/usr/bin/env python3
"""看板大表里点开一格后的编辑区:列出这一格能改的东西,改完回填到对应的 JSON / YAML / Python 文件。

每个审核条目(``<stage>`` 或 ``<stage>/<task>``)能改的"槽"(``slots``)由该阶段的产物推出来::

    seed     task/risk_seeds.json 里这个种子
    ingest   task/robosafejanus_assets.json[<group>](整组的下载命令)、入库报告
    design   设计文件里该任务的 instruction、S1、S2(S1 / S2 并排),sim 里该任务(不含 success)、core_risk
    layout   task/RoboSafeJanus/config/<task>.yml、tasks/<task>.py 及其 *_X_* 组模块
    expert   experts/<task>.py 及其组模块、sim 里该任务的 success
    eval     提交时附的报告(.md / .json)

槽的种类(``kind``):``json``(JSON 文件里的一段,按选择器 ``a.b`` / ``[k=v]`` 定位)、``jsonstr``(JSON 里的一个字符串,
直接编辑文字)、``json-nosuccess``(sim 里的任务,保存时保留原来的 success)、``text``(整个文本文件)。
保存时检查格式(JSON / YAML / Python 语法),并比较打开时的版本:文件在打开之后被别人改过就拒绝,要重新打开。
JSON 按文件原来的缩进写回。只能写这里列出的槽(页面只传槽的编号),不能写任意路径。

改了已批准 / 待审的格子,这一格会变成「失效」/「提交后已改」,在审核条目里点「重新提交」;下游的闸门随之关闭。

assets 格的「填资产链接」(``assets_load`` / ``assets_save``):一行一个物体(label + 链接),链接换成带 commit 的
``hf download`` 命令,整组写回 ``task/robosafejanus_assets.json[<group>]``;有一行不对就整组不写。

命令行(调试用)::

    python scripts/internal/pipeline_edit.py slots <group> <key>          # 如 design/put_phone_toaster
    python scripts/internal/pipeline_edit.py show <group> <key> <slot>
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsj_pipeline as rp  # noqa: E402

ROOT = rp.ROOT
TEXT_SUFFIXES = (".py", ".yml", ".yaml", ".md", ".json", ".txt")
WRITABLE = ("task/", "data/pipeline/")  # 只允许写这些目录下的文件(相对仓库根)
MAX_BYTES = 2 << 20


class EditError(Exception):
    """不能读 / 写,消息直接给用户看。"""


def _rel_ok(path_s: str) -> bool:
    path = (ROOT / path_s).resolve()
    try:
        rel = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return False
    return rel.startswith(WRITABLE) or path_s.startswith(WRITABLE)


def _slot(label: str, path: str, selector: str = "", kind: str = "", note: str = "", pair: str = "") -> dict:
    if not kind:
        kind = "json" if selector else "text"
    return {"label": label, "path": path, "selector": selector, "kind": kind, "note": note, "pair": pair}


def slots(state: dict, key: str) -> list[dict]:
    """这一格能改的槽(顺序即页面上的顺序);编号是在列表里的下标。"""
    g = state["group"]
    stage, _, task = key.partition("/")
    out: list[dict] = []
    if stage == "seed":
        out.append(_slot("风险种子", "task/risk_seeds.json", f"[name={g}]"))
    elif stage == "ingest":
        out.append(_slot("资产下载命令(整组)", "task/robosafejanus_assets.json", g,
                         note="label → hf download 命令;改了 ingest 会失效,要重新下载 / 提交(也可以点「改资产链接」)"))
        report = (state["stages"].get(key) or {}).get("report")
        if report:
            out.append(_slot("入库报告", report))
    elif stage == "design":
        design = f"task/robosafejanus_tasks/{g}.json"
        out.append(_slot("指令 instruction", design, f"tasks.{task}.instruction", kind="jsonstr"))
        out.append(_slot("S1(安全设置变体)", design, f"tasks.{task}.S1", pair="s"))
        out.append(_slot("S2(题目)", design, f"tasks.{task}.S2", pair="s"))
        out.append(_slot("sim 里该任务(资产 / setup / S1 安全设置;success 在 expert 格改)",
                         f"task/robosafejanus_sim/{g}.json", f"tasks.{task}", kind="json-nosuccess"))
        out.append(_slot("core_risk(整组)", design, "core_risk", kind="jsonstr", note="改了整组的 design 都要重新提交"))
    elif stage in ("layout", "expert"):
        for spec in rp.default_artifacts(state, stage, task):
            path_s, _, selector = spec.partition("#")
            if selector:
                out.append(_slot("sim 里该任务的 success", path_s, selector,
                                 note="成功判定(check 列表);也可以用 task_success_spec.py 回填"))
            else:
                name = Path(path_s).name
                shared = "组模块,同组其他任务共用" if "_X_" in name else ""
                out.append(_slot(name, path_s, note=shared))
    else:  # eval 等:提交时附的文本报告
        entry = state["stages"].get(key) or {}
        for spec in [entry.get("report"), *entry.get("artifacts", {})]:
            if spec and "#" not in spec and not spec.startswith("state:") and spec.endswith(TEXT_SUFFIXES) \
                    and (ROOT / spec).is_file():
                out.append(_slot(Path(spec).name, spec))
    for i, s in enumerate(out):
        s["id"] = i
    return out


# ---------------------------------------------------------------------------
# JSON 选择器
# ---------------------------------------------------------------------------


def _parts(selector: str) -> list[str]:
    return [p for p in selector.split(".") if p]


def _get(data, selector: str):
    return rp._select(data, selector) if selector else data


def _set(data, selector: str, value):
    """把 ``selector`` 指的位置换成 ``value``(中间的字典键不存在时新建;列表元素必须已存在)。"""
    parts = _parts(selector)
    if not parts:
        return value
    node = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        m = re.fullmatch(r"\[(\w+)=(.+)\]", part)
        if m:
            k, v = m.groups()
            if not isinstance(node, list):
                raise EditError(f"{selector}: {part} 不是列表")
            idx = next((j for j, x in enumerate(node) if isinstance(x, dict) and str(x.get(k)) == v), None)
            if idx is None:
                raise EditError(f"{selector}: 列表里没有 {k} == {v}")
            if last:
                node[idx] = value
            else:
                node = node[idx]
        else:
            if not isinstance(node, dict):
                raise EditError(f"{selector}: {part} 的上一级不是对象")
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
    return data


def _indent(text: str) -> int:
    m = re.search(r"\n( +)\S", text)
    return len(m.group(1)) if m else 2


def _dump(data, indent: int) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# 读 / 写
# ---------------------------------------------------------------------------


def _version(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_slot(slot: dict) -> dict:
    """``{text, version, exists, editable, lang}``;文件不存在时不能编辑(要新建请让 Claude 做这一格或在终端里做)。"""
    path = ROOT / slot["path"]
    lang = {".py": "python", ".yml": "yaml", ".yaml": "yaml", ".md": "markdown"}.get(path.suffix, "json")
    info = {"lang": lang if slot["kind"] == "text" else ("text" if slot["kind"] == "jsonstr" else "json")}
    if not path.is_file():
        return {**info, "text": "", "version": "", "exists": False, "editable": False,
                "why": f"{slot['path']} 还不存在(让 Claude 做这一格,或在终端里新建)"}
    if path.stat().st_size > MAX_BYTES:
        return {**info, "text": "", "version": "", "exists": True, "editable": False, "why": "文件太大,不在页面上编辑"}
    raw = path.read_text(encoding="utf-8")
    if slot["kind"] == "text":
        return {**info, "text": raw, "version": _version(raw), "exists": True, "editable": _rel_ok(slot["path"])}
    try:
        data = json.loads(raw)
    except ValueError as error:
        return {**info, "text": raw, "version": "", "exists": True, "editable": False,
                "why": f"{slot['path']} 现在不是合法 JSON:{error}"}
    value = _get(data, slot["selector"])
    if slot["kind"] == "json-nosuccess" and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != "success"}
    if slot["kind"] == "jsonstr":
        text = value if isinstance(value, str) else ("" if value is None else json.dumps(value, ensure_ascii=False))
    else:
        text = "" if value is None else json.dumps(value, indent=2, ensure_ascii=False)
    return {**info, "text": text, "version": _version(text), "exists": value is not None,
            "editable": _rel_ok(slot["path"])}


def _check_text(path: Path, text: str) -> None:
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as error:
            raise EditError(f"Python 语法错误:第 {error.lineno} 行 {error.msg}") from None
    elif path.suffix in (".yml", ".yaml"):
        import yaml

        try:
            yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise EditError(f"YAML 格式错误:{error}") from None
    elif path.suffix == ".json":
        try:
            json.loads(text)
        except ValueError as error:
            raise EditError(f"JSON 格式错误:{error}") from None


def write_slot(slot: dict, text: str, version: str) -> str:
    """写回;``version`` 是打开时 ``read_slot`` 给的版本,和现在的不同就拒绝。返回给用户看的说明。"""
    current = read_slot(slot)
    if not current["editable"]:
        raise EditError(current.get("why") or "这一项不能在页面上编辑")
    if current["version"] != version:
        raise EditError(f"{slot['path']} 在你打开之后被改过了(可能是 Claude 或终端里改的),关掉对话框重新打开再改")
    if text == current["text"]:
        return "没有改动"
    path = ROOT / slot["path"]
    if slot["kind"] == "text":
        _check_text(path, text)
        path.write_text(text, encoding="utf-8")
        return f"已写回 {slot['path']}"
    if slot["kind"] == "jsonstr":
        value = text
    else:
        try:
            value = json.loads(text)
        except ValueError as error:
            raise EditError(f"JSON 格式错误:{error}") from None
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if slot["kind"] == "json-nosuccess":
        if not isinstance(value, dict):
            raise EditError("sim 里的任务必须是一个对象 {...}")
        old = _get(data, slot["selector"]) or {}
        if "success" in value:
            raise EditError("success 在 expert 格里改,这里请去掉 success")
        if isinstance(old, dict) and "success" in old:  # 放回原来的位置,不打乱键的顺序
            items = list(value.items())
            pos = min(list(old).index("success"), len(items))
            items.insert(pos, ("success", old["success"]))
            value = dict(items)
    data = _set(data, slot["selector"], value)
    path.write_text(_dump(data, _indent(raw)), encoding="utf-8")
    return f"已写回 {slot['path']} 的 {slot['selector']}"


# ---------------------------------------------------------------------------
# 看板接口
# ---------------------------------------------------------------------------


NAME_RE = re.compile(r"[A-Za-z0-9_]+")


def _entry(group: str, key: str) -> tuple[dict, str]:
    if not NAME_RE.fullmatch(group or ""):
        raise EditError("组名不合法")
    stage, _, task = (key or "").partition("/")
    if stage not in rp.STAGES or (task and not NAME_RE.fullmatch(task)) or bool(task) != (stage in rp.TASK_STAGES):
        raise EditError("格子不合法")
    if not rp.state_path(group).exists():
        raise EditError(f"{group}: 还没有流水线状态")
    return rp.load_state(group), key


def load(group: str, key: str) -> dict:
    """页面打开一格时:这一格的槽和内容、当前审核状态。"""
    state, key = _entry(group, key)
    status, _ = rp.entry_status(state, key)
    items = []
    for s in slots(state, key):
        items.append({**{k: s[k] for k in ("id", "label", "path", "selector", "kind", "note", "pair")}, **read_slot(s)})
    warn = {"approved": "这一格已批准:保存后会变成「失效」,下游的闸门关闭,要重新提交、批准",
            "awaiting_review": "这一格在待审:保存后会变成「提交后已改」,要在审核条目里点「重新提交」"}.get(status, "")
    return {"group": group, "key": key, "status": status, "warn": warn, "busy": False, "slots": items}


def save(group: str, key: str, slot_id: int, text: str, version: str) -> dict:
    state, key = _entry(group, key)
    items = slots(state, key)
    if not isinstance(slot_id, int) or not 0 <= slot_id < len(items):
        raise EditError("没有这一项")
    message = write_slot(items[slot_id], text, version)
    state = rp.load_state(group)
    status, why = rp.entry_status(state, key)
    after = read_slot(items[slot_id])
    return {"message": message, "status": status, "status_label": rp.STATUS_LABEL.get(status) or "未开始",
            "why": why, "version": after["version"], "text": after["text"]}


# ---------------------------------------------------------------------------
# assets 格的「填资产链接」:一行一个物体(label + 链接),回填 task/robosafejanus_assets.json[<group>]
# ---------------------------------------------------------------------------

ALLOWED_REPOS = rp.ALLOWED_REPOS
LABEL_RE = re.compile(r"[a-z][a-z0-9_]*")
SHA_RE = re.compile(r"[0-9a-f]{40}")
# huggingface.co / hf-mirror.com 网页链接(tree / blob / resolve),以及 SimReady Central 给的 hf://...
HF_LINK = re.compile(r"(?:hf://|https?://)?(?:www\.)?(?:huggingface\.co/|hf-mirror\.com/)?datasets/"
                     r"([^/\s]+/[^/\s]+)/(?:tree|blob|resolve)/([^/\s]+)/([^?#\s]+)")
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def _resolve_revision(repo: str, rev: str) -> str:
    """链接里是分支名(如 main)时查出 commit;查不到就让用户换带 commit 的链接。"""
    url = f"{MIRROR}/api/datasets/{repo}/revision/{urllib.parse.quote(rev, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=20) as res:
            sha = json.loads(res.read()).get("sha") or ""
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise EditError(f"链接里的版本是 {rev}(不是 commit),去 {MIRROR} 查 commit 失败:{error};"
                        "换成带 commit 的链接或直接贴 hf download 命令") from None
    if not SHA_RE.fullmatch(sha):
        raise EditError(f"查不到 {repo} 的 {rev} 对应的 commit")
    return sha


def _command(repo: str, rev: str, folder: str) -> str:
    local = ("robodojo" if repo == rp.ROBODOJO_REPO else "simready") + f"-assets-{rev[:8]}"
    return f"hf download {repo} --repo-type dataset --revision {rev} --include '{folder}/*' --local-dir {local}"


def _asset_dir(repo: str, path: str) -> str:
    parts = [x for x in path.strip("/").split("/") if x]
    if repo == rp.ROBODOJO_REPO:  # Assets/Object/RoboDojo/<类型>/<名字>/<编号>
        if len(parts) < 6 or parts[:3] != ["Assets", "Object", "RoboDojo"]:
            raise EditError("RoboDojo 官方的链接要指到具体模型目录,如 Assets/Object/RoboDojo/Rigid/bread/00000")
        return "/".join(parts[:6])
    if not parts:
        raise EditError("链接里没有资产目录")
    return parts[0]  # SimReady 库:第一级目录就是一个资产(如 D00738、food_egg_chicken_b03)


def link_to_command(link: str, robodojo_rev: str | None) -> str:
    """把用户贴的链接换成 ``hf download`` 命令;已经是命令时只检查格式。不合规时抛 ``EditError``。"""
    link = link.strip()
    if link.startswith("hf download"):
        try:
            args = shlex.split(link)
        except ValueError as error:
            raise EditError(f"命令解析不了:{error}") from None
        opt = {k: args[i + 1] for i, k in enumerate(args[:-1]) if k.startswith("--")}
        repo = args[2] if len(args) > 2 else ""
        if repo not in ALLOWED_REPOS:
            raise EditError(f"来源 {repo or '(空)'} 不允许(只能是 RoboDojo 官方和 4 个 SimReady 库)")
        if opt.get("--repo-type") != "dataset" or not SHA_RE.fullmatch(opt.get("--revision", "")) \
                or not opt.get("--include"):
            raise EditError("命令要带 --repo-type dataset、--revision <40 位 commit> 和 --include '<目录>/*'")
        _asset_dir(repo, opt["--include"].rstrip("*"))
        return " ".join(link.split())
    m = HF_LINK.search(urllib.parse.unquote(link))
    if m:
        repo, rev, path = m.groups()
        if repo not in ALLOWED_REPOS:
            raise EditError(f"来源 {repo} 不允许(只能是 RoboDojo 官方和 4 个 SimReady 库)")
        folder = _asset_dir(repo, path)
        return _command(repo, rev if SHA_RE.fullmatch(rev) else _resolve_revision(repo, rev), folder)
    if link.startswith("Assets/Object/RoboDojo/"):
        if not robodojo_rev:
            raise EditError("不知道 RoboDojo 官方用哪个 revision:贴完整的 huggingface 链接或 hf download 命令")
        return _command(rp.ROBODOJO_REPO, robodojo_rev, _asset_dir(rp.ROBODOJO_REPO, link))
    raise EditError("认不出这个链接:贴 huggingface.co / hf-mirror.com 上资产目录的网页链接、SimReady Central 的 "
                    "hf://... 路径、Assets/Object/RoboDojo/... 目录或 hf download 命令")


def describe_command(cmd: str) -> str:
    try:
        args = shlex.split(cmd)
    except ValueError:
        return ""
    opt = {k: args[i + 1] for i, k in enumerate(args[:-1]) if k.startswith("--")}
    repo = args[2] if len(args) > 2 else ""
    kind = "RoboDojo 官方" if repo == rp.ROBODOJO_REPO else f"SimReady · {repo.split('/')[0]}"
    return f"{kind} · {opt.get('--include', '').rstrip('*').rstrip('/')} · {opt.get('--revision', '')[:8]}"


def seed_labels(group: str) -> list[str]:
    """从种子里猜的 label(括号里的英文名):目标物体、交互物体、风险 / 安全变体,外加 S1.2 的 moving_obstacle。"""
    try:
        seeds = rp.read_json(rp.ROOT / "task" / "risk_seeds.json")
    except (OSError, ValueError):
        return []
    seed = next((s for s in seeds if isinstance(s, dict) and s.get("name") == group), None) or {}
    words = [seed.get("target_object"), seed.get("interact_object")]
    for key in ("risk_variation", "safe_variation"):
        words += (seed.get(key) or {}).get("elements") or []
    out = []
    for w in words:
        m = re.search(r"[(（]\s*([A-Za-z][A-Za-z0-9 _\-]*)\s*[)）]", str(w or ""))
        if m:
            label = re.sub(r"[\s\-]+", "_", m.group(1).strip()).lower()
            if label not in out:
                out.append(label)
    return out + (["moving_obstacle"] if "moving_obstacle" not in out else [])


def _assets_version(sources: dict) -> str:
    return _version(json.dumps(sources, ensure_ascii=False))


def assets_load(group: str) -> dict:
    """「填资产链接」打开时:该组现有的 label → 命令、按种子猜的 label、种子摘要。"""
    state, _ = _entry(group, "seed")
    data = rp.read_json(rp.ASSETS_JSON) if rp.ASSETS_JSON.exists() else {}
    sources = data.get(group) or {}
    seed_status, _ = rp.entry_status(state, "seed")
    warn = "" if seed_status == "approved" else "seed 还没批准:批准后才能填资产"
    ingest, _ = rp.entry_status(state, "ingest")
    if ingest in ("approved", "awaiting_review") and not warn:
        warn = "ingest 已" + ("批准" if ingest == "approved" else "提交") + ":改了链接 ingest 会失效,要重新下载 / 提交"
    rows = [{"label": k, "link": v, "info": describe_command(v)} for k, v in sources.items()]
    return {"group": group, "rows": rows, "suggest": [x for x in seed_labels(group) if x not in sources],
            "version": _assets_version(sources), "warn": warn,
            "can_save": seed_status == "approved"}


def assets_save(group: str, rows: list, version: str, dry_run: bool = False) -> dict:
    """检查每一行并(``dry_run`` 为假时)写回;有一行不对就整体不写,``errors`` 按行号给原因。"""
    state, _ = _entry(group, "seed")
    if not isinstance(rows, list):
        raise EditError("rows 不合法")
    data = rp.read_json(rp.ASSETS_JSON) if rp.ASSETS_JSON.exists() else {}
    old = data.get(group) or {}
    rd_rev = None
    try:
        import asset_candidates

        rd_rev = asset_candidates.robodojo_revision(data)
    except Exception:  # noqa: BLE001 - 只是给 Assets/Object/RoboDojo/... 补 revision
        rd_rev = None
    out, errors, new = [], {}, {}
    for i, row in enumerate(rows):
        label = str((row or {}).get("label") or "").strip()
        link = str((row or {}).get("link") or "").strip()
        if not label and not link:
            continue
        try:
            if not LABEL_RE.fullmatch(label):
                raise EditError("label 用英文小写 snake_case(如 metal_tray),将来就是 sim assets 的键")
            if label in new:
                raise EditError(f"label {label} 重复了")
            if not link:
                if label in old:
                    raise EditError("链接清空了:要删掉这个物体请点 ×")
                out.append({"row": i, "label": label, "link": "", "skip": True})  # 按种子列出来还没填的,不写
                continue
            cmd = old[label] if old.get(label) == link else link_to_command(link, rd_rev)
            repo = shlex.split(cmd)[2]
            if repo != rp.ROBODOJO_REPO:  # 同一个 label 在别的组也用 SimReady 时必须是同一个模型
                clash = [g for g, s in data.items() if g != group and isinstance(s, dict) and s.get(label)
                         and s[label] != cmd and shlex.split(s[label])[2:3] != [rp.ROBODOJO_REPO]]
                if clash:
                    raise EditError(f"{', '.join(clash)} 组的 {label} 用的是另一个 SimReady 模型(挂进 Assets 都叫 "
                                    f"SimReady_{label},会冲突):用同一个链接,或者换个 label 名")
            new[label] = cmd
            out.append({"row": i, "label": label, "link": cmd, "info": describe_command(cmd)})
        except EditError as error:
            errors[i] = str(error)
            out.append({"row": i, "label": label, "link": link, "error": str(error)})
    added = [k for k in new if k not in old]
    changed = [k for k in new if k in old and new[k] != old[k]]
    removed = [k for k in old if k not in new]
    parts = (f"新增 {', '.join(added)}" if added else "", f"改了 {', '.join(changed)}" if changed else "",
             f"删掉 {', '.join(removed)}" if removed else "")
    diff = "; ".join(x for x in parts if x) or "没有改动"
    result = {"rows": out, "errors": errors, "diff": diff, "version": _assets_version(old)}
    if errors or dry_run:
        return result
    seed_status, _ = rp.entry_status(state, "seed")
    if seed_status != "approved":
        raise EditError("seed 还没批准,先批准 seed 再填资产")
    if _assets_version(old) != version:
        raise EditError("资产 JSON 里这个组在你打开之后被改过了(可能是 Claude 或终端里改的),关掉重新打开再填")
    if diff == "没有改动":
        return {**result, "message": "没有改动"}
    raw = rp.ASSETS_JSON.read_text(encoding="utf-8") if rp.ASSETS_JSON.exists() else "{}"
    data[group] = new
    rp.ASSETS_JSON.write_text(_dump(data, _indent(raw)), encoding="utf-8")
    state = rp.load_state(group)
    message = f"已写回 task/{rp.ASSETS_JSON.name}[{group}]:{diff}"
    if rp.entry_status(state, "ingest")[0] in ("stale", "changed"):
        message += "。ingest 已失效,要重新下载 / 提交"
    elif rp.entry_status(state, "ingest")[0] == "none":
        message += "。下一步:ingest 格的「⬇ 下载资产」"
    return {**result, "message": message, "version": _assets_version(new)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slots")
    s.add_argument("group")
    s.add_argument("key")
    s = sub.add_parser("show")
    s.add_argument("group")
    s.add_argument("key")
    s.add_argument("slot", type=int)
    a = ap.parse_args()
    try:
        data = load(a.group, a.key)
    except EditError as error:
        print(error, file=sys.stderr)
        return 2
    if a.cmd == "slots":
        print(f"{a.group} {a.key}: {data['status']} {data['warn']}")
        for s in data["slots"]:
            flag = "可编辑" if s["editable"] else "只读:" + s.get("why", "")
            where = s["path"] + ("#" + s["selector"] if s["selector"] else "")
            print(f"  [{s['id']}] {s['label']}  {where}  ({s['kind']}, {flag})")
    else:
        print(data["slots"][a.slot]["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
