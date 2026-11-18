# ruff: noqa: E501  -- 主要是 HTML / CSS / JS 模板,长行按页面结构排版
"""看板的流水线部分,由 ``sync_status.py`` 和 ``rsj_pipeline.py`` 生成。看板只有一个页面 ``data/PIPELINE.html``
(sync_status 生成),内容只有一张流水线大表(``table_html``):每行一个任务,从左到右是任务构造 seed … expert
(审核阶段)、数据收集与上传 collect / exchange / upload(``JOB_HEADS``,不审核)、模型评测 布局 / 模型 / eval / 结果
(``WIDE_COLUMNS``,不审核;模型 / 结果 / 布局与视频、远端状态由 sync_status 写进 ``CELLS_FILE``)。

- 点状态格:屏幕中间弹出这一阶段的审核条目(检查、改动、批准 / 驳回);点组名:弹出风险种子和全部阶段。
  条目预先写在隐藏的 ``#gstores`` 里,脚本挪进对话框 ``#pipe-dialog``(``ROUTE_JS``),关闭时放回;
  地址是 ``#e-<group>-<key>`` / ``#g/<group>``。所有按钮(Claude / 采集 / 转换 / 上传)也是弹出对话框。
- 审核格底色是审核状态,字是这一格做出来的东西(``stage_info``:物体数、入库数、题数、成功率…);任务格底色是层级。
- assets 列不审核、按组一格(``assets_cell``):seed 批准后有「🔗 填资产链接」(``link_button`` → ``#assets-dialog`` /
  ``ASSETS_JS`` → ``/api/assets/load|save`` → ``pipeline_edit.assets_save``),一行一个物体填链接,保存就算完成。
- ingest 按组一格,「⬇ 下载资产」(``ingest_button``,group_jobs ``ingest``)后台下载 / 转换 / 入库,完了提交并自动批准(入库不审核)。
- 点开后条目里有编辑区(``EDIT_JS`` → ``/api/edit/load|save`` → ``pipeline_edit.py``):这一格的 JSON / YAML / Python
  直接改,保存后回填到对应文件。
- collect / exchange / upload 格写实际数据(``job_info``:条数 + 视频、v3.0 / v2.1 条数、远端同步),expert 批准后可点。
- 任务构造各格的「▶ Claude」(``claude_button``)只给提示词(``claude_prompt``,弹出 ``#prompt-dialog`` / ``PROMPT_JS``),
  复制到终端里的 Claude Code 交互着做,不在后台起会话;只在未开始 / 驳回 / 失效 / 提交后已改的格子上有。
  未开始(从零开始)的格子点格子本身就直接弹出提示词(``data-direct``)。
- 失效 / 提交后已改的格子另有「✎ 人工审核」(``manual_button``):弹出审核条目,在编辑区里改、看相对 HEAD 的改动,
  点「✔ 人工审核通过」(``reapprove`` = ``resubmit --approve``,重新跑检查,通过即批准),不必交给 Claude 重做。
- 「采集」「转换」「上传」按钮由 sync_status 页面脚本处理:点了弹出计划对话框,确认后后台运行 ``group_jobs.py``。
- 模型格只有下载(已下载 / 「⬇ 下载」)和 W&B 同步(已上传 / 「⟳ 同步 W&B」),由 sync_status ``model_item_html`` 写。
- eval 列按 S1 变体分子列(``S1_VARIANTS``,2026-11-16 起;``variant_cell``,不审核):评过的格写 success / n /
  视频按钮(``36.0% (n=50)🎬 50``,sync_status 写在 ``eval:<S1.x>:<policy>``),没评过的写这一行模型在这个变体上
  已评几条;都带「▶ 评测」。目前只有 S1.0 / S1.1(``EVAL_VARIANTS``)接进了评测(``eval_button`` → group_jobs ``eval``,
  模型下载完后出现,评完不提交);其余变体按钮灰着,设计里为 null 的写「不适用」。
  变体没布局 / 还没接入时格子里另有「▶ Claude」(``variant_button``):提示词让 Claude 把这个任务这个变体的
  设计(sim JSON 的 S1.x assets / safety)、布局和评测接入做完,每个任务只在第一个模型行上出现一次。
- 表尾是还没开始的种子,带「从种子开始」。

片段 ``data/pipeline/_fragments/table.html`` 页面生成时内嵌;打开时先读同目录的 ``table.gen``(片段内容的 hash),
和内嵌的不一样(页面上操作后只重画了片段)才再读一次整个片段,所以操作完立刻看到新状态,平时不用重复下载。
旧地址 ``TASK_BUILD.html`` / ``SYNC_STATUS.html`` / ``MODELS_EVAL.html`` 是跳到 ``PIPELINE.html`` 的小页面。

用 ``sync_status.py --serve`` 打开时可以在页面上操作(``/api/pipeline/<action>``,见 ``ACTIONS``):

    init      从种子新建组(提交 seed 待审)
    approve   批准某阶段(可附说明)
    reject    驳回某阶段(必须写意见,Claude 重做时会读)
    resubmit  沿用上次的附件 / 指标重新提交(后台运行检查)
    reapprove 人工审核过:重新提交并直接批准(失效 / 提交后已改的格子,后台运行检查)
    pick      选定某个候选资产写进 robosafejanus_assets.json(ingest 会失效,要重新下载)

页面上的操作都转成对应的命令行(``rsj_pipeline.py`` / ``asset_candidates.py``),与在终端里执行完全相同。
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsj_pipeline as rp  # noqa: E402

PAGE = "PIPELINE.html"
FRAGMENT_DIR = "pipeline/_fragments"
FRAGMENTS = ("table",)
OLD_FRAGMENTS = ("overview", "build", "data", "eval")  # 以前分标签时的片段,render 时删掉
# 旧页面地址(旧链接、书签还能用,跳到 PIPELINE.html)
OLD_PAGES = ("TASK_BUILD.html", "SYNC_STATUS.html", "MODELS_EVAL.html")
# (键, 名称, 旧标签, 阶段, 说明)
PHASES = (
    ("build", "任务构造", "#build", ("seed", "assets", "ingest", "design", "layout", "expert"),
     "风险种子 → 任务列表与资产 → 设计 / sim JSON → 布局与指令 → 专家脚本"),
    ("data", "数据收集与上传", "#data", (),
     "批量采集 → LeRobot 转换 → 上传 ModelScope(格子里点按钮,不审核)"),
    ("eval", "模型评测", "#eval", (),
     "评测布局 → 下载模型 → 评测(格子里点按钮,不审核)"),
)
PHASE_OF = {stage: phase for phase in PHASES for stage in phase[3]}
# 动作 -> 在后台运行(检查可能要几分钟)还是同步运行
ACTIONS = {"init": False, "approve": False, "reject": False, "resubmit": True, "reapprove": True, "pick": False}
NAME_RE = re.compile(r"[A-Za-z0-9_]+")
CAND_RE = re.compile(r"[A-Za-z0-9_:/.\-]+")
STATUS_CLASS = {"approved": "good", "awaiting_review": "warn", "changed": "bad", "rejected": "bad",
                "stale": "stale", "none": "idle"}
# Claude 负责的阶段:页面上提示怎么让 Claude 开始 / 重做
CLAUDE_STAGES = {"design": "design 阶段", "layout": "布局阶段", "expert": "专家调试阶段"}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# 页面动作(供 sync_status 的 HTTP 接口调用)
# ---------------------------------------------------------------------------


def action_command(action: str, payload: dict) -> list[str]:
    """把页面请求换成命令行;参数不合法时抛 ``ValueError``。"""
    if action not in ACTIONS:
        raise ValueError(f"未知操作 {action}")
    group = str(payload.get("group") or "")
    if not NAME_RE.fullmatch(group):
        raise ValueError("组名不合法")
    py = [sys.executable, "-u"]
    if action == "init":
        return [*py, str(HERE / "rsj_pipeline.py"), "init", group]
    if action == "pick":
        label, cand = str(payload.get("label") or ""), str(payload.get("candidate") or "")
        if not NAME_RE.fullmatch(label) or not CAND_RE.fullmatch(cand):
            raise ValueError("label / 候选不合法")
        return [*py, str(HERE / "asset_candidates.py"), "pick", group, label, cand]
    stage = str(payload.get("stage") or "")
    if stage not in rp.STAGES:
        raise ValueError("阶段不合法")
    cmd = [*py, str(HERE / "rsj_pipeline.py"), "resubmit" if action == "reapprove" else action, group, stage]
    if action == "reapprove":
        cmd.append("--approve")
    task = payload.get("task")
    if task:
        if not NAME_RE.fullmatch(str(task)):
            raise ValueError("任务名不合法")
        cmd += ["--task", str(task)]
    note = str(payload.get("note") or "").strip()
    if action == "reject" and not note:
        raise ValueError("驳回需要写意见")
    if note:
        cmd += ["--note", note]
    return cmd


def run_action(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def load_seeds() -> tuple[list[dict], str]:
    """``(种子列表, 错误)``;risk_seeds.json 正在编辑、暂时不是合法 JSON 时页面照常生成。"""
    if not rp.SEEDS_JSON.exists():
        return [], ""
    try:
        return rp.read_json(rp.SEEDS_JSON), ""
    except (json.JSONDecodeError, OSError) as error:
        return [], f"{rp.SEEDS_JSON.name} 读取失败: {error}"


def load_snapshot(out_dir: Path) -> dict:
    """sync_status 的上一次快照(数据 / 评测统计);没有时为空。"""
    try:
        return json.loads((out_dir / "SYNC_STATUS.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _group_json(directory: Path, group: str) -> dict:
    path = directory / f"{group}.json"
    try:
        return rp.read_json(path).get("tasks", {}) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def task_info(state: dict) -> list[dict]:
    """组里每个任务一行(顺序同任务草稿):``name / level / instruction / owner``。

    层级取设计文件 S2.3 的 ground_truth,指令取设计文件的 ``instruction``,``owner`` 为 sim 里 ``same_as`` 的归属组;
    还没写设计时为空。没有任务草稿时返回一行占位(``name`` 为空)。
    """
    design, sim = _group_json(rp.DESIGN_DIR, state["group"]), _group_json(rp.SIM_DIR, state["group"])
    rows = []
    for name in state.get("draft_tasks") or []:
        spec = design.get(name) or {}
        gt = ((spec.get("S2") or {}).get("S2.3") or {}).get("ground_truth")
        rows.append({"name": name, "level": ("risky" if gt == "unsafe" else "safe") if gt else "",
                     "instruction": spec.get("instruction") or "",
                     "owner": (sim.get(name) or {}).get("same_as")})
    return rows or [{"name": "", "level": "", "instruction": "", "owner": None}]


def stage_keys(state: dict, stage: str) -> list[tuple[str, str | None]]:
    if stage in rp.TASK_STAGES:
        return [(f"{stage}/{t}", t) for t in rp.stage_tasks(state, stage)]
    return [(stage, None)]


def stage_href(group: str, key: str) -> str:
    """某个组某个阶段的审核条目(页面上点了在屏幕中间弹出)。"""
    return f"#e-{group}-{key}"


# ---------------------------------------------------------------------------
# 通用片段
# ---------------------------------------------------------------------------


def badge(status: str, why: str = "") -> str:
    label = rp.STATUS_LABEL.get(status) or "未开始"
    return f'<span class="badge {STATUS_CLASS.get(status, "idle")}" title="{esc(why)}">{esc(label)}</span>'


def checks_html(entry: dict) -> str:
    items = []
    for ok, msg in entry.get("checks", []):
        first, _, rest = str(msg).partition("\n")
        body = f"<pre>{esc(msg)}</pre>" if rest else esc(first)
        items.append(f'<li class="{"ok" if ok else "ng"}">{"✅" if ok else "❌"} {body}</li>')
    return f'<ul class="checks">{"".join(items)}</ul>' if items else ""


def notes_html(entry: dict) -> str:
    kinds = {"submit": "提交", "approve": "批准", "reject": "驳回", "force": "强制"}
    rows = [f'<li><small>{esc(n["at"])}</small> <b>{kinds.get(n["kind"], n["kind"])}</b> {esc(n["text"])}</li>'
            for n in entry.get("notes", []) if n.get("text") or n["kind"] != "submit"]
    return f'<ul class="notes">{"".join(rows)}</ul>' if rows else ""


def report_link(report: str | None, out_dir: Path) -> str:
    if not report:
        return ""
    path = (rp.ROOT / report).resolve()
    try:
        rel = path.relative_to(out_dir.resolve())
        return f'<a href="{esc(rel.as_posix())}" target="_blank">附件 {esc(report)}</a>'
    except ValueError:
        return f"附件 <code>{esc(report)}</code>"


def entry_html(state: dict, stage: str, key: str, task: str | None, out_dir: Path) -> str:
    """一个阶段(按任务的阶段为其中一个任务)的状态、检查、改动、记录和操作按钮。"""
    g = state["group"]
    entry = state["stages"].get(key)
    status, why = rp.entry_status(state, key)
    title = f"<code>{esc(task)}</code>" if task else f"<b>{esc(stage)}</b> <small>{esc(rp.STAGE_DESC[stage])}</small>"
    anchor = f'id="e-{esc(g)}-{esc(key)}"'
    data = f'data-group="{esc(g)}" data-stage="{esc(stage)}"' + (f' data-task="{esc(task)}"' if task else "")
    task_suffix = f" ({esc(task)})" if task else ""
    # 编辑区:打开对话框时由 EDIT_JS 读 /api/edit/load 填进来,改完回填到对应的文件(pipeline_edit.py)
    editbox = f'<div class="editbox" data-group="{esc(g)}" data-key="{esc(key)}"></div>'
    if stage == "ingest":
        editbox = (f'<p class="alinkrow">{link_button(state, "🔗 改资产链接")}<small class="muted">改了链接 ingest 会失效,'
                   f'要重新下载 / 提交</small></p>' if link_button(state) else "") + editbox
    if not entry:
        problems = rp.upstream_problems(state, stage, task)
        if problems:
            hint = "<p><small>等待上游:" + esc("; ".join(problems)) + "</small></p>"
        elif stage == "ingest":
            hint = "<p>可以开始:点表里 ingest 格的「⬇ 下载资产」,后台下载、转换、入库,完了自动批准(入库不审核)。</p>"
        elif stage in CLAUDE_STAGES:
            btn = claude_button(state, stage, task) if stage in BUILD_STAGES else ""
            hint = (f'<p>可以开始。点 {btn or "终端里的 Claude"} 拿提示词,贴到终端里的 Claude Code 交互着做'
                    f'(或者直接说<code class="copy">开始 {esc(g)} 的{CLAUDE_STAGES[stage]}{task_suffix}</code>)。</p>')
        else:
            hint = "<p>可以开始。</p>"
        return f'<div class="entry" {anchor}><div class="ehead">{title} {badge(status)}</div>{hint}{editbox}</div>'

    facts = [f"提交 {esc(entry.get('submitted_at', ''))}"]
    if entry.get("approved_at"):
        facts.append(f"批准 {esc(entry['approved_at'])}")
    facts += [f"{esc(k)} = <b>{esc(v)}</b>" for k, v in entry.get("metrics", {}).items()]
    link = report_link(entry.get("report"), out_dir)
    if link:
        facts.append(link)
    parts = [f'<div class="ehead">{title} {badge(status, why)} <small>{" · ".join(facts)}</small></div>']
    if why:
        parts.append(f'<p class="why">{esc(why)}</p>')
    parts.append(editbox)
    parts.append(notes_html(entry))
    parts.append(checks_html(entry))
    if status in ("awaiting_review", "changed", "rejected", "stale"):
        diff = rp.git_diff(entry.get("artifacts", {}))
        if diff:
            parts.append(f'<details class="fold"><summary>相对 HEAD 的改动</summary><pre class="diff">{esc(diff)}</pre></details>')
    arts = "".join(f"<li><code>{esc(s)}</code> <small>{esc(h)}</small></li>" for s, h in entry.get("artifacts", {}).items())
    if arts:
        parts.append(f'<details class="fold"><summary>产物({len(entry["artifacts"])})</summary><ul>{arts}</ul></details>')

    buttons = []
    if status == "awaiting_review":
        buttons.append(f'<button class="act primary" data-action="approve" {data}>批准</button>')
        buttons.append(f'<button class="act" data-action="reject" {data}>驳回</button>')
    if status in ("changed", "stale"):
        # 人工编辑 / 审核:编辑区里改好、看过改动后直接批准;也可以只重新提交等以后审,或交给 Claude 重做
        buttons.append(f'<button class="act primary" data-action="reapprove" {data} '
                       f'title="重新跑检查,通过即批准(你已经人工看过 / 改过)">✔ 人工审核通过</button>')
        buttons.append(f'<button class="act" data-action="resubmit" {data} title="重新跑检查,变成待审">重新提交(待审)</button>')
    if status == "rejected":
        buttons.append(f'<button class="act" data-action="resubmit" {data}>改好了,重新提交</button>')
    if stage in BUILD_STAGES and task:
        buttons.append(claude_button(state, stage, task))
    buttons = [b for b in buttons if b]
    if buttons:
        note = ('<textarea class="note" rows="2" placeholder="说明 / 驳回意见(驳回必填,Claude 重做时会逐条读)">'
                "</textarea>" if status == "awaiting_review" else "")
        if status in ("changed", "stale"):
            note = ('<p class="manual"><b>人工编辑 / 审核</b>:在上面编辑区直接改这一格的文件(保存即写回),'
                    '再看下面「相对 HEAD 的改动」;确认没问题点「✔ 人工审核通过」(重新跑检查,通过即批准)。'
                    '不想自己改就点「↻ 重做」拿提示词交给 Claude。</p>'
                    '<textarea class="note" rows="2" placeholder="审核说明(可选,记进审核记录)"></textarea>')
        parts.insert(parts.index(editbox) + 1, f'<div class="actions">{note}<div class="btns">{"".join(buttons)}</div></div>')
    return f'<div class="entry" {anchor}>{"".join(p for p in parts if p)}</div>'


# ---------------------------------------------------------------------------
# 流水线大表
# ---------------------------------------------------------------------------


LEGEND = ('<p class="legend"><span class="badge good">通过</span><span class="badge warn">待审</span>'
          '<span class="badge bad">驳回 / 提交后已改</span><span class="badge stale">失效</span><span class="badge idle">未开始</span>'
          '<span class="legend-item lvl-safe">safe 任务</span><span class="legend-item lvl-risky">risky 任务</span>'
          '<small>任务构造格子底色是审核状态,字是这一格做出来的东西(物体数、入库数、题数、成功率…);'
          'collect / exchange / upload / eval 不审核,绿 = 做完,蓝 = 做了一部分;'
          'seed 按组(跨行合并),其余阶段每个任务各审一次;多个组都有的任务从 layout 起各组显示同一份(在哪一行审都一样)</small></p>')

# 一次 render 内复用的文件读取(sync_status 常驻,每次 render 开头清空)
_CACHE: dict = {}


def _cached(key, fn):
    if key not in _CACHE:
        try:
            _CACHE[key] = fn()
        except (OSError, ValueError, KeyError, TypeError):
            _CACHE[key] = None
    return _CACHE[key]


def group_sources(group: str) -> dict:
    """资产 JSON 里这个组的 ``{label: hf download 命令}``。"""
    return (_cached("assets", lambda: rp.read_json(rp.ASSETS_JSON)) or {}).get(group) or {}


def _ingested(group: str, label: str) -> tuple[bool | None, str]:
    """``(是否已入库, 入库后的目录)``;这个组的资产 JSON 里没有这个 label 时为 ``(None, "")``。"""
    targets = _cached(("ingest", group), lambda: {t["label"]: t for t in rp.ingest_targets(group)}) or {}
    t = targets.get(label)
    if not t:
        return None, ""
    where = t["paths"][-1] if t["paths"] else t["dir"]
    if t["repo"] == rp.ROBODOJO_REPO:
        return bool(list((rp.ROOT / t["dir"]).glob("object.usd*"))), where
    return all((rp.ROOT / p / "object.usd").exists() for p in t["paths"]) and len(t["paths"]) > 1, where


def stage_info(state: dict, stage: str, task: str) -> dict | None:
    """格子里显示的"做出了什么":``{main, lines: [(文字, class)], small, tip}``;算不出来时 None。

    ``lines`` 一行一项(assets / ingest 是资产名字,seed 的 risky / safe,design 的 S1 / S2)。
    """
    g = state["group"]
    entry = state["stages"].get(f"{stage}/{task}" if task else stage) or {}
    metrics = entry.get("metrics") or {}
    info = {"main": "", "lines": [], "small": "", "tip": ""}
    if stage == "seed":
        seed = next((s for s in (_cached("seeds", lambda: load_seeds()[0]) or []) if s.get("name") == g), None)
        if not seed:
            return None
        risky = len((seed.get("risk_variation") or {}).get("elements") or [])
        safe = len((seed.get("safe_variation") or {}).get("elements") or [])
        return {**info, "main": seed.get("risk_category") or "种子", "lines": [(f"risky {risky}", ""), (f"safe {safe}", "")]}
    if stage == "assets":  # 不是审核阶段,assets_cell 用
        sources = group_sources(g)
        if not sources:
            return None
        official = [k for k, v in sources.items() if rp.ROBODOJO_REPO in (v or "")]
        lines = [(k, "" if k in official else "sr") for k in sources]
        rest = len(sources) - len(official)
        small = " · ".join(x for x in (f"官方 {len(official)}" if official else "", f"SimReady {rest}" if rest else "") if x)
        tip = "\n".join(f"{k}: {'RoboDojo 官方' if k in official else 'SimReady'}" for k in sources)
        return {**info, "lines": lines, "small": small, "tip": tip}
    if stage == "ingest":
        labels = list(group_sources(g))
        if not labels:
            return None
        lines, tip, done = [], [], 0
        for k in labels:
            ok, where = _ingested(g, k)
            done += bool(ok)
            lines.append((("✅ " if ok else "❌ ") + k, "" if ok else "ng"))
            tip.append(f"{k}: {where or '?'}")
        return {**info, "lines": lines, "small": f"{done}/{len(labels)} 入库", "tip": "\n".join(tip)}
    if stage == "design":
        spec = (_cached(("design", g), lambda: _group_json(rp.DESIGN_DIR, g)) or {}).get(task) or {}
        if not spec:
            return None
        s2 = len(spec.get("S2") or {})
        s1 = len([k for k in (spec.get("S1") or {}) if k.startswith("S1.")])
        text = spec.get("instruction") or ""
        return {**info, "lines": [(f"S1 {s1} 项", ""), (f"S2 {s2} 题", "")],
                "tip": text if isinstance(text, str) else " / ".join(map(str, text))}
    if stage == "layout":
        path = rp.MIRROR / "config" / f"{task}.yml"
        text = _cached(("yml", task), lambda: path.read_text(encoding="utf-8")) if path.exists() else None
        if not text:
            return None
        n = sum(len([x for x in m.split(",") if x.strip()]) for m in re.findall(r"label:\s*\[([^\]]*)\]", text))
        return {**info, "main": f"{n} 个物体",
                "small": "固定布局模板" if re.search(r"--layout_template\s", text) else "随机布局"}
    if stage == "expert":
        rate = metrics.get("success_rate")
        numeric = rate not in (None, "adopted")
        if metrics.get("smoke") == "passed":  # 已通过冒烟测试(2026-11-15 起格子按这个写)
            return {**info, "main": "冒烟通过", "small": f"小批量 {rate}" if numeric else "",
                    "tip": "已通过冒烟测试" + (f";小批量成功率 {rate}" if numeric else "")}
        if rate is None:
            return None
        if rate == "adopted":
            return {**info, "main": "流水线前完成", "small": "adopt"}
        return {**info, "main": f"成功率 {rate}", "small": "小批量"}
    if metrics:
        return {**info, "main": " · ".join(f"{k} {v}" for k, v in metrics.items())}
    return None


SYNC_MARK = {"synced": "✅", "local_only": "⬆️", "diff": "⚠️", "remote_only": "⬇️", "missing": "❌", "unknown": "❔"}


def _lerobot_counts(task: str) -> tuple[dict, bool]:
    """``({"v30": 条数, "v21": 条数}, v2.1 是否早于 v3.0)``。"""
    base = rp.ROOT / "data" / f"{rp.BENCHMARK_NAME}-Lerobot"
    infos = {v: base / v / rp._task_rel(task) / "meta" / "info.json" for v in ("v30", "v21")}
    got = {v: (rp.read_json(p).get("total_episodes") or 0) if p.exists() else 0 for v, p in infos.items()}
    older = all(p.exists() for p in infos.values()) and infos["v21"].stat().st_mtime < infos["v30"].stat().st_mtime
    return got, older


def job_info(col: str, task: str, cell: dict) -> tuple[str, str, str, str]:
    """collect / exchange / upload 格(不审核):``(底色, 主文字, 小字, 按钮不能点的原因)``,原因为空表示可以点。"""
    n = rp._hdf5_count(task)
    shards = (rp.ROOT / "data" / rp.BENCHMARK_NAME / "_shards" / task).exists()
    collected = n >= rp.COLLECT_TARGET and not shards
    if col == "collect":
        klass = "good" if collected else ("todo" if n or shards else "idle")
        small = " · ".join(x for x in ((cell.get("sync") or {}).get("hdf5_size", ""), "有分片(没采完)" if shards else "") if x)
        return klass, f"{n}/{rp.COLLECT_TARGET} 条", small, ""
    got, older = _lerobot_counts(task)
    exported = collected and got["v30"] == n and got["v21"] == n and not older
    if col == "exchange":
        klass = "good" if exported else ("todo" if any(got.values()) else "idle")
        small = "v2.1 早于 v3.0,要重新转换" if older else ("和 hdf5 一致" if exported else (f"hdf5 {n} 条" if n else ""))
        return klass, f"v3.0 {got['v30']}\nv2.1 {got['v21']}", small, "" if collected else "先采够 " + str(rp.COLLECT_TARGET) + " 条"
    sync = cell.get("sync") or {}
    states = [sync.get(k) for k in ("hdf5", "v30", "v21")]
    small = "\n".join(f"{name} {SYNC_MARK.get(st, '❔')}" for name, st in zip(("HDF5", "v3.0", "v2.1"), states))
    if not sync:
        main, klass = "—", "idle"
    elif all(st == "synced" for st in states):
        main, klass = "已同步", "good"
    elif any(st in ("local_only", "diff") for st in states):
        main, klass = "待上传", "todo"
    else:
        main, klass = "远端没有", "idle"
    return klass, main, small, "" if exported else "先转换 LeRobot"


def status_cell(state: dict, key: str, span: int = 1, extra: str = "") -> str:
    """一格:底色是审核状态;通过的格子写做出了什么,其他状态写状态 + 做出了什么,未开始时写实际数据(如果有)。"""
    g = state["group"]
    status, why = rp.entry_status(state, key)
    stage, _, task = key.partition("/")
    try:
        info = stage_info(state, stage, task) if status != "none" else None
    except (OSError, ValueError, KeyError, TypeError):
        info = None
    info = info or {"main": "", "lines": [], "small": "", "tip": ""}
    if status == "approved":
        main = info["main"] or ("" if info["lines"] else "通过")
    else:
        main = "·" if status == "none" else (rp.STATUS_LABEL.get(status) or status)
    parts = [f'<span class="cmain">{esc(main)}</span>'] if main else []
    if info["lines"]:
        items = "".join(f'<span class="{c}">{esc(t)}</span>' for t, c in info["lines"])
        parts.append(f'<span class="clines">{items}</span>')
    if info["small"]:
        parts.append(f"<small>{esc(info['small'])}</small>")
    klass = STATUS_CLASS[status]
    rs = f' rowspan="{span}"' if span > 1 else ""
    tip = "\n".join(x for x in (" · ".join(y for y in (rp.STATUS_LABEL.get(status) or "未开始", why) if y),
                                info["tip"]) if x)
    return (f'<td class="st {klass}"{rs} title="{esc(tip)}">'
            f'<a href="{stage_href(g, key)}">{"".join(parts)}</a>{extra}</td>')


# 总览宽表里穿插的数据列:sync_status 每次重建时把每个任务的格子写进 CELLS_FILE
CELLS_FILE = "pipeline/_fragments/cells.json"
DATA_HEADS = {"eval_layout": ("评测布局", "seed 0"), "model:*": ("模型", "每个模型一行(ModelScope 模型仓库里有哪些 policy 就列哪些):下载 · W&B 同步")}
# 数据收集与上传:不审核,格子里写实际数据 + 按钮(job_cell)
JOB_HEADS = {"collect": ("collect", "批量采集 · 视频"), "exchange": ("exchange", "转换 LeRobot v3.0 + v2.1"),
             "upload": ("upload", "上传 ModelScope")}
# 大表列头的中文名(英文阶段名放在悬停提示里)
COL_LABEL = {"seed": "种子", "assets": "资产", "ingest": "入库", "design": "设计", "layout": "布局", "expert": "专家",
             "collect": "采集", "exchange": "转换", "upload": "上传", "eval": "评测"}
# (大类, 列):列是阶段名、JOB_HEADS 或 DATA_HEADS 的键(不能重名)。
# 评测列按 S1 执行测试分子列(task/Task_Design.md「S1 任务设计」):S1.0 是原任务,其余是加了安全设置的变体
S1_VARIANTS = {"S1.0": "原任务", "S1.1": "Static collision", "S1.2": "Dynamic collision",
               "S1.3": "Motion boundary", "S1.4": "Human interaction", "S1.5": "Manipulation stability"}
# 已经接进 robodojo.sh eval 的 S1 变体;其余变体的「▶ 评测」先灰着(sim JSON 里的 S1.x 场景 / safety 检查还没实现)
EVAL_VARIANTS = ("S1.0", "S1.1")  # S1.1:robodojo.sh eval --variant(src/eval_client/variants.py,2026-11-16)
# 模型评测:评测布局在最前;``*:*`` 列按模型分行——每个任务每个模型(cells 的 ``_policies``:ModelScope 上的 policy)一行,
# 同一模型的模型 / 评测在同一行,其他列上下合并。格子数据在 cells 的 ``model:<policy>``(下载 + W&B 同步);
# 评测格(``eval_cell``,不审核)每个 S1 变体一个子列,写这一行模型在这个变体上已评几条,评过的直接写
# success / n + 视频按钮(cells 的 ``eval:<S1.x>:<policy>``),「▶ 评测」只评这一行的模型。
WIDE_COLUMNS = (
    *[("build", s) for s in PHASES[0][3]],
    *[("data", c) for c in JOB_HEADS],
    ("eval", "eval_layout"), ("eval", "model:*"), *[("eval", f"eval:{v}:*") for v in S1_VARIANTS],
)


assert not (set(DATA_HEADS) | set(JOB_HEADS)) & set(rp.STAGES), "数据列的键与阶段重名"


def _rowspan(td: str, k: int) -> str:
    """按模型分行时,不分模型的任务格上下合并 ``k`` 行(``data-t`` 让 CSS 保留它的下边线)。"""
    return td if k <= 1 else td.replace("<td", f'<td rowspan="{k}" data-t', 1)


def load_cells(out_dir: Path) -> dict:
    try:
        return json.loads((out_dir / CELLS_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cell_policies(cells: dict | None) -> tuple[str, ...]:
    """宽表按模型分行的模型种类:sync_status 按 ModelScope 模型仓库写进 cells 的 ``_policies``;
    没有时(旧的 cells.json)退回 group_jobs.eval_policies()。"""
    import group_jobs as gj

    return tuple((cells or {}).get("_policies") or gj.eval_policies())


def _blocked(problems: list[str]) -> str:
    """闸门没过时按钮的提示:前几条上游问题。"""
    words = [re.sub(r": (\w+)", lambda m: ": " + (rp.STATUS_LABEL.get(m[1]) or "未提交"), p, count=1) for p in problems]
    more = f" 等 {len(words)} 项" if len(words) > 3 else ""
    return "要先批准:" + "; ".join(words[:3]) + more


BUILD_STAGES = ("design", "layout", "expert")  # 格子里是「▶ Claude」(给提示词);assets 用户自己填,ingest 是下载按钮


def assets_cell(state: dict, span: int, buttons: bool) -> str:
    """assets 列(按组一格,不审核):资产名字 + 「填资产链接」。底色:填好绿、没填蓝、来源不对红。"""
    g = state["group"]
    try:
        info = stage_info(state, "assets", "")
    except (OSError, ValueError, KeyError, TypeError):
        info = None
    problems = rp.assets_problems(g) if info else []
    klass = "bad" if problems else ("good" if info else "todo")
    parts = []
    if info:
        items = "".join(f'<span class="{c}">{esc(t)}</span>' for t, c in info["lines"])
        parts.append(f'<span class="clines">{items}</span><small>{esc(info["small"])}</small>')
    else:
        parts.append('<small>资产由你来填</small>')
    if problems:
        parts.append(f'<small class="ng">{esc(problems[0].split(": ", 1)[-1])}</small>')
    if buttons:
        parts.append(link_button(state, "🔗 改资产链接" if info else "🔗 填资产链接"))
    rs = f' rowspan="{span}"' if span > 1 else ""
    tip = "不审核:填好链接保存就算完成,下一步是 ingest 的「⬇ 下载资产」" + ("\n" + info["tip"] if info else "")
    return f'<td class="st ac {klass}"{rs} title="{esc(tip)}">{"".join(parts)}</td>'


def ingest_button(state: dict) -> str:
    """ingest 格的「⬇ 下载资产」:未开始 / 驳回 / 失效 / 提交后已改时有,点了弹出计划,确认后后台运行 group_jobs.py ingest。"""
    status, _ = rp.entry_status(state, "ingest")
    if status not in ("none", "rejected", "stale", "changed"):
        return ""
    g = esc(state["group"])
    problems = rp.upstream_problems(state, "ingest", None)
    label = "⬇ 下载资产" if status == "none" else "↻ 重新下载"
    tip = ("; ".join(p.split(": ", 1)[-1] for p in problems) if problems else
           "后台下载 SimReady 原始模型、转换成 robodojo_format、挂进 Assets,写入库报告后提交,检查通过自动批准")
    dis = " disabled data-blocked=1" if problems else ""
    return (f'<button class="gact" data-kind="ingest" data-group="{g}" data-task="{GROUP_TASK}" title="{esc(tip)}"{dis}>'
            f'{label}</button><div class="gjob" data-group="{g}" data-task="{GROUP_TASK}" data-kind="ingest"></div>')


GROUP_TASK = "__group__"  # 按组的后台任务(ingest)在 group_jobs 里的任务名(group_jobs.GROUP_TASK)


def layouts_button(state: dict, task: str) -> str:
    """评测布局格的「⊕ 生成布局」:不到 50 个时有;layout 没批准时灰掉。点了弹出计划(同组缺布局的一起生成),
    确认后后台运行 group_jobs.py layouts(Isaac 里查稳定 seed,再写 50 个布局)。"""
    import group_jobs as gj

    if task not in rp.own_tasks(state) or gj.eval_layouts(task) >= gj.EVAL_NUM:
        return ""
    g = state["group"]
    blocked = gj.layouts_blocked(state, task)
    tip = blocked or f"在 Isaac 里查能稳定沉降的 seed,写 {gj.EVAL_NUM} 个评测布局(同组缺布局的任务一起,共用 seed)"
    dis = " disabled data-blocked=1" if blocked else ""
    return (f'<button class="gact" data-kind="layouts" data-group="{esc(g)}" data-task="{esc(task)}" '
            f'title="{esc(tip)}"{dis}>⊕ 生成布局</button>'
            f'<div class="gjob" data-group="{esc(g)}" data-task="{esc(task)}" data-kind="layouts"></div>')


def eval_button(state: dict, task: str, policy: str = "", variant: str = "") -> str:
    """eval 格的「▶ 评测」:这个任务的模型(ModelScope 上有的)下载完、评测布局够 50 个才出现;expert 没批准时灰掉。

    点了弹出计划(group_jobs ``eval``:要评的模型、已评 / 续评、GPU),确认后后台跑 ``robodojo.sh eval``。
    ``policy`` 给出时(宽表按模型分开的 eval 格)只管这个模型。``variant``(S1.1 …)时评这个 S1 变体:
    有它的布局就出现,条数 = 布局个数。评完不提交(eval 不审核)。"""
    import group_jobs as gj

    g = state["group"]
    downloading = _cached("downloading", gj._downloading) or set()
    try:
        models = [m for m in gj.eval_models(g, task, downloading) if not policy or m["policy"] == policy]
    except (OSError, ValueError, KeyError):
        return ""
    ready = [m for m in models if m["ready"]]
    target = gj.eval_target(task, variant)
    if not ready or (gj.eval_layouts(task) < gj.EVAL_NUM if not variant else not target):
        return ""
    blocked = gj.eval_blocked(state, g, task, downloading, policy, variant)
    names = "、".join(f"{m['policy']}{'(zero-shot)' if m['zero_shot'] else ''}" for m in ready)
    each = "" if policy else "各"
    what = f"{variant}(加障碍物,记成功率 + 违规率)" if variant else ""
    tip = blocked or f"用已下载的 {names} {each}评 {what}{target} 条(点了先列计划、选 GPU,确认后后台运行)"
    dis = " disabled data-blocked=1" if blocked else ""
    pol = f' data-policy="{esc(policy)}"' if policy else ""
    pol += f' data-variant="{esc(variant)}"' if variant else ""
    return (f'<button class="gact" data-kind="eval" data-group="{esc(g)}" data-task="{esc(task)}"{pol} '
            f'title="{esc(tip)}"{dis}>▶ 评测</button>'
            f'<div class="gjob" data-group="{esc(g)}" data-task="{esc(task)}" data-kind="eval"{pol}></div>')


def task_variants(state: dict, task: str) -> set[str]:
    """这个任务设计里有的 S1 变体(``S1.x`` 为 null 的不适用);S1.0 总有。"""
    g = state["group"]
    spec = (_cached(("design", g), lambda: _group_json(rp.DESIGN_DIR, g)) or {}).get(task) or {}
    return {"S1.0"} | {k for k, v in (spec.get("S1") or {}).items() if k.startswith("S1.") and v is not None}


# 每个 S1 变体:一句话含义(task/Task_Design.md)、要先问用户的事
VARIANT_BRIEF = {
    "S1.1": ("在交互物附近放静态障碍物,测是否碰到", "障碍物放哪"),
    "S1.2": ("执行中让动态障碍物进入机械臂路径,测是否避让或停止",
             "它怎么动(路线、速度、出现时机、停留)"),  # 设计只定用哪个障碍物,怎么动和用户沟通(2026-11-17)
    "S1.3": ("设禁区,测机械臂和夹持物是否一直在允许区域内", "禁区范围"),
    "S1.4": ("人手在同一区域操作 / 递交 / 接受物体,测交互是否安全", "人手放哪、怎么动"),
    "S1.5": ("改被抓物体的摩擦 / 重量,测是否抓稳不掉", ""),  # 条件设计里已定
}


def variant_prompt(state: dict, task: str, variant: str) -> str:
    """S1 变体格子的「▶ Claude」提示词(精简):含义、设计里定的东西、先问用户的事、确认后几步。
    做完格子里直接能「▶ 评测」(S1 变体不审核)。"""
    g = state["group"]
    sim = (_cached(("sim", g), lambda: _group_json(rp.SIM_DIR, g)) or {}).get(task) or {}
    owner = sim.get("same_as")
    if owner:  # 共享任务:仿真细节在归属组
        sim = (_cached(("sim", owner), lambda: _group_json(rp.SIM_DIR, owner)) or {}).get(task) or {}
    spec = (sim.get("S1") or {}).get(variant) or {}
    assets = spec.get("assets") or {}
    todo = [c.get("fn") for c in spec.get("safety") or [] if (c or {}).get("impl") != "existing"]
    what, ask = VARIANT_BRIEF.get(variant, ("", "设计没写清的"))
    lines = [f"做 {g} 组 {task} 的 {variant}({S1_VARIANTS.get(variant, '')}):{what}。"]
    used = "、".join(f"{k}({v or '还没有模型'})" for k, v in assets.items())
    if used:
        lines.append(f"按设计用 {used},见 task/robosafejanus_sim/{owner or g}.json。")
    if spec.get("params"):
        lines.append(f"按设计的条件 {json.dumps(spec['params'], ensure_ascii=False)},见 task/robosafejanus_sim/{owner or g}.json。")
    lines += [f"先和我沟通{ask},给 1–2 个方案,我确认后再做,别自己定。" if ask else "设计没写清的先问我。",
              "", "确认后:" if ask else "要做:"]
    layout = (f"python src/data_collect/eval_layouts.py variant --task {task} --variant S1.1 ..." if variant == "S1.1"
              else f"{task}__{variant}_<i>.json(eval_layouts.py 要补这个变体的生成方式)")
    steps = [f"生成布局 {layout}" + ("" if variant in ("S1.1", "S1.5") else
                                     f",评测时每步驱动(只在 {variant} 生效,reset / soft_reset 复位)")]
    if todo:
        steps.append(f"实现 {variant}.safety 里的 todo({'、'.join(dict.fromkeys(todo))}),改 impl: \"existing\"")
    if variant not in EVAL_VARIANTS:
        steps.append(f"把 {variant} 加进 pipeline_page.py 的 EVAL_VARIANTS")
    steps.append(f"跑 1 条 robodojo.sh eval --task {task} --variant {variant},确认违规能触发;汇报视频路径")
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
    return "\n".join(lines)


def variant_button(state: dict, task: str, variant: str) -> str:
    """S1 变体格的「▶ Claude」:点了弹出提示词(#prompt-dialog),做完布局 / safety 回来看板直接评。"""
    title = f"{variant} · {task} · {state['group']}"
    tip = "给出提示词,复制到终端里的 Claude Code 交互着做这个 S1 变体,做完格子里就能评"
    return (f'<button type="button" class="act small cprompt" data-title="{esc(title)}" '
            f'data-prompt="{esc(variant_prompt(state, task, variant))}" title="{esc(tip)}">▶ Claude</button>')


def variant_cell(state: dict, task: str, policy: str, variant: str, buttons: bool, result: str = "",
                 first: bool = True) -> str:
    """宽表某个 S1 变体的 eval 格:S1.0 走 ``eval_cell``;设计里没有这个变体时写「不适用」;
    已接入但没布局、以及还没接进评测的变体带「▶ Claude」(每个任务只在第一个模型行上有一次)。"""
    import group_jobs as gj

    if variant not in task_variants(state, task):
        return '<td class="pc st idle" title="设计里这个任务没有这个 S1 变体"><small>不适用</small></td>'
    if variant in EVAL_VARIANTS:
        extra = ""
        base = "" if variant == "S1.0" else variant  # S1.0 的布局就叫 <task>_<i>.json
        if buttons and first and not gj.eval_layouts(task, base):
            extra = variant_button(state, task, variant)
        return eval_cell(state, task, policy, buttons, result, base, extra)
    tip = (f"{variant} {S1_VARIANTS.get(variant, '')}:评测还没接入"
           "(robodojo.sh eval 只跑 S1.0;sim JSON 里这个变体的场景 / safety 检查要先实现)")
    btn = (f'<button class="gact" data-kind="eval" data-group="{esc(state["group"])}" data-task="{esc(task)}" '
           f'data-policy="{esc(policy)}" data-variant="{esc(variant)}" title="{esc(tip)}" disabled data-blocked=1>'
           '▶ 评测</button>') if buttons else ""
    more = variant_button(state, task, variant) if buttons and first else ""
    return (f'<td class="pc st jc idle" title="{esc(tip)}"><span class="jmain">—</span><br>'
            f'<small>未接入</small>{btn}{more}</td>')


def eval_cell(state: dict, task: str, policy: str, buttons: bool, result: str = "", variant: str = "",
              extra: str = "") -> str:
    """宽表 S1.0 的 eval 格(按模型分行,不审核):评过的直接写 ``result``(sync_status 写在 cells 的
    ``eval:S1.0:<policy>``:success / n / 视频按钮);没评过写这一行模型已评几条(同一个 ckpt 取 ``eval_time``
    最大的一次)。两种都带「▶ 评测」。底色:评满绿、评了一部分蓝、没评 / 没模型灰。
    ``variant``(S1.1 …)时是这个变体的格子:目标条数 = 它的布局个数,没有布局时写「没有布局」。
    ``extra`` 是另加的按钮(变体没布局时的「▶ Claude」)。"""
    import group_jobs as gj

    downloading = _cached("downloading", gj._downloading) or set()
    try:
        model = next((m for m in gj.eval_models(state["group"], task, downloading) if m["policy"] == policy), None)
        done = max((n for _, n in gj.eval_runs(task, policy, model["ckpt"], variant)), default=0) if model else 0
    except (OSError, ValueError, KeyError):
        model, done = None, 0
    target = gj.eval_target(task, variant)
    if model is None:
        main, klass, small = "—", "idle", "还没有模型"
    elif not target:
        main, klass, small = "—", "idle", f"没有 {variant} 布局"
    else:
        klass = "good" if done >= target else ("todo" if done else "idle")
        main = f"{done}/{target} 条"
        small = "zero-shot" if model["zero_shot"] else ""
    if result and model is not None and done < target:  # 没评满:结果块下面补一句进度
        result += f'<small class="muted">已评 {done}/{target}</small>'
    btn = eval_button(state, task, policy, variant) if buttons else ""
    if result:  # sync_status 已经写成 .eval-item 块(每个 ckpt 一块),直接用
        body = result
    else:
        body = f'<span class="jmain">{esc(main)}</span>'
        if small:
            body += f"<br><small>{esc(small)}</small>"
    return f'<td class="pc st jc {klass}">{body}{btn}{extra}</td>'


def link_button(state: dict, label: str = "🔗 填资产链接") -> str:
    """assets 格的「填资产链接」:seed 批准后出现,点了弹出 #assets-dialog(ASSETS_JS),回填 robosafejanus_assets.json。"""
    if rp.entry_status(state, "seed")[0] != "approved":
        return ""
    n = len((_cached("assets", lambda: rp.read_json(rp.ASSETS_JSON)) or {}).get(state["group"]) or {})
    tip = (f"已填 {n} 个物体,点开看 / 改" if n else "一行一个物体:label + 资产链接(huggingface 网页链接、hf://... 或 hf download 命令)")
    return (f'<button type="button" class="act small alink" data-group="{esc(state["group"])}" title="{esc(tip)}">'
            f'{esc(label)}</button>')


def claude_prompt(state: dict, stage: str, task: str | None) -> str:
    """「▶ Claude」给的提示词:贴进终端里的 Claude Code(仓库目录),和用户交互着做这一格,做完提交审核后停下。

    ``task`` 为空时是还没有任务草稿的组的 design(先拟任务列表)。内容随这一格的状态变:驳回时带上驳回意见,
    失效 / 提交后已改时说明原因。
    """
    g = state["group"]
    if not task:
        return (f"/rsj-{stage} {g}\n\n{g} 还没有任务列表:先按种子拟任务列表(任务名、risky / safe、用到哪些资产 label)"
                "给我确认,确认后 set-tasks,再逐个任务写设计、提交 design 审核后停下。有需要我决定的地方先问我。")
    key = f"{stage}/{task}"
    status, why = rp.entry_status(state, key)
    entry = state["stages"].get(key) or {}
    lines = [f"/rsj-{stage} {g} {task}", "",
             f"只做 {g} 组 {task} 任务的 {stage} 阶段,提交时用 --task {task}。有需要我决定的地方先问我,"
             "做完提交审核后停下(不要批准)。"]
    if status == "rejected":
        notes = [n["text"] for n in entry.get("notes", []) if n.get("kind") == "reject" and n.get("text")]
        lines += ["", "这一格被驳回了,驳回意见:", *(f"- {n}" for n in notes[-1:]),
                  f"(全部记录:python scripts/internal/rsj_pipeline.py notes {g} {stage} --task {task})",
                  "逐条按意见改好再提交,汇报时逐条说明怎么改的。"]
    elif status == "stale":
        lines += ["", f"上游批准后又变了,这一格已失效({why}):按现在的上游重新做,再提交。"]
    elif status == "changed":
        lines += ["", f"提交后产物又改过了({why}):检查这些改动对不对,没问题就重新提交。"]
    return "\n".join(lines)


# 「▶ Claude」只出现在要做 / 重做的格子上(已批准 / 待审的没有)
PROMPT_STATUS = ("none", "rejected", "stale", "changed")
PROMPT_LABEL = {"none": "▶ Claude", "rejected": "↻ 按意见重做", "stale": "↻ 重做", "changed": "▶ Claude"}


def claude_button(state: dict, stage: str, task: str | None) -> str:
    """任务构造某一格的「▶ Claude」:点了弹出提示词(#prompt-dialog,PROMPT_JS),复制到终端里的 Claude 交互着做。
    不在后台起会话。上游没批准时按钮灰掉;已批准 / 待审的格子没有。"""
    g = state["group"]
    if task:
        status, _ = rp.entry_status(state, f"{stage}/{task}")
        if status not in PROMPT_STATUS:
            return ""
        problems = rp.upstream_problems(state, stage, task) if status == "none" else []
        label = PROMPT_LABEL.get(status, "▶ Claude")
    else:  # 还没有任务草稿的组的 design:先拟任务列表
        problems = rp.upstream_problems(state, stage, None)
        label = "▶ Claude"
    title = f"{stage}{' · ' + task if task else ''} · {g}"
    tip = _blocked(problems) if problems else \
        "给出提示词,复制到终端里的 Claude Code 交互着做这一格"
    dis = " disabled" if problems else ""
    return (f'<button type="button" class="act small cprompt" data-title="{esc(title)}" '
            f'data-prompt="{esc(claude_prompt(state, stage, task))}" title="{esc(tip)}"{dis}>{esc(label)}</button>')


def manual_button(state: dict, stage: str, task: str) -> str:
    """失效 / 提交后已改的格子的「✎ 人工审核」:打开这一格的审核条目(编辑区 + 改动 + 「✔ 人工审核通过」)。"""
    status, why = rp.entry_status(state, f"{stage}/{task}")
    if status not in ("stale", "changed"):
        return ""
    tip = f"{why}\n自己改 / 看改动,确认后直接批准" if why else "自己改 / 看改动,确认后直接批准"
    return (f'<a class="act small mreview" href="{stage_href(state["group"], f"{stage}/{task}")}" '
            f'title="{esc(tip)}">✎ 人工审核</a>')


JOB_LABEL = {"collect": ("▶ 采集", "补采到目标条数(不审核)"),
             "exchange": ("⇄ 转换", "导出 LeRobot v3.0 + v2.1(不审核)"),
             "upload": ("⬆ 上传", "先 dry-run 给你看清单,确认后上传 HDF5 + LeRobot 到 ModelScope(不审核)")}


def job_cell(state: dict, col: str, task: str, cells: dict, buttons: bool) -> str:
    """collect / exchange / upload 格:写实际数据,按钮在 expert 批准后可点(点了弹出计划,确认后后台运行 group_jobs.py)。"""
    try:
        klass, main, small, blocked = job_info(col, task, cells.get(task) or {})
    except (OSError, ValueError, KeyError, TypeError):
        klass, main, small, blocked = "idle", "—", "", ""
    expert, _ = rp.entry_status(state, f"expert/{task}")
    if expert != "approved":
        blocked = "expert 要先批准(现在:" + (rp.STATUS_LABEL.get(expert) or "未提交") + ")"
    extra = ""
    if col == "collect" and (cells.get(task) or {}).get("video", "—") != "—":
        extra += f'<div class="jvideo">{cells[task]["video"]}</div>'
    if buttons:
        g, t = esc(state["group"]), esc(task)
        label, tip = JOB_LABEL[col]
        dis = " disabled data-blocked=1" if blocked else ""
        extra += (f'<button class="gact" data-kind="{col}" data-group="{g}" data-task="{t}" title="{esc(blocked or tip)}"{dis}>'
                  f'{label}</button><div class="gjob" data-group="{g}" data-task="{t}" data-kind="{col}"></div>')
    sub = f"<br><small>{_lines(small)}</small>" if small else ""
    return f'<td class="st jc {klass}"><span class="jmain">{_lines(main)}</span>{sub}{extra}</td>'


def _lines(text: str) -> str:
    """多项上下排(job_info 里用换行分隔)。"""
    return esc(text).replace("\n", "<br>")


def _wbr(name: str) -> str:
    """长名字在下划线后可以换行(组 / 任务列窄一些)。"""
    return esc(name).replace("_", "_<wbr>")


def policy_cell(state: dict, by_group: dict, t: dict, col: str, policy: str, cells: dict, buttons: bool,
                first: bool = True) -> str:
    """按模型分行的一格:``model`` 是 sync_status 写的数据,``eval:<S1.x>`` 是 ``variant_cell``(不审核:
    评过写 success / n + 视频,没评过写已评几条)+ 只评 ``policy`` 的「▶ 评测」。same_as 引用的任务照搬归属组的格子。
    ``first`` 是这一行的第一个模型行(S1 变体的「▶ Claude」只放这里一次)。"""
    tag = f'<span class="polname">{esc(policy)}</span>'
    if col.startswith("eval:"):
        src = state if t["name"] in rp.own_tasks(state) else by_group.get(t["owner"] or "")
        if not t["name"] or src is None:
            return '<td class="st idle pc">·</td>'
        result = (cells.get(t["name"]) or {}).get(f"{col}:{policy}") or ""
        return variant_cell(src, t["name"], policy, col.split(":", 1)[1], buttons, result, first)
    value = cells.get(t["name"], {}).get(f"{col}:{policy}") if t["name"] else None
    lead = tag + "<br>" if col == "model" else ""
    return f'<td class="dc pc">{lead}{value or "<small>—</small>"}</td>'


def group_task_table(states: list[dict], stages, *, phases: bool = False, filterable: bool = False,
                     buttons: bool = False, cells: dict | None = None, seeds: list[dict] | None = None) -> str:
    """每行一个任务的表,``#`` / 组 / 组级阶段(seed)跨行合并,按任务的阶段每行一格。

    ``cells`` 给出时是总览的宽表:``WIDE_COLUMNS``(流水线各阶段中间穿插 HDF5 / LeRobot / 视频 / 模型 / 评测的数据列);
    ``phases`` 时表头多一行大类,``buttons`` 时任务构造各格带「Claude」按钮、collect / exchange / upload 格带「采集」「转换」「上传」,
    ``seeds`` 是表尾"未开始的种子"行(带「从种子开始」)。任务格的底色是层级(safe / risky)。
    每组一个 ``<tbody>``,``filterable`` 时带 ``data-search``。
    """
    if not states and not seeds:
        return '<p class="empty">还没有组,task/risk_seeds.json 里也没有种子。</p>'
    columns = list(WIDE_COLUMNS) if cells is not None else [(PHASE_OF[s][0], s) for s in stages]
    fixed = ["#", "组", "任务"]

    def head(col: str) -> str:  # 只写中文列名,不写副标题(英文名和说明放在悬停提示里)
        if col in JOB_HEADS:
            name, sub = JOB_HEADS[col]
            return f'<th title="{esc(name)} · {esc(sub)}">{esc(COL_LABEL.get(col, name))}</th>'
        if col in DATA_HEADS:
            name, sub = DATA_HEADS[col]
            return f'<th class="dh" title="{esc(sub)}">{esc(name)}</th>'
        if col.startswith("eval:"):
            v = col.split(":")[1]
            note = "" if v in EVAL_VARIANTS else ";评测还没接入"
            return (f'<th class="sub" title="{esc(v)} {esc(S1_VARIANTS.get(v, ""))} · 评测进度'
                    f'(不审核;每个模型一行,按钮只评这一行的模型{note})">{esc(v)}</th>')
        desc = "资产链接(你来填,不审)" if col == "assets" else rp.STAGE_DESC[col]
        return f'<th title="{esc(col)} · {esc(desc)}">{esc(COL_LABEL.get(col, col))}</th>'

    # 评测的 S1 子列:列头多一行(评测跨子列,其他列头上下合并)
    n_sub = sum(1 for _, c in columns if c.startswith("eval:"))
    if n_sub:
        col_heads, sub_heads, opened = "", "", False
        for _, c in columns:
            if c.startswith("eval:"):
                if not opened:
                    col_heads += (f'<th colspan="{n_sub}" title="eval · 每个 S1 执行测试一个子列(不审核)">'
                                  f'{esc(COL_LABEL["eval"])}</th>')
                    opened = True
                sub_heads += head(c)
            else:
                col_heads += head(c).replace("<th", '<th rowspan="2"', 1)
        head_rows = f"<tr>{col_heads}</tr><tr>{sub_heads}</tr>"
    else:
        head_rows = "<tr>" + "".join(head(c) for _, c in columns) + "</tr>"
    depth = 2 if n_sub else 1
    if phases:
        head1 = "".join(f'<th rowspan="{depth + 1}" class="lhead">{h}</th>' for h in fixed)
        for p in PHASES:
            n = sum(1 for ph, _ in columns if ph == p[0])
            if n:
                head1 += f'<th colspan="{n}" class="phead ph-{p[0]}">{esc(p[1])}</th>'
        thead = f"<tr>{head1}</tr>{head_rows}"
    else:
        lead = "".join(f'<th rowspan="{depth}" class="lhead">{h}</th>' if depth > 1 else f'<th class="lhead">{h}</th>'
                       for h in fixed)
        thead = head_rows.replace("<tr>", f"<tr>{lead}", 1)
    bodies = []
    by_group = {s["group"]: s for s in states}
    policies = cell_policies(cells) if any(c.endswith(":*") for _, c in columns) else ()
    k = max(1, len(policies))  # 每个任务占几行
    for index, state in enumerate(states, 1):
        g = state["group"]
        tasks = task_info(state)
        n = len(tasks)
        rs = f' rowspan="{n * k}"' if n * k > 1 else ""
        rows = []
        for i, t in enumerate(tasks):
            row = []
            sub_rows: list[list[str]] = [[] for _ in policies]  # 按模型分的格子:每个模型一行
            if i == 0:
                adopted = ' <span class="badge idle" title="流水线之前完成,登记进流水线">adopt</span>' if state.get("adopted") else ""
                row.append(f'<td class="group-index"{rs}>{index}</td>')
                row.append(f'<td class="gname"{rs}><a href="#g/{esc(g)}"><code>{_wbr(g)}</code></a>{adopted}</td>')
            if t["name"]:
                lvl = f' lvl-{t["level"]}' if t["level"] else ""
                tip = f'{t["level"] or "层级未定(还没写设计)"} · {t["instruction"]}'.strip(" ·")
                if t["owner"]:
                    tip += f'\nlayout 起和 {t["owner"]} 组的同一份'
                row.append(_rowspan(f'<td class="tname{lvl}" title="{esc(tip)}"><code>{_wbr(t["name"])}</code></td>', k))
            else:
                row.append(_rowspan('<td class="tname"><small class="muted">还没有任务草稿</small></td>', k))
            for _, col in columns:
                if col.endswith(":*"):  # 按模型分行:第 j 个模型的格子放进第 j 行
                    for j, policy in enumerate(policies):
                        sub_rows[j].append(policy_cell(state, by_group, t, col[:-2], policy, cells or {}, buttons,
                                                       first=j == 0))
                    continue
                before = len(row)
                if col in JOB_HEADS:
                    # same_as 引用的任务在归属组采 / 转 / 传:照搬归属组的格子
                    src = state if t["name"] in rp.own_tasks(state) else by_group.get(t["owner"] or "")
                    if not t["name"] or src is None:
                        row.append('<td class="st idle">·</td>')
                    else:
                        row.append(job_cell(src, col, t["name"], cells or {}, buttons))
                elif col in DATA_HEADS:
                    value = (cells or {}).get(t["name"], {}).get(col) if t["name"] else None
                    extra = ""
                    if buttons and col == "eval_layout" and t["name"]:
                        src = state if t["name"] in rp.own_tasks(state) else by_group.get(t["owner"] or "")
                        extra = layouts_button(src, t["name"]) if src else ""
                    row.append(f'<td class="dc">{value or "<small>—</small>"}{extra}</td>')
                elif col == "assets":
                    if i == 0:
                        row.append(assets_cell(state, n * k, buttons))
                elif col == "design" and not t["name"]:  # 还没有任务列表:design 先拟任务
                    btn = claude_button(state, "design", None) if buttons else ""
                    row.append(f'<td class="st todo">{btn}<small>先拟任务列表</small></td>' if btn else '<td class="st idle">·</td>')
                elif col in rp.TASK_STAGES:
                    # same_as 的任务 layout 起不在本组审:照搬归属组那一格(同一个审核条目、同一个按钮)
                    src = state
                    if t["name"] and t["name"] not in rp.stage_tasks(state, col):
                        src = by_group.get(t["owner"] or "")
                    if not t["name"] or src is None or t["name"] not in rp.stage_tasks(src, col):
                        row.append('<td class="st idle">·</td>')
                    else:
                        extra = claude_button(src, col, t["name"]) if buttons and col in BUILD_STAGES else ""
                        if buttons and col in BUILD_STAGES:
                            extra = manual_button(src, col, t["name"]) + extra
                        td = status_cell(src, f"{col}/{t['name']}", extra=extra)
                        # 从零开始(未开始且按钮可点):点格子直接弹出 Claude 提示词(PROMPT_JS)
                        if (col in BUILD_STAGES and "cprompt" in extra and " disabled" not in extra
                                and rp.entry_status(src, f"{col}/{t['name']}")[0] == "none"):
                            td = td.replace("<td", "<td data-direct", 1)
                        row.append(td)
                elif i == 0:
                    extra = ingest_button(state) if buttons and col == "ingest" else ""
                    row.append(status_cell(state, col, span=n * k, extra=extra))
                    continue  # 组级格子已经跨整组
                if len(row) > before and col not in ("assets",):
                    row[-1] = _rowspan(row[-1], k)
            last = " tlast" if i == n - 1 else ""
            if not policies:
                rows.append(f'<tr>{"".join(row)}</tr>')
            for j, cells_j in enumerate(sub_rows):  # 第一个模型的行带上不分模型的格子
                lead = "".join(row) if j == 0 else ""
                cls = ("tfirst" if j == 0 else "psub") + (" plast" if j == k - 1 else "") + last
                rows.append(f'<tr class="{cls}">{lead}{"".join(cells_j)}</tr>')
        search = ""
        if filterable:
            words = " ".join([g] + [f"{t['name']} {t['level']} {t['instruction']}" for t in tasks]).lower()
            search = f' data-search="{esc(words)}"'
        bodies.append(f'<tbody class="grp" data-group="{esc(g)}"{search}>{"".join(rows)}</tbody>')
    rest = len(columns) - 1
    for seed in seeds or []:
        name = esc(seed["name"])
        words = esc(" ".join(str(seed.get(k, "")) for k in ("name", "core_risk", "risk_category")).lower())
        bodies.append(
            f'<tbody class="grp seedrow" data-group="{name}" data-search="{words}"><tr><td class="group-index"></td>'
            f'<td class="gname"><code>{_wbr(seed["name"])}</code><br><small>未开始</small></td>'
            f'<td class="tname">{esc(seed.get("core_risk", ""))}</td>'  # 不带 sdesc:它的 min-width 会把任务列撑宽
            f'<td class="st idle"><button class="act small" data-action="init" data-group="{name}">从种子开始</button></td>'
            f'<td class="sdesc" colspan="{rest}"><small>{esc(seed.get("risk_category", ""))}</small></td></tr></tbody>')
    wide = " wide" if cells is not None else ""
    return f'<div class="mwrap"><table class="pipe{wide}"><thead>{thead}</thead>{"".join(bodies)}</table></div>{LEGEND}'


def seed_html(seed: dict | None) -> str:
    if not seed:
        return '<p class="empty">risk_seeds.json 里没有这个种子</p>'

    def elems(key):
        return " · ".join(esc(e) for e in (seed.get(key) or {}).get("elements") or []) or "—"

    return f"""<p><b>{esc(seed.get('core_risk', ''))}</b>
<small>{esc(seed.get('risk_category', ''))} · {esc(seed.get('background', ''))}</small></p>
<p class="muted">{esc(seed.get('risk_reasoning', ''))}</p>
<p><span class="badge lvl-risky">risky</span> {elems('risk_variation')}</p>
<p><span class="badge lvl-safe">safe</span> {elems('safe_variation')}</p>"""


def group_store_html(state: dict, seeds: list[dict], out_dir: Path) -> str:
    """一个组的风险种子和全部审核条目;页面上点格子 / 组名时由脚本挪进弹出的对话框。"""
    g = state["group"]
    seed = next((s for s in seeds if s.get("name") == g), None)
    adopted = ' <span class="badge idle">adopt:流水线之前完成</span>' if state.get("adopted") else ""
    entries = [entry_html(state, stage, key, task, out_dir)
               for stage in rp.STAGES for key, task in stage_keys(state, stage)]
    return (f'<div class="gstore" data-group="{esc(g)}"><div class="entry gseed"><div class="ehead"><b>风险种子</b>{adopted}</div>'
            f'{seed_html(seed)}</div>{"".join(entries)}</div>')


def table_html(out_dir: Path) -> str:
    """看板唯一的内容:流水线大表 + 隐藏的审核条目(点格子弹出)。"""
    with rp.memo_scope():
        return _table_html(out_dir)


def _table_html(out_dir: Path) -> str:
    _CACHE.clear()
    states = rp.all_states()
    seeds, seed_error = load_seeds()
    banner = f'<p class="badge bad">{esc(seed_error)}</p>' if seed_error else ""
    started = {s["group"] for s in states}
    unstarted = [s for s in seeds if s.get("name") and s["name"] not in started]
    table = group_task_table(states, rp.STAGES, phases=True, buttons=True, filterable=True,
                             cells=load_cells(out_dir), seeds=unstarted)
    stores = "".join(group_store_html(s, seeds, out_dir) for s in states)
    return f"""{banner}<div class="filterbar"><input id="filter" type="search" placeholder="筛选组 / 任务 / 种子…(如 heat / risky / microwave)" autocomplete="off">
<span id="filter-count" class="muted"></span>
<span class="muted">每行一个任务(任务格底色:绿 safe / 红 risky)· 左右滚动看全部列 · 点格子里的字弹出该阶段的审核(批准 / 驳回)·
点组名弹出风险种子和全部阶段 · 任务构造各格点「▶ Claude」拿提示词,贴到终端里的 Claude 交互着做 ·
collect / exchange / upload / eval 格点「采集」「转换」「上传」「评测」直接做(不审核)</span></div>
<div id="glist">{table}</div><div id="gstores" hidden>{stores}</div>"""


def fragments(out_dir: Path) -> dict[str, str]:
    return {"table": table_html(out_dir)}


def _gen(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def fragment_box(key: str, out_dir: Path) -> str:
    """页面里放片段的容器:内嵌上次写好的片段(静态打开也能看);打开时脚本比一下 ``<key>.gen``,变了才重读。"""
    try:
        inner = (out_dir / FRAGMENT_DIR / f"{key}.html").read_text(encoding="utf-8")
        gen = _gen(inner)
    except OSError:
        inner, gen = '<p class="empty">流水线片段还没生成</p>', ""
    return (f'<div class="pipe-ui pipe-frag" data-frag="{FRAGMENT_DIR}/{key}.html" data-gen="{gen}">'
            f'{inner}</div>')


# 点审核格 / 组名弹出的对话框:条目从隐藏的 #gstores 挪进来,关闭时放回(ROUTE_JS)
DIALOG_HTML = ('<dialog id="pipe-dialog" class="pipe-ui"><div class="pdhead"><code id="pipe-dialog-group"></code>'
               '<a id="pipe-dialog-all" href="#">看这个组的全部阶段</a><span class="grow"></span>'
               '<span id="pipe-dialog-warn" class="ewarn" hidden></span>'
               '<button type="button" class="act small" id="pipe-dialog-close">关闭</button></div>'
               '<div class="pdbody" id="pipe-dialog-body"></div></dialog>')


# 「▶ Claude」的提示词对话框(PROMPT_JS)
PROMPT_DIALOG_HTML = (
    '<dialog id="prompt-dialog" class="pipe-ui"><div class="pdhead"><b>给 Claude 的提示词</b> <code id="prompt-dialog-title"></code>'
    '<span class="grow"></span><button type="button" class="act small" id="prompt-dialog-close">关闭</button></div>'
    '<div class="pdbody"><p class="muted">复制下面的提示词,贴到仓库目录(<code>RoboSafeJanus/</code>)里打开的 Claude Code。'
    'Claude 会和你交互着做这一格,做完提交审核后停下,回到看板审核。框里可以先改再复制。</p>'
    '<textarea id="prompt-text" rows="10" spellcheck="false"></textarea>'
    '<div class="abar"><button type="button" class="act primary" id="prompt-copy">复制</button>'
    '<span id="prompt-msg" class="muted"></span></div></div></dialog>')

# assets 格「填资产链接」的对话框(ASSETS_JS)
ASSETS_DIALOG_HTML = (
    '<dialog id="assets-dialog" class="pipe-ui"><div class="pdhead"><b>资产链接</b> <code id="assets-dialog-group"></code>'
    '<span class="grow"></span><button type="button" class="act small" id="assets-dialog-close">关闭</button></div>'
    '<div class="pdbody"><p class="muted">一行一个物体:左边 label(英文 snake_case,将来是 sim 里 assets 的键),'
    '右边贴链接,可以是 huggingface.co / hf-mirror.com 上资产目录的网页链接、SimReady Central 的 <code>hf://...</code> 路径、'
    '<code>Assets/Object/RoboDojo/...</code> 目录,或者 <code>hf download ...</code> 命令。保存时换成带 commit 的 '
    '<code>hf download</code> 命令写进 <code>task/robosafejanus_assets.json</code>。来源只能是 RoboDojo 官方和 4 个 SimReady 库。</p>'
    '<p id="assets-dialog-warn" class="ewarn" hidden></p>'
    '<div id="assets-rows"></div>'
    '<div class="abar"><button type="button" class="act small" id="assets-add">+ 加一行</button>'
    '<span id="assets-suggest" class="muted"></span></div>'
    '<div class="abar"><button type="button" class="act" id="assets-check">检查</button>'
    '<button type="button" class="act primary" id="assets-save" disabled>保存</button>'
    '<span id="assets-msg" class="muted"></span></div></div></dialog>')


def merged_assets() -> str:
    """页面用的样式和脚本:组件样式、读取片段、弹出审核条目、审核按钮。"""
    return (f'<style>{scoped_css()}</style><div id="pipe-toast" class="pipe-ui" hidden></div>{DIALOG_HTML}'
            f'{ASSETS_DIALOG_HTML}{PROMPT_DIALOG_HTML}<script>{ROUTE_JS}{EDIT_JS}{ACTION_JS}{ASSETS_JS}{PROMPT_JS}{FIT_JS}</script>')


# ---------------------------------------------------------------------------
# 样式 / 脚本
# ---------------------------------------------------------------------------

# 组件样式都限定在 .pipe-ui 下,嵌入其他看板页时不影响原有样式
COMPONENT_CSS = """
  .badge.stale { background: rgba(130, 80, 223, .16); color: #8250df; }
  .badge.idle { background: var(--panel-2); color: var(--faint); }
  button.act { font: inherit; font-size: 13px; background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 12px; cursor: pointer; }
  button.act:hover { border-color: var(--faint); background: var(--hover); }
  button.act.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.act.small { padding: 2px 9px; font-size: 12px; }
  button.act:disabled { opacity: .5; cursor: wait; }
  .muted { color: var(--muted); font-size: 12px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--line); }
  .dot.good { background: var(--good); } .dot.warn { background: var(--warn); } .dot.bad { background: var(--bad-fg); }
  .dot.stale { background: #8250df; }
  .entry { border-top: 1px solid var(--line); margin-top: 10px; padding-top: 10px; scroll-margin-top: 12px; }
  .entry.flash, .pgroup.flash { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 4px; }
  .ehead { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .why { color: var(--bad-fg); font-size: 12px; margin: 4px 0; }
  ul.checks, ul.notes { list-style: none; padding: 0; margin: 6px 0; font-size: 13px; }
  ul.checks li.ng { color: var(--bad-fg); }
  ul.checks pre, pre.diff { background: var(--panel-2); border: 1px solid var(--line); border-radius: 7px; padding: 8px;
    overflow: auto; white-space: pre-wrap; font: 11px/1.5 var(--mono); margin: 4px 0; max-height: 360px; }
  details.fold { margin-top: 8px; } details.fold summary { cursor: pointer; color: var(--muted); font-size: 13px; }
  details.pipe-fold { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; margin-bottom: 16px; }
  details.pipe-fold > summary { color: var(--text); font-weight: 600; }
  .actions { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
  .actions .btns { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  textarea.note { font: inherit; font-size: 13px; background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 7px; padding: 6px 10px; width: 100%; resize: vertical; }
  code.copy { background: var(--panel-2); padding: 1px 6px; border-radius: 4px; cursor: copy; }
  .pgroup { border-top: 1px solid var(--border); margin-top: 12px; padding-top: 10px; scroll-margin-top: 12px; }
  .ghead { display: flex; align-items: center; gap: 8px; }
  details.pgroup:not([open]) > summary { color: var(--muted); }
  .mwrap { overflow-x: auto; }
  table.pipe { margin-bottom: 4px; }
  table.pipe th, table.pipe td { text-align: center; vertical-align: middle; white-space: nowrap; }
  /* 单元格允许换行(按钮不换):表格自然宽度小一些,FIT_JS 再把整张表等比缩到一屏 */
  table.pipe td, table.pipe th { white-space: normal; padding: 4px 5px; }
  table.pipe td button { white-space: nowrap; }
  table.pipe th.rowhead { text-align: left; }
  table.pipe th.phead { background: var(--panel); border-bottom: 2px solid var(--border); }
  table.pipe td.st a { display: block; color: inherit; text-decoration: none; }
  table.pipe tbody td.st.good { background: var(--good-bg); color: var(--good-fg); }
  table.pipe tbody td.st.warn { background: var(--warn-bg); color: var(--warn-fg); }
  table.pipe tbody td.st.bad { background: var(--bad-bg); color: var(--bad-fg); }
  table.pipe tbody td.st.stale { background: rgba(130, 80, 223, .14); }
  table.pipe tbody td.st.idle { color: var(--faint); }
  .done { color: var(--good-fg); }
  .legend { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
  table.pipe th.lhead, table.pipe td.gname, table.pipe td.tname, table.pipe td.instr, table.pipe td.nstep { text-align: left; }
  table.pipe td.group-index { color: var(--faint); text-align: right; width: 28px; font-variant-numeric: tabular-nums; }
  table.pipe td.instr { white-space: normal; min-width: 240px; max-width: 420px; color: var(--muted); font-size: 12px; }
  table.pipe tbody.grp + tbody.grp > tr:first-child > * { border-top: 2px solid var(--border); }
  table.pipe tbody.grp > tr:last-child > * { border-bottom: 0; }
  table.pipe td[rowspan] { border-bottom: 0; }
  table.pipe.wide td.dc { text-align: left; white-space: nowrap; font-size: 12px; }
  table.pipe.wide td.dc small { font-size: 11px; }
  table.pipe td.st.jc .eval-item { margin: 2px 0 4px; }
  table.pipe td.st.jc .eval-item .video-btn { margin-left: 4px; }
  table.pipe td.st.jc .eval-item .ng { color: var(--bad-fg); }  /* S1 变体:有违规时违规率标红 */
  table.pipe.wide td.dc button.gact { display: block; margin: 5px auto 0; }
  table.pipe th.dh { background: var(--panel-2); }
  table.pipe th.sub { font-size: 11px; font-weight: 500; }
  table.pipe th.phead.ph-data { box-shadow: inset 0 3px 0 var(--accent); }
  table.pipe th.phead.ph-eval { box-shadow: inset 0 3px 0 var(--good); }
  /* 模型评测按模型分行:同一任务的两个模型之间虚线,任务之间实线;合并的任务格保留下边线 */
  table.pipe td[rowspan][data-t] { border-bottom: 1px solid var(--line); }
  table.pipe tbody.grp > tr.tlast > td[data-t] { border-bottom: 0; }
  table.pipe tr:not(.plast) > td.pc { border-bottom: 1px dashed var(--border); }
  table.pipe td .polname { font-weight: 600; font-size: 11px; color: var(--muted); }
  table.pipe td.st button.gact { display: block; margin: 5px auto 0; }  /* 按钮单独一行,不跟在小字后面 */
  table.pipe td.st .gjob { margin: 3px auto 0; text-align: center; }
  table.pipe td.st.jc .jmain { font-weight: 500; }
  table.pipe td.st.jc .jvideo { margin-top: 4px; }
  table.pipe td.st.jc .jvideo .video-btn { font-size: 12px; }
  details.pgroup > summary { cursor: pointer; list-style-position: inside; }
  #pipe-toast { position: fixed; right: 16px; bottom: 16px; max-width: min(560px, calc(100% - 32px)); background: var(--panel);
    border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; box-shadow: 0 8px 28px rgba(0,0,0,.2);
    font-size: 13px; white-space: pre-wrap; max-height: 50vh; overflow: auto; z-index: 50; }
  #pipe-toast.err { border-color: var(--bad-fg); }
"""

PAGE_CSS = """
  @media (prefers-color-scheme: dark) { .badge.stale { color: #d2a8ff; } }
  @media (max-width: 800px) { body { padding: 16px; } }
  .filterbar { display: flex; align-items: center; gap: 10px; margin: 0 0 10px; flex-wrap: wrap; }
  #filter { width: min(360px, 100%); font: inherit; font-size: 13px; background: var(--panel); color: var(--text);
    border: 1px solid var(--border); border-radius: 7px; padding: 6px 10px; }
  table.pipe tr.hit > td.tname { background: var(--warn-bg); }
  .box p, .gseed p { margin: 4px 0; }
  h3 { font-size: 13px; margin: 0 0 6px; color: var(--muted); font-weight: 500; }
  #pipe-dialog { width: min(1200px, calc(100% - 48px)); max-height: calc(100vh - 64px); padding: 0; display: none; }
  #pipe-dialog[open] { display: flex; flex-direction: column; }
  #pipe-dialog::backdrop { background: rgba(0, 0, 0, .35); }
  #pipe-dialog .pdhead { display: flex; gap: 10px; align-items: center; padding: 12px 18px; border-bottom: 1px solid var(--border); }
  #pipe-dialog .pdhead code { font-size: 14px; font-weight: 600; }
  #pipe-dialog .pdhead .grow { flex: 1; }
  #pipe-dialog .pdbody { overflow: auto; padding: 12px 18px 16px; display: flex; flex-direction: column; gap: 10px; }
  .editbox { margin: 8px 0 4px; }
  .editbox .etop { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 6px; }
  .editbox .etop.ng, .editbox .esmsg.ng { color: var(--bad-fg); }
  .ewarn { color: var(--warn-fg); background: var(--warn-bg); border-radius: 6px; padding: 1px 8px; font-size: 12px; }
  .editbox .eslot { margin: 0 0 10px; min-width: 0; }
  .editbox .eshead { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; font-size: 13px; margin-bottom: 3px; }
  .editbox .eshead code { font-size: 11px; color: var(--muted); }
  .editbox .eshead small { color: var(--muted); }
  .editbox textarea.etext { width: 100%; box-sizing: border-box; font: 12px/1.5 var(--mono); background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 7px; padding: 8px 10px; resize: vertical; tab-size: 2; white-space: pre; overflow: auto; }
  .editbox textarea.etext:focus { outline: 2px solid var(--accent); outline-offset: -1px; background: var(--panel); }
  .editbox textarea.etext[readonly] { opacity: .7; }
  .editbox .esbar { display: flex; gap: 10px; align-items: center; margin-top: 4px; font-size: 12px; }
  .editbox .esmsg { color: var(--muted); } .editbox .esmsg.ok { color: var(--good-fg); }
  .editbox .epair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 760px) { .editbox .epair { grid-template-columns: 1fr; } }
  .editbox .ebottom { display: flex; gap: 8px; }
  #pipe-dialog .pdbody > .entry { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin: 0; }
  table.pipe tbody td.st.todo { background: rgba(9, 105, 218, .08); color: var(--accent); }
  table.pipe td.st.todo small { color: var(--muted); }
  table.pipe td.active { outline: 2px solid var(--accent); outline-offset: -2px; }
  table.pipe th.lhead { padding: 5px 6px; }
  table.pipe td.group-index { padding: 5px 4px; width: 18px; }
  table.pipe td.gname, table.pipe td.tname { white-space: normal; padding: 5px 7px; line-height: 1.3; }
  table.pipe td.gname, table.pipe td.tname { width: 116px; min-width: 116px; max-width: 116px; }  /* 组列和任务列一样宽 */
  table.pipe td.gname code, table.pipe td.tname code { font-size: 11.5px; background: none; padding: 0; }
  table.pipe tbody td.tname.lvl-safe { background: var(--safe-bg); box-shadow: inset 3px 0 0 var(--safe-fg); }
  table.pipe tbody td.tname.lvl-risky { background: var(--risky-bg); box-shadow: inset 3px 0 0 var(--risky-fg); }
  table.pipe td.st small { opacity: .85; }
  table.pipe td.st a .cmain, table.pipe td.st a .clines, table.pipe td.st a small { display: block; }
  table.pipe td.st .clines { font-size: 11.5px; line-height: 1.35; }
  table.pipe td.st .clines > span { display: block; }
  table.pipe td.st .clines > span.sr { font-style: italic; }
  table.pipe td.st .clines > span.ng { color: var(--bad-fg); }
  table.pipe tbody.seedrow td { color: var(--muted); }
  table.pipe td.st.ac .clines, table.pipe td.st.ac small { display: block; }
  table.pipe td.st.ac small.ng { color: var(--bad-fg); }
  table.pipe.wide td.dc .mdl { margin: 2px 0 6px; }
  table.pipe.wide td.dc .mdl small.wb { display: block; margin-top: 3px; }
  table.pipe.wide td.dc .mdl-btn { display: block; margin-top: 3px; font: inherit; font-size: 12px; background: var(--panel);
    color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 2px 8px; cursor: pointer; }
  table.pipe.wide td.dc .mdl-btn:hover:not(:disabled) { border-color: var(--accent); }
  table.pipe.wide td.dc .mdl-btn:disabled { opacity: .45; cursor: not-allowed; }
  table.pipe td.st button.alink { margin-top: 5px; display: block; margin-left: auto; margin-right: auto; }
  table.pipe td.st.todo small { display: block; margin-top: 3px; }
  .alinkrow { display: flex; gap: 10px; align-items: center; margin: 6px 0; }
  #prompt-dialog { width: min(760px, calc(100% - 48px)); max-height: calc(100vh - 64px); padding: 0; display: none; }
  #prompt-dialog[open] { display: flex; flex-direction: column; }
  #prompt-dialog::backdrop { background: rgba(0, 0, 0, .35); }
  #prompt-dialog .pdhead { display: flex; gap: 10px; align-items: center; padding: 12px 18px; border-bottom: 1px solid var(--border); }
  #prompt-dialog .pdhead .grow { flex: 1; }
  #prompt-dialog .pdbody { overflow: auto; padding: 12px 18px 16px; display: flex; flex-direction: column; gap: 8px; }
  #prompt-dialog .pdbody > p { margin: 0; }
  #prompt-dialog textarea { width: 100%; box-sizing: border-box; font: 12.5px/1.5 var(--mono); background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 7px; padding: 8px 10px; resize: vertical; }
  #prompt-dialog .abar { display: flex; gap: 8px; align-items: center; }
  table.pipe td.st button.cprompt { margin-top: 5px; display: block; margin-left: auto; margin-right: auto; }
  table.pipe td.st a.mreview { display: block; width: max-content; margin: 5px auto 0; padding: 2px 9px; font-size: 12px;
    background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 6px; }
  table.pipe td.st a.mreview:hover { border-color: var(--faint); background: var(--hover); }
  table.pipe td.st[data-direct] > a { cursor: copy; }
  .entry p.manual { margin: 0 0 6px; font-size: 13px; }
  #assets-dialog { width: min(1000px, calc(100% - 48px)); max-height: calc(100vh - 64px); padding: 0; display: none; }
  #assets-dialog[open] { display: flex; flex-direction: column; }
  #assets-dialog::backdrop { background: rgba(0, 0, 0, .35); }
  #assets-dialog .pdhead { display: flex; gap: 10px; align-items: center; padding: 12px 18px; border-bottom: 1px solid var(--border); }
  #assets-dialog .pdhead .grow { flex: 1; }
  #assets-dialog .pdbody { overflow: auto; padding: 12px 18px 16px; display: flex; flex-direction: column; gap: 8px; }
  #assets-dialog .pdbody > p { margin: 0; }
  #assets-dialog .arow { display: grid; grid-template-columns: 150px 1fr auto; gap: 6px 8px; align-items: start; }
  #assets-dialog .arow input { font: 12px/1.4 var(--mono); background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 8px; min-width: 0; box-sizing: border-box; width: 100%; }
  #assets-dialog .arow input:focus { outline: 2px solid var(--accent); outline-offset: -1px; background: var(--panel); }
  #assets-dialog .arow .ainfo { grid-column: 2 / 4; font-size: 11.5px; color: var(--muted); min-height: 0; margin-top: -3px; }
  #assets-dialog .arow .ainfo.ng { color: var(--bad-fg); }
  #assets-dialog .arow .ainfo.ok { color: var(--good-fg); }
  #assets-dialog .abar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #assets-dialog #assets-msg.ok { color: var(--good-fg); } #assets-dialog #assets-msg.ng { color: var(--bad-fg); }
  #assets-dialog #assets-suggest button { margin-left: 4px; }
  @media (max-width: 640px) { #assets-dialog .arow { grid-template-columns: 1fr auto; } #assets-dialog .arow .alink-in { grid-column: 1 / 3; } }
  table.pipe tbody.seedrow td.sdesc { white-space: normal; text-align: left; min-width: 260px; font-size: 12px; }
"""


def scoped_css() -> str:
    """组件和页面样式的每个选择器加上 ``.pipe-ui`` 前缀(放进看板页时不影响原有样式;``#id`` 和 ``@`` 规则不加)。"""
    def prefix(m):
        text = m.group(1)
        lead = text[: len(text) - len(text.lstrip())]
        if text.strip().startswith("@"):
            return text + "{"
        sels = [sel.strip() for sel in text.split(",")]
        return lead + ", ".join(sel if sel.startswith("#") else f".pipe-ui {sel}" for sel in sels) + " {"

    return re.sub(r"([^{}]+)\{", prefix, COMPONENT_CSS + PAGE_CSS)


ACTION_JS = """
(function () {
  if (window.__pipeActions) return;
  window.__pipeActions = true;
  const toast = document.getElementById('pipe-toast');
  function say(text, err) {
    const d = document.getElementById('pipe-dialog');  // 对话框开着时 toast 放进对话框,不被遮罩挡住
    const host = d && d.open ? d : document.body;
    if (toast.parentElement !== host) host.append(toast);
    toast.textContent = text;
    toast.className = 'pipe-ui' + (err ? ' err' : '');
    toast.hidden = false;
  }
  async function waitJob() {
    for (;;) {
      await new Promise((r) => setTimeout(r, 2500));
      const res = await fetch('api/jobs').then((r) => r.json()).catch(() => null);
      const job = res && res.jobs && res.jobs.pipeline;
      if (!job || job.state !== 'running') return job;
      say('检查中…\\n' + (job.log_tail || '').slice(-1500));
    }
  }
  document.addEventListener('click', async (ev) => {
    const copy = ev.target.closest('.pipe-ui code.copy');
    if (copy) {
      if (navigator.clipboard) navigator.clipboard.writeText(copy.textContent);
      say('已复制:' + copy.textContent);
      return;
    }
    const btn = ev.target.closest('.pipe-ui button.act[data-action]');
    if (!btn) return;
    const d = btn.dataset;
    const box = btn.closest('.entry');
    const noteEl = box && box.querySelector('textarea.note');
    const note = noteEl ? noteEl.value.trim() : '';
    if (d.action === 'reject' && !note) { say('驳回需要在文本框里写意见', true); if (noteEl) noteEl.focus(); return; }
    const key = d.stage + (d.task ? '/' + d.task : '');
    const what = { init: '从种子新建 ' + d.group, approve: '批准 ' + key, reject: '驳回 ' + key,
      resubmit: '重新提交 ' + key, reapprove: '人工审核通过 ' + key + '(重新跑检查,通过即批准)',
      pick: d.label + ' 选用 ' + d.candidate }[d.action];
    if (!confirm(d.group + ':' + what + '?')) return;
    btn.disabled = true;
    say('执行中…');
    try {
      const res = await fetch('api/pipeline/' + d.action, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ group: d.group, stage: d.stage, task: d.task, label: d.label, candidate: d.candidate, note: note }) });
      const data = await res.json().catch(() => ({ error: '服务端没有返回 JSON(页面需要用 sync_status.py --serve 打开)' }));
      if (!res.ok || data.error) { say((data.error || '失败') + (data.output ? '\\n\\n' + data.output : ''), true); btn.disabled = false; return; }
      if (data.started) {
        const job = await waitJob();
        if (job && job.state === 'failed') { say('检查未通过:\\n' + (job.log_tail || ''), true); btn.disabled = false; return; }
      }
      say((data.output || '完成') + '\\n刷新中…');
      setTimeout(() => location.reload(), 600);
    } catch (e) {
      say('请求失败:' + e + '\\n页面需要用 sync_status.py --serve 打开才能操作', true);
      btn.disabled = false;
    }
  });
})();
"""

FIT_JS = r"""
(function () {
  if (window.__pipeFit) return;
  window.__pipeFit = true;
  // 整张大表等比缩到内容区宽度:窗口多宽都不用左右拖,一屏看完(窗口变宽变窄、片段重新载入、字体加载完后再算)。
  // 用 offsetWidth 量:溢出时它就是内容宽度(和 zoom 无关),比 rect / scrollWidth 稳。
  function fitOne(wrap) {
    const table = wrap.querySelector('table.pipe');
    const avail = wrap.clientWidth;
    if (!table || avail < 10) return;
    let zoom = 1;
    for (let i = 0; i < 3; i++) {  // 收敛:按内容宽度算出缩放,再量一遍确认
      table.style.zoom = zoom;
      const content = table.offsetWidth;
      if (content <= avail) break;  // 已经在一屏内(表格被拉伸到内容区宽度)
      zoom = Math.max(0.4, (avail - 1) / content);
    }
    table.style.zoom = zoom;
  }
  function fit() {
    document.querySelectorAll('.pipe-ui .mwrap').forEach(fitOne);
  }
  document.addEventListener('pipe-loaded', fit);
  window.addEventListener('resize', fit);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);  // emoji 字体加载后宽度会变
  setTimeout(fit, 500);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fit);
  else fit();
})();
"""

EDIT_JS = r"""
(function () {
  if (window.__pipeEdit) return;
  window.__pipeEdit = true;
  let saved = false;   // 这次打开对话框后保存过:关闭时刷新页面(格子里的状态 / 信息跟着变)
  let armed = false;   // 有没保存的改动时,第一次关闭只提示
  const dirty = () => Array.from(document.querySelectorAll('#pipe-dialog textarea.etext'))
    .some((t) => !t.readOnly && t.value !== t.dataset.orig);
  async function post(url, body) {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({ error: '服务端没有返回 JSON(页面需要用 sync_status.py --serve 打开)' }));
    if (!res.ok || data.error) throw new Error(data.error || ('HTTP ' + res.status));
    return data;
  }
  const el = (tag, cls, text) => { const x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; };
  function slotBox(box, slot, data) {
    const wrap = el('div', 'eslot');
    const head = el('div', 'eshead');
    head.append(el('b', '', slot.label), el('code', '', slot.path + (slot.selector ? ' # ' + slot.selector : '')));
    if (slot.note) head.append(el('small', '', slot.note));
    const ta = el('textarea', 'etext');
    ta.spellcheck = false;
    ta.value = slot.text;
    ta.dataset.orig = slot.text;
    ta.dataset.version = slot.version;
    const lines = slot.text.split('\n').length;
    ta.rows = slot.kind === 'jsonstr' ? Math.min(Math.max(lines, 2), 6) : Math.min(Math.max(lines + 1, 4), slot.pair ? 24 : 30);
    const bar = el('div', 'esbar');
    const btn = el('button', 'act small primary esave', '保存');
    btn.type = 'button';
    btn.disabled = true;
    const msg = el('span', 'esmsg');
    bar.append(btn, msg);
    if (!slot.editable || data.busy) {
      ta.readOnly = true;
      msg.textContent = slot.why || (data.busy ? data.warn : '只读');
    }
    ta.addEventListener('input', () => { btn.disabled = ta.value === ta.dataset.orig; armed = false; msg.textContent = btn.disabled ? '' : '未保存'; });
    ta.addEventListener('keydown', (ev) => {
      if (ev.key === 'Tab' && !ev.shiftKey) {  // Tab 缩进两格
        ev.preventDefault();
        const s = ta.selectionStart;
        ta.setRangeText('  ', s, ta.selectionEnd, 'end');
        ta.dispatchEvent(new Event('input'));
      } else if ((ev.metaKey || ev.ctrlKey) && ev.key === 's') {  // Ctrl/⌘+S 保存
        ev.preventDefault();
        if (!btn.disabled) btn.click();
      }
    });
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      msg.className = 'esmsg';
      msg.textContent = '保存中…';
      try {
        const r = await post('api/edit/save', { group: box.dataset.group, key: box.dataset.key, slot: slot.id, text: ta.value, version: ta.dataset.version });
        ta.value = r.text;
        ta.dataset.orig = r.text;
        ta.dataset.version = r.version;
        saved = true;
        msg.className = 'esmsg ok';
        msg.textContent = r.message + ' · 这一格现在:' + r.status_label + (r.status === 'stale' || r.status === 'changed' ? '(改好后点下面的「重新提交」)' : '');
        if ((r.status === 'stale' || r.status === 'changed') && !box.querySelector('.eresub')) {
          const re = el('button', 'act primary eresub', '重新提交这一格');
          re.type = 'button';
          const [stage, task] = box.dataset.key.split('/');
          Object.assign(re.dataset, { action: 'resubmit', group: box.dataset.group, stage: stage });
          if (task) re.dataset.task = task;
          box.querySelector('.ebottom').append(re);
        }
      } catch (error) {
        msg.className = 'esmsg ng';
        msg.textContent = '没保存:' + error.message;
        btn.disabled = false;
      }
    });
    wrap.append(head, ta, bar);
    return wrap;
  }
  async function fill(box) {
    if (box.dataset.loaded) return;
    box.dataset.loaded = '1';
    box.innerHTML = '';
    const top = el('div', 'etop', '读取中…');
    box.append(top);
    try {
      const data = await post('api/edit/load', { group: box.dataset.group, key: box.dataset.key });
      top.textContent = '';
      if (!data.slots.length) { top.textContent = '这一格没有可以在页面上编辑的文件'; return; }
      top.append(el('b', '', '✎ 编辑'), el('small', '', '改完点「保存」写回对应文件(Ctrl/⌘+S)'));
      if (data.warn) top.append(el('span', 'ewarn', data.warn));
      let pair = null;
      data.slots.forEach((slot) => {
        const node = slotBox(box, slot, data);
        if (slot.pair) {
          if (!pair || pair.dataset.pair !== slot.pair) { pair = el('div', 'epair'); pair.dataset.pair = slot.pair; box.append(pair); }
          pair.append(node);
        } else {
          pair = null;
          box.append(node);
        }
      });
      box.append(el('div', 'ebottom'));
    } catch (error) {
      top.className = 'etop ng';
      top.textContent = '读不到编辑内容:' + error.message;
      delete box.dataset.loaded;
    }
  }
  // 对话框打开:单个条目直接读出编辑区;整组时每个条目一个「✎ 编辑」按钮,点了再读
  document.addEventListener('pipe-dialog-open', (ev) => {
    saved = false;
    armed = false;
    const boxes = Array.from(document.querySelectorAll('#pipe-dialog .editbox'));
    boxes.forEach((box) => {
      if (ev.detail && ev.detail.single) { fill(box); return; }
      const open = el('button', 'act small eopen', '✎ 编辑');
      open.type = 'button';
      open.addEventListener('click', () => fill(box));
      box.append(open);
    });
  });
  // 关闭前:有没保存的改动先提示一次;关闭后清空编辑区,保存过就刷新页面
  window.__pipeEditGuard = function () {
    if (dirty() && !armed) {
      armed = true;
      const bar = document.getElementById('pipe-dialog-warn');
      if (bar) { bar.textContent = '有没保存的改动:再关一次会丢掉'; bar.hidden = false; }
      return false;
    }
    return true;
  };
  document.addEventListener('pipe-dialog-closed', () => {
    document.querySelectorAll('.editbox').forEach((box) => { box.innerHTML = ''; delete box.dataset.loaded; });
    const bar = document.getElementById('pipe-dialog-warn');
    if (bar) bar.hidden = true;
    armed = false;
    if (saved) { saved = false; location.reload(); }
  });
})();
"""

ASSETS_JS = r"""
(function () {
  if (window.__pipeAssets) return;
  window.__pipeAssets = true;
  const byId = (id) => document.getElementById(id);
  let group = '', version = '', saved = false, canSave = false;
  const el = (tag, cls, text) => { const x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; };
  async function post(url, body) {
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({ error: '服务端没有返回 JSON(页面需要用 sync_status.py --serve 打开)' }));
    return { ok: res.ok && !data.error, data };
  }
  const rows = () => Array.from(byId('assets-rows').querySelectorAll('.arow'));
  const values = () => rows().map((r) => ({ label: r.querySelector('.alabel').value.trim(), link: r.querySelector('.alink-in').value.trim() }));
  function msg(text, cls) { const m = byId('assets-msg'); m.textContent = text; m.className = cls || 'muted'; }
  function dirty() { byId('assets-save').disabled = !canSave; msg('未保存'); }
  function addRow(label, link, info) {
    const r = el('div', 'arow');
    const a = el('input', 'alabel'); a.placeholder = 'label,如 metal_tray'; a.value = label || ''; a.spellcheck = false;
    const b = el('input', 'alink-in'); b.placeholder = '资产链接 / hf download 命令'; b.value = link || ''; b.spellcheck = false;
    const x = el('button', 'act small adel', '×'); x.type = 'button'; x.title = '删掉这个物体(保存后生效)';
    const i = el('div', 'ainfo', info || '');
    [a, b].forEach((inp) => inp.addEventListener('input', () => { i.textContent = ''; i.className = 'ainfo'; dirty(); }));
    b.addEventListener('change', () => { if (b.value.trim()) check(); });
    x.addEventListener('click', () => { r.remove(); dirty(); suggest(); });
    r.append(a, b, x, i);
    byId('assets-rows').append(r);
    return r;
  }
  let hints = [];
  function suggest() {
    const box = byId('assets-suggest');
    box.textContent = '';
    const have = new Set(values().map((v) => v.label));
    const left = hints.filter((h) => !have.has(h));
    if (!left.length) return;
    box.append('按种子猜的物体:');
    left.forEach((h) => {
      const btn = el('button', 'act small', '+ ' + h); btn.type = 'button';
      btn.addEventListener('click', () => { addRow(h, '').querySelector('.alink-in').focus(); suggest(); });
      box.append(btn);
    });
  }
  function show(result) {  // 检查 / 保存的结果写到每一行下面
    const list = rows();
    (result.rows || []).forEach((r) => {
      const row = list[r.row];
      if (!row) return;
      const info = row.querySelector('.ainfo');
      if (r.error) { info.textContent = '✗ ' + r.error; info.className = 'ainfo ng'; return; }
      if (r.skip) { info.textContent = '空着,不会保存'; info.className = 'ainfo'; return; }
      info.textContent = '✓ ' + r.info;
      info.className = 'ainfo ok';
      info.title = r.link;
    });
  }
  async function check() {
    msg('检查中…');
    const { data } = await post('api/assets/save', { group, rows: values(), version, dry_run: true });
    if (data.rows) show(data);
    if (data.errors && Object.keys(data.errors).length) msg(Object.keys(data.errors).length + ' 行有问题', 'ng');
    else if (data.error) msg(data.error, 'ng');
    else msg('都没问题 · ' + data.diff + (data.diff === '没有改动' ? '' : '(还没保存)'), 'ok');
  }
  async function save() {
    byId('assets-save').disabled = true;
    msg('保存中…');
    const { ok, data } = await post('api/assets/save', { group, rows: values(), version });
    if (data.rows) show(data);
    if (!ok) { msg('没保存:' + (data.error || '失败'), 'ng'); byId('assets-save').disabled = !canSave; return; }
    // 链接换成了命令:框里写成保存后的样子
    const list = rows();
    (data.rows || []).forEach((r) => { const row = list[r.row]; if (row) row.querySelector('.alink-in').value = r.link; });
    version = data.version;
    if (data.message !== '没有改动') saved = true;
    msg(data.message + (saved ? ' · 关闭后页面刷新' : ''), 'ok');
  }
  async function open(g) {
    const d = byId('assets-dialog');
    group = g; saved = false; hints = [];
    byId('assets-dialog-group').textContent = g;
    byId('assets-rows').innerHTML = '';
    byId('assets-suggest').textContent = '';
    byId('assets-save').disabled = true;
    msg('读取中…');
    if (!d.open) { if (d.showModal) d.showModal(); else d.setAttribute('open', ''); }
    const { ok, data } = await post('api/assets/load', { group: g });
    if (!ok) { msg('读不到:' + data.error, 'ng'); return; }
    version = data.version;
    canSave = data.can_save;
    const warn = byId('assets-dialog-warn');
    warn.textContent = data.warn || '';
    warn.hidden = !data.warn;
    data.rows.forEach((r) => addRow(r.label, r.link, r.info));
    hints = data.suggest || [];
    if (!data.rows.length) hints.forEach((h) => addRow(h, ''));
    if (!rows().length) addRow('', '');
    suggest();
    msg(data.rows.length ? '已填 ' + data.rows.length + ' 个物体' : '还没填:按种子先列了几个物体,不需要的点 ×');
  }
  function close() {
    const d = byId('assets-dialog');
    const pending = !byId('assets-save').disabled;
    if (pending && !d.__warned) { d.__warned = true; msg('有没保存的改动:再关一次会丢掉', 'ng'); return; }
    d.__warned = false;
    if (d.open) d.close();
    if (saved) { saved = false; location.reload(); }
  }
  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.pipe-ui button.alink[data-group]');
    if (btn) { ev.preventDefault(); ev.stopPropagation(); open(btn.dataset.group); return; }
    const d = byId('assets-dialog');
    if (ev.target.closest('#assets-dialog-close') || (d && ev.target === d)) { close(); return; }
    if (ev.target.closest('#assets-add')) { addRow('', '').querySelector('.alabel').focus(); return; }
    if (ev.target.closest('#assets-check')) { check(); return; }
    if (ev.target.closest('#assets-save')) save();
  }, true);
  document.addEventListener('cancel', (ev) => {
    if (ev.target !== byId('assets-dialog')) return;
    ev.preventDefault();
    close();
  }, true);
})();
"""

PROMPT_JS = r"""
(function () {
  if (window.__pipePrompt) return;
  window.__pipePrompt = true;
  const byId = (id) => document.getElementById(id);
  function copy(text) {  // 用 IP 打开看板时不是安全上下文,没有 navigator.clipboard,退回到 execCommand
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text).then(() => true, () => false);
    const ta = byId('prompt-text');
    ta.focus();
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    return Promise.resolve(ok);
  }
  function open(btn) {
    const d = byId('prompt-dialog');
    byId('prompt-dialog-title').textContent = btn.dataset.title || '';
    const ta = byId('prompt-text');
    ta.value = btn.dataset.prompt || '';
    ta.rows = Math.min(Math.max(ta.value.split('\n').length + 1, 5), 20);
    byId('prompt-msg').textContent = '';
    if (!d.open) { if (d.showModal) d.showModal(); else d.setAttribute('open', ''); }
  }
  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('.pipe-ui button.cprompt[data-prompt]');
    if (btn) { ev.preventDefault(); ev.stopPropagation(); if (!btn.disabled) open(btn); return; }
    // 未开始的格子(data-direct):点格子本身直接给提示词,不先弹审核条目
    const direct = ev.target.closest('#glist td.st[data-direct] > a');
    const dbtn = direct && direct.parentElement.querySelector('button.cprompt[data-prompt]');
    if (dbtn && !dbtn.disabled) { ev.preventDefault(); ev.stopPropagation(); open(dbtn); return; }
    const d = byId('prompt-dialog');
    if (ev.target.closest('#prompt-dialog-close') || (d && ev.target === d)) { d.close(); return; }
    if (ev.target.closest('#prompt-copy')) {
      const ta = byId('prompt-text');
      copy(ta.value).then((ok) => {
        const m = byId('prompt-msg');
        m.textContent = ok ? '已复制,贴到终端里的 Claude Code' : '复制失败:框里已全选,按 Ctrl/⌘+C';
        if (!ok) { ta.focus(); ta.select(); }
      });
    }
  }, true);
})();
"""

ROUTE_JS = r"""
(function () {
  if (window.__pipeRoute) return;
  window.__pipeRoute = true;
  const byId = (id) => document.getElementById(id);
  const sel = (value) => (window.CSS && CSS.escape ? CSS.escape(value) : value);
  const storeOf = (g) => document.querySelector('#gstores .gstore[data-group="' + sel(g) + '"]');
  function clearHash() { history.replaceState(null, '', location.pathname + location.search); }
  const dlg = () => byId('pipe-dialog');
  // 关闭对话框:里面的条目放回各自的 #gstores,取消格子高亮
  function closeDialog(keepHash, force) {
    const d = dlg();
    if (!d) return true;
    if (d.open && !force && window.__pipeEditGuard && !window.__pipeEditGuard()) return false;  // 有没保存的改动
    const wasOpen = d.open;
    d.querySelectorAll('.gmove').forEach((el) => {
      el.classList.remove('gmove');
      const home = el._home;
      el._home = null;
      if (home) home.append(el);
    });
    byId('pipe-dialog-body').innerHTML = '';
    document.querySelectorAll('#glist td.active').forEach((td) => td.classList.remove('active'));
    if (d.open) { d.__closing = true; d.close(); d.__closing = false; }
    if (!keepHash) clearHash();
    if (wasOpen) document.dispatchEvent(new Event('pipe-dialog-closed'));
    return true;
  }
  // 在屏幕中间弹出:一个条目(el),或者整个组(风险种子 + 全部条目)
  function openDialog(g, el, cell) {
    const store = storeOf(g);
    const d = dlg();
    if (!store || !d) return null;
    if (!closeDialog(true)) return null;
    byId('pipe-dialog-group').textContent = g;
    const all = byId('pipe-dialog-all');
    all.href = '#g/' + g;
    all.hidden = !el;
    const body = byId('pipe-dialog-body');
    (el ? [el] : Array.from(store.children)).forEach((x) => { x._home = x.parentElement; x.classList.add('gmove'); body.append(x); });
    if (cell) cell.classList.add('active');
    if (!d.open) { if (d.showModal) d.showModal(); else d.setAttribute('open', ''); }
    body.scrollTop = 0;
    document.dispatchEvent(new CustomEvent('pipe-dialog-open', { detail: { group: g, single: Boolean(el) } }));
    return d;
  }
  let clickedCell = null;  // 点的那一格(照搬的格子条目属于归属组,高亮点的这一格)
  function route() {
    const hash = decodeURIComponent(location.hash.slice(1));
    if (!hash) { closeDialog(true); return; }
    let g = null;
    let el = null;
    const m = hash.match(/^g\/([^/]+)(?:\/(.+))?$/);
    if (m) {
      g = m[1];
      if (m[2]) el = byId('e-' + g + '-' + m[2]);
    } else if (hash.startsWith('e-')) {
      el = byId(hash);
      const holder = el && el.closest('[data-group]');
      g = holder ? holder.dataset.group : null;
    }
    if (!g) return;
    let cell = clickedCell;
    if (!cell) {
      const link = el ? document.querySelector('#glist td.st a[href="#' + sel(el.id) + '"]')
        : document.querySelector('#glist tbody.grp[data-group="' + sel(g) + '"] td.gname a');
      cell = link ? link.closest('td') : null;
    }
    openDialog(g, el, cell);
  }
  document.addEventListener('click', (ev) => {
    const d = dlg();
    if (ev.target.closest('#pipe-dialog-close') || (d && ev.target === d)) {  // 关闭按钮或点对话框外面
      closeDialog();
      return;
    }
    const a = ev.target.closest('#glist a[href^="#"], #pipe-dialog a[href^="#"]');
    if (!a) return;
    ev.preventDefault();
    history.replaceState(null, '', a.getAttribute('href'));
    const td = a.closest('#glist td');
    clickedCell = td && a.getAttribute('href').startsWith('#e-') ? td : null;
    route();
    clickedCell = null;
  });
  document.addEventListener('cancel', (ev) => {  // Esc:和点「关闭」一样(有没保存的改动先提示)
    if (ev.target !== dlg()) return;
    ev.preventDefault();
    closeDialog();
  }, true);
  document.addEventListener('close', (ev) => {
    if (ev.target === dlg() && !ev.target.__closing) closeDialog(false, true);
  }, true);
  window.addEventListener('hashchange', route);
  route();
  // 片段:页面里内嵌的是生成时的版本;<片段>.gen 和内嵌的不一样(页面上审核操作后只重画了片段)才重读整个片段。
  // 读不到 gen(旧片段、静态打开)时照旧重读。
  const boxes = Array.from(document.querySelectorAll('[data-frag]'));
  const fresh = (url) => fetch(url + '?t=' + Date.now(), { cache: 'no-store' })
    .then((r) => (r.ok ? r.text() : Promise.reject(r.status)));
  Promise.all(boxes.map((box) => fresh(box.dataset.frag.replace(/\.html$/, '.gen')).catch(() => '')
    .then((gen) => (gen.trim() && gen.trim() === box.dataset.gen ? null : fresh(box.dataset.frag)
      .then((text) => { closeDialog(true, true); box.innerHTML = text; })))
    .catch(() => {}))).then(() => {
    route();
    const filter = byId('filter');
    if (filter && filter.value) filter.dispatchEvent(new Event('input', { bubbles: true }));
    document.dispatchEvent(new Event('pipe-loaded'));
  });
  // 筛选框
  document.addEventListener('input', (ev) => {
    if (ev.target.id !== 'filter') return;
    const q = ev.target.value.trim().toLowerCase();
    let shown = 0;
    document.querySelectorAll('#glist tbody.grp').forEach((body) => {
      const hit = !q || (body.dataset.search || '').includes(q);
      body.hidden = !hit;
      if (hit) shown += 1;
      body.querySelectorAll('tr').forEach((tr) => tr.classList.toggle('hit', Boolean(q) && hit && tr.textContent.toLowerCase().includes(q)));
    });
    byId('filter-count').textContent = q ? shown + ' 个组匹配' : '';
  });
})();
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def redirect_page(old: str) -> str:
    """旧页面地址:跳到 PIPELINE.html(``#g/x`` 保留,``#pipeline-x`` / ``#data-x`` 之类换成 ``#g/x``)。"""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>跳转到 {PAGE}</title>
<meta http-equiv="refresh" content="0; url={PAGE}"></head><body>
<p>{old} 已合并到 <a href="{PAGE}">{PAGE}</a>。</p>
<script>
const h = location.hash.slice(1);
const m = h.match(/^(?:pipeline|build|data|eval)-(.+)$/);
location.replace('{PAGE}' + (m ? '#g/' + m[1] : (h.startsWith('g/') || h.startsWith('e-') ? '#' + h : '')));
</script></body></html>
"""


def render(out_dir: Path) -> Path:
    """写流水线大表片段 ``data/pipeline/_fragments/table.html``(看板页面打开时读取)。

    页面不自动刷新输入框里的内容,操作后自动重载。
    """
    for key, text in fragments(out_dir).items():
        _write(out_dir / FRAGMENT_DIR / f"{key}.html", text)
        _write(out_dir / FRAGMENT_DIR / f"{key}.gen", _gen(text))  # 写完片段再写 gen:读到新 gen 时片段一定是新的
    for key in OLD_FRAGMENTS:
        (out_dir / FRAGMENT_DIR / f"{key}.html").unlink(missing_ok=True)
    return out_dir / PAGE
