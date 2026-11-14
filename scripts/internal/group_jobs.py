#!/usr/bin/env python3
"""看板大表里的后台任务:ingest 格的「下载资产」(按组),collect / exchange / upload 格的「采集」「转换」「上传」(按任务)。

任务在独立会话里后台运行,看板重启不会中断。状态在 ``<RSJ_PIPELINE_DIR>/_jobs/<group>.<task>.json``,
日志在 ``data/.cache/jobs/task_<task>_<kind>_<时间>.log``(``RSJ_PIPELINE_DIR`` 的上一级)。每个任务同一时间只跑一个,
转换 / 上传同一时间只有一个(LeRobot 转换和上传脚本共用缓存);不同任务的采集可以同时跑在不同的卡上。

按钮对应四种任务(``kind``;任务构造各格的「▶ Claude」不在这里,只给出提示词让用户贴进终端的 Claude 交互着做,
见 ``pipeline_page.claude_prompt``。2026-11-13 前有过后台无人值守的 ``claude`` 任务,旧记录还能在 status 里看到):

    ingest   「下载资产」(按组,任务名记作 ``__group__``):资产链接填好(seed 批准、来源合规)后,
             ``source ext_assets/integrate/pxrenv.sh`` + ``simready_assets.py --group <g> --report``
             下载 SimReady 原始模型、转换成 robodojo_format、挂进 Assets;全部入库后自动提交 ingest 给用户审
             (有没入库的就不提交)
    collect  「采集」:该任务的 expert 批准后,补采到 RSJ_COLLECT_TARGET 条(``collect_data_parallel.sh``;有残留分片时
             按原条数续采;任务 YAML 注释里写了 ``--layout_template <路径>`` 的自动带上)
    export   「转换」(exchange 格):采够后导出 LeRobot v3.0 + v2.1(``export_lerobot.sh --format both``)
    upload   「上传」:转换完、条数一致后,``--task`` 上传 HDF5 和 LeRobot(先 dry-run 给用户看,默认 ``--no_delete``),
             再核对远端
    eval     「评测」(eval 格):expert 批准、模型下载完、评测布局够 50 个后,用已下载的 Pi_0 / Pi_05 checkpoint
             各评 50 条(``robodojo.sh eval``,seed 0,arx_x5,joint);risky 没有自己的模型时用同组 safe 模型 zero-shot。
             没评满的接着同一个 ``ROBODOJO_RUN_ID`` 续评,评满的跳过。每条流水线用一对 GPU(policy + sim),
             填几对就并行几条。评完不提交:报告和提交 eval 审核让 Claude 做(/rsj-eval)
    layouts  「生成布局」(评测布局格):layout 批准后,在 Isaac 里分批(每批 20 个 seed,一个进程)检查哪些 seed
             能稳定沉降(``eval_layouts.py check``),取前 50 个写评测布局(``eval_layouts.py write``)。同组还缺布局的
             自有任务一起生成,共用同一批 seed(safe / risky 逐条可比);已有一部分(不到 50 个)的会被替换

采集 / 转换 / 上传不审核(2026-11-13 起),做完不提交;只要求该任务的 expert 已批准。命令行(看板调用的就是这些函数)::

    python scripts/internal/group_jobs.py plan ingest <group> __group__
    python scripts/internal/group_jobs.py plan collect|exchange|upload <group> <task>
    python scripts/internal/group_jobs.py start collect <group> <task> --gpus 0,1
    python scripts/internal/group_jobs.py start exchange <group> <task>
    python scripts/internal/group_jobs.py start upload <group> <task> [--delete]
    python scripts/internal/group_jobs.py start eval <group> <task> --gpus 2,3,5,7
    python scripts/internal/group_jobs.py start layouts <group> <task> --gpus 2
    python scripts/internal/group_jobs.py status
    python scripts/internal/group_jobs.py stop <group> <task>
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsj_pipeline as rp  # noqa: E402

ROOT = rp.ROOT
JOB_DIR = rp.STATE_DIR / "_jobs"
LOG_DIR = rp.STATE_DIR.parent / ".cache" / "jobs"  # 看板的任务日志目录(测试时随 RSJ_PIPELINE_DIR 走)
SHARDS = ROOT / "data" / rp.BENCHMARK_NAME / "_shards"
LEROBOT = ROOT / "data" / f"{rp.BENCHMARK_NAME}-Lerobot"
HDF5_REPO = os.environ.get("MS_DATA_REPO_ID", "zjushine/RoboSafeJanus-Benchmark")
LEROBOT_REPO = os.environ.get("MS_LEROBOT_REPO_ID", f"{HDF5_REPO}-Lerobot")
START_SEED = 100000
SEED_STRIDE = int(os.environ.get("ROBODOJO_SEED_STRIDE", "1000"))
FREE_GPU_MIB = 2000  # 显存占用低于这个值算空闲卡
TEMPLATE_RE = re.compile(r"--layout_template\s+([^\s。,,;;))]+)")
KIND_LABEL = {"ingest": "下载资产", "collect": "采集", "export": "转换 LeRobot", "upload": "上传 ModelScope",
              "eval": "评测", "layouts": "生成布局", "claude": "Claude"}
DATA_BUTTONS = ("collect", "exchange", "upload")  # 不审核,expert 批准后直接做
BUTTONS = ("ingest", *DATA_BUTTONS, "eval", "layouts")
# 评测(rsj-eval / docs/model_evaluation.md 的固定协议)
EVAL_POLICIES = ("Pi_0", "Pi_05")
EVAL_NUM = 50
EVAL_SEED = 0
EVAL_CFG = "arx_x5"
MODEL_DIR = ROOT / "data" / "RoboSafeJanus-Model"  # download_models_from_modelscope.py 的 --local_dir
EVAL_RESULT = ROOT / "eval_result" / "RoboDojo"
EVAL_LAYOUT = ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / EVAL_CFG / str(EVAL_SEED)
EVAL_LOGS = ROOT / "eval_logs"
LAYOUT_CACHE = ROOT / "data" / ".cache" / "eval_layouts"  # check 写的稳定 seed 列表(每批一个文件)
LAYOUT_CHUNK = 10  # 每个 Isaac 进程检查多少个 seed:启动约 1 分钟,一个 seed 约 20 秒;块小、分到多张卡上更快
LAYOUT_MAX_SEED = 200  # 最多检查到这个 seed,还凑不够 50 个就失败
LAYOUT_CHUNK_TIMEOUT = 1200  # 一批超过 20 分钟算卡住,杀掉,这批 seed 不要
GROUP_TASK = "__group__"  # 按组的任务(ingest)的任务名
PXRENV = "ext_assets/integrate/pxrenv.sh"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
NAME_RE = re.compile(r"[A-Za-z0-9_]+")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def job_id(group: str, task: str) -> str:
    return f"{group}.{task}"


# ---------------------------------------------------------------------------
# 任务记录
# ---------------------------------------------------------------------------


def job_path(jid: str) -> Path:
    return JOB_DIR / f"{jid}.json"


def read_job(jid: str) -> dict | None:
    try:
        return json.loads(job_path(jid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_job(job: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = job_path(job_id(job["group"], job["task"]))
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_job(group: str, task: str, **fields) -> None:
    job = read_job(job_id(group, task)) or {"group": group, "task": task}
    job.update(fields)
    write_job(job)


def _alive(pid: int | None) -> bool:
    """pid 还在,且确实是本脚本的 run 进程(防 pid 复用)。"""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        args = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True
    return "group_jobs.py" in args and " run " in args


def job_label(job: dict) -> str:
    if job.get("kind") == "claude":
        return f"Claude {job.get('stage', '')}"
    return KIND_LABEL.get(job.get("kind"), job.get("kind"))


def job_state(job: dict) -> str:
    """running / done / failed / stopped / interrupted(进程没了但没写结果)。"""
    if job.get("returncode") is not None:
        if job.get("stopped"):
            return "stopped"
        return "done" if job["returncode"] == 0 else "failed"
    return "running" if _alive(job.get("pid")) else "interrupted"


def log_tail(path: str | None, limit: int = 6000, lines: int = 40) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - limit))
            text = fp.read().decode("utf-8", "replace")
    except OSError:
        return ""
    return "\n".join(text.replace("\r", "\n").splitlines()[-lines:])


def all_jobs(tail: bool = True) -> dict[str, dict]:
    """``{<group>.<task>: 任务记录}``(旧的按组记录没有 task,跳过)。"""
    out = {}
    if not JOB_DIR.exists():
        return out
    for path in sorted(JOB_DIR.glob("*.json")):
        job = read_job(path.stem)
        if not job or not job.get("task"):
            continue
        job["state"] = job_state(job)
        job["label"] = job_label(job)
        if tail:
            job["progress"] = job_progress(job)
        out[path.stem] = job
    return out


def running(kinds=None, exclude: str | None = None) -> list[dict]:
    return [j for jid, j in all_jobs(tail=False).items()
            if j["state"] == "running" and jid != exclude and (kinds is None or j.get("kind") in kinds)]


# ---------------------------------------------------------------------------
# 进度(看板任务面板只显示进度,不显示日志)
# ---------------------------------------------------------------------------

_UPLOAD_CACHE: dict[str, dict] = {}
_RE_TAG = re.compile(r"^\[(upload_to_modelscope|upload_lerobot)\] local:")
_RE_FILES = re.compile(r"Uploading (\d+) file\(s\) in (\d+) batch")
_RE_SIZE = re.compile(r"\] total: upload .*?([\d.]+) ([KMGT]?)i?B\)")
_RE_BAR = re.compile(r"^\[Uploading (.+?)\]:\s+\d+%\|[^|]*\|\s*([\d.]+)([kMG]?)/")
_RE_COMMIT = re.compile(r"Batch (\d+)/(\d+): committed (\d+) file")
_RE_VERIFY = re.compile(r"^\[(upload_to_modelscope|upload_lerobot)\] (verified|done)")
UPLOAD_NAME = {"upload_to_modelscope": "HDF5", "upload_lerobot": "LeRobot"}
UNIT = {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}


def upload_progress(log: str | None) -> dict | None:
    """ModelScope 上传日志的进度:``{text, done, total}``。日志很大(每个文件一条 tqdm),只读新增的部分。

    进度条 = 已提交的文件 / 要上传的文件(按批次提交,一批 256 个);文字里另写已经传了多少字节
    (每个文件进度条见过的最大值之和,比批次更细)。"""
    if not log:
        return None
    c = _UPLOAD_CACHE.setdefault(log, {"pos": 0, "rest": "", "phase": "", "files": 0, "batches": 0,
                                       "batch": 0, "committed": 0, "bytes": {}, "size": 0, "finished": []})
    try:
        with open(log, "rb") as fp:
            fp.seek(c["pos"])
            chunk = fp.read()
            c["pos"] = fp.tell()
    except OSError:
        return None
    text = c["rest"] + chunk.decode("utf-8", "replace")
    lines = re.split(r"[\r\n]", text)
    c["rest"] = lines.pop()
    for line in lines:
        line = re.sub(r"^(?:\x1b\[A)+", "", line)  # tqdm 多行进度条的光标上移
        if m := _RE_BAR.match(line):
            c["bytes"][m[1]] = max(c["bytes"].get(m[1], 0), float(m[2]) * UNIT[m[3]])
        elif m := _RE_COMMIT.search(line):
            c["batch"], c["batches"] = int(m[1]), int(m[2])
            c["committed"] += int(m[3])
        elif m := _RE_TAG.match(line):
            c.update(phase=UPLOAD_NAME[m[1]], files=0, batches=0, batch=0, committed=0, bytes={}, size=0)
        elif m := _RE_SIZE.search(line):
            c["size"] = float(m[1]) * UNIT[m[2]]
        elif m := _RE_FILES.search(line):
            c["files"], c["batches"] = int(m[1]), int(m[2])
        elif (m := _RE_VERIFY.match(line)) and UPLOAD_NAME[m[1]] not in c["finished"]:
            c["finished"].append(UPLOAD_NAME[m[1]])
    if not c["phase"]:
        return None
    prefix = "".join(f"{x} ✓ · " for x in c["finished"] if x != c["phase"])
    if c["phase"] in c["finished"]:
        return {"text": prefix + f"上传 {c['phase']} 完成,核对中", "done": 1, "total": 1}
    # 进度条按字节(HDF5 是传完才一起提交,按提交的文件数会一直是 0);文字里另写提交了几批
    sent, size = sum(c["bytes"].values()), c["size"]
    bits = [f"上传 {c['phase']}"]
    if size:
        bits.append(f"{min(sent, size) / 1e9:.1f}/{size / 1e9:.1f} GB")
    elif c["files"]:  # 小文件按批提交,没有字节数时才写
        bits.append(f"已提交 {min(c['committed'], c['files'])}/{c['files']} 个文件")
    if not size:  # 没有总大小时退回按提交的文件数
        return {"text": prefix + " · ".join(bits), "done": min(c["committed"], c["files"]), "total": c["files"]}
    return {"text": prefix + " · ".join(bits), "done": int(min(sent, size)), "total": int(size)}


def collect_progress(task: str) -> dict:
    n = rp._hdf5_count(task)
    shard = sum(1 for _ in (SHARDS / task).glob("w*/**/*.hdf5")) if (SHARDS / task).exists() else 0
    have = min(n + shard, rp.COLLECT_TARGET)
    return {"text": f"已采 {have}/{rp.COLLECT_TARGET} 条", "done": have, "total": rp.COLLECT_TARGET}


def eval_progress(group: str, task: str) -> dict | None:
    """每个要评的模型评了几条(取同一 ckpt 下条数最多的那次,和续评的口径一样)。"""
    try:
        models = [m for m in eval_models(group, task, set()) if m["step"]]
    except (OSError, ValueError, KeyError):
        return None
    if not models:
        return None
    counts = [(m["policy"], min(EVAL_NUM, max((r[1] for r in eval_runs(task, m["policy"], m["ckpt"])), default=0)))
              for m in models]
    return {"text": " · ".join(f"{p} {n}/{EVAL_NUM}" for p, n in counts),
            "done": sum(n for _, n in counts), "total": EVAL_NUM * len(counts)}


def job_progress(job: dict) -> dict | None:
    """任务面板 / 格子里的进度 ``{text, done, total}``(total 为 0 时只有文字);算不出来时 None(只显示步骤)。"""
    try:
        kind, task = job.get("kind"), job.get("task", "")
        if kind == "eval":
            return eval_progress(job["group"], task)
        if kind == "collect":
            return collect_progress(task)
        if kind == "upload":
            return upload_progress(job.get("log"))
        if kind == "layouts":
            return layouts_progress(job["group"], job)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------


class PlanError(Exception):
    """不能开始(闸门没过、任务在跑等),消息直接给用户看。"""


def label_problem(problem: str) -> str:
    """``layout/x: none`` → ``layout/x: 未提交``(状态换成页面上的中文)。"""
    return re.sub(r": (\w+)", lambda m: ": " + (rp.STATUS_LABEL.get(m[1]) or "未提交"), problem, count=1)


def _problems(head: str, problems: list[str]) -> PlanError:
    words = [label_problem(p) for p in problems]
    return PlanError(head + "\n" + "\n".join(f"  - {p}" for p in words))


def _seeds(path: Path) -> list[int]:
    try:
        return [int(s) for s in path.read_text().split()]
    except (OSError, ValueError):
        return []


def layout_template(task: str) -> str | None:
    """任务 YAML 注释里写的 ``--layout_template <路径>``(layout 阶段约定的写法)。"""
    for base in (rp.MIRROR / "config", ROOT / "task" / "RoboDojo" / "config"):
        path = base / f"{task}.yml"
        if path.exists():
            m = TEMPLATE_RE.search(path.read_text(encoding="utf-8"))
            return m.group(1) if m else None
    return None


def resume_target(task: str) -> int | None:
    """残留分片的原条数:各 worker 日志里第一个 ``target=<n>`` 之和。"""
    total, found = 0, False
    for log in sorted((SHARDS / task / "logs").glob("w*.log")):
        m = re.search(r"\] task=\S+ target=(\d+)", log.read_text(encoding="utf-8", errors="replace"))
        if m:
            total += int(m.group(1))
            found = True
    return total if found else None


def task_collect_plan(task: str) -> dict:
    n = rp._hdf5_count(task)
    cdir = rp._bench_dir(task)
    item = {"task": task, "have": n, "target": rp.COLLECT_TARGET, "episode_num": 0, "start_seed": None,
            "template": layout_template(task), "resume": False}
    if (SHARDS / task).exists():
        num = resume_target(task)
        if num is None:
            raise PlanError(f"有残留分片 _shards/{task},但从日志里读不出原条数,请让 Claude 处理")
        item.update(episode_num=num, resume=True)
        return item
    if n >= rp.COLLECT_TARGET:
        return item
    item["episode_num"] = rp.COLLECT_TARGET - n
    used = _seeds(cdir / "seed.txt") + _seeds(cdir / "next_seed.txt")
    if used:  # 新分片的 seed 段放到已有 seed 之后,合并时不会撞
        item["start_seed"] = START_SEED + max(0, -(-(max(used) + 1 - START_SEED) // SEED_STRIDE)) * SEED_STRIDE
    return item


def gpu_usage() -> list[dict]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "used": int(parts[1]), "total": int(parts[2]), "util": parts[3]})
    return gpus


def _state(group: str, task: str, stage: str) -> dict:
    if not NAME_RE.fullmatch(group) or not NAME_RE.fullmatch(task):
        raise PlanError("组名 / 任务名不合法")
    if not rp.state_path(group).exists():
        raise PlanError(f"{group}: 还没有流水线状态")
    state = rp.load_state(group)
    tasks = rp.own_tasks(state) if stage in (*DATA_BUTTONS, "layouts") else rp.stage_tasks(state, stage)
    if task not in tasks:
        raise PlanError(f"{task} 不是 {group} 的自有任务(same_as 引用的在归属组里做)")
    return state


def _expert_ok(state: dict, task: str, what: str) -> None:
    """采集 / 转换 / 上传只要求该任务的 expert 已批准且仍有效。"""
    status, why = rp.entry_status(state, f"expert/{task}")
    if status != "approved":
        label = rp.STATUS_LABEL.get(status) or "未提交"
        raise PlanError(f"{task} 还不能{what}:expert 要先批准(现在:{label}{',' + why if why else ''})")


def _collected(task: str) -> str | None:
    """没采完的原因;None 表示 hdf5 够了且没有残留分片。"""
    n = rp._hdf5_count(task)
    if (SHARDS / task).exists():
        return f"有残留分片 _shards/{task}(采集没跑完),先点「采集」续采"
    if n < rp.COLLECT_TARGET:
        return f"只有 {n}/{rp.COLLECT_TARGET} 条,先点「采集」采够"
    return None


def _busy(group: str, task: str) -> None:
    job = read_job(job_id(group, task))
    if job and job_state(job) == "running":
        what = job_label(job)
        raise PlanError(f"{task} 已有任务在运行:{what}(开始于 {job.get('started_at')})")


def collect_plan(group: str, task: str) -> dict:
    state = _state(group, task, "collect")
    _expert_ok(state, task, "采集")
    t = task_collect_plan(task)
    jid = job_id(group, task)
    busy_gpus = {g for j in running(("collect",), exclude=jid) for g in j.get("gpus", [])}
    gpus = gpu_usage()
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy_gpus]
    if t["resume"]:
        what = f"续采残留分片(原 {t['episode_num']} 条)"
    elif t["episode_num"]:
        what = f"补采 {t['episode_num']} 条" + (f",seed 从 {t['start_seed']} 起" if t["start_seed"] else "")
    else:
        what = "已够,不用采"
    tpl = f",模板 {t['template']}" if t["template"] else ""
    lines = [f"{task}:采到 {rp.COLLECT_TARGET} 条", f"已有 {t['have']} 条 → {what}{tpl}"]
    if gpus:
        def gpu_text(g: dict) -> str:
            busy = "(别的任务在用)" if g["index"] in busy_gpus else ""
            return f"{g['index']}:{g['used'] // 1024}/{g['total'] // 1024}G{busy}"

        lines += ["", "GPU 显存占用:" + "  ".join(gpu_text(g) for g in gpus)]
    if t["episode_num"]:
        per_gpu = os.environ.get("ROBODOJO_PER_GPU", "3")
        lines += ["", f"用下面填的所有卡(每卡 {per_gpu} 个 worker,一个约 9 GB)。中断后再点一次就续采。",
                  "采完不用审核,接着点 exchange 格的「转换」。"]
    return {"group": group, "task": task, "kind": "collect", "plan": t, "free_gpus": free,
            "need_gpus": bool(t["episode_num"]), "nothing": not t["episode_num"], "text": "\n".join(lines)}


def export_needed(task: str) -> str | None:
    """需要导出的原因;None 表示 v3.0 / v2.1 都和 hdf5 条数一致且 v2.1 不早于 v3.0。"""
    n = rp._hdf5_count(task)
    infos = {v: LEROBOT / v / rp._task_rel(task) / "meta" / "info.json" for v in ("v30", "v21")}
    got = {v: rp.read_json(p).get("total_episodes") if p.exists() else None for v, p in infos.items()}
    if got["v30"] != n or got["v21"] != n:
        return f"hdf5 {n} 条,v30 {got['v30']},v21 {got['v21']}"
    if infos["v21"].stat().st_mtime < infos["v30"].stat().st_mtime:
        return "v21 早于 v30"
    return None


def _upload_cmd(script: str, repo: str, task: str, *extra: str) -> list[str]:
    return [sys.executable, "-u", script, "--repo_id", repo, "--task", task, *extra]


UPLOADERS = (("HDF5", "src/data_collect/upload_to_modelscope.py", HDF5_REPO),
             ("LeRobot", "src/data_collect/upload_lerobot_to_modelscope.py", LEROBOT_REPO))


def _no_other_transfer(group: str, task: str) -> None:
    busy = running(("export", "upload"), exclude=job_id(group, task))
    if busy:
        raise PlanError(f"{busy[0]['task']} 正在{KIND_LABEL[busy[0]['kind']]},等它结束")


def export_plan(group: str, task: str) -> dict:
    """「转换」:采够后导出 LeRobot v3.0 + v2.1。"""
    state = _state(group, task, "exchange")
    _expert_ok(state, task, "转换")
    reason = _collected(task)
    if reason:
        raise PlanError(f"{task} 还不能转换:{reason}")
    _no_other_transfer(group, task)
    need = export_needed(task)
    lines = [f"{task}:导出 LeRobot v3.0 + v2.1(export_lerobot.sh --format both)", "",
             f"现在:{need}" if need else "已经转换过,v3.0 / v2.1 条数都和 hdf5 一致,不用再转",
             "", "转换完不用审核,接着点 upload 格的「上传」。" if need else ""]
    return {"group": group, "task": task, "kind": "export", "reason": need, "nothing": not need,
            "text": "\n".join(lines).rstrip()}


def upload_plan(group: str, task: str, dry_run: bool = True) -> dict:
    """「上传」:转换完后 dry-run 给用户看,确认后上传 HDF5 + LeRobot。"""
    state = _state(group, task, "upload")
    _expert_ok(state, task, "上传")
    reason = _collected(task)
    if reason:
        raise PlanError(f"{task} 还不能上传:{reason}")
    need = export_needed(task)
    if need:
        raise PlanError(f"{task} 还不能上传:LeRobot 没转换好({need}),先点 exchange 格的「转换」")
    _no_other_transfer(group, task)
    lines = [f"{task}:上传 HDF5 + LeRobot 到 ModelScope(只动这个任务的目录)", ""]
    deletes = uploads = 0
    if dry_run:
        for label, script, repo in UPLOADERS:
            proc = subprocess.run(_upload_cmd(script, repo, task, "--dry_run"), cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=1800)
            out = proc.stdout + proc.stderr
            if proc.returncode != 0:
                tail = "\n".join(out.splitlines()[-30:])
                raise PlanError(f"{label} dry-run 失败(exit {proc.returncode}):\n{tail}")
            keep = [ln for ln in out.splitlines() if ln.startswith(("  upload", "  delete", "  new", "  changed"))
                    or "total:" in ln or "--task" in ln]
            m = re.search(r"delete (\d+) remote files", out)
            deletes += int(m.group(1)) if m else 0
            m = re.search(r"total: upload (\d+)(?: new \+ (\d+) changed)?", out)
            uploads += sum(int(x) for x in m.groups() if x) if m else 0
            lines += [f"[{label}] {repo}", *(keep or out.splitlines()[-12:]), ""]
    if dry_run and not uploads and not deletes:
        lines.append("远端已经和本地一致;确认后只做一次逐文件核对。")
    lines.append("上传后逐文件核对 sha256,再跑 check_task_alignment --only modelscope。")
    if deletes:
        lines.append(f"⚠ 远端有 {deletes} 个本地已没有的文件:默认保留(--no_delete),勾选下面的框才删除。")
    return {"group": group, "task": task, "kind": "upload", "deletes": deletes, "nothing": False,
            "text": "\n".join(lines)}


def _ingest_cmd(group: str, *extra: str) -> list[str]:
    """``simready_assets.py`` 要 pxr:先 source pxrenv.sh(激活 RoboDojo 环境),走 hf-mirror。"""
    script = ("python", "ext_assets/integrate/simready_assets.py", "--group", group, *extra)
    args = " ".join(shlex.quote(x) for x in script)
    return ["bash", "-c", f"source {PXRENV} && export HF_ENDPOINT={shlex.quote(HF_ENDPOINT)} && {args}"]


def ingest_report(group: str) -> str:
    return rp.rel(rp.STATE_DIR / group / "ingest_report.md")


def ingest_plan(group: str, task: str) -> dict:
    """ingest 格的「下载资产」:资产链接填好后按组下载 / 转换 / 入库,完了提交 ingest 审核。"""
    if task != GROUP_TASK or not NAME_RE.fullmatch(group):
        raise PlanError("ingest 是按组的(任务名用 __group__)")
    if not rp.state_path(group).exists():
        raise PlanError(f"{group}: 还没有流水线状态")
    state = rp.load_state(group)
    status, why = rp.entry_status(state, "ingest")
    if status == "awaiting_review":
        raise PlanError("ingest 已提交,等你审核(点 ingest 格)")
    if status == "approved":
        raise PlanError("ingest 已批准;改了资产链接后会失效,那时再下载")
    problems = rp.upstream_problems(state, "ingest", None)
    if problems:
        raise _problems("还不能下载:", problems)
    try:
        proc = subprocess.run(_ingest_cmd(group, "--dry_run"), cwd=str(ROOT), capture_output=True, text=True,
                              timeout=300)
    except subprocess.TimeoutExpired:
        raise PlanError("dry-run 超时(5 分钟)") from None
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        raise PlanError("dry-run 失败:\n" + "\n".join(out.splitlines()[-40:]))
    label = rp.STATUS_LABEL.get(status) or "未开始"
    lines = [f"{group}:下载 SimReady 原始模型 → 转换成 robodojo_format → 挂进 Assets/Object/RoboDojo"
             f"(ingest 现在:{label}{',' + why if why else ''})", "",
             "计划(simready_assets.py --dry_run):", *("  " + ln for ln in out.splitlines()[-60:]), "",
             f"全部入库后写入库报告 {ingest_report(group)} 并提交 ingest 给你审;有资产没入库就不提交,看日志里的 ❌。",
             f"走 {HF_ENDPOINT};Palatial 资产几百 MB,可能要几分钟到几十分钟。已入库且来源一致的跳过。"]
    return {"group": group, "task": task, "kind": "ingest", "nothing": False, "text": "\n".join(lines)}


def task_level(task: str) -> str:
    """``safe_tasks`` / ``risky_tasks``(任务定义里的层级)。"""
    sys.path.insert(0, str(ROOT))
    from src.data_collect.benchmark_layout import task_groups

    return (task_groups().get(task) or (None, "safe_tasks"))[1]


def _downloading() -> set[tuple[str, str]]:
    """正在跑的 ``download_models_from_modelscope.py`` 的 ``(policy, 模型目录名)``("*" 表示没限定)。"""
    found = set()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("download_models_from_modelscope.py") for a in args):
            value = {f: args[i + 1] for i, f in enumerate(args[:-1]) if f in ("--policy", "--task")}
            found.add((value.get("--policy", "*"), value.get("--task", "*")))
    return found


def _complete_step(run_dir: Path) -> str | None:
    """run 目录里下载完整的最新 step(没有 ``*.incomplete`` 临时文件);没有时 None。"""
    steps = sorted((d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
    for step in reversed(steps):
        if not any(step.rglob("*.incomplete")) and any(step.rglob("*")):
            return step.name
    return None


def eval_models(group: str, task: str, downloading: set | None = None) -> list[dict]:
    """这个任务要评的模型:每个 policy 一个 ``{policy, key, ckpt, step, zero_shot, ready, why}``。

    先找同名单任务模型(``<policy>/<task>/``),再找同组同层级的组模型(``<group>_<level>``);risky 都没有时用
    同组 safe 组模型 zero-shot。``ready`` = 本地下载完整、``checkpoints/<run>`` 链接在、没有在下载。"""
    level = task_level(task)
    keys = [(task, False), (f"{group}_{level}", False)]
    if level == "risky_tasks":
        keys.append((f"{group}_safe_tasks", True))
    downloading = _downloading() if downloading is None else downloading
    out = []
    for policy in EVAL_POLICIES:
        for key, zero_shot in keys:
            base = MODEL_DIR / policy / key
            runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
            if not runs:
                continue
            run = runs[-1]
            item = {"policy": policy, "key": key, "ckpt": run.name, "step": _complete_step(run),
                    "zero_shot": zero_shot, "ready": False, "why": ""}
            link = ROOT / "XPolicyLab" / "policy" / policy / "checkpoints" / run.name
            if any(p in (policy, "*") and t in (key, "*") for p, t in downloading):
                item["why"] = "还在下载"
            elif not item["step"]:
                item["why"] = "没下载完(在模型列点「⬇ 下载」)"
            elif not link.exists():
                item["why"] = f"缺 checkpoints/{run.name} 链接(重新点一次下载会补上)"
            else:
                item["ready"] = True
            out.append(item)
            break
    return out


def eval_layouts(task: str) -> int:
    return len(list(EVAL_LAYOUT.glob(f"{task}_*.json")))


def eval_runs(task: str, policy: str, ckpt: str) -> list[tuple[str, int]]:
    """已有的评测 ``[(run_id, eval_time)]``,按时间排。"""
    base = EVAL_RESULT / task / policy / EVAL_CFG / f"{EVAL_SEED}_ckpt_name={ckpt},action_type=joint"
    out = []
    for path in sorted(base.glob("*/_result.json")) if base.is_dir() else []:
        try:
            out.append((path.parent.name, int(json.loads(path.read_text(encoding="utf-8")).get("eval_time") or 0)))
        except (OSError, ValueError):
            continue
    return out


def eval_blocked(state: dict, group: str, task: str, downloading: set | None = None) -> str:
    """评测按钮不能点的原因(看板 eval 格用);空串表示可以点。"""
    status, why = rp.entry_status(state, f"expert/{task}")
    if status != "approved":
        return "expert 要先批准(现在:" + (rp.STATUS_LABEL.get(status) or "未提交") + ")"
    models = eval_models(group, task, downloading)
    if not models:
        return "还没有模型:远端有了以后在模型列点「⬇ 下载」"
    if not any(m["ready"] for m in models):
        return "模型还不能用:" + ";".join(f"{m['policy']} {m['why']}" for m in models)
    n = eval_layouts(task)
    if n < EVAL_NUM:
        return f"评测布局只有 {n}/{EVAL_NUM} 个(让 Claude 按 /rsj-eval 第 2 步补齐)"
    return ""


def eval_plan(group: str, task: str) -> dict:
    """eval 格的「评测」:已下载的 Pi_0 / Pi_05 各评 50 条,没评满的续评。"""
    state = _state(group, task, "eval")
    reason = eval_blocked(state, group, task)
    if reason:
        raise PlanError(f"{task} 还不能评测:{reason}")
    queue, lines = [], [f"{task}:每个模型评 {EVAL_NUM} 条(seed {EVAL_SEED},{EVAL_CFG},joint,policy-env uv)", ""]
    for m in eval_models(group, task):
        name = f"{m['policy']} · {m['ckpt']}(step {m['step'] or '?'}{',zero-shot' if m['zero_shot'] else ''})"
        if not m["ready"]:
            lines.append(f"  ⏭ {name}:{m['why']},这次不评")
            continue
        runs = eval_runs(task, m["policy"], m["ckpt"])
        best = max(runs, key=lambda r: r[1]) if runs else None
        if best and best[1] >= EVAL_NUM:
            lines.append(f"  ✅ {name}:已评满 {best[1]} 条({best[0]}),跳过")
            continue
        resume = best[0] if best and best[1] > 0 else None
        lines.append(f"  ▶ {name}:" + (f"续评 {best[0]}(已有 {best[1]}/{EVAL_NUM})" if resume else "新评"))
        queue.append({**m, "run_id": resume})
    if not queue:
        lines += ["", "都评满了,不用再跑。报告和提交审核让 Claude 做:/rsj-eval " + group]
        return {"group": group, "task": task, "kind": "eval", "queue": [], "nothing": True, "text": "\n".join(lines)}
    jid = job_id(group, task)
    busy_gpus = {g for j in running(("collect", "eval"), exclude=jid) for g in j.get("gpus", [])}
    gpus = gpu_usage()
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy_gpus]
    if gpus:
        def gpu_text(g: dict) -> str:
            busy = "(别的任务在用)" if g["index"] in busy_gpus else ""
            return f"{g['index']}:{g['used'] // 1024}/{g['total'] // 1024}G{busy}"

        lines += ["", "GPU 显存占用:" + "  ".join(gpu_text(g) for g in gpus)]
    lines += ["", "下面填的卡两两一对(policy 卡 + sim 卡),一对跑一条,填几对就并行几条;只填 1 张时 policy 和 sim 共用。",
              f"每条约 1–2 小时。中断后再点一次就续评;日志在 {rp.rel(EVAL_LOGS)}/。",
              f"评完不用审核这一步;报告和提交 eval 审核让 Claude 做:/rsj-eval {group}"]
    return {"group": group, "task": task, "kind": "eval", "queue": queue, "free_gpus": free[: 2 * len(queue)],
            "need_gpus": True, "nothing": False, "text": "\n".join(lines)}


def layouts_blocked(state: dict, task: str) -> str:
    """生成布局按钮不能点的原因(看板评测布局格用);空串表示可以点。"""
    status, why = rp.entry_status(state, f"layout/{task}")
    if status != "approved":
        return "layout 要先批准(现在:" + (rp.STATUS_LABEL.get(status) or "未提交") + ")"
    return ""


def _layout_busy(group: str, task: str, exclude: str | None = None) -> None:
    """同组有正在生成布局的任务也包含 ``task`` 时不能再开(两个 write 会互相覆盖)。

    ``exclude`` 是自己这条任务记录:后台运行时会再算一次计划,不能把自己算成"正在生成"。"""
    for j in running(("layouts",), exclude=exclude):
        if j["group"] == group and task in (j.get("tasks") or [j["task"]]):
            raise PlanError(f"{task} 的评测布局正在生成(和 {j['task']} 一起,开始于 {j.get('started_at')})")


def layouts_plan(group: str, task: str) -> dict:
    """评测布局格的「生成布局」:这个任务和同组还缺布局、layout 已批准的自有任务一起生成 50 个评测布局。"""
    state = _state(group, task, "layouts")
    reason = layouts_blocked(state, task)
    if reason:
        raise PlanError(f"{task} 还不能生成评测布局:{reason}")
    n = eval_layouts(task)
    if n >= EVAL_NUM:
        return {"group": group, "task": task, "kind": "layouts", "tasks": [], "nothing": True,
                "text": f"{task} 已有 {n} 个评测布局,不用生成(要重做请让 Claude 处理,评测结果会跟着失效)"}
    jid = job_id(group, task)
    _layout_busy(group, task, exclude=jid)
    tpl = layout_template(task)
    tasks = [task] + [t for t in rp.own_tasks(state) if t != task and eval_layouts(t) < EVAL_NUM
                      and not layouts_blocked(state, t) and layout_template(t) == tpl]
    for t in tasks[1:]:
        _layout_busy(group, t, exclude=jid)
    lines = [f"生成评测布局:{EVAL_NUM} 个 / 任务({rp.rel(EVAL_LAYOUT)}/<任务>_<i>.json)", ""]
    for t in tasks:
        have = eval_layouts(t)
        lines.append(f"  ▶ {t}" + (f"(已有 {have} 个,会被替换)" if have else "") + ("  ← 你点的" if t == task else ""))
    if len(tasks) > 1:
        lines += ["", "同组还缺布局的任务一起生成,用同一批 seed:",
                  "同一个 seed 在各任务里固定件和物体位置相同,safe / risky 逐条可比。"]
    skipped = [t for t in rp.own_tasks(state) if eval_layouts(t) < EVAL_NUM and t not in tasks]
    if skipped:
        lines.append("这次不生成:" + "、".join(f"{t}({layouts_blocked(state, t) or '布局模板不同'})" for t in skipped))
    lines += ["", f"做法:在 Isaac 里检查哪些 seed 能稳定沉降(eval_layouts.py check),每个进程查 {LAYOUT_CHUNK} 个,",
              f"分到下面填的所有卡上一起跑;所有任务都稳定的 seed 凑够 {EVAL_NUM} 个就写布局(eval_layouts.py write)。",
              f"最多查到 seed {LAYOUT_MAX_SEED};以前查过的 seed 不再查。"
              "评测 seed 段是 0..N,不会和采集(100000+)重叠。" + (f"布局模板 {tpl}。" if tpl else ""),
              "一个进程约 21 GB 显存,同一张卡写两次就在这张卡上跑两个。卡越多越快;"
              f"{len(tasks)} 个任务约要查 {len(tasks) * 60} 个 seed,一个 seed 约 20 秒。",
              "生成完评测格的「▶ 评测」才会出现。"]
    busy_gpus = {g for j in running(("collect", "eval", "layouts"), exclude=jid) for g in j.get("gpus", [])}
    gpus = gpu_usage()
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy_gpus]
    if gpus:
        lines += ["", "GPU 显存占用:" + "  ".join(
            f"{g['index']}:{g['used'] // 1024}/{g['total'] // 1024}G" for g in gpus)]
    return {"group": group, "task": task, "kind": "layouts", "tasks": tasks, "template": tpl,
            "free_gpus": free, "need_gpus": True, "nothing": False, "text": "\n".join(lines)}


def _stable_file(task: str, start: int) -> Path:
    return LAYOUT_CACHE / f"{task}_{start}_{start + LAYOUT_CHUNK - 1}.json"


def checked_seeds(task: str) -> tuple[set[int], set[int]]:
    """``(查过的 seed, 其中稳定的)``:读缓存里所有 ``<task>_<起>_<止>.json``(块大小改过也能接着用)。"""
    checked: set[int] = set()
    stable: set[int] = set()
    pattern = re.compile(re.escape(task) + r"_(\d+)_(\d+)\.json")
    for path in LAYOUT_CACHE.glob(f"{task}_*_*.json") if LAYOUT_CACHE.is_dir() else []:
        m = pattern.fullmatch(path.name)
        if not m:
            continue
        checked |= set(range(int(m[1]), int(m[2]) + 1))
        stable |= set(rp.read_json(path).get("stable") or [])
    return checked, stable


def common_stable(tasks: list[str], extra: dict[str, set[int]] | None = None) -> set[int]:
    """所有任务都稳定的 seed;``extra`` 是正在查的 seed(估算"查完最多能有多少"时当作稳定)。"""
    common = None
    for t in tasks:
        stable = checked_seeds(t)[1] | (extra or {}).get(t, set())
        common = stable if common is None else common & stable
    return common or set()


def layouts_progress(group: str, job: dict) -> dict | None:
    """生成布局的进度:每个任务查了几个 seed、稳定了几个,所有任务都稳定的凑到几个;写完后是布局个数。"""
    tasks = job.get("tasks") or [job.get("task")]
    if all(eval_layouts(t) >= EVAL_NUM for t in tasks):
        return {"text": "、".join(f"{t} {eval_layouts(t)} 个" for t in tasks), "done": 1, "total": 1}
    bits = []
    for t in tasks:
        checked, stable = checked_seeds(t)
        bits.append(f"{t} 查了 {len(checked)} 个 seed,稳定 {len(stable)}")
    have = min(len(common_stable(tasks)), EVAL_NUM)
    return {"text": " · ".join(bits) + f" · 共同稳定 {have}/{EVAL_NUM}", "done": have, "total": EVAL_NUM}


def _check_chunk(task: str, start: int, gpu: int, template: str | None) -> subprocess.Popen:
    LAYOUT_CACHE.mkdir(parents=True, exist_ok=True)
    cmd = ["timeout", "--kill-after=30", str(LAYOUT_CHUNK_TIMEOUT), sys.executable, "-u",
           "src/data_collect/eval_layouts.py", "check", "--task_name", task, "--start_seed", str(start),
           "--num", str(LAYOUT_CHUNK), "--out", str(_stable_file(task, start)), "--enable_cameras", "--headless"]
    if template:
        cmd += ["--layout_template", template]
    print(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}", flush=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMNI_KIT_ACCEPT_EULA": "YES",
           "PYTHONWARNINGS": "ignore::UserWarning",  # 和 collect_data.sh 一样:仓库根目录要在 PYTHONPATH(import env.*)
           "PYTHONPATH": f"{ROOT}:{ROOT / 'XPolicyLab'}:{os.environ.get('PYTHONPATH', '')}"}
    log = open(LAYOUT_CACHE / f"{task}_{start}.log", "w", encoding="utf-8")  # noqa: SIM115 - 子进程结束后随进程回收
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)


def run_layouts(group: str, task: str, gpus: str) -> int:
    """把 (任务, 10 个 seed) 一块一块分给空着的卡(``gpus`` 里同一张卡写两次就跑两个),从小 seed 往大查;
    已查过的 seed 跳过。正在查的都算稳定也凑不够 50 个时才继续派新块,凑够就等手上的查完再写布局。"""
    p = layouts_plan(group, task)  # 再确认一次闸门
    tasks, tpl = p["tasks"], p["template"]
    slots = [int(x) for x in gpus.split(",") if x.strip()] or [0]
    busy: dict[int, tuple[str, int, subprocess.Popen]] = {}  # 槽位 -> (任务, 起始 seed, 进程)

    def todo():
        for start in range(0, LAYOUT_MAX_SEED, LAYOUT_CHUNK):
            for t in tasks:
                chunk = set(range(start, start + LAYOUT_CHUNK))
                if chunk <= checked_seeds(t)[0] or any(b[:2] == (t, start) for b in busy.values()):
                    continue
                yield t, start

    def in_flight() -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        for t, start, _ in busy.values():
            out.setdefault(t, set()).update(range(start, start + LAYOUT_CHUNK))
        return out

    while True:
        # 正在查的都当作稳定还凑不够,就把空槽位填上
        for slot in [i for i in range(len(slots)) if i not in busy]:
            if len(common_stable(tasks, in_flight())) >= EVAL_NUM:
                break
            unit = next(todo(), None)
            if unit is None:
                break
            busy[slot] = (*unit, _check_chunk(unit[0], unit[1], slots[slot], tpl))
            time.sleep(5)  # 错开 Isaac 启动
        if not busy:
            have = len(common_stable(tasks))
            if have >= EVAL_NUM:
                break
            _step(group, task, f"查到 seed {LAYOUT_MAX_SEED} 只有 {have} 个所有任务都稳定的 seed,不够 {EVAL_NUM} 个"
                               "(场景可能放不下,让 Claude 看一下)")
            return 1
        running_text = "、".join(f"{t} {s}–{s + LAYOUT_CHUNK - 1}(GPU {slots[i]})" for i, (t, s, _) in busy.items())
        update_job(group, task, step=f"检查中:{running_text}")
        time.sleep(5)
        for slot, (t, start, proc) in list(busy.items()):
            code = proc.poll()
            if code is None:
                continue
            del busy[slot]
            if not _stable_file(t, start).exists():
                # 卡住被杀或崩溃:这块 seed 记成都不稳定,接着查别的
                print(f"[group_jobs] {t} seed {start}–{start + LAYOUT_CHUNK - 1} 没检查完(exit {code}),这块不用",
                      flush=True)
                _stable_file(t, start).write_text(json.dumps({"task": t, "stable": [], "failed": code}))
            else:
                n = len(rp.read_json(_stable_file(t, start)).get("stable") or [])
                print(f"[group_jobs] {now()} {t} seed {start}–{start + LAYOUT_CHUNK - 1}:稳定 {n}/{LAYOUT_CHUNK}",
                      flush=True)
    _step(group, task, f"写评测布局({len(tasks)} 个任务 × {EVAL_NUM})")
    # write 取所有文件的交集:同一任务的多批要先合并成一个文件,否则批与批之间也会取交集
    merged = []
    for t in tasks:
        seeds = sorted(checked_seeds(t)[1])
        path = LAYOUT_CACHE / f"{t}_stable.json"
        path.write_text(json.dumps({"task": t, "stable": seeds}))
        merged.append(str(path))
    os.environ["PYTHONPATH"] = f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"  # write 也 import src.* / utils.*
    cmd = [sys.executable, "src/data_collect/eval_layouts.py", "write", "--tasks", ",".join(tasks),
           "--stable", ",".join(merged), "--eval_num", str(EVAL_NUM), "--eval_seed", str(EVAL_SEED)]
    if tpl:
        cmd += ["--layout_template", tpl]
    if any(eval_layouts(t) for t in tasks):
        cmd.append("--overwrite")  # 已有的不到 50 个,计划里写了会替换
    if _sh(cmd) != 0 or any(eval_layouts(t) < EVAL_NUM for t in tasks):
        _step(group, task, "写布局失败")
        return 1
    _step(group, task, "生成完:" + "、".join(f"{t} {eval_layouts(t)} 个" for t in tasks))
    return 0


def plan(button: str, group: str, task: str, dry_run: bool = True) -> dict:
    if button == "ingest":
        return ingest_plan(group, task)
    if button == "eval":
        return eval_plan(group, task)
    if button == "layouts":
        return layouts_plan(group, task)
    if button == "collect":
        return collect_plan(group, task)
    if button == "exchange":
        return export_plan(group, task)
    return upload_plan(group, task, dry_run=dry_run)


# ---------------------------------------------------------------------------
# 启动 / 停止
# ---------------------------------------------------------------------------


def start(button: str, group: str, task: str, gpus: str = "", delete: bool = False) -> dict:
    """重新算一遍计划,通过后在新会话里后台运行 ``run``;返回任务记录。"""
    _busy(group, task)
    p = plan(button, group, task, dry_run=False)
    if p["nothing"]:
        raise PlanError("没有要做的事")
    kind = p["kind"]
    args = [kind, group, task]
    gpu_list: list[int] = []
    if kind in ("collect", "eval", "layouts") and p["need_gpus"]:
        try:
            gpu_list = [int(x) for x in re.split(r"[,\s]+", gpus.strip()) if x]
        except ValueError:
            raise PlanError(f"GPU 列表写错了:{gpus!r}(例:0,1,2)") from None
        if not gpu_list:
            raise PlanError("要填用哪几张 GPU(例:0,1)")
        args += ["--gpus", ",".join(map(str, gpu_list))]
    if kind == "upload" and delete:
        args.append("--delete")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"task_{task}_{kind}_{datetime.now():%Y%m%d_%H%M%S}.log"
    job = {"group": group, "task": task, "kind": kind, "pid": None, "log": str(log), "started_at": now(),
           "tasks": p.get("tasks") or [task],
           "finished_at": None, "returncode": None, "step": "启动中", "gpus": gpu_list, "delete": bool(delete),
           "args": args}
    write_job(job)
    with open(log, "w", encoding="utf-8") as fp:
        fp.write(f"[group_jobs] {now()} {job_label(job)} {group}/{task}\n{p['text']}\n\n")
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "run", *args], cwd=str(ROOT),
                                stdout=fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    update_job(group, task, pid=proc.pid)
    return read_job(job_id(group, task))


def stop(group: str, task: str) -> str:
    job = read_job(job_id(group, task))
    if not job or job_state(job) != "running":
        raise PlanError(f"{task} 没有在运行的任务")
    os.killpg(job["pid"], signal.SIGTERM)  # 新会话:pgid == pid,连同采集 worker 一起停
    update_job(group, task, stopped=True, step="已停止", returncode=-15, finished_at=now())
    return f"已停止 {task} 的{job_label(job)};再点一次按钮会续上"


# ---------------------------------------------------------------------------
# 运行(在后台会话里)
# ---------------------------------------------------------------------------


def _step(group: str, task: str, text: str) -> None:
    print(f"\n[group_jobs] {now()} {text}", flush=True)
    update_job(group, task, step=text)


def _sh(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def run_collect(group: str, task: str, gpus: str) -> int:
    t = collect_plan(group, task)["plan"]
    if not t["episode_num"]:
        return 0
    _step(group, task, f"采集({t['have']} → {rp.COLLECT_TARGET})")
    cmd = ["bash", "scripts/collect_data_parallel.sh", task, str(t["episode_num"]), gpus]
    if t["start_seed"]:
        cmd += ["--start_seed", str(t["start_seed"])]
    if t["template"]:
        cmd += ["--layout_template", t["template"]]
    code = _sh(cmd)
    n = rp._hdf5_count(task)
    if code != 0 or _collected(task):
        _step(group, task, f"没采完:{n}/{rp.COLLECT_TARGET} 条(exit {code});分片保留,再点一次续采")
        return 1
    _step(group, task, f"采够 {n} 条")
    return 0


def run_ingest(group: str) -> int:
    ingest_plan(group, GROUP_TASK)  # 再确认一次闸门
    _step(group, GROUP_TASK, "下载 / 转换 / 入库")
    if _sh(_ingest_cmd(group, "--report", ingest_report(group))) != 0:
        _step(group, GROUP_TASK, "有资产没入库(看日志里的 ❌),没有提交")
        return 1
    _step(group, GROUP_TASK, "提交 ingest")
    if _sh([sys.executable, str(HERE / "rsj_pipeline.py"), "submit", group, "ingest", "--report", ingest_report(group),
            "--note", "看板「下载资产」"]) != 0:
        _step(group, GROUP_TASK, "检查没过,没有提交(看日志)")
        return 1
    _step(group, GROUP_TASK, "已提交,等你审核")
    return 0


def run_export(group: str, task: str) -> int:
    if not export_plan(group, task)["reason"]:
        return 0
    _step(group, task, "转换 LeRobot v3.0 + v2.1")
    if _sh(["bash", "scripts/export_lerobot.sh", task, "--format", "both"]) != 0 or export_needed(task):
        _step(group, task, "转换失败")
        return 1
    _step(group, task, "转换完成")
    return 0


def run_upload(group: str, task: str, delete: bool) -> int:
    upload_plan(group, task, dry_run=False)  # 再确认一次闸门
    extra = [] if delete else ["--no_delete"]
    for label, script, repo in UPLOADERS:
        _step(group, task, f"上传 {label}")
        if _sh(_upload_cmd(script, repo, task, *extra)) != 0:
            return 1
    _step(group, task, "核对远端")
    return _sh([sys.executable, str(HERE / "check_task_alignment.py"), "--only", "modelscope", "--task", task,
                "--modelscope", HDF5_REPO])


def run_eval(group: str, task: str, gpus: str) -> int:
    queue = eval_plan(group, task)["queue"]  # 再确认一次闸门,重新算续评
    ids = [int(x) for x in gpus.split(",") if x.strip()]
    pairs = [(ids[i], ids[i + 1] if i + 1 < len(ids) else ids[i]) for i in range(0, len(ids), 2)] or [(0, 0)]
    EVAL_LOGS.mkdir(exist_ok=True)
    todo, procs, results = list(queue), [], []
    _step(group, task, f"评测 {len(queue)} 个模型({len(pairs)} 条并行)")
    while todo or procs:
        used = {pair for pair, *_ in procs}
        for pair in [p for p in pairs if p not in used]:
            if not todo:
                break
            m = todo.pop(0)
            log = EVAL_LOGS / f"{task}_{m['policy']}_p{pair[0]}_{datetime.now():%Y%m%d_%H%M%S}.log"
            cmd = ["bash", "scripts/robodojo.sh", "eval", "--policy-dir", f"XPolicyLab/policy/{m['policy']}",
                   "--task", task, "--ckpt", m["ckpt"], "--policy-env", "uv", "--action-type", "joint",
                   "--env-cfg", EVAL_CFG, "--eval-num", str(EVAL_NUM), "--seed", str(EVAL_SEED),
                   "--policy-gpu", str(pair[0]), "--env-gpu", str(pair[1])]
            env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
            if m["run_id"]:
                env["ROBODOJO_RUN_ID"] = m["run_id"]
            print(f"$ {'ROBODOJO_RUN_ID=' + m['run_id'] + ' ' if m['run_id'] else ''}{' '.join(cmd)}\n  日志 {log}",
                  flush=True)
            fp = open(log, "w", encoding="utf-8")  # noqa: SIM115 - 进程结束后关
            procs.append((pair, m, subprocess.Popen(cmd, cwd=str(ROOT), stdout=fp, stderr=subprocess.STDOUT,
                                                    env=env, stdin=subprocess.DEVNULL), fp))
            _step(group, task, "评测中:" + "、".join(f"{x[1]['policy']}(GPU {x[0][0]}/{x[0][1]})" for x in procs))
            time.sleep(8)  # 错开启动,policy server 抢端口 / Isaac 同时初始化容易出问题
        for item in list(procs):
            pair, m, proc, fp = item
            if proc.poll() is None:
                continue
            fp.close()
            procs.remove(item)
            runs = eval_runs(task, m["policy"], m["ckpt"])
            n = max((r[1] for r in runs), default=0)
            results.append((m, proc.returncode, n))
            print(f"[group_jobs] {now()} {m['policy']} 结束 exit {proc.returncode},评了 {n}/{EVAL_NUM} 条", flush=True)
        time.sleep(5)
    short = [f"{m['policy']} {n}/{EVAL_NUM}" for m, _, n in results if n < EVAL_NUM]
    done = "、".join(f"{m['policy']} {n} 条" for m, _, n in results)
    if short:
        _step(group, task, f"没评满:{'、'.join(short)}(再点一次续评)")
        return 1
    _step(group, task, f"评完:{done};报告和提交审核让 Claude 做(/rsj-eval {group})")
    return 0


def run(kind: str, group: str, task: str, gpus: str = "", delete: bool = False) -> int:
    update_job(group, task, pid=os.getpid())
    code = 1
    try:
        if kind == "ingest":
            code = run_ingest(group)
        elif kind == "collect":
            code = run_collect(group, task, gpus)
        elif kind == "export":
            code = run_export(group, task)
        elif kind == "eval":
            code = run_eval(group, task, gpus)
        elif kind == "layouts":
            code = run_layouts(group, task, gpus)
        else:
            code = run_upload(group, task, delete)
    except PlanError as error:
        print(f"[group_jobs] {error}", flush=True)
    finally:
        job = read_job(job_id(group, task)) or {}
        if not job.get("stopped"):
            update_job(group, task, returncode=code, finished_at=now(),
                       step=(job.get("step") if kind in ("ingest", "eval", "layouts") else "完成") if code == 0
                       else (job.get("step") or "") + "(失败)")
        print(f"\n[group_jobs] {now()} exit {code}", flush=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "start"):
        s = sub.add_parser(name)
        s.add_argument("button", choices=BUTTONS)
        s.add_argument("group")
        s.add_argument("task")
        if name == "start":
            s.add_argument("--gpus", default="")
            s.add_argument("--delete", action="store_true", help="上传时删除远端多余文件(默认保留)")
    s = sub.add_parser("run", help="内部用:后台会话里执行")
    s.add_argument("kind", choices=("ingest", "collect", "export", "upload", "eval", "layouts"))
    s.add_argument("group")
    s.add_argument("task")
    s.add_argument("--gpus", default="")
    s.add_argument("--delete", action="store_true")
    sub.add_parser("status")
    s = sub.add_parser("stop")
    s.add_argument("group")
    s.add_argument("task")
    a = ap.parse_args()
    try:
        if a.cmd == "plan":
            print(plan(a.button, a.group, a.task)["text"])
        elif a.cmd == "start":
            job = start(a.button, a.group, a.task, a.gpus, a.delete)
            print(f"已启动 {job_label(job)} {a.group}/{a.task}:pid {job['pid']},日志 {job['log']}")
        elif a.cmd == "run":
            return run(a.kind, a.group, a.task, a.gpus, a.delete)
        elif a.cmd == "status":
            for jid, j in all_jobs(tail=False).items():
                print(f"{jid}: {j['label']} {j['state']} · {j.get('step')} · {j.get('started_at')} · {j.get('log')}")
        else:
            print(stop(a.group, a.task))
    except PlanError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
