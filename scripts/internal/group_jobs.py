#!/usr/bin/env python3
"""看板大表里的后台任务:ingest 格的「下载资产」(按组),collect / exchange / upload 格的「采集」「转换」「上传」(按任务)。

任务在独立会话里后台运行,看板重启不会中断。状态在 ``<RSJ_PIPELINE_DIR>/_jobs/<group>.<task>.json``(按模型的评测是
``<group>.<task>.<policy>.json``),日志在 ``data/.cache/jobs/task_<task>_<kind>_<时间>.log``(``RSJ_PIPELINE_DIR`` 的
上一级)。每个任务同一时间只跑一个;按模型的评测各有一条记录,同一任务的不同模型可以同时评(也可以和采集同时跑),
同一个模型、或"评所有模型"和单模型之间仍互斥。
转换同一时间只有一个(共用扁平视图),上传同一时间只有一个(共用 ModelScope 缓存和硬链接视图);
转换 A 和上传 B 可以同时跑。不同任务的采集可以同时跑在不同的卡上。

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
             各评 50 条(``robodojo.sh eval``,seed 0,arx_x5,动作空间取 deploy.yml);risky 没有自己的模型时
             用同组 safe 模型 zero-shot。
             没评满的接着同一个 ``ROBODOJO_RUN_ID`` 续评,评满的跳过。每条流水线用一对 GPU(policy + sim),
             填几对就并行几条。评测不审核(2026-11-15 起),评完不提交
    layouts  「生成布局」(评测布局格):layout 批准后,在 Isaac 里分批(每批 20 个 seed,一个进程)检查哪些 seed
             能稳定沉降(``eval_layouts.py check``),取前 50 个写评测布局(``eval_layouts.py write``)。同组还缺布局的
             自有任务一起生成,共用同一批 seed(safe / risky 逐条可比);已有一部分(不到 50 个)的会被替换

采集和评测先选机器(``docs/servers.json``,``servers.py``):本机直接跑;远端机器走 SSH——同步代码
(``servers.CODE_DIRS`` + ``data/grasp_pose``,评测再加该任务的评测布局和要续评的结果,都不加 ``--delete``)→ 远端
``setsid nohup`` 运行(目录 ``data/.rsj_remote/<task>_<kind>_<时间>``,日志 ``<目录>.log``,退出码 ``<目录>/exit_code``)→
本地后台进程每分钟看一次远端进度、把远端日志接到本地日志 → 跑完把结果拉回。采集边采边拉:远端以 ``--keep_shards``
启动,本地每轮把新完成的分片增量 rsync 到 ``data/.cache/remote_pull/<run>/shards/``,传输和采集重叠;采完补齐分片、
按远端合并的 ``seed.txt`` 核对,再用 ``merge_shards.py`` 接到本地数据之后(编号接续、seed 去重),分片对不上时退回
整包拉 ``<rdir>/<task>``。评测拉回 ``eval_result/RoboDojo/<task>/`` 和 ``eval_logs/``。
远端没跑完时再点一次、选同一台机器就续上(远端还在跑就接着看,跑完了就拉回);采集没拉回前不能换机器(seed 段会撞)。
远端的数据不删,模型不自动同步(缺的在计划里列出来)。评测前会把布局 / 任务 YAML 用到、远端缺的资产(含它们
引用到的原件)自动同步过去,并把写死的本机绝对路径改成远端仓库路径;采集仍然只在计划里列出来。

采集 / 转换 / 上传(2026-11-13 起)和评测(2026-11-15 起)不审核,做完不提交;只要求该任务的 expert 已批准。
命令行(看板调用的就是这些函数)::

    python scripts/internal/group_jobs.py plan ingest <group> __group__
    python scripts/internal/group_jobs.py plan collect|exchange|upload <group> <task>
    python scripts/internal/group_jobs.py start collect <group> <task> --gpus 0,1 [--machine autodl-A18] [--per_gpu 2]
    python scripts/internal/group_jobs.py start exchange <group> <task>
    python scripts/internal/group_jobs.py start upload <group> <task> [--delete]
    python scripts/internal/group_jobs.py start eval <group> <task> --gpus 2,3,5,7 [--machine autodl-A18]
    python scripts/internal/group_jobs.py start layouts <group> <task> --gpus 2
    python scripts/internal/group_jobs.py status
    python scripts/internal/group_jobs.py stop <group> <task> [--policy Pi_0]
"""

from __future__ import annotations

import argparse
import fcntl
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsj_pipeline as rp  # noqa: E402
import servers as sv  # noqa: E402

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
TEMPLATE_RE = re.compile(r"--layout_template\s+([^\s,;()。，；（）]+)")
KIND_LABEL = {"ingest": "下载资产", "collect": "采集", "export": "转换 LeRobot", "upload": "上传 ModelScope",
              "eval": "评测", "layouts": "生成布局", "claude": "Claude"}
DATA_BUTTONS = ("collect", "exchange", "upload")  # 不审核,expert 批准后直接做
BUTTONS = ("ingest", *DATA_BUTTONS, "eval", "layouts")
# 评测(docs/model_evaluation.md 的固定协议)
EVAL_POLICIES = ("Pi_0", "Pi_05")  # 只决定排序(这两个在前);要评哪些模型看 eval_policies():ModelScope 上有的都评
SNAPSHOT = ROOT / "data" / "SYNC_STATUS.json"  # sync_status 写的快照,models = ModelScope 模型仓库里的模型
# 评测时策略服务要的额外环境变量:本地从环境 / data/.cache/policy_env.json 取,远端从 docs/servers.json 的
# 机器项 "policy_env" 取
POLICY_ENV_VARS = {"LingBot_VLA": {"QWEN25_PATH": "Qwen2.5-VL-3B-Instruct 权重目录"}}
POLICY_ENV_FILE = ROOT / "data" / ".cache" / "policy_env.json"
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
MACHINE_KINDS = ("collect", "eval")  # 这两种先选机器(docs/servers.json)
REMOTE_DIR = "data/.rsj_remote"  # 远端仓库下每次运行的目录
PULL_DIR = ROOT / "data" / ".cache" / "remote_pull"  # 远端采集拉回来先放这里,再 merge_shards 合并
REMOTE_POLL = 60  # 秒
ASSET_ROOT = "Assets/Object/RoboDojo"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def job_id(group: str, task: str, policy: str = "", variant: str = "") -> str:
    """任务记录的键:``<group>.<task>``;按模型的评测是 ``<group>.<task>.<policy>``(同一任务的不同模型能同时评),
    S1 变体的评测再加 ``@<variant>``(``<group>.<task>.<policy>@S1.1``;S1.0 不加,和以前一样)。"""
    sub = policy + (f"@{variant}" if variant else "")
    return f"{group}.{task}.{sub}" if sub else f"{group}.{task}"


def job_key(job: dict) -> str:
    """记录自己的键:只有 eval 带 policy / variant(别的 kind 的总是空)。"""
    if job.get("kind") != "eval":
        return job_id(job["group"], job["task"])
    return job_id(job["group"], job["task"], job.get("policy") or "", job.get("variant") or "")


def norm_variant(variant: str) -> str:
    """评测变体:``""`` / ``S1.0`` = 原任务(返回空串),否则 ``S1.1`` … ``S1.5``。"""
    variant = (variant or "").strip()
    if variant in ("", "S1.0"):
        return ""
    if not re.fullmatch(r"S1\.[1-5]", variant):
        raise PlanError(f"变体写错了:{variant!r}(S1.0 … S1.5)")
    return variant


_RUN_POLICY = ""  # run 进程里:按模型评测时是模型名,update_job / _step 按它找自己的记录(别的进程总是空)
_RUN_VARIANT = ""  # run 进程里:S1 变体评测时是变体名


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


def write_job(job: dict, jid: str | None = None) -> None:
    """写到 ``jid``(默认按记录内容算的键;改旧记录时传读出来的键,别写到新位置)。"""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = job_path(jid or job_key(job))
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_job(group: str, task: str, **fields) -> None:
    job = read_job(job_id(group, task, _RUN_POLICY, _RUN_VARIANT)) or {"group": group, "task": task}
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
    try:  # Linux:直接读命令行,不用每个任务起一次 ps(看板每 5 秒轮询)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").replace("\0", " ")
        return "group_jobs.py" in args and " run " in args
    except OSError:
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
    label = KIND_LABEL.get(job.get("kind"), job.get("kind"))
    if job.get("machine") and job["machine"] != sv.default_key():
        label = f"{label}@{job['machine']}"
    who = " ".join(x for x in (job.get("policy"), job.get("variant")) if x)
    return f"{who} {label}" if who else label


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


_DONE_PROGRESS: dict[tuple[str, int], dict | None] = {}  # 已结束任务的进度(按记录文件 mtime;记录一改就重算)


def all_jobs(tail: bool = True) -> dict[str, dict]:
    """``{<group>.<task>[.<policy>]: 任务记录}``(旧的按组记录没有 task,跳过)。

    ``tail`` 时算进度:已结束的任务进度不会再变,同一份记录只算一次(看板每 5 秒轮询一次)。"""
    out = {}
    if not JOB_DIR.exists():
        return out
    with rp.memo_scope():
        for path in sorted(JOB_DIR.glob("*.json")):
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            job = read_job(path.stem)
            if not job or not job.get("task"):
                continue
            job["state"] = job_state(job)
            job["label"] = job_label(job)
            if tail:
                key = (path.stem, mtime)
                if job["state"] == "running":
                    job["progress"] = job_progress(job)
                elif key in _DONE_PROGRESS:
                    job["progress"] = _DONE_PROGRESS[key]
                else:
                    job["progress"] = _DONE_PROGRESS[key] = job_progress(job)
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


_EXPORT_CACHE: dict[str, dict] = {}
_RE_EXPORT_PHASE = re.compile(r"^\[export_lerobot\] (v30|v21) ")
_RE_EXPORT_BAR = re.compile(r"(?:^Converting |^\[v30_to_v21\] videos).*?:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def export_progress(log: str | None) -> dict | None:
    """LeRobot 转换日志的进度:v3.0 按 episode(转换器的 tqdm),v2.1 按视频(切片重编码的 tqdm)。

    两段各自一条进度条(和上传一样,换阶段从 0 开始),文字前面写已完成的阶段。日志里夹着大量
    SVT/ffmpeg 输出,只读新增的部分。"""
    if not log:
        return None
    c = _EXPORT_CACHE.setdefault(log, {"pos": 0, "rest": "", "phase": "", "done": 0, "total": 0, "finished": []})
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
        line = re.sub(r"^(?:\x1b\[A)+", "", line)
        if m := _RE_EXPORT_PHASE.match(line):
            if c["phase"] and c["phase"] not in c["finished"]:
                c["finished"].append(c["phase"])
            c.update(phase=m[1], done=0, total=0)
        elif (m := _RE_EXPORT_BAR.search(line)) and c["phase"]:
            c["done"], c["total"] = int(m[1]), int(m[2])
        elif line.startswith("[export_lerobot] failed"):
            c["failed"] = True
    if not c["phase"] or (c.get("failed") and not c["total"]):  # 没开始就失败:让面板写失败原因
        return None
    name = {"v30": "v3.0", "v21": "v2.1"}
    prefix = "".join(f"{name[x]} ✓ · " for x in c["finished"])
    unit = "条" if c["phase"] == "v30" else "个视频"
    if not c["total"]:
        return {"text": prefix + f"转 {name[c['phase']]} 准备中", "done": 0, "total": 0}
    return {"text": prefix + f"转 {name[c['phase']]} {c['done']}/{c['total']} {unit}",
            "done": c["done"], "total": c["total"]}


def collect_progress(task: str) -> dict:
    n = rp._hdf5_count(task)
    shard = sum(1 for _ in (SHARDS / task).glob("w*/**/*.hdf5")) if (SHARDS / task).exists() else 0
    have = min(n + shard, rp.COLLECT_TARGET)
    return {"text": f"已采 {have}/{rp.COLLECT_TARGET} 条", "done": have, "total": rp.COLLECT_TARGET}


def eval_progress(group: str, task: str, policy: str = "", variant: str = "") -> dict | None:
    """每个要评的模型评了几条(取同一 ckpt 下条数最多的那次,和续评的口径一样);``policy`` 给出时只看这个模型。"""
    try:
        models = [m for m in eval_models(group, task, set()) if m["step"] and (not policy or m["policy"] == policy)]
    except (OSError, ValueError, KeyError):
        return None
    if not models:
        return None
    target = eval_target(task, variant)
    counts = [(m["policy"], min(target, max((r[1] for r in eval_runs(task, m["policy"], m["ckpt"], variant)),
                                            default=0)))
              for m in models]
    return {"text": " · ".join(f"{p} {n}/{target}" for p, n in counts),
            "done": sum(n for _, n in counts), "total": target * len(counts)}


def job_progress(job: dict) -> dict | None:
    """任务面板 / 格子里的进度 ``{text, done, total}``(total 为 0 时只有文字);算不出来时 None(只显示步骤)。"""
    try:
        kind, task = job.get("kind"), job.get("task", "")
        r = job.get("remote")
        if kind == "collect" and r and not r.get("merged"):
            have = min(rp._hdf5_count(task) + int(r.get("count") or 0), rp.COLLECT_TARGET)
            return {"text": f"{r['machine']} 已采 {r.get('count') or 0}/{r['episode_num']} · 合计 {have}/{rp.COLLECT_TARGET} 条",
                    "done": have, "total": rp.COLLECT_TARGET}
        if kind == "eval" and r and r.get("counts") and job_state(job) == "running":
            counts = r["counts"]
            target = eval_target(task, job.get("variant") or "")
            return {"text": f"{r['machine']} · " + " · ".join(f"{p} {n}/{target}" for p, n in counts.items()),
                    "done": sum(counts.values()), "total": target * len(counts)}
        if kind == "eval":
            return eval_progress(job["group"], task, job.get("policy") or "", job.get("variant") or "")
        if kind == "collect":
            return collect_progress(task)
        if kind == "upload":
            return upload_progress(job.get("log"))
        if kind == "export":
            return export_progress(job.get("log"))
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


def machine_cfg(machine: str = "") -> dict:
    """``docs/servers.json`` 里的一台机器(空 = default_machine);配置不全时 PlanError。"""
    try:
        m = sv.get(machine or sv.default_key())
    except sv.ServerError as error:
        raise PlanError(str(error)) from None
    why = sv.problems(m)
    if why:
        raise PlanError(f"{m['key']} 还不能用:{'; '.join(why)}")
    return m


def job_machine(job: dict) -> str:
    """任务跑在哪台机器上(以前的记录没有 machine,都是本机)。"""
    return job.get("machine") or sv.default_key()


def busy_gpus(kinds, jid: str, machine: str) -> set[int]:
    """同一台机器上别的任务占着的卡。"""
    return {g for j in running(kinds, exclude=jid) if job_machine(j) == machine for g in j.get("gpus", [])}


def machine_gpus(m: dict) -> list[dict]:
    if sv.is_local(m):
        return gpu_usage()
    try:
        return sv.gpu_usage(m)
    except sv.ServerError as error:
        raise PlanError(str(error)) from None


def gpu_lines(gpus: list[dict], busy: set[int], m: dict) -> list[str]:
    def gpu_text(g: dict) -> str:
        mark = "(别的任务在用)" if g["index"] in busy else ""
        return f"{g['index']}:{g['used'] // 1024}/{g['total'] // 1024}G{mark}"

    return ["", f"{m['key']} GPU 显存占用:" + "  ".join(gpu_text(g) for g in gpus)] if gpus else []


def _machine_fields(m: dict) -> dict:
    """计划里带给页面的机器信息(选择框 + 当前选的)。"""
    try:
        options = sv.machines()
    except sv.ServerError:
        options = []
    return {"machine": m["key"], "machine_local": sv.is_local(m), "machines": options}


def task_asset_dirs(task: str) -> list[str]:
    """任务 YAML 里用到的物体资产目录 ``Assets/Object/RoboDojo/<Section>/<name>/<index>``(本地存在的)。"""
    import yaml

    path = rp.MIRROR / "config" / f"{task}.yml"
    if not path.exists():
        path = ROOT / "task" / "RoboDojo" / "config" / f"{task}.yml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = []
    for section, items in cfg.items():
        for item in items if isinstance(items, list) else []:
            for cat in (item.get("category") or []) if isinstance(item, dict) else []:
                for idx in cat.get("index") or []:
                    rel = f"{ASSET_ROOT}/{section}/{cat.get('name')}/{int(idx):05d}"
                    if (ROOT / rel).exists():
                        out.append(rel)
    return sorted(set(out))


LAYOUT_ASSET_SECTIONS = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
USD_REF_RE = re.compile(r"@([^@\n]{1,500}?\.usd[a]?)@")
ABS_ASSET_RE = re.compile(
    r"(/[A-Za-z0-9_./+@\-]*(?:data/RoboSafeJanus-Asset|ext_assets|Assets)/[A-Za-z0-9_./+@\-]+)")
ASSET_SCAN_MAX_BYTES = 8 * 1024 * 1024


@rp.memoized
def _layout_index() -> dict[str, list[str]]:
    """评测布局目录 ``{<名字>: [<名字>_<i>.json, ...]}``(看板渲染时整张表共用一次列目录)。"""
    index: dict[str, list[str]] = {}
    try:
        names = sorted(os.listdir(EVAL_LAYOUT))
    except OSError:
        return index
    for n in names:
        match = re.fullmatch(r"(.+)_\d+\.json", n)
        if match:
            index.setdefault(match.group(1), []).append(n)
    return index


def eval_layout_files(task: str, variant: str = "") -> list[Path]:
    """评测布局文件 ``<task>_<i>.json``;变体是 ``<task>__<variant>_<i>.json``(src/eval_client/variants.py)。"""
    name = f"{task}__{variant}" if variant else task
    return [EVAL_LAYOUT / n for n in _layout_index().get(name, [])]


def layout_asset_dirs(task: str, variant: str = "") -> list[str]:
    """评测布局里用到的物体资产目录(本地存在的;和 ``load_saved_layout`` 的路径规则一致)。"""
    out = set()
    for path in eval_layout_files(task, variant):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for section in LAYOUT_ASSET_SECTIONS:
            cats = cfg.get(section)
            if not isinstance(cats, dict):
                continue
            for cat, insts in cats.items():
                for inst in insts or []:
                    if not isinstance(inst, dict) or inst.get("category_idx") is None:
                        continue
                    root = "Clutter" if inst.get("type") == "cluttered" else section
                    rel = f"{ASSET_ROOT}/{root}/{cat}/{int(inst['category_idx']):05d}"
                    if (ROOT / rel).exists():
                        out.add(rel)
    return sorted(out)


def _local_to_rel(path: Path) -> str | None:
    """本地绝对路径 → 仓库相对路径(``Assets`` / ``ext_assets`` 的软链接解析到 ``data/RoboSafeJanus-Asset``)。"""
    try:
        resolved = path.resolve()
    except OSError:
        return None
    for name in ("Assets", "ext_assets"):
        try:
            return f"{name}/{resolved.relative_to((ROOT / name).resolve())}"
        except ValueError:
            pass
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return None


def _normalize_asset_rel(rel: str) -> str:
    """``data/RoboSafeJanus-Asset/{Assets,ext_assets}/...`` → ``{Assets,ext_assets}/...``(远端两者等价,去重)。"""
    prefix = "data/RoboSafeJanus-Asset/"
    if rel.startswith(prefix):
        rest = rel[len(prefix):]
        if rest.startswith(("Assets/", "ext_assets/")):
            return rest
    return rel


def _ref_to_rel(ref: str, base_dir: Path) -> str | None:
    """资产文件里的一个引用(绝对 / 相对)→ 仓库相对路径;映射不了或不在本仓库时返回 None。"""
    ref = ref.strip().strip("@")
    if not ref:
        return None
    if os.path.isabs(ref):
        prefix = str(ROOT.resolve()).rstrip("/") + "/"
        if ref.startswith(prefix):
            return _normalize_asset_rel(ref[len(prefix):])
        # 换机器构建的资产 / 旧文档里的前缀:按资产根标记剥离
        for marker in ("/data/RoboSafeJanus-Asset/", "/ext_assets/", "/Assets/"):
            i = ref.find(marker)
            if i >= 0:
                return _normalize_asset_rel(ref[i + 1:])
        return None
    try:
        rel = _local_to_rel((base_dir / ref).resolve())
    except OSError:
        return None
    return _normalize_asset_rel(rel) if rel else None


def _asset_deps(rel: str) -> list[str]:
    """资产目录里引用到的、不在本目录内的其它资产目录(SimReady 包装 usd、metadata、curobo.yml)。"""
    base = ROOT / rel
    try:
        own = base.resolve()
    except OSError:
        return []
    out = set()
    for f in base.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in (".usd", ".usda", ".json", ".yml", ".yaml"):
            continue
        try:
            if f.stat().st_size > ASSET_SCAN_MAX_BYTES:
                continue  # 二进制 crate 原件:纹理等同目录一起传,不解析
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        refs = [m.group(1) for m in USD_REF_RE.finditer(text)]
        refs += [m.group(0) for m in ABS_ASSET_RE.finditer(text)]
        for ref in refs:
            dep = _ref_to_rel(ref, f.parent)
            if not dep:
                continue
            local = ROOT / dep
            if not local.exists():
                continue
            try:
                if own == local.resolve() or own in local.resolve().parents:
                    continue  # 本目录内的文件,跟着目录一起传
            except OSError:
                continue
            # 文件引用传所在目录(纹理等一起带上);目录引用直接传
            out.add(dep if local.is_dir() else str(Path(dep).parent))
    return sorted(out)


def asset_sync_plan(task: str, variant: str = "") -> list[str]:
    """评测该任务 / 变体要的资产目录:布局 + 任务 YAML + 它们引用到的原件目录(递归去重)。"""
    todo = list(layout_asset_dirs(task, variant)) + task_asset_dirs(task)
    seen: set[str] = set()
    while todo:
        rel = todo.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        if len(seen) > 300:
            raise PlanError(f"{task} 的资产引用太多(超过 300 个目录),先检查布局 / 资产")
        todo.extend(d for d in _asset_deps(rel) if d not in seen)
    return sorted(seen)


def asset_label(rel: str) -> str:
    """资产目录 → 短名字(``.../SimReady_hazelnut_s300/00000`` → ``SimReady_hazelnut_s300``)。"""
    path = Path(rel)
    return path.parent.name if path.name.isdigit() else path.name


def _asset_probes(task: str, variant: str = "") -> dict[str, str]:
    """``{要在远端查的路径: 资产目录}``:目录里本地有的 metadata.json / object.usd(z),都没有时查目录本身。"""
    probes: dict[str, str] = {}
    for rel in asset_sync_plan(task, variant):
        files = [f"{rel}/{name}" for name in ("metadata.json", "object.usd", "object.usdz")
                 if (ROOT / rel / name).exists()]
        for probe in files or [rel]:
            probes[probe] = rel
    return probes


def missing_asset_dirs(m: dict, task: str, variant: str = "", absent: set[str] | None = None) -> list[str]:
    """远端缺的资产目录(仓库相对):目录不在,或目录里的 metadata.json / object.usd(z) 不全。
    ``absent`` 是已经一次查好的远端缺的路径(eval_plan 把资产和模型合成一次 SSH),没有时自己查。"""
    probes = _asset_probes(task, variant)
    missing = [p for p in probes if p in absent] if absent is not None else sv.missing_paths(m, list(probes))
    return sorted({probes.get(path, path) for path in missing})


def _rewrite_asset_paths(m: dict, dirs: list[str]) -> None:
    """刚同步过去的资产里写死的本机仓库绝对路径 → 远端仓库路径(否则 usd 引用加载不到)。"""
    repo = str(m["repo_path"]).rstrip("/") + "/"
    replacements = [(str(ROOT.resolve()).rstrip("/") + "/", repo),
                    ("/home/xuancun/RoboSafeJanus/", repo)]  # 旧构建机的路径( docs/data_collection.md 8.6 )
    includes = " ".join(f"--include='*.{ext}'" for ext in ("usd", "usda", "json", "yml", "yaml"))
    lines = []
    for rel in dirs:
        q = shlex.quote(rel)
        for old, new in replacements:
            # grep 没匹配时退出 1,不能让 set -o pipefail 把整段判失败;-I 跳过二进制 crate
            lines.append(f"( set +o pipefail; grep -I -rlZF {includes} {shlex.quote(old)} {q} 2>/dev/null "
                         f"| xargs -0 -r sed -i {shlex.quote(f's#{old}#{new}#g')} ) || exit 1")
        # 有些 NFS 上 sed -i 生成的新文件权限是 0000
        lines.append(f"find {q} -type f -perm 0000 -exec chmod 664 {{}} + || exit 1")
    proc = sv.run(m, "; ".join(lines), timeout=300)
    if proc.returncode != 0:
        raise PlanError(f"改写 {m['key']} 上资产的绝对路径失败:{(proc.stderr or proc.stdout).strip()[-300:]}")


def sync_assets(m: dict, dirs: list[str]) -> None:
    """把本地 ``dirs``(仓库相对)同步到远端,并改写写死的本机绝对路径。"""
    if not dirs:
        return
    parents = sorted({str(Path(d).parent) for d in dirs})
    proc = sv.run(m, "mkdir -p " + " ".join(shlex.quote(p) for p in parents), timeout=120)
    if proc.returncode != 0:
        raise PlanError(f"在 {m['key']} 上建资产目录失败:{(proc.stderr or proc.stdout).strip()[-300:]}")
    dest = f"{sv.target(m)}:{m['repo_path']}/"
    for rel in dirs:
        src = ROOT / rel
        if not src.exists():
            raise PlanError(f"本地缺资产 {rel},没法同步到 {m['key']}")
        if _rsync(m, "-L", f"{src}/", f"{dest}{rel}/") != 0:
            raise PlanError(f"同步资产 {rel} 到 {m['key']} 失败")
    _rewrite_asset_paths(m, dirs)


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
    tasks = rp.own_tasks(state) if stage in (*DATA_BUTTONS, "layouts", "eval") else rp.stage_tasks(state, stage)
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


def _busy(group: str, task: str, kind: str = "", policy: str = "", variant: str = "") -> None:
    """同一条记录还在跑时不能再开。按模型的评测(``<group>.<task>.<policy>``)只和同一模型、
    以及这个任务"评所有模型"的评测互斥;"评所有模型"还要等这个任务的单模型评测都结束。
    (2026-11-16 前按模型的评测也记在 ``<group>.<task>``、带 policy 字段,这种只和同一模型互斥。)"""
    jobs = [read_job(job_id(group, task, policy, variant))]
    if kind == "eval" and variant:  # 变体评测只和同一模型、同一变体的互斥(结果目录分开)
        jobs += [j for j in running(("eval",)) if j["group"] == group and j["task"] == task
                 and j.get("variant") == variant and (not policy or j.get("policy", "") in ("", policy))]
    elif kind == "eval" and policy:
        task_job = read_job(job_id(group, task)) or {}
        if task_job.get("kind") == "eval" and task_job.get("policy", "") in ("", policy):
            jobs.append(task_job)
    elif kind == "eval":
        jobs += [j for j in running(("eval",)) if j["group"] == group and j["task"] == task and j.get("policy")
                 and not j.get("variant")]
    for job in jobs:
        if job and job_state(job) == "running":
            raise PlanError(f"{task} 已有任务在运行:{job_label(job)}(开始于 {job.get('started_at')})")


def pending_remote_collect(group: str, task: str) -> dict | None:
    """上一次远端采集还没合并回本地的记录 ``{machine, dir, episode_num, start_seed, template, ...}``。"""
    job = read_job(job_id(group, task)) or {}
    r = job.get("remote")
    return r if job.get("kind") == "collect" and r and not r.get("merged") else None


def collect_plan(group: str, task: str, machine: str = "") -> dict:
    state = _state(group, task, "collect")
    _expert_ok(state, task, "采集")
    m = machine_cfg(machine)
    pending = pending_remote_collect(group, task)
    if pending and pending["machine"] != m["key"]:
        raise PlanError(f"{task} 在 {pending['machine']} 上还有一批没拉回(远端 {pending['dir']}):"
                        f"选 {pending['machine']} 再点一次,接着看 / 拉回 / 续采;拉回之前不能换机器(seed 段会撞)")
    if not sv.is_local(m):
        return remote_collect_plan(group, task, m, pending)
    t = task_collect_plan(task)
    jid = job_id(group, task)
    busy = busy_gpus(("collect",), jid, m["key"])
    gpus = gpu_usage()
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy]
    if t["resume"]:
        what = f"续采残留分片(原 {t['episode_num']} 条)"
    elif t["episode_num"]:
        what = f"补采 {t['episode_num']} 条" + (f",seed 从 {t['start_seed']} 起" if t["start_seed"] else "")
    else:
        what = "已够,不用采"
    tpl = f",模板 {t['template']}" if t["template"] else ""
    lines = [f"{task}:在 {m['key']}(本机)上采到 {rp.COLLECT_TARGET} 条", f"已有 {t['have']} 条 → {what}{tpl}"]
    lines += gpu_lines(gpus, busy, m)
    if t["episode_num"]:
        lines += ["", "用下面填的所有卡,每卡跑「每卡 worker 数」个(一个约 14–20 GB)。中断后再点一次就续采。",
                  "采完不用审核,接着点 exchange 格的「转换」。"]
    return {"group": group, "task": task, "kind": "collect", "plan": t, "free_gpus": free,
            "per_gpu": sv.per_gpu_default(m), **_machine_fields(m),
            "need_gpus": bool(t["episode_num"]), "nothing": not t["episode_num"], "text": "\n".join(lines)}


def remote_collect_plan(group: str, task: str, m: dict, pending: dict | None) -> dict:
    """在远端机器上采:续上次没拉回的,或者按本地已有的条数补采(seed 段接在本地已用的 seed 之后)。"""
    jid = job_id(group, task)
    if pending:
        t = {"task": task, "have": rp._hdf5_count(task), "target": rp.COLLECT_TARGET,
             "episode_num": pending["episode_num"], "start_seed": pending["start_seed"],
             "template": pending.get("template"), "resume": True}
        what = (f"续上次 {pending.get('started_at', '')} 在 {m['key']} 上的采集({pending['episode_num']} 条,"
                f"远端 {pending['dir']}):还在跑就接着看,跑完了就拉回合并,没跑完就续采")
    else:
        if (SHARDS / task).exists():
            raise PlanError(f"本地有残留分片 _shards/{task}(本机采集没跑完):先选本机续采完,再用远端补")
        t = task_collect_plan(task)
        what = (f"补采 {t['episode_num']} 条,seed 从 {t['start_seed'] or START_SEED} 起" if t["episode_num"]
                else "已够,不用采")
    tpl = f",模板 {t['template']}" if t["template"] else ""
    lines = [f"{task}:在 {m['key']}(远端 {sv.target(m)}:{m['repo_path']})上采到 {rp.COLLECT_TARGET} 条",
             f"本地已有 {t['have']} 条 → {what}{tpl}"]
    if not t["episode_num"]:
        return {"group": group, "task": task, "kind": "collect", "plan": t, **_machine_fields(m),
                "need_gpus": False, "nothing": True, "text": "\n".join(lines)}
    try:
        missing = sv.missing_paths(m, task_asset_dirs(task))
    except sv.ServerError as error:
        raise PlanError(str(error)) from None
    if missing:
        raise PlanError(f"{m['key']} 上缺这个任务的资产(资产文件里有本机绝对路径,不自动同步;按 "
                        "docs/data_collection.md 8.6 节传过去或在远端从 ModelScope 拉):\n"
                        + "\n".join(f"  - {p}" for p in missing))
    busy = busy_gpus(("collect", "eval"), jid, m["key"])
    gpus = machine_gpus(m)
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy]
    lines += gpu_lines(gpus, busy, m)
    lines += ["", "步骤(后台自动做,关掉页面也继续):",
              f"  1. 同步代码 {' '.join(sv.CODE_DIRS)} + data/grasp_pose 到远端(rsync,不加 --delete)",
              f"  2. 远端 collect_data_parallel.sh 采 {t['episode_num']} 条(用下面填的卡,每卡「每卡 worker 数」个)",
              "  3. 边采边把新完成的分片拉回本地,采完核对 seed 后 merge_shards.py 接到本地数据之后(编号接续,seed 去重)",
              "远端的数据不删。中断后再点一次、选同一台机器就续上;采完不用审核,接着点 exchange 格的「转换」。"]
    return {"group": group, "task": task, "kind": "collect", "plan": t, "free_gpus": free,
            "per_gpu": sv.per_gpu_default(m), **_machine_fields(m), "remote_dir": (pending or {}).get("dir"),
            "need_gpus": True, "nothing": False, "text": "\n".join(lines)}


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


# 同类任务串行(排队):两个转换会互相删扁平视图 RoboSafeJanus-Benchmark-flat 的软链接,两个上传共用
# ``.cache/modelscope`` 断点 / 哈希缓存和硬链接视图目录。转换 A 和上传 B 只读写各自的任务目录,可以同时跑。
# 按钮随时能点:后台进程先排队(任务记录 ``waiting``),按点击顺序(started_at)轮到了再做。
QUEUED_KINDS = ("export", "upload")
QUEUE_POLL = 5  # 排队时每几秒看一次轮到没有


def _queue_ahead(kind: str, jid: str, mine: tuple[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """``kind`` 这一类排在 ``jid`` 前面的任务:``(正在做的, 排队中且先点的)``。``mine`` = (started_at, jid),
    None 时(还没点)所有排队中的都算在前面。"""
    jobs = [(j_id, j) for j_id, j in all_jobs(tail=False).items()
            if j.get("kind") == kind and j["state"] == "running" and j_id != jid]
    active = [j for _, j in jobs if not j.get("waiting")]
    waiting = sorted(((j.get("started_at") or "", j_id), j) for j_id, j in jobs if j.get("waiting"))
    ahead = [j for key, j in waiting if mine is None or key < mine]
    return active, ahead


def _queue_note(group: str, task: str, kind: str) -> str:
    """计划里写一行排队情况;前面没人时空串。"""
    active, ahead = _queue_ahead(kind, job_id(group, task))
    if not active and not ahead:
        return ""
    names = [f"{j['task']}(进行中)" for j in active] + [f"{j['task']}(排队)" for j in ahead]
    return (f"⏳ 前面还有 {len(names)} 个{KIND_LABEL[kind]}:{'、'.join(names)}。"
            f"确认后排队,按点击顺序一个个做。")


def _wait_turn(group: str, task: str, kind: str):
    """后台 run 进程:排队直到前面同类的都做完,拿到 ``<kind>.lock``(进程退出自动释放)后返回锁文件。"""
    jid = job_id(group, task)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(LOG_DIR / f".{kind}.lock", "w")  # noqa: SIM115 - 持有到进程结束
    mine = ((read_job(jid) or {}).get("started_at") or now(), jid)
    shown = None
    while True:
        active, ahead = _queue_ahead(kind, jid, mine)
        if not active and not ahead:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:  # 前一个刚拿到锁、还没改记录
                pass
            else:
                update_job(group, task, waiting=False)
                return lock
        first = (active or ahead or [{}])[0]
        text = (f"排队中:前面 {len(active) + len(ahead)} 个"
                + (f"(正在{KIND_LABEL[kind]} {first['task']})" if active else ""))
        if text != shown:
            _step(group, task, text)
            shown = text
        time.sleep(QUEUE_POLL)


def export_plan(group: str, task: str) -> dict:
    """「转换」:采够后导出 LeRobot v3.0 + v2.1。"""
    state = _state(group, task, "exchange")
    _expert_ok(state, task, "转换")
    reason = _collected(task)
    if reason:
        raise PlanError(f"{task} 还不能转换:{reason}")
    need = export_needed(task)
    queue = _queue_note(group, task, "export") if need else ""
    lines = [f"{task}:导出 LeRobot v3.0 + v2.1(export_lerobot.sh --format both)", "",
             f"现在:{need}" if need else "已经转换过,v3.0 / v2.1 条数都和 hdf5 一致,不用再转",
             *(["", queue] if queue else []),
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
    queue = _queue_note(group, task, "upload")
    lines = [f"{task}:上传 HDF5 + LeRobot 到 ModelScope(只动这个任务的目录)", "", *([queue, ""] if queue else [])]
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
    """ingest 格的「下载资产」:资产链接填好后按组下载 / 转换 / 入库,完了提交 ingest(检查通过自动批准)。"""
    if task != GROUP_TASK or not NAME_RE.fullmatch(group):
        raise PlanError("ingest 是按组的(任务名用 __group__)")
    if not rp.state_path(group).exists():
        raise PlanError(f"{group}: 还没有流水线状态")
    state = rp.load_state(group)
    status, why = rp.entry_status(state, "ingest")
    if status == "awaiting_review":
        raise PlanError("ingest 已提交、还没批准(旧的待审条目,点 ingest 格批准)")
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


def _is_step(name: str) -> bool:
    """checkpoint step 目录:纯数字、step_<n> / global_step_<n>(LingBot_VLA)或 checkpoint-<n>(GR00T_N17),
    或 run 根目录下的权重文件 checkpoint_step_<n>.safetensors(OpenWAM);
    和 download_models_from_modelscope.py / sync_status.py 的 STEP_RE 一致。"""
    return bool(re.match(
        r"^(?:(?:checkpoint[_-])?(?:global_)?step|checkpoint)[_-]?\d+(?:\.(?:safetensors|pt|pth|ckpt|bin))?$|^\d+$",
        name, re.IGNORECASE))


def _step_num(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


@rp.memoized
def _complete_step(run_dir: Path) -> str | None:
    """run 目录里下载完整的最新 step(没有 ``*.incomplete`` 临时文件);没有时 None。"""
    steps = sorted((d for d in run_dir.iterdir() if _is_step(d.name)), key=lambda d: _step_num(d.name))
    for step in reversed(steps):
        if step.is_file():  # 单文件 step(OpenWAM):下载中是 <name>.incomplete,下完才改名
            if step.stat().st_size > 0:
                return step.name
        elif not any(step.rglob("*.incomplete")) and any(step.rglob("*")):
            return step.name
    return None


@rp.memoized
def eval_policies() -> tuple[str, ...]:
    """要评的模型 = ModelScope 模型仓库里有的 policy(读 sync_status 的快照;Pi_0 / Pi_05 在前)。
    快照读不到时退回本地 data/RoboSafeJanus-Model/ 下已有的。只算 XPolicyLab/policy/ 里有适配的。"""
    found: set[str] = set()
    try:
        found = {m["policy"] for m in json.loads(SNAPSHOT.read_text(encoding="utf-8")).get("models") or []}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not found and MODEL_DIR.is_dir():
        found = {d.name for d in MODEL_DIR.iterdir() if d.is_dir()}
    found = {p for p in found if (ROOT / "XPolicyLab" / "policy" / p).is_dir()}
    return (*[p for p in EVAL_POLICIES if p in found], *sorted(found - set(EVAL_POLICIES)))


@rp.memoized
def _deploy_cfg(policy: str) -> dict:
    import yaml

    try:
        path = ROOT / "XPolicyLab" / "policy" / policy / "deploy.yml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def policy_env(policy: str) -> str:
    """``robodojo.sh eval --policy-env``:deploy.yml 有 ``policy_uv_env_path`` 的是 ``uv``(Pi_0 / Pi_05),
    否则是 conda 环境(deploy.yml 的 ``policy_conda_env``,没写时用小写的 policy 名,如 lingbot_vla)。"""
    cfg = _deploy_cfg(policy)
    if cfg.get("policy_uv_env_path"):
        return "uv"
    return str(cfg.get("policy_conda_env") or policy.lower())


def policy_action_type(policy: str) -> str:
    """``robodojo.sh eval --action-type``:deploy.yml 的 ``action_type``(没写 / null 时 ``joint``)。

    OpenWAM 是 EE 空间策略(``ee``,model.py 里非 ee 直接报错);结果目录名里的 ``action_type=``
    也跟着它走,否则评完看板找不到结果(``eval_runs`` 只认这个目录)。"""
    value = _deploy_cfg(policy).get("action_type")
    return str(value).strip() if value else "joint"


def policy_env_vars(policy: str, m: dict | None = None) -> tuple[dict, list[str]]:
    """``(要加的环境变量, 缺的说明)``;``m`` 是远端机器时从它的 ``policy_env`` 取。"""
    need = POLICY_ENV_VARS.get(policy) or {}
    if not need:
        return {}, []
    if m is not None and not sv.is_local(m):
        have = dict(m.get("policy_env") or {})
        where = f"docs/servers.json 的 {m['key']}.policy_env"
    else:
        try:
            have = json.loads(POLICY_ENV_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            have = {}
        have = {**have, **{k: os.environ[k] for k in need if os.environ.get(k)}}
        where = rp.rel(POLICY_ENV_FILE)
    got = {k: str(have[k]) for k in need if have.get(k)}
    return got, [f"缺 {k}({desc},写进 {where})" for k, desc in need.items() if k not in got]


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
    for policy in eval_policies():
        for key, zero_shot in keys:
            base = MODEL_DIR / policy / key
            runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []
            if not runs:
                continue
            run = runs[-1]
            item = {"policy": policy, "key": key, "ckpt": run.name, "step": _complete_step(run),
                    "zero_shot": zero_shot, "ready": False, "why": ""}
            link = ROOT / "XPolicyLab" / "policy" / policy / "checkpoints" / run.name
            env_missing = policy_env_vars(policy)[1]
            if any(p in (policy, "*") and t in (key, "*") for p, t in downloading):
                item["why"] = "还在下载"
            elif env_missing:
                item["why"] = ";".join(env_missing)
            elif not item["step"]:
                item["why"] = "没下载完(在模型列点「⬇ 下载」)"
            elif not link.exists():
                item["why"] = f"缺 checkpoints/{run.name} 链接(重新点一次下载会补上)"
            elif (link / "lingbotvla_cli.yaml").exists() and not (link / "checkpoints").exists():
                # LingBot_VLA 评测找 <run>/checkpoints/global_step_*,下载脚本会建成这种布局
                item["why"] = f"checkpoints/{run.name} 是旧的整目录链接(重新点一次下载会改好)"
            else:
                item["ready"] = True
            out.append(item)
            break
    return out


def eval_layouts(task: str, variant: str = "") -> int:
    """评测布局个数(``eval_layout_files``)。"""
    return len(eval_layout_files(task, variant))


def eval_target(task: str, variant: str = "") -> int:
    """要评满几条:原任务 ``EVAL_NUM``;变体是它的布局个数(放不下障碍物的 S1.0 布局跳过了,不到 50)。"""
    return eval_layouts(task, variant) if variant else EVAL_NUM


def eval_result_base(task: str, policy: str, ckpt: str, variant: str = "") -> Path:
    """``eval_result/RoboDojo/<task>/<policy>/<config>/<seed>_ckpt_name=<ckpt>,action_type=<ee|joint>[,variant=S1.x]``。
    ``action_type`` 取 deploy.yml(``policy_action_type``),和 ``main.py`` 用 ``additional_info`` 拼的目录一致。"""
    suffix = f",variant={variant}" if variant else ""
    return (EVAL_RESULT / task / policy / EVAL_CFG
            / f"{EVAL_SEED}_ckpt_name={ckpt},action_type={policy_action_type(policy)}{suffix}")


def eval_runs(task: str, policy: str, ckpt: str, variant: str = "") -> list[tuple[str, int]]:
    """已有的评测 ``[(run_id, eval_time)]``,按时间排。"""
    base = eval_result_base(task, policy, ckpt, variant)
    out = []
    for path in sorted(base.glob("*/_result.json")) if base.is_dir() else []:
        try:
            out.append((path.parent.name, int(json.loads(path.read_text(encoding="utf-8")).get("eval_time") or 0)))
        except (OSError, ValueError):
            continue
    return out


def eval_manifest(task: str, policy: str, ckpt: str, run_id: str, variant: str = "") -> Path:
    """``run_id`` 的续评清单 ``_resume_<run_id>.json``(和结果目录同级;src/eval_client/main.py 只认它)。"""
    return eval_result_base(task, policy, ckpt, variant) / f"_resume_{run_id}.json"


def eval_resume_run(task: str, policy: str, ckpt: str, target: int, variant: str = "") -> tuple[str, int] | None:
    """能续评的那次 ``(run_id, eval_time)``:没评满且续评清单还在。

    评到自己 ``--eval-num`` 的运行(比如 5 条的 pilot)结束时清单被删了;拿它的 run_id 再跑,
    main.py 找不到清单会在同一个目录从第 0 条从头评、把已有结果盖掉,所以这种只能新评。"""
    runs = [r for r in eval_runs(task, policy, ckpt, variant)
            if 0 < r[1] < target and eval_manifest(task, policy, ckpt, r[0], variant).is_file()]
    return max(runs, key=lambda r: r[1]) if runs else None


def eval_blocked(state: dict, group: str, task: str, downloading: set | None = None, policy: str = "",
                 variant: str = "") -> str:
    """评测按钮不能点的原因(看板 eval 格用);空串表示可以点。``policy`` 给出时只看这个模型。"""
    status, why = rp.entry_status(state, f"expert/{task}")
    if status != "approved":
        return "expert 要先批准(现在:" + (rp.STATUS_LABEL.get(status) or "未提交") + ")"
    models = [m for m in eval_models(group, task, downloading) if not policy or m["policy"] == policy]
    if not models:
        return "还没有模型:远端有了以后在模型列点「⬇ 下载」"
    if not any(m["ready"] for m in models):
        return "模型还不能用:" + ";".join(f"{m['policy']} {m['why']}" for m in models)
    if variant:
        if not eval_layouts(task, variant):
            return (f"还没有 {variant} 评测布局(python src/data_collect/eval_layouts.py variant --task {task} "
                    f"--variant {variant} ...)")
        return ""
    n = eval_layouts(task)
    if n < EVAL_NUM:
        return f"评测布局只有 {n}/{EVAL_NUM} 个(在评测布局格点「⊕ 生成布局」)"
    return ""


def policy_venv(policy: str) -> str:
    """policy 的 uv 环境目录(相对仓库根):deploy.yml 的 ``policy_uv_env_path``(相对 policy 目录)。"""
    pdir = Path("XPolicyLab") / "policy" / policy
    env = Path(str(_deploy_cfg(policy).get("policy_uv_env_path") or "openpi")).expanduser()
    return str(env if env.is_absolute() else pdir / env)


def _remote_needs(m: dict, item: dict) -> dict[str, str]:
    """远端评这个模型要有的路径(仓库相对,按本地的 ckpt / step):``venv`` uv / conda 环境、``link``
    checkpoint 链接、``model`` 模型文件。"""
    env = policy_env(item["policy"])
    return {
        "venv": (f"{policy_venv(item['policy'])}/.venv/bin/activate" if env == "uv"
                 else f"{m['conda_root']}/envs/{env}/bin/python"),
        "link": f"XPolicyLab/policy/{item['policy']}/checkpoints/{item['ckpt']}",
        "model": f"{rp.rel(MODEL_DIR)}/{item['policy']}/{item['key']}/{item['ckpt']}/{item['step']}",
    }


def remote_eval_missing(m: dict, item: dict, absent: set[str] | None = None) -> list[str]:
    """远端评这个模型还缺什么:``_remote_needs`` 里的路径、策略要的环境变量(servers.json 的 policy_env)。
    ``absent`` 同 ``missing_asset_dirs``。"""
    need = list(_remote_needs(m, item).values())
    missing = [p for p in need if p in absent] if absent is not None else sv.missing_paths(m, need)
    return missing + policy_env_vars(item["policy"], m)[1]


def _ssh_hint(m: dict) -> str:
    """``docs/servers.json`` 里那台机器的 ssh 命令(没写 ``ssh.command`` 时自己拼)。"""
    ssh = m.get("ssh") or {}
    command = str(ssh.get("command") or "").strip()
    if command:
        return command
    port = f"-p {ssh['port']} " if ssh.get("port") else ""
    return f"ssh {port}{sv.target(m)}"


def remote_fix_lines(m: dict, missing_items: list[tuple[dict, list[str]]]) -> list[str]:
    """远端缺 uv 环境 / checkpoint 链接 / 模型文件时怎么补:一条 ssh + 每个模型一条下载命令,看板里直接抄。"""
    out = [f"要在 {m['key']} 上补(和本地同一个 ckpt / step):",
           f"    {_ssh_hint(m)}"]
    conda = ((m.get("conda_env") or {}).get("name") or "").strip()
    if conda:  # 下载脚本要 modelscope,装在 conda 环境里(远端默认 python 不一定有)
        out.append(f"    conda activate {conda}")
    out.append(f"    cd {m['repo_path']}")
    notes = []
    for item, missing in missing_items:
        need = _remote_needs(m, item)
        env = policy_env(item["policy"])
        venv = need["venv"]
        if need["model"] in missing or need["link"] in missing:
            cmd = ["python", "scripts/download_models_from_modelscope.py", "--policy", item["policy"],
                   "--task", item["key"], "--run", item["ckpt"]]
            if item["step"]:
                cmd += ["--step", str(item["step"])]
            out.append("    " + " ".join(cmd))
        if venv in missing:
            kind = " uv 环境" if env == "uv" else f" conda 环境 {env}"
            notes.append(f"  {item['policy']} 的{kind}也没有({venv}):先按 docs/model_evaluation.md §3.2 在远端装好")
    out.append('  说明:下载脚本会顺带建 checkpoints/<run> 软链接;远端没装 modelscope 先 '
               'pip install -U "modelscope>=1.40",私有仓库要先 modelscope login --token <token>'
               '(或设 MODELSCOPE_API_TOKEN)。')
    if m.get("shared_storage_with"):
        out.append(f"  {m['key']} 和 {m['shared_storage_with']} 共用同一份存储,下完两台都能用。")
    out += notes
    out.append("  补完回来再点一次「评测」。")
    return out


def eval_plan(group: str, task: str, policy: str = "", machine: str = "", variant: str = "") -> dict:
    """eval 格的「评测」:ModelScope 上有的模型(已下载的)各评 50 条,没评满的续评;``policy`` 给出时只评这个模型。

    ``machine`` 是远端机器时,模型 / 策略环境要远端已经有(按本地的 ckpt / step 查),缺的这次不评;
    布局 / 任务 YAML 用到而远端缺的资产,评测前会自动同步(计划 ``missing_assets``)。
    ``variant``(S1.1 …)时评这个 S1 变体:它自己的布局各评一遍(条数 = 布局个数),结果单独存。"""
    variant = norm_variant(variant)
    if policy and policy not in eval_policies():
        raise PlanError(f"ModelScope 上没有 {policy} 的模型(有:{'、'.join(eval_policies()) or '无'})")
    m = machine_cfg(machine)
    remote = not sv.is_local(m)
    state = _state(group, task, "eval")
    reason = eval_blocked(state, group, task, policy=policy, variant=variant)
    if reason:
        raise PlanError(f"{task} {policy + ' ' if policy else ''}{variant + ' ' if variant else ''}还不能评测:{reason}")
    who = f"{policy} " if policy else "每个模型"
    where = f"在 {m['key']}(远端 {sv.target(m)}:{m['repo_path']})上" if remote else f"在 {m['key']}(本机)上"
    target = eval_target(task, variant)
    what = (f"{variant} 变体评 {target} 条({target} 个 {variant} 布局,layout_id 和 S1.0 对应;"
            "每步查安全检查,记成功率 + 违规率)" if variant else f"评 {EVAL_NUM} 条")
    kinds = sorted({policy_action_type(it["policy"]) for it in eval_models(group, task)
                    if not policy or it["policy"] == policy}) or ["joint"]
    queue, lines = [], [f"{task}:{where}{who}{what}(seed {EVAL_SEED},{EVAL_CFG},{'/'.join(kinds)})", ""]
    models = [it for it in eval_models(group, task) if not policy or it["policy"] == policy]
    absent: set[str] | None = None
    if remote:  # 资产和每个模型要的路径合成一次 SSH 查
        probe = [*_asset_probes(task, variant),
                 *(p for it in models if it["ready"] for p in _remote_needs(m, it).values())]
        try:
            absent = set(sv.missing_paths(m, probe))
        except sv.ServerError as error:
            raise PlanError(str(error)) from None
    missing_assets = missing_asset_dirs(m, task, variant, absent) if remote else []
    if missing_assets:
        names = "、".join(asset_label(p) for p in missing_assets[:6]) + (" 等" if len(missing_assets) > 6 else "")
        lines.append(f"  📦 {m['key']} 上缺 {len(missing_assets)} 个资产目录,评测前会自动同步:{names}")
    missing_items: list[tuple[dict, list[str]]] = []
    for item in models:
        name = f"{item['policy']} · {item['ckpt']}(step {item['step'] or '?'}{',zero-shot' if item['zero_shot'] else ''})"
        if not item["ready"]:
            lines.append(f"  ⏭ {name}:{item['why']},这次不评")
            continue
        runs = eval_runs(task, item["policy"], item["ckpt"], variant)
        best = max(runs, key=lambda r: r[1]) if runs else None
        if best and best[1] >= target:
            lines.append(f"  ✅ {name}:已评满 {best[1]} 条({best[0]}),跳过")
            continue
        partial = eval_resume_run(task, item["policy"], item["ckpt"], target, variant)
        resume = partial[0] if partial else None
        if remote:
            missing = remote_eval_missing(m, item, absent)
            if missing:
                lines.append(f"  ⏭ {name}:{m['key']} 上缺 " + "、".join(missing) + ",这次不评")
                missing_items.append((item, missing))
                continue
        if resume:
            how = f"续评 {resume}(已有 {partial[1]}/{target})"
        elif best and best[1] > 0:
            how = f"新评({best[0]} 的 {best[1]} 条已经按自己的条数评完、没有续评清单,接不上)"
        else:
            how = "新评"
        lines.append(f"  ▶ {name}:{how}")
        queue.append({**item, "run_id": resume})
    fix = remote_fix_lines(m, missing_items) if missing_items else []
    if not queue:
        if fix:
            tail = [f"{m['key']} 上没有能评的模型,先按下面补,再点一次:", "", *fix]
        elif remote and any(ln.startswith("  ⏭") for ln in lines):
            tail = [f"{m['key']} 上没有能评的模型:先在远端装好 uv 环境、下好模型(和本地同一个 ckpt / step),再点一次。"]
        else:
            tail = ["都评满了,不用再跑。"]
        return {"group": group, "task": task, "kind": "eval", "policy": policy, "variant": variant, "queue": [],
                "missing_assets": missing_assets, "nothing": True, **_machine_fields(m),
                "text": "\n".join([*lines, "", *tail])}
    jid = job_id(group, task, policy, variant)
    busy = busy_gpus(("collect", "eval", "layouts"), jid, m["key"])
    gpus = machine_gpus(m)
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy]
    if fix:
        lines += ["", "上面 ⏭ 的模型这次不评,要在远端补完再点一次:", *fix]
    lines += gpu_lines(gpus, busy, m)
    lines += ["", "下面填的卡两两一对(policy 卡 + sim 卡),一对跑一条,填几对就并行几条;只填 1 张时 policy 和 sim 共用。",
              f"每条约 1–2 小时。中断后再点一次就续评;日志在 {rp.rel(EVAL_LOGS)}/。"]
    if remote:
        lines += [f"远端:先同步代码 + 这个任务的评测布局 + 缺的资产(自动) + 要续评的结果,"
                  f"评完把 eval_result/RoboDojo/{task}/ 和日志拉回本地。"]
    lines += ["评测不审核,评完不用提交。"]
    return {"group": group, "task": task, "kind": "eval", "policy": policy, "variant": variant, "queue": queue,
            "free_gpus": free[: 2 * len(queue)], "missing_assets": missing_assets, **_machine_fields(m),
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
    busy = busy_gpus(("collect", "eval", "layouts"), jid, sv.default_key())  # 生成布局只在本机跑
    gpus = gpu_usage()
    free = [g["index"] for g in gpus if g["used"] < FREE_GPU_MIB and g["index"] not in busy]
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


def plan(button: str, group: str, task: str, dry_run: bool = True, policy: str = "", machine: str = "",
         variant: str = "") -> dict:
    if button == "ingest":
        return ingest_plan(group, task)
    if button == "eval":
        return eval_plan(group, task, policy, machine, variant)
    if button == "layouts":
        return layouts_plan(group, task)
    if button == "collect":
        return collect_plan(group, task, machine)
    if button == "exchange":
        return export_plan(group, task)
    return upload_plan(group, task, dry_run=dry_run)


# ---------------------------------------------------------------------------
# 启动 / 停止
# ---------------------------------------------------------------------------


def start(button: str, group: str, task: str, gpus: str = "", delete: bool = False, policy: str = "",
          machine: str = "", per_gpu: int | None = None, variant: str = "") -> dict:
    """重新算一遍计划,通过后在新会话里后台运行 ``run``;返回任务记录。``policy`` 只用于 eval(只评这个模型);
    ``machine`` 只用于 collect / eval(``docs/servers.json`` 的键,空 = 默认机器);``per_gpu`` 只用于 collect。"""
    policy = policy if button == "eval" else ""
    variant = norm_variant(variant) if button == "eval" else ""
    _busy(group, task, "eval" if button == "eval" else "", policy, variant)
    if button not in MACHINE_KINDS:
        machine, per_gpu = "", None
    machine = machine_cfg(machine)["key"] if button in MACHINE_KINDS else ""
    p = plan(button, group, task, dry_run=False, policy=policy, machine=machine, variant=variant)
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
    if policy:
        args += ["--policy", policy]
    if variant:
        args += ["--variant", variant]
    if machine:
        args += ["--machine", machine]
    if per_gpu:
        if not 1 <= int(per_gpu) <= 8:
            raise PlanError(f"每卡 worker 数要在 1–8 之间(填了 {per_gpu})")
        args += ["--per_gpu", str(int(per_gpu))]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"task_{task}_{kind}_{datetime.now():%Y%m%d_%H%M%S}.log"
    job = {"group": group, "task": task, "kind": kind, "pid": None, "log": str(log), "started_at": now(),
           "tasks": p.get("tasks") or [task],
           "finished_at": None, "returncode": None, "step": "启动中", "gpus": gpu_list, "delete": bool(delete),
           "policy": policy, "variant": variant, "machine": machine, "args": args}
    if kind in QUEUED_KINDS:
        job.update(waiting=True, step="排队中")
    old = read_job(job_id(group, task)) or {}
    if kind == "collect" and old.get("remote") and not old["remote"].get("merged"):
        job["remote"] = old["remote"]  # 上次远端没拉回的:同一台机器接着做(collect_plan 已经查过是同一台)
    write_job(job)
    with open(log, "w", encoding="utf-8") as fp:
        fp.write(f"[group_jobs] {now()} {job_label(job)} {group}/{task}\n{p['text']}\n\n")
        proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "run", *args], cwd=str(ROOT),
                                stdout=fp, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    job = read_job(job_key(job)) or job  # run 进程可能已经改过(排队 / 步骤)
    job["pid"] = proc.pid
    write_job(job)
    return job


def stop(group: str, task: str, policy: str = "", variant: str = "") -> str:
    """停掉 ``<group>.<task>``(``policy`` / ``variant`` 给出时是这个模型 / 变体的评测)那条记录的任务。"""
    variant = norm_variant(variant)
    jid = job_id(group, task, policy, variant)
    job = read_job(jid)
    if not job or job_state(job) != "running":
        raise PlanError(f"{task} {policy + ' ' if policy else ''}{variant + ' ' if variant else ''}没有在运行的任务")
    note = ""
    r = job.get("remote")
    if r and r.get("pid"):  # 远端的也停掉(setsid 起的:pgid == pid)
        try:
            m = machine_cfg(r["machine"])
            sv.run(m, f"kill -TERM -- -{int(r['pid'])} 2>/dev/null; sleep 2; kill -KILL -- -{int(r['pid'])} 2>/dev/null; true",
                   timeout=40)
            note = f",{r['machine']} 上的进程也停了(远端数据保留)"
        except (PlanError, sv.ServerError) as error:
            note = f";⚠ 远端没停掉:{error}"
    os.killpg(job["pid"], signal.SIGTERM)  # 新会话:pgid == pid,连同采集 worker 一起停
    job.update(stopped=True, step="已停止", returncode=-15, finished_at=now())
    write_job(job, jid)
    return f"已停止 {task} 的{job_label(job)}{note};再点一次按钮会续上"


# ---------------------------------------------------------------------------
# 运行(在后台会话里)
# ---------------------------------------------------------------------------


def _step(group: str, task: str, text: str) -> None:
    print(f"\n[group_jobs] {now()} {text}", flush=True)
    update_job(group, task, step=text)


def _sh(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    # 子脚本(collect_data.sh 等)直接调 `python`:当前解释器目录放到 PATH 最前,
    # 看板没 conda activate 启动时也能用到 RoboDojo 环境
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PATH": path})


def run_collect(group: str, task: str, gpus: str, machine: str = "", per_gpu: int | None = None) -> int:
    m = machine_cfg(machine)
    if not sv.is_local(m):
        return run_collect_remote(group, task, gpus, m, per_gpu)
    t = collect_plan(group, task, m["key"])["plan"]
    if not t["episode_num"]:
        return 0
    if per_gpu:
        os.environ["ROBODOJO_PER_GPU"] = str(per_gpu)
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
    _step(group, GROUP_TASK, "已提交并自动批准(入库不审核)")
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


def _eval_cmd(item: dict, task: str, pair: tuple[int, int], target: int, variant: str) -> list[str]:
    """一个模型的 ``robodojo.sh eval`` 命令(本机 / 远端一样)。"""
    cmd = ["bash", "scripts/robodojo.sh", "eval", "--policy-dir", f"XPolicyLab/policy/{item['policy']}",
           "--task", task, "--ckpt", item["ckpt"], "--policy-env", policy_env(item["policy"]),
           "--action-type", policy_action_type(item["policy"]),
           "--env-cfg", EVAL_CFG, "--eval-num", str(target), "--seed", str(EVAL_SEED),
           "--policy-gpu", str(pair[0]), "--env-gpu", str(pair[1])]
    return cmd + (["--variant", variant] if variant else [])


def run_eval(group: str, task: str, gpus: str, policy: str = "", machine: str = "", variant: str = "") -> int:
    m = machine_cfg(machine)
    if not sv.is_local(m):
        return run_eval_remote(group, task, gpus, policy, m, variant)
    queue = eval_plan(group, task, policy, m["key"], variant)["queue"]  # 再确认一次闸门,重新算续评
    target = eval_target(task, variant)
    tag = f"_{variant}" if variant else ""
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
            log = EVAL_LOGS / f"{task}{tag}_{m['policy']}_p{pair[0]}_{datetime.now():%Y%m%d_%H%M%S}.log"
            cmd = _eval_cmd(m, task, pair, target, variant)
            env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", **policy_env_vars(m["policy"])[0]}
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
            runs = eval_runs(task, m["policy"], m["ckpt"], variant)
            n = max((r[1] for r in runs), default=0)
            results.append((m, proc.returncode, n))
            print(f"[group_jobs] {now()} {m['policy']} 结束 exit {proc.returncode},评了 {n}/{target} 条", flush=True)
        time.sleep(5)
    short = [f"{m['policy']} {n}/{target}" for m, _, n in results if n < target]
    done = "、".join(f"{m['policy']} {n} 条" for m, _, n in results)
    if short:
        _step(group, task, f"没评满:{'、'.join(short)}(再点一次续评)")
        return 1
    _step(group, task, f"评完:{done}")
    return 0


# ---------------------------------------------------------------------------
# 远端机器(docs/servers.json):同步 → 远端后台运行 → 轮询 → 拉回
# ---------------------------------------------------------------------------


def _rsync(m: dict, *args: str) -> int:
    return _sh(["rsync", "-a", "--partial", "-e", sv.rsync_rsh(m), *args])


def _rsync_data(m: dict, *args: str) -> int:
    """增量拉分片:不带 ``--partial``——远端还在写的文件传一半也不留副本,下一轮整份重传,不会拿半个文件当续传基础。"""
    return _sh(["rsync", "-a", "-e", sv.rsync_rsh(m), *args])


SHARD_EXCLUDES = ("--exclude", "/logs/", "--exclude", "video_fail/")  # 分片里 worker 日志和失败视频不拉(merge 也不用)


def remote_shards(m: dict, rdir: str, task: str) -> bool:
    """远端这次运行的采集分片还在(``--keep_shards`` 启动,合并后不删)。"""
    out = sv.run(m, f"ls -d {shlex.quote(f'{rdir}/_shards/{task}')}/w* 2>/dev/null | head -1", timeout=60).stdout
    return bool(out.strip())


def remote_seeds(m: dict, rdir: str, task: str) -> set[int]:
    """远端合并好的 ``<rdir>/<task>/<config>/seed.txt``:这批采到的全部 seed(核对分片拉全了没有)。"""
    path = shlex.quote(f"{rdir}/{task}/{rp.CONFIG_NAME}/seed.txt")
    return {int(s) for s in sv.run(m, f"cat {path} 2>/dev/null", timeout=60).stdout.split() if s.isdigit()}


def shard_seeds(root: Path, task: str) -> set[int]:
    """``<root>/w*/<task>/<config>/seed.txt`` 的并集。"""
    seeds: set[int] = set()
    for path in root.glob(f"w*/{task}/{rp.CONFIG_NAME}/seed.txt"):
        seeds.update(_seeds(path))
    return seeds


def sync_code(m: dict) -> None:
    """代码 + 抓取缓存同步到远端仓库(不加 --delete;task/RoboDojo 的相对软链接原样保留)。"""
    dest = f"{sv.target(m)}:{m['repo_path']}/"
    if _rsync(m, "--checksum", "--exclude", "__pycache__", *sv.CODE_DIRS, dest) != 0:
        raise PlanError(f"同步代码到 {m['key']} 失败")
    if (ROOT / "data" / "grasp_pose").is_dir():
        sv.run(m, "mkdir -p data", timeout=30)
        if _rsync(m, "data/grasp_pose", f"{dest}data/") != 0:
            raise PlanError(f"同步 data/grasp_pose 到 {m['key']} 失败")


def remote_launch(m: dict, rdir: str, inner: str) -> int:
    """远端 ``setsid nohup`` 跑 ``inner``(已在 conda 环境里),日志 ``<rdir>.log``,结束写 ``<rdir>/exit_code``。
    已经在跑时不重复启动,返回正在跑的 pid。"""
    body = f"{sv.conda_activate(m)} && export OMNI_KIT_ACCEPT_EULA=YES && {{ {inner} ; }}; echo $? > {shlex.quote(rdir)}/exit_code"
    q = shlex.quote
    script = f"""mkdir -p {q(rdir)}
if [ -f {q(rdir)}.pid ] && kill -0 "$(cat {q(rdir)}.pid)" 2>/dev/null; then echo "PID $(cat {q(rdir)}.pid)"; exit 0; fi
rm -f {q(rdir)}/exit_code
echo "[group_jobs] $(date -Iseconds) start on $(hostname)" >> {q(rdir)}.log
setsid nohup bash -c {q(body)} >> {q(rdir)}.log 2>&1 < /dev/null &
echo $! > {q(rdir)}.pid
echo "PID $!"
"""
    out = sv.run(m, script, timeout=60).stdout
    found = re.search(r"PID (\d+)", out)
    if not found:
        raise PlanError(f"{m['key']} 上启动失败:{out.strip()[-300:]}")
    return int(found.group(1))


def remote_status(m: dict, rdir: str, offset: int, count_cmd: str = "") -> dict:
    """``{alive, exit, size, new, count}``:进程还在不在、退出码、日志新增的最后几十行、``count_cmd`` 的输出。"""
    q = shlex.quote
    log = q(f"{rdir}.log")
    script = f"""pid=$(cat {q(rdir)}.pid 2>/dev/null || echo 0)
if kill -0 "$pid" 2>/dev/null; then echo "@@ALIVE 1"; else echo "@@ALIVE 0"; fi
echo "@@EXIT $(cat {q(rdir)}/exit_code 2>/dev/null)"
n=$(stat -c %s {log} 2>/dev/null || echo 0)
echo "@@SIZE $n"
{('echo "@@COUNT $(' + count_cmd + ')"') if count_cmd else ''}
if [ "$n" -gt {offset} ]; then tail -c +{offset + 1} {log} | head -c $((n - {offset})) | tr '\\r' '\\n' | tail -n 40; fi
"""
    out = sv.run(m, script, timeout=90).stdout
    info = {"alive": False, "exit": None, "size": offset, "count": None, "new": []}
    for line in out.splitlines():
        if line.startswith("@@ALIVE "):
            info["alive"] = line.endswith("1")
        elif line.startswith("@@EXIT "):
            value = line[7:].strip()
            info["exit"] = int(value) if value.lstrip("-").isdigit() else None
        elif line.startswith("@@SIZE "):
            info["size"] = int(line[7:].strip() or offset)
        elif line.startswith("@@COUNT "):
            info["count"] = line[8:].strip()
        else:
            info["new"].append(line)
    return info


def remote_watch(group: str, task: str, m: dict, rdir: str, what: str, count_cmd: str = "", on_count=None,
                 on_tick=None) -> int:
    """每分钟看一次远端,把远端日志新增的部分接到本地日志;返回远端退出码(进程没了又没写退出码时 -1)。
    ``on_tick(count)`` 每轮调一次(采集用它增量拉分片,失败不影响继续看)。
    SSH 偶尔断开不算失败,连着 10 次连不上才放弃(远端照样在跑,再点一次按钮会接上)。"""
    offset, failures = 0, 0
    while True:
        try:
            info = remote_status(m, rdir, offset, count_cmd)
            failures = 0
        except sv.ServerError as error:
            failures += 1
            print(f"[group_jobs] {now()} {error}(第 {failures} 次)", flush=True)
            if failures >= 10:
                raise PlanError(f"连不上 {m['key']},不再等;远端进程不受影响,恢复后再点一次按钮接上") from None
            time.sleep(REMOTE_POLL)
            continue
        if info["new"]:
            print("\n".join(f"  [{m['key']}] {ln}" for ln in info["new"]), flush=True)
        offset = info["size"]
        if on_count and info["count"] is not None:
            on_count(info["count"])
        if on_tick:
            try:
                on_tick(info["count"])
            except (sv.ServerError, OSError) as error:
                print(f"[group_jobs] {now()} 增量拉回跳过:{error}", flush=True)
        if not info["alive"]:
            return info["exit"] if info["exit"] is not None else -1
        update_job(group, task, step=f"{what}({m['key']} 上运行中)")
        time.sleep(REMOTE_POLL)


def run_collect_remote(group: str, task: str, gpus: str, m: dict, per_gpu: int | None) -> int:
    p = collect_plan(group, task, m["key"])  # 再确认一次闸门;有没拉回的就续那一批
    t = p["plan"]
    if not t["episode_num"]:
        return 0
    old = pending_remote_collect(group, task)
    r = dict(old) if old else {"machine": m["key"], "dir": f"{REMOTE_DIR}/{task}_collect_{datetime.now():%Y%m%d_%H%M%S}",
                               "episode_num": t["episode_num"], "start_seed": t["start_seed"] or START_SEED,
                               "template": t["template"], "started_at": now()}
    rdir = r["dir"]
    update_job(group, task, remote=r)
    # 已合并的(<rdir>/<task>/<config>/data)和分片里的取大的:采到一半只有分片,采完两边都有(--keep_shards)
    q = shlex.quote
    count_cmd = (f"a=$(ls {q(f'{rdir}/{task}/{rp.CONFIG_NAME}/data')} 2>/dev/null | grep -c '[.]hdf5$'); "
                 f"b=$(find {q(f'{rdir}/_shards/{task}')} -path '*/{task}/{rp.CONFIG_NAME}/data/*.hdf5' 2>/dev/null"
                 " | grep -c .); echo $(( a > b ? a : b ))")
    info = remote_status(m, rdir, 0) if old else {"alive": False, "exit": None}
    if info["alive"]:
        _step(group, task, f"{m['key']} 上还在采,接着看({rdir})")
    elif info["exit"] == 0:
        _step(group, task, f"{m['key']} 上已采完,直接拉回")
    else:
        _step(group, task, f"同步代码到 {m['key']}")
        sync_code(m)
        n = per_gpu or sv.per_gpu_default(m)
        # --keep_shards:采完远端合并后分片不删,本地才能边采边拉、按分片合并
        cmd = ["bash", "scripts/collect_data_parallel.sh", task, str(r["episode_num"]), gpus,
               "--save_dir", rdir, "--start_seed", str(r["start_seed"]), "--keep_shards"]
        if r.get("template"):
            cmd += ["--layout_template", r["template"]]
        inner = f"export ROBODOJO_PER_GPU={n} && " + " ".join(shlex.quote(c) for c in cmd)
        _step(group, task, f"{m['key']} 上采集 {r['episode_num']} 条(GPU {gpus},每卡 {n} 个 worker)")
        print(f"$ [{m['key']}] {inner}", flush=True)
        r["pid"] = remote_launch(m, rdir, inner)
        update_job(group, task, remote=r)

    # 采集期间就把新完成的分片增量拉回,和采集重叠,别等跑完才整包传(远端出口限速时这一步能省十几分钟)
    staging = PULL_DIR / Path(rdir).name
    shard_dst = staging / "shards"
    shard_src = f"{sv.target(m)}:{m['repo_path']}/{rdir}/_shards/{task}/"
    pulled_upto = {"count": -1}

    def pull_shards(force: bool = False) -> int:
        """增量拉分片:rsync 只传新增 / 变化的文件;采集期间传了一半的文件下一轮整份重传。"""
        count = int(r.get("count") or 0)
        if not force and count <= pulled_upto["count"]:
            return 0
        shard_dst.mkdir(parents=True, exist_ok=True)
        code = _rsync_data(m, *SHARD_EXCLUDES, shard_src, f"{shard_dst}/")
        if code == 0:
            pulled_upto["count"] = count
        return code

    def on_count(value: str) -> None:
        if value.isdigit():
            r["count"] = int(value)
            update_job(group, task, remote=r)

    def on_tick(value: str | None) -> None:
        if value and value.isdigit() and int(value) > 0 and int(value) > pulled_upto["count"] and pull_shards() == 0:
            print(f"[group_jobs] {now()} 已增量拉回 {value} 条分片", flush=True)

    code = remote_watch(group, task, m, rdir, f"采集 {r['episode_num']} 条", count_cmd, on_count, on_tick)
    if code != 0:
        _step(group, task, f"{m['key']} 上没采完(exit {code}):远端分片保留在 {rdir},再点一次、选 {m['key']} 续采")
        return 1
    # 拉回:优先补齐分片(<staging>/shards/w*/<task>/<config>,和整包是同一种 merge_shards 输入);
    # 分片不在(老记录、分片被删)或没拉全时,退回整包拉 <rdir>/<task>[_failed]/ 到 <staging>/w0/
    _step(group, task, f"补齐从 {m['key']} 拉回的分片")
    merge_root = staging
    try:
        if remote_shards(m, rdir, task):
            pull_shards(force=True)
            want = remote_seeds(m, rdir, task)
            missing = sorted(want - shard_seeds(shard_dst, task)) if want else []
            if want and not missing:
                merge_root = shard_dst
            else:
                why = "远端没有合并好的 seed.txt" if not want else f"分片缺 {len(missing)} 条(如 {missing[:3]})"
                _step(group, task, f"{why},整包拉回")
    except sv.ServerError as error:
        _step(group, task, f"分片核对失败({error}),整包拉回")
    if merge_root is staging:
        (staging / "w0").mkdir(parents=True, exist_ok=True)
        _step(group, task, f"从 {m['key']} 拉回数据")
        src = f"{sv.target(m)}:{m['repo_path']}/{rdir}/"
        if _rsync(m, "--include", f"/{task}/***", "--include", f"/{task}_failed/***", "--exclude", "*",
                  src, f"{staging / 'w0'}/") != 0:
            _step(group, task, "拉回失败(远端数据还在);再点一次重新拉")
            return 1
        pulled = len(list((staging / "w0" / task / rp.CONFIG_NAME / "data").glob("*.hdf5")))
        expected = set(_seeds(staging / "w0" / task / rp.CONFIG_NAME / "seed.txt"))
    else:
        pulled = len(list(shard_dst.glob(f"w*/{task}/{rp.CONFIG_NAME}/data/*.hdf5")))
        expected = shard_seeds(shard_dst, task)
    remote_n = int(r.get("count") or 0)
    if remote_n and pulled < remote_n:
        _step(group, task, f"拉回的条数不对:本地 {pulled},远端 {remote_n};再点一次重新拉")
        return 1
    _step(group, task, f"合并 {pulled} 条到本地")
    # --keep_shards:先不删拉回的副本,确认远端的 seed 都进了本地 seed.txt 再删(合并按 seed 去重,重跑安全)
    if _sh([sys.executable, "src/data_collect/merge_shards.py", "--task_name", task, "--shard_root", str(merge_root),
            "--save_dir", f"data/{rp.BENCHMARK_NAME}", "--config_name", rp.CONFIG_NAME, "--keep_shards"]) != 0:
        _step(group, task, f"合并失败:拉回的数据在 {rp.rel(staging)},让 Claude 看一下")
        return 1
    local_seeds = set(_seeds(rp._bench_dir(task) / "seed.txt"))
    lost = [x for x in sorted(expected) if x not in local_seeds]
    if lost:
        _step(group, task, f"合并后本地缺 {len(lost)} 个远端 seed(如 {lost[:3]}),拉回的数据留在 {rp.rel(staging)}")
        return 1
    shutil.rmtree(staging, ignore_errors=True)
    r["merged"] = True
    update_job(group, task, remote=r)
    n = rp._hdf5_count(task)
    if _collected(task):
        _step(group, task, f"合并后 {n}/{rp.COLLECT_TARGET} 条,还不够;再点一次补采")
        return 1
    _step(group, task, f"采够 {n} 条({m['key']} 采的 {pulled} 条已合并;远端数据保留在 {rdir})")
    return 0


def run_eval_remote(group: str, task: str, gpus: str, policy: str, m: dict, variant: str = "") -> int:
    dest = f"{sv.target(m)}:{m['repo_path']}"
    result_rel = rp.rel(EVAL_RESULT / task)
    # 先把远端已有的结果拉回来(上次评完但拉回失败的、远端评得更多的);--update:本地更新的不盖
    _step(group, task, f"先拉回 {m['key']} 上已有的评测结果")
    (EVAL_RESULT / task).mkdir(parents=True, exist_ok=True)
    if sv.run(m, f"test -d {shlex.quote(result_rel)}", timeout=30).returncode == 0 \
            and _rsync(m, "--update", f"{dest}/{result_rel}/", f"{EVAL_RESULT / task}/") != 0:
        raise PlanError(f"从 {m['key']} 拉回已有评测结果失败,没开评(避免把远端更全的结果当成没评)")
    plan = eval_plan(group, task, policy, m["key"], variant)  # 再确认一次闸门(含远端模型检查),重新算续评
    queue, missing_assets = plan["queue"], plan.get("missing_assets") or []
    target = eval_target(task, variant)
    tag = f"_{variant}" if variant else ""
    if not queue:
        _step(group, task, f"{m['key']} 上的结果拉回后已评满,不用再跑")
        return 0
    rdir = f"{REMOTE_DIR}/{task}_eval_{datetime.now():%Y%m%d_%H%M%S}"
    r = {"machine": m["key"], "dir": rdir, "started_at": now(),
         "counts": {item["policy"]: 0 for item in queue}}
    update_job(group, task, remote=r)
    _step(group, task, f"同步代码 + 评测布局到 {m['key']}")
    sync_code(m)
    layout_rel = rp.rel(EVAL_LAYOUT)
    sv.run(m, f"mkdir -p {shlex.quote(layout_rel)} {shlex.quote(rp.rel(EVAL_LOGS))}", timeout=30)
    # <task>_*.json 也包括 S1 变体的 <task>__<variant>_*.json
    if _rsync(m, "--include", f"{task}_*.json", "--exclude", "*", f"{EVAL_LAYOUT}/", f"{dest}/{layout_rel}/") != 0:
        raise PlanError(f"同步评测布局到 {m['key']} 失败")
    if missing_assets:  # 缺的资产 + 它们引用到的原件;传完把写死的本机绝对路径改成远端仓库路径
        _step(group, task, f"同步 {len(missing_assets)} 个缺少的资产到 {m['key']}")
        sync_assets(m, missing_assets)
    for item in queue:  # 要续评的结果 + 续评清单先推过去(ROBODOJO_RUN_ID 接着同一个目录评;main.py 靠清单续)
        if item["run_id"]:
            base = eval_result_base(task, item["policy"], item["ckpt"], variant)
            rel = rp.rel(base / item["run_id"])
            manifest = eval_manifest(task, item["policy"], item["ckpt"], item["run_id"], variant)
            sv.run(m, f"mkdir -p {shlex.quote(rel)}", timeout=30)
            # --update:远端比本地新的文件不盖(远端可能评得更多)
            if (_rsync(m, "--update", f"{base / item['run_id']}/", f"{dest}/{rel}/") != 0
                    or _rsync(m, "--update", str(manifest), f"{dest}/{rp.rel(manifest)}") != 0):
                raise PlanError(f"推送续评结果 {rel} 到 {m['key']} 失败")
    ids = [int(x) for x in gpus.split(",") if x.strip()]
    pairs = [(ids[i], ids[i + 1] if i + 1 < len(ids) else ids[i]) for i in range(0, len(ids), 2)] or [(0, 0)]
    lanes: list[list[str]] = [[] for _ in pairs]  # 每对卡一条流水线,模型轮流分到各条上,同一条上依次评
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    for i, item in enumerate(queue):
        pair = pairs[i % len(pairs)]
        cmd = _eval_cmd(item, task, pair, target, variant)
        log = f"{rp.rel(EVAL_LOGS)}/{task}{tag}_{item['policy']}_p{pair[0]}_{stamp}.log"
        run_id = f"ROBODOJO_RUN_ID={shlex.quote(item['run_id'])} " if item["run_id"] else ""
        run_id += "".join(f"{k}={shlex.quote(v)} " for k, v in policy_env_vars(item["policy"], m)[0].items())
        lanes[i % len(pairs)].append(f"{run_id}{' '.join(shlex.quote(c) for c in cmd)} > {shlex.quote(log)} 2>&1"
                                     f" || fail=1")
    body = "fail=0; " + " ".join(f"( {'; '.join(lane)}; exit $fail ) & sleep 8;" for lane in lanes if lane)
    inner = body + " ok=0; for j in $(jobs -p); do wait $j || ok=1; done; (exit $ok)"  # 不能 exit:后面要写退出码
    _step(group, task, f"{m['key']} 上评测 {len(queue)} 个模型({len(pairs)} 条并行)")
    print(f"$ [{m['key']}] {inner}", flush=True)
    r["pid"] = remote_launch(m, rdir, inner)
    update_job(group, task, remote=r)
    # 进度:远端各模型最新一次 _result.json 的 eval_time
    probes = []
    for item in queue:
        base = rp.rel(eval_result_base(task, item["policy"], item["ckpt"], variant))
        probes.append(f"echo -n \"{item['policy']}=$(cat {shlex.quote(base)}/*/_result.json 2>/dev/null"
                      " | grep -o '\"eval_time\": *[0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1) \"")
    count_cmd = "; ".join(probes)

    def on_count(value: str) -> None:
        for part in value.split():
            name, _, n = part.partition("=")
            if name in r["counts"] and n.isdigit():
                r["counts"][name] = int(n)
        update_job(group, task, remote=r)

    code = remote_watch(group, task, m, rdir, f"评测 {len(queue)} 个模型", count_cmd, on_count)
    _step(group, task, f"从 {m['key']} 拉回评测结果和日志")
    (EVAL_RESULT / task).mkdir(parents=True, exist_ok=True)
    pulled = _rsync(m, f"{dest}/{result_rel}/", f"{EVAL_RESULT / task}/")
    _rsync(m, "--include", f"{task}{tag}_*_{stamp}.log", "--exclude", "*", f"{dest}/{rp.rel(EVAL_LOGS)}/", f"{EVAL_LOGS}/")
    if pulled != 0:
        _step(group, task, f"拉回评测结果失败(远端结果还在 {result_rel});再点一次会重新拉")
        return 1
    counts = {item["policy"]: max((n for _, n in eval_runs(task, item["policy"], item["ckpt"], variant)), default=0)
              for item in queue}
    short = [f"{p} {n}/{target}" for p, n in counts.items() if n < target]
    if short or code != 0:
        _step(group, task, f"没评满:{'、'.join(short) or '(远端 exit ' + str(code) + ')'};再点一次续评")
        return 1
    _step(group, task, "评完:" + "、".join(f"{p} {n} 条" for p, n in counts.items())
          + f"({m['key']})")
    return 0


def run(kind: str, group: str, task: str, gpus: str = "", delete: bool = False, policy: str = "",
        machine: str = "", per_gpu: int | None = None, variant: str = "") -> int:
    global _RUN_POLICY, _RUN_VARIANT
    _RUN_POLICY = policy if kind == "eval" else ""
    _RUN_VARIANT = norm_variant(variant) if kind == "eval" else ""
    update_job(group, task, pid=os.getpid())
    code = 1
    try:
        if kind in QUEUED_KINDS:
            _turn = _wait_turn(group, task, kind)  # noqa: F841 - 锁要拿到进程结束
        if kind == "ingest":
            code = run_ingest(group)
        elif kind == "collect":
            code = run_collect(group, task, gpus, machine, per_gpu)
        elif kind == "export":
            code = run_export(group, task)
        elif kind == "eval":
            code = run_eval(group, task, gpus, policy, machine, _RUN_VARIANT)
        elif kind == "layouts":
            code = run_layouts(group, task, gpus)
        else:
            code = run_upload(group, task, delete)
    except (PlanError, sv.ServerError) as error:
        print(f"[group_jobs] {error}", flush=True)
        update_job(group, task, step=str(error).splitlines()[0][:200])
    finally:
        job = read_job(job_id(group, task, _RUN_POLICY, _RUN_VARIANT)) or {}
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
        s.add_argument("--policy", default="", help="eval 时只评这个模型(Pi_0 / Pi_05);默认都评")
        s.add_argument("--variant", default="", help="eval 时评这个 S1 变体(S1.1 …);默认 S1.0(原任务)")
        s.add_argument("--machine", default="", help="collect / eval 在哪台机器上跑(docs/servers.json 的键;默认 default_machine)")
        if name == "start":
            s.add_argument("--gpus", default="")
            s.add_argument("--delete", action="store_true", help="上传时删除远端多余文件(默认保留)")
            s.add_argument("--per_gpu", type=int, default=None, help="collect:每卡几个 worker")
    s = sub.add_parser("run", help="内部用:后台会话里执行")
    s.add_argument("kind", choices=("ingest", "collect", "export", "upload", "eval", "layouts"))
    s.add_argument("group")
    s.add_argument("task")
    s.add_argument("--gpus", default="")
    s.add_argument("--delete", action="store_true")
    s.add_argument("--policy", default="")
    s.add_argument("--variant", default="")
    s.add_argument("--machine", default="")
    s.add_argument("--per_gpu", type=int, default=None)
    sub.add_parser("status")
    s = sub.add_parser("stop")
    s.add_argument("group")
    s.add_argument("task")
    s.add_argument("--policy", default="", help="停这个模型的评测(按模型的评测记录)")
    s.add_argument("--variant", default="", help="停这个 S1 变体的评测")
    a = ap.parse_args()
    try:
        if a.cmd == "plan":
            print(plan(a.button, a.group, a.task, policy=a.policy, machine=a.machine, variant=a.variant)["text"])
        elif a.cmd == "start":
            job = start(a.button, a.group, a.task, a.gpus, a.delete, a.policy, a.machine, a.per_gpu, a.variant)
            print(f"已启动 {job_label(job)} {a.group}/{a.task}:pid {job['pid']},日志 {job['log']}")
        elif a.cmd == "run":
            return run(a.kind, a.group, a.task, a.gpus, a.delete, a.policy, a.machine, a.per_gpu, a.variant)
        elif a.cmd == "status":
            for jid, j in all_jobs(tail=False).items():
                where = f" @{j['machine']}" if j.get("machine") else ""
                print(f"{jid}: {j['label']}{where} {j['state']} · {j.get('step')} · {j.get('started_at')} · {j.get('log')}")
        else:
            print(stop(a.group, a.task, a.policy, a.variant))
    except PlanError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
