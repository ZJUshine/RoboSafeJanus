#!/usr/bin/env python3
"""``docs/servers.json`` 里的机器:看板的「采集」「评测」先让用户选在哪台机器上跑(``group_jobs``)。

``role: local`` 的机器(仓库所在的这台)直接本地运行;``role: remote`` 的机器走 SSH(``BatchMode``,要先配好免密):
同步代码(``rsync``,不加 ``--delete``)→ 远端 ``setsid nohup`` 运行 → 轮询日志 → 跑完再把结果拉回本地。
远端机器要在 servers.json 里写全 ``ssh``(user / host / port)、``repo_path``、``conda_root``、``conda_env.name``,
否则在选择框里灰掉。模型不自动同步(缺的在计划里列出来)。评测前会把该任务缺的资产自动同步过去并改写资产里
写死的本机绝对路径;采集不自动同步,远端缺资产时计划里列出来,先按 ``docs/data_collection.md`` 8.6 节准备好。

命令行(调试用)::

    python scripts/internal/servers.py list            # 机器列表、能不能用
    python scripts/internal/servers.py gpus autodl-A18 # 远端 GPU 显存占用
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SERVERS_JSON = ROOT / "docs" / "servers.json"
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6")
# 采集每个 worker 实测 14–20 GB 显存(reference-collect-servers);按每卡显存估每卡几个 worker
WORKER_GIB = 20
CODE_DIRS = ("env", "env_cfg", "task", "src", "utils", "scripts")  # 采集 / 评测要同步过去的代码(RoboDojo 自有部分)


class ServerError(Exception):
    """机器配置不全、连不上或远端命令失败;消息直接给用户看。"""


def load() -> dict:
    try:
        return json.loads(SERVERS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ServerError(f"读不了 {SERVERS_JSON.relative_to(ROOT)}:{error}") from None


def default_key() -> str:
    data = load()
    return data.get("default_machine") or next(iter(data.get("machines") or {}), "")


def get(key: str) -> dict:
    machines = load().get("machines") or {}
    if key not in machines:
        raise ServerError(f"servers.json 里没有机器 {key!r}(有:{'、'.join(machines)})")
    return {"key": key, **machines[key]}


def is_local(m: dict) -> bool:
    return m.get("role") == "local"


def problems(m: dict) -> list[str]:
    """远端机器还缺哪些配置(为空表示能用)。"""
    if is_local(m):
        return []
    ssh = m.get("ssh") or {}
    out = [f"ssh.{k}" for k in ("user", "host", "port") if not ssh.get(k)]
    out += [k for k in ("repo_path", "conda_root") if not m.get(k)]
    if not (m.get("conda_env") or {}).get("name"):
        out.append("conda_env.name")
    return [f"servers.json 里没写 {', '.join(out)}"] if out else []


def gpu_summary(m: dict) -> str:
    parts = []
    for g in m.get("gpus") or []:
        mem = f" {g['memory_gib_per_gpu']}G" if g.get("memory_gib_per_gpu") else ""
        parts.append(f"{g.get('count') or '?'}× {g.get('model') or '?'}{mem}")
    return "、".join(parts) or "GPU 未知"


def per_gpu_default(m: dict) -> int:
    """每卡几个采集 worker 的默认值:显存 / 20 GB,1..4;显存未知时 1。"""
    mem = max((g.get("memory_gib_per_gpu") or 0 for g in m.get("gpus") or []), default=0)
    return max(1, min(4, int(mem // WORKER_GIB))) if mem else 1


def machines() -> list[dict]:
    """选择框用:``[{key, label, local, usable, why, gpus, per_gpu, default}]``。"""
    data = load()
    default = data.get("default_machine")
    out = []
    for key, raw in (data.get("machines") or {}).items():
        m = {"key": key, **raw}
        why = problems(m)
        out.append({"key": key, "label": f"{key}({'本机' if is_local(m) else '远端'})",
                    "description": m.get("description") or "", "local": is_local(m), "usable": not why,
                    "why": "; ".join(why), "gpus": gpu_summary(m), "per_gpu": per_gpu_default(m),
                    "default": key == default})
    return out


def target(m: dict) -> str:
    ssh = m["ssh"]
    return f"{ssh['user']}@{ssh['host']}"


def ssh_argv(m: dict) -> list[str]:
    return ["ssh", *SSH_OPTS, "-p", str(m["ssh"]["port"]), target(m)]


def rsync_rsh(m: dict) -> str:
    return " ".join(["ssh", *SSH_OPTS, "-p", str(m["ssh"]["port"])])


def conda_activate(m: dict) -> str:
    return (f"source {shlex.quote(m['conda_root'])}/etc/profile.d/conda.sh && "
            f"conda activate {shlex.quote(m['conda_env']['name'])}")


def run(m: dict, script: str, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess:
    """在远端用 bash 跑一段脚本(从 stdin 传,不用操心引号),工作目录是 ``repo_path``。"""
    body = f"set -o pipefail\ncd {shlex.quote(m['repo_path'])} || exit 97\n{script}\n"
    try:
        proc = subprocess.run([*ssh_argv(m), "bash", "-s"], input=body, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ServerError(f"{m['key']}:SSH 超时({timeout} 秒)") from None
    if check and proc.returncode == 255:
        raise ServerError(f"{m['key']}:连不上({(proc.stderr or '').strip().splitlines()[-1:] or ['ssh 失败']})")
    if check and proc.returncode == 97:
        raise ServerError(f"{m['key']}:远端没有仓库目录 {m['repo_path']}")
    return proc


def gpu_usage(m: dict) -> list[dict]:
    """``[{index, used, total, util}]``(MiB);和 group_jobs.gpu_usage 同样的格式。"""
    proc = run(m, "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                  "--format=csv,noheader,nounits", timeout=40)
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "used": int(parts[1]), "total": int(parts[2]), "util": parts[3]})
    if not gpus:
        raise ServerError(f"{m['key']}:nvidia-smi 没有输出({(proc.stderr or '').strip()[-200:]})")
    return gpus


def missing_paths(m: dict, paths: list[str]) -> list[str]:
    """远端 ``repo_path`` 下不存在的路径(相对仓库根)。"""
    if not paths:
        return []
    script = "for p in " + " ".join(shlex.quote(p) for p in paths) + '; do [ -e "$p" ] || echo "$p"; done'
    return [ln for ln in run(m, script, timeout=60).stdout.splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("gpus").add_argument("machine")
    a = ap.parse_args()
    try:
        if a.cmd == "list":
            for m in machines():
                print(f"{m['label']:<24} {'可用' if m['usable'] else '不可用:' + m['why']} · {m['gpus']}"
                      f" · 每卡 {m['per_gpu']} 个采集 worker{' · 默认' if m['default'] else ''}")
        else:
            for g in gpu_usage(get(a.machine)):
                print(f"GPU {g['index']}: {g['used']}/{g['total']} MiB, util {g['util']}%")
    except ServerError as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
