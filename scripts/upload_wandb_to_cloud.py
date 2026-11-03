"""把本地下载的 wandb run 目录上传到 W&B 官方(``wandb.ai``)。

配合 ``scripts/download_wandb_from_modelscope.py`` 使用:先从 ModelScope 模型仓库把
``<POLICY>/<task>/<run>/wandb/<run-dir>/`` 下载到 ``data/RoboSafeJanus-Wandb``,再运行本脚本执行
``wandb sync``。训练时是 ``run-*``(在线模式目录),新版 CLI 默认跳过,需要 ``--include-online``。

凭证(任一):``WANDB_API_KEY`` 环境变量;或 ``data/.cache/wandb/api_key`` 文件;
或 ``--api_key_file``。entity 默认用 key 所属账号,可用 ``--entity`` / ``WANDB_ENTITY`` 覆盖。

上传状态记在 ``data/.cache/wandb/synced.json``(不进 run 目录,避免污染数据),用于跳过已上传的 run
(``--all`` 重传,``--force`` 给 wandb sync 加 ``--include-synced``)。

本机没装 wandb 时,脚本会用 ``uv`` 在 ``data/.cache/wandb/venv`` 建一个专用环境
(``--venv_dir`` 可改),或退回 ``uvx --from wandb wandb``;也可用 ``WANDB_CLI`` 指定命令。

用法::

    python scripts/upload_wandb_to_cloud.py --list                 # 只看本地有哪些 run、哪些已传
    python scripts/upload_wandb_to_cloud.py --dry_run              # 传什么,不真的上传
    python scripts/upload_wandb_to_cloud.py
    python scripts/upload_wandb_to_cloud.py --policy Pi_05 --entity my-team
"""

import argparse
from datetime import datetime
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^(?:offline-)?run-\d{8}_\d{6}-[A-Za-z0-9]+$")
URL_RE = re.compile(r"https?://[^\s]*(?:/runs/|/r/)[A-Za-z0-9]+")
BRACKET_RE = re.compile(r"\[([^/\s\]]+)/([^/\s\]]+)/([A-Za-z0-9]+)\]")

parser = argparse.ArgumentParser()
parser.add_argument("--local_dir", default=str(REPO_ROOT / "data" / "RoboSafeJanus-Wandb"),
                    help="download_wandb_from_modelscope.py 的 --local_dir")
parser.add_argument("--policy", nargs="*", default=["*"], help="策略名 fnmatch 模式")
parser.add_argument("--task", nargs="*", default=["*"], help="任务/组模型目录名 fnmatch 模式")
parser.add_argument("--run", nargs="*", default=["*"], help="run 目录名 fnmatch 模式")
parser.add_argument("--path", nargs="*", default=None,
                    help="run 前缀(相对 local_dir,形如 Pi_05/<task>/<run>/wandb/<run-dir>)")
parser.add_argument("--only_missing", action="store_true", default=True,
                    help="只上传还没上传过的(默认行为);--all 关闭")
parser.add_argument("--all", dest="only_missing", action="store_false", help="重传所有本地 run")
parser.add_argument("--force", action="store_true", help="给 wandb sync 加 --include-synced")
parser.add_argument("--dry_run", action="store_true", help="只打印将要执行的命令")
parser.add_argument("--list", action="store_true", help="只列出本地 run 和上传状态")
parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                    help="覆盖 entity;默认用 API key 所属账号")
parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"), help="覆盖 project(默认用 run 自带)")
parser.add_argument("--base_url", default=os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
                    help="W&B API 地址;默认官方 https://api.wandb.ai"
                         "(训练 run 里可能记着别的部署地址,必须显式覆盖)")
parser.add_argument("--proxy", default=os.environ.get("WANDB_PROXY"),
                    help="上传走 HTTP(S) 代理,如 http://127.0.0.1:7899;默认 WANDB_PROXY")
parser.add_argument("--api_key_file", default=None, help="默认 data/.cache/wandb/api_key")
parser.add_argument("--state_file", default=None, help="默认 data/.cache/wandb/synced.json")
parser.add_argument("--venv_dir", default=None, help="wandb 专用 venv;默认 data/.cache/wandb/venv")
parser.add_argument("--timeout", type=int, default=3 * 3600, help="单个 run 上传超时(秒)")
args = parser.parse_args()

local_dir = Path(args.local_dir).expanduser().resolve()
cache_dir = local_dir.parent / ".cache" / "wandb"
api_key_file = Path(args.api_key_file).expanduser() if args.api_key_file else cache_dir / "api_key"
state_file = Path(args.state_file).expanduser() if args.state_file else cache_dir / "synced.json"
venv_dir = Path(args.venv_dir).expanduser() if args.venv_dir else cache_dir / "venv"


def load_state() -> dict:
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def collect_runs() -> list[dict]:
    runs = []
    if not local_dir.is_dir():
        return runs
    for candidate in sorted(local_dir.glob("*/*/*/wandb/*")):
        if not candidate.is_dir() or not RUN_RE.match(candidate.name):
            continue
        policy, task, run = candidate.relative_to(local_dir).parts[:3]
        if not (matches(policy, args.policy) and matches(task, args.task) and matches(run, args.run)):
            continue
        rel = candidate.relative_to(local_dir).as_posix()
        if args.path and not any(rel == p.rstrip("/") or rel.startswith(p.rstrip("/") + "/") for p in args.path):
            continue
        files = [path for path in candidate.rglob("*") if path.is_file()]
        runs.append({
            "policy": policy,
            "task": task,
            "run": run,
            "run_dir": rel,
            "path": str(candidate),
            "name": candidate.name,
            "mode": "offline" if candidate.name.startswith("offline-run-") else "online",
            "files": len(files),
            "size": sum(path.stat().st_size for path in files),
            "has_wandb_file": any(path.suffix == ".wandb" for path in files),
        })
    return runs


def find_wandb() -> tuple[list[str], str]:
    """返回 (命令前缀, 说明)。"""
    override = os.environ.get("WANDB_CLI")
    if override:
        return override.split(), f"WANDB_CLI={override}"
    exe = shutil.which("wandb")
    if exe:
        return [exe], exe
    try:
        import wandb  # noqa: F401
    except ImportError:
        pass
    else:
        return [sys.executable, "-m", "wandb"], f"{sys.executable} -m wandb"
    target = venv_dir / "bin" / "wandb"
    if target.is_file():
        return [str(target)], str(target)
    uv = shutil.which("uv")
    if not uv:
        return [], "没找到 wandb CLI,也没有 uv;请先 pip install wandb"
    if args.dry_run:
        return [], f"会用 uv 在 {venv_dir} 安装 wandb(首次约 1 分钟)"
    print(f"[upload_wandb] installing wandb into {venv_dir} (first run only)")
    subprocess.check_call([uv, "venv", str(venv_dir)])
    subprocess.check_call([uv, "pip", "install", "--python", str(venv_dir / "bin" / "python"), "wandb"])
    return [str(target)], str(target)


def supports_online(cmd: list[str]) -> bool:
    try:
        result = subprocess.run([*cmd, "sync", "--help"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return True
    text = (result.stdout or "") + (result.stderr or "")
    return "--include-online" in text


def api_key() -> str | None:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    if api_key_file.is_file():
        key = api_key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return None


def run_url(output: str, base_url: str, run_dir: str) -> tuple[str, str]:
    """(官方 run URL, 输出里其它部署的 URL)。

    ``wandb sync`` 会按 run 文件里记的旧部署地址打印 "View run at ..."(训练机可能用的是
    CoreWeave 自建 W&B),所以优先取和 ``--base_url`` 同域名的地址,否则用输出里的
    ``[entity/project/run_id]`` 拼官方地址。
    """
    host = urlparse(base_url).netloc
    urls = URL_RE.findall(output)
    for url in urls:
        if urlparse(url).netloc == host:
            return url, ""
    match = BRACKET_RE.search(output)
    source = urls[0] if urls else ""
    if not match:
        return source, ""
    ui_host = host[4:] if host.startswith("api.") else host
    return f"https://{ui_host}/{match.group(1)}/{match.group(2)}/runs/{match.group(3)}", source


runs = collect_runs()
if not runs:
    sys.exit(f"[upload_wandb] no local wandb runs under {local_dir}")

state = load_state()
for item in runs:
    item["uploaded"] = state.get(item["run_dir"], {}).get("synced_at")
    item["url"] = state.get(item["run_dir"], {}).get("url", "")

if args.list:
    for item in runs:
        status = f"✅ {item['uploaded']}" if item["uploaded"] else "· 未上传"
        print(f"  {item['run_dir']}  ({item['files']} files, {item['size'] / 1e6:.1f} MB, "
              f"{item['mode']}) {status}")
    print(f"[upload_wandb] {len(runs)} runs, {sum(1 for item in runs if item['uploaded'])} uploaded")
    sys.exit(0)

selected = [item for item in runs if not (args.only_missing and item["uploaded"])]
if not selected:
    sys.exit("[upload_wandb] nothing to upload (all runs already uploaded; use --all to resend)")

key = api_key()
if not key:
    sys.exit("[upload_wandb] missing WANDB_API_KEY: export WANDB_API_KEY=... "
             f'or write it to {api_key_file}')
if not all(item["has_wandb_file"] for item in selected):
    broken = [item["run_dir"] for item in selected if not item["has_wandb_file"]]
    sys.exit(f"[upload_wandb] run dirs without .wandb file (download incomplete?): {', '.join(broken[:5])}")

cmd, source = find_wandb()
print(f"[upload_wandb] wandb CLI: {source or 'not found'}")
print(f"[upload_wandb] {len(selected)}/{len(runs)} runs -> {args.entity or 'default entity'}")
if args.entity:
    os.environ.setdefault("WANDB_ENTITY", args.entity)
if not cmd:
    sys.exit(f"[upload_wandb] {source}")

env = dict(os.environ, WANDB_API_KEY=key, WANDB_BASE_URL=args.base_url)
if args.proxy:
    env.update({"HTTP_PROXY": args.proxy, "HTTPS_PROXY": args.proxy, "ALL_PROXY": args.proxy})
online_flag = supports_online(cmd)
print(f"[upload_wandb] target: {args.base_url}" + (f" via {args.proxy}" if args.proxy else ""))

failed = []
for index, item in enumerate(selected, 1):
    command = [*cmd, "sync", "--yes"]
    if online_flag:
        command.append("--include-online")
    if args.force:
        command.append("--include-synced")
    if args.entity:
        command += ["--entity", args.entity]
    if args.project:
        command += ["--project", args.project]
    command.append(item["path"])
    print(f"\n===== [{index}/{len(selected)}] {item['run_dir']} ({item['size'] / 1e6:.1f} MB) =====")
    print("  " + " ".join(command))
    if args.dry_run:
        continue
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        failed.append(item["run_dir"])
        print(f"[upload_wandb] timeout after {args.timeout}s")
        continue
    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(output.strip().splitlines()[-12:])
    print(tail)
    if result.returncode != 0:
        failed.append(item["run_dir"])
        print(f"[upload_wandb] FAILED: exit {result.returncode}")
        continue
    if "wandb: ERROR" in output or re.search(r"^Error:", output, re.M):
        failed.append(item["run_dir"])
        print("[upload_wandb] FAILED: wandb reported an error (exit code was 0)")
        continue
    match_url, source_url = run_url(output, args.base_url, item["run_dir"])
    state[item["run_dir"]] = {
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "url": match_url,
        "base_url": args.base_url,
        "source_url": source_url,
    }
    save_state(state)
    if not match_url:
        print("[upload_wandb] warning: synced but no run URL in the output")
    print(f"[upload_wandb] OK {match_url}".rstrip())

if args.dry_run:
    print(f"[upload_wandb] dry-run: {len(selected)} runs would be uploaded")
    sys.exit(0)
if failed:
    sys.exit(f"[upload_wandb] failed {len(failed)} runs: {', '.join(failed[:5])}")
print(f"[upload_wandb] uploaded {len(selected)} runs; state -> {state_file}")
