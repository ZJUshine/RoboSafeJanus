"""把本地下载的 wandb run 目录上传到 W&B 官方(``wandb.ai``)。

配合 ``scripts/download_wandb_from_modelscope.py`` 使用:先从 ModelScope 模型仓库把
``<POLICY>/<task>/<run>/wandb/<run-dir>/`` 下载到 ``data/RoboSafeJanus-Wandb``,再运行本脚本执行
``wandb sync``。训练时是 ``run-*``(在线模式目录),新版 CLI 默认跳过,需要 ``--include-online``。

凭证(任一):``WANDB_API_KEY`` 环境变量;或 ``data/.cache/wandb/api_key`` 文件;
或 ``--api_key_file``。entity 默认用 key 所属账号,可用 ``--entity`` / ``WANDB_ENTITY`` 覆盖。

上传状态记在 ``data/.cache/wandb/synced.json``(不进 run 目录,避免污染数据),用于跳过已上传的 run
(``--all`` 重传,``--force`` 给 wandb sync 加 ``--include-synced``)。W&B 上删过的 run id 留墓碑,
``wandb sync`` 不能再用同一个 id 建 run:遇到这种情况自动换一个新 id 重传一次,并把新 id 记进状态文件
(``upload_id``),以后重同步继续用它。

本机没装 wandb 时,脚本会用 ``uv`` 在 ``data/.cache/wandb/venv`` 建一个专用环境
(``--venv_dir`` 可改),或退回 ``uvx --from wandb wandb``;也可用 ``WANDB_CLI`` 指定命令。

用法::

    python scripts/upload_wandb_to_cloud.py --list                 # 只看本地有哪些 run、哪些已传
    python scripts/upload_wandb_to_cloud.py --dry_run              # 传什么,不真的上传
    python scripts/upload_wandb_to_cloud.py
    python scripts/upload_wandb_to_cloud.py --policy Pi_05 --entity my-team
"""

import argparse
import base64
from datetime import datetime
import fnmatch
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import string
import subprocess
import sys
from urllib.parse import urlparse
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^(?:offline-)?run-\d{8}_\d{6}-[A-Za-z0-9]+$")
# 训练机写进 run 文件的 project 名不一定和官方仓库统一:同一个 run 续训后 project 可能改名
# (LingBot 的续训 run 自带 ``LingBotVLA``,仓库里统一叫 ``RoboSafeJanus-lingbot-vla``),不覆盖就会
# 把续训那段传到另一个 project,曲线被拆成两半。没给 ``--project`` 时按 policy 用这里的默认值。
PROJECT_BY_POLICY = {"LingBot_VLA": "RoboSafeJanus-lingbot-vla"}
URL_RE = re.compile(r"https?://[^\s]*(?:/runs/|/r/)[A-Za-z0-9]+")
BRACKET_RE = re.compile(r"\[([^/\s\]]+)/([^/\s\]]+)/([A-Za-z0-9]+)\]")
# 在 W&B 上删过的 run id 会留墓碑,``wandb sync`` 再建同一个 id 会 410:
# ``run <id> was previously created and deleted; try a new run id``
TOMBSTONE_RE = re.compile(r"run ([A-Za-z0-9]+) was previously created and deleted")

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
parser.add_argument("--proxy", default=os.environ.get("WANDB_PROXY", "http://127.0.0.1:49623") or None,
                    help="上传走 HTTP(S) 代理;默认 WANDB_PROXY,没设时用本机 clash 混合端口 "
                         "http://127.0.0.1:49623(WANDB_PROXY='' 表示不走代理)")
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


def same_run(a: dict, b: dict) -> bool:
    """同一个 W&B run 的不同段(训练中断续训):同一个训练目录,目录名最后的 run id 一样。"""
    return (a["policy"], a["task"], a["run"]) == (b["policy"], b["task"], b["run"]) and \
        a["name"].rsplit("-", 1)[-1] == b["name"].rsplit("-", 1)[-1]


def wandb_bytes(path: str) -> int:
    """run 目录里 ``.wandb`` 文件的大小:记进状态文件,远端换成新文件(大小变了)时看板会提示重新同步。"""
    return sum(f.stat().st_size for f in Path(path).glob("*.wandb"))


def wandb_query(text: str, key: str) -> dict:
    """W&B GraphQL 只读查询(走 --proxy)。"""
    handlers = [urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})] if args.proxy else []
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/graphql", data=json.dumps({"query": text}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()})
    with urllib.request.build_opener(*handlers).open(request, timeout=30) as response:
        return json.loads(response.read()).get("data") or {}


def uploaded_run(output: str, key: str) -> tuple[str, int | None]:
    """上传后官方 W&B 上这个 run 的 ``(网页地址, summary 里的 _step)``;查不到时 ``("", None)``。

    ``wandb sync`` 打印的是 ``[project/run_id]`` 和训练机部署(如 CoreWeave)的地址,entity 要另查。"""
    match = re.search(r"\[([^/\s\]]+)/([A-Za-z0-9]+)\] setting up run", output)
    if not match:
        return "", None
    project, run_id = match.groups()
    try:
        entity = args.entity or wandb_query("{ viewer { entity } }", key)["viewer"]["entity"]
        host = urlparse(args.base_url).netloc
        url = f"https://{host[4:] if host.startswith('api.') else host}/{entity}/{project}/runs/{run_id}"
        run = wandb_query(f'{{ project(name: "{project}", entityName: "{entity}") '
                          f'{{ run(name: "{run_id}") {{ summaryMetrics }} }} }}', key)["project"]["run"]
        if not run:
            return "", None
        return url, json.loads(run.get("summaryMetrics") or "{}").get("_step")
    except (OSError, ValueError, KeyError, TypeError):
        return "", None


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
if not args.project:
    mapped = sorted({PROJECT_BY_POLICY[i["policy"]] for i in selected if i["policy"] in PROJECT_BY_POLICY})
    if mapped:
        print(f"[upload_wandb] project 按 policy 指定(训练机自带的 project 名不统一): {', '.join(mapped)}")

def sync_command(item: dict, run_id: str = "") -> list[str]:
    command = [*cmd, "sync", "--yes"]
    if online_flag:
        command.append("--include-online")
    if args.force:
        command.append("--include-synced")
    if args.entity:
        command += ["--entity", args.entity]
    project = args.project or PROJECT_BY_POLICY.get(item["policy"])
    if project:
        command += ["--project", project]
    if run_id:
        command += ["--id", run_id]
    command.append(item["path"])
    return command


def run_sync(command: list[str]) -> tuple[int, str] | None:
    """跑一次 ``wandb sync``;超时返回 None。"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    output = (result.stdout or "") + (result.stderr or "")
    print("\n".join(output.strip().splitlines()[-12:]))
    return result.returncode, output


def unused_run_id() -> str:
    """新的 run id(8 位小写字母+数字),避开本地 run 目录名和状态文件里已经用过的 id。"""
    taken = {r["name"].rsplit("-", 1)[-1] for r in runs}
    taken |= {s.get("upload_id", "") for s in state.values()}
    while True:
        candidate = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        if candidate not in taken:
            return candidate


failed, partial = [], []
for index, item in enumerate(selected, 1):
    saved = state.get(item["run_dir"]) or {}
    run_id = saved.get("upload_id", "")  # 被删过、换过 id 的 run 后续重同步继续用同一个新 id
    command = sync_command(item, run_id)
    print(f"\n===== [{index}/{len(selected)}] {item['run_dir']} ({item['size'] / 1e6:.1f} MB) =====")
    print("  " + " ".join(command))
    if args.dry_run:
        continue
    attempt = run_sync(command)
    if attempt is None:
        failed.append(item["run_dir"])
        print(f"[upload_wandb] timeout after {args.timeout}s")
        continue
    returncode, output = attempt
    tombstone = TOMBSTONE_RE.search(output)
    if tombstone and (not run_id or tombstone.group(1) == run_id):
        run_id = unused_run_id()
        print(f"[upload_wandb] run {tombstone.group(1)} 在 W&B 上被删过(id 留了墓碑,不能重建),"
              f"换新 id {run_id} 重传")
        command = sync_command(item, run_id)
        print("  " + " ".join(command))
        attempt = run_sync(command)
        if attempt is None:
            failed.append(item["run_dir"])
            print(f"[upload_wandb] timeout after {args.timeout}s")
            continue
        returncode, output = attempt
    if returncode != 0:
        failed.append(item["run_dir"])
        print(f"[upload_wandb] FAILED: exit {returncode}")
        continue
    # .wandb 文件结尾有一条没写完的记录:训练机在训练还没结束时就把 wandb/ 传上了 ModelScope。
    # 写完的部分已经传上去了,记成"部分上传",等远端换成完整文件后再同步
    incomplete = "incomplete record" in output
    errors = [line for line in output.splitlines()
              if line.startswith("wandb: ERROR") and "incomplete record" not in line]
    if errors or re.search(r"^Error:", output, re.M):
        failed.append(item["run_dir"])
        print("[upload_wandb] FAILED: wandb reported an error (exit code was 0)")
        continue
    match_url, source_url = run_url(output, args.base_url, item["run_dir"])
    official, step = uploaded_run(output, key)
    if official and urlparse(match_url).netloc != urlparse(official).netloc:
        match_url, source_url = official, source_url or match_url
    state[item["run_dir"]] = {
        "synced_at": datetime.now().isoformat(timespec="seconds"),
        "url": match_url,
        "base_url": args.base_url,
        "source_url": source_url,
        "wandb_bytes": wandb_bytes(item["path"]),
    }
    if run_id:  # 原 id 在 W&B 上删过,这个 run 传到了新 id 下
        state[item["run_dir"]]["upload_id"] = run_id
    later = [r for r in runs if r["run_dir"] > item["run_dir"] and same_run(r, item)]
    if incomplete:
        state[item["run_dir"]].update({"partial": True, "last_step": step})
        if later:  # 训练中断后续训:前面的段结尾本来就不完整,后面的段接着传
            print(f"[upload_wandb] 这一段结尾不完整(训练中断过,到 step {step}),后面还有 {len(later)} 段同一个 run")
        else:
            partial.append(f"{item['run_dir']}" + (f"(到 step {step})" if step is not None else ""))
            print(f"[upload_wandb] PARTIAL: .wandb 文件不完整,只传了写完的部分"
                  f"{f'(到 step {step})' if step is not None else ''};训练机重新上传完整的 wandb/ 后再同步")
    save_state(state)
    if not match_url:
        print("[upload_wandb] warning: synced but no run URL in the output")
    print(f"[upload_wandb] OK {match_url}".rstrip())

if args.dry_run:
    print(f"[upload_wandb] dry-run: {len(selected)} runs would be uploaded")
    sys.exit(0)
if partial:  # 看板任务面板把 [status] 这一行当成结果显示
    steps = sorted({m for p in partial for m in re.findall(r"step \d+", p)})
    print(f"[status] 部分上传 {len(partial)} 个 run" + (f"(到 {'、'.join(steps)})" if steps else "")
          + ":远端 wandb 文件不完整")
if failed:
    sys.exit(f"[upload_wandb] failed {len(failed)} runs: {', '.join(failed[:5])}")
print(f"[upload_wandb] uploaded {len(selected)} runs; state -> {state_file}")
