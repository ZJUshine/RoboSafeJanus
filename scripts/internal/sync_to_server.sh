#!/usr/bin/env bash
# 把本仓库的采集 / 评测环境同步到外部服务器(docs/data_collection.md §8.6 的可执行版)。
#
# 只上传代码和抓取缓存;资产(约 40G)在服务器上用 scripts/init_assets.sh 从 ModelScope 拉,
# 因为本地到外部服务器的上行通常只有几百 KB/s,传资产要按天算。
#
# 用法:
#   bash scripts/internal/sync_to_server.sh --host <host> --port <port> --repo <远端仓库路径> \
#       [--user root] [--with-assets] [--dry-run] [--force-driver]
#
# 例:
#   bash scripts/internal/sync_to_server.sh --host connect.weste.seetacloud.com --port 37231 \
#       --repo /root/autodl-fs/RoboDojo --with-assets
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }

HOST="" PORT="" USER_NAME="root" REPO="" WITH_ASSETS=0 DRY_RUN=0 FORCE_DRIVER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --with-assets) WITH_ASSETS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force-driver) FORCE_DRIVER=1; shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) error "未知参数: $1" ;;
  esac
done
[[ -n "$HOST" && -n "$PORT" && -n "$REPO" ]] || error "必须给 --host / --port / --repo(见 --help)"

TARGET="${USER_NAME}@${HOST}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
sshx() { ssh -p "$PORT" "${SSH_OPTS[@]}" "$TARGET" "$@"; }
RSYNC_SSH="ssh -p ${PORT} ${SSH_OPTS[*]}"
RSYNC_EXTRA=()
[[ "$DRY_RUN" == 1 ]] && RSYNC_EXTRA+=(-n)

# ── 1. 远端自检:驱动、GPU、磁盘、conda 环境、显卡库 ────────────────────────────
info "[1/6] 远端自检 ($TARGET:$PORT)"
REMOTE_INFO="$(sshx 'set -e
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd, -)"
echo "threads=$(nproc)"
echo "mem_gib=$(free -g | awk "/^Mem:/{print \$2}")"
echo "glibc=$(ldd --version | head -1 | awk "{print \$NF}")"
echo "conda_env=$(ls -d $HOME/miniconda3/envs/RoboDojo 2>/dev/null || echo missing)"
echo "isaacsim=$(ls -d $HOME/miniconda3/envs/RoboDojo/lib/python3.11/site-packages/isaacsim 2>/dev/null || echo missing)"
echo "libegl=$(ldconfig -p | grep -c "libEGL.so.1" || true)"
df -BG --output=avail "$(dirname "$REPO")" 2>/dev/null | tail -1 | sed "s/^/repo_avail=/" || echo "repo_avail=?"
')" || error "ssh 连不上 $TARGET:$PORT"
echo "$REMOTE_INFO" | sed 's/^/    /'

DRIVER="$(echo "$REMOTE_INFO" | sed -n 's/^driver=//p')"
case "$DRIVER" in
  580.*) ;;
  *) [[ "$FORCE_DRIVER" == 1 ]] || error "驱动 $DRIVER 不是 580.x:Isaac Sim 5.1 在 595.x 分支一启动渲染就段错误
    (librtx.scenedb.plugin.so)。换 580.x 的机器,或确认后加 --force-driver。" ;;
esac
echo "$REMOTE_INFO" | grep -q '^libegl=0$' && \
  warn "远端缺 libEGL.so.1,需要先 apt-get install -y libegl1 libglu1-mesa libgl1 libxt6"

# ── 2. 代码 ─────────────────────────────────────────────────────────────────────
info "[2/6] 同步代码 → $REPO"
sshx "mkdir -p '$REPO'"
# third_party 只需 IsaacLab(editable 安装)和 curobo;GraspGenX / RoboTwin 只在本地离线用。
# XPolicyLab 只传代码:policy/*/openpi/.venv 是各机器自己 install.sh 建的,checkpoints 评测时再拉。
(
  cd "$CURRENT_DIR"
  rsync -a --partial --info=progress2 "${RSYNC_EXTRA[@]}" \
    --exclude __pycache__ --exclude .git --exclude .venv \
    --exclude 'third_party/GraspGenX' --exclude 'third_party/RoboTwin' \
    --exclude 'XPolicyLab/policy/*/checkpoints' \
    -e "$RSYNC_SSH" \
    env env_cfg task src utils scripts docs docker third_party XPolicyLab \
    pyproject.toml README.md AGENTS.md CLAUDE.md \
    "$TARGET:$REPO/"
)

# ── 3. 抓取缓存 ────────────────────────────────────────────────────────────────
info "[3/6] 同步抓取缓存"
sshx "mkdir -p '$REPO/data'"
[[ -d "$CURRENT_DIR/data/grasp_pose" ]] && \
  rsync -a --partial "${RSYNC_EXTRA[@]}" -e "$RSYNC_SSH" \
    "$CURRENT_DIR/data/grasp_pose" "$TARGET:$REPO/data/"

# ── 4. ModelScope 凭据(私有资产仓库要用) ──────────────────────────────────────
if [[ -d "$HOME/.modelscope/credentials" ]]; then
  info "[4/6] 同步 ModelScope 凭据"
  sshx "mkdir -p ~/.modelscope"
  rsync -a "${RSYNC_EXTRA[@]}" -e "$RSYNC_SSH" \
    "$HOME/.modelscope/credentials/" "$TARGET:~/.modelscope/credentials/"
else
  warn "[4/6] 本地没有 ~/.modelscope/credentials,跳过;远端拉私有资产前需先 modelscope login"
fi

# ── 5. 资产(远端从 ModelScope 拉,约 40G) ─────────────────────────────────────
if [[ "$WITH_ASSETS" == 1 ]]; then
  info "[5/6] 远端下载资产(后台,约 40G)"
  sshx "cd '$REPO' && if [ -d Assets/Robots ] && [ -d Assets/Eval_Layout ]; then
           echo 'ALREADY_PRESENT';
         elif [[ $DRY_RUN == 1 ]]; then
           echo 'WOULD_RUN: bash scripts/init_assets.sh';
         else
           nohup bash scripts/init_assets.sh > '$REPO/init_assets.log' 2>&1 &
           echo 'STARTED pid='\$!;
         fi"
  echo "    日志:$REPO/init_assets.log(下载完再跑冒烟)"
else
  info "[5/6] 跳过资产(加 --with-assets 让远端从 ModelScope 拉)"
fi

# ── 6. 收尾检查清单 ────────────────────────────────────────────────────────────
info "[6/6] 远端下一步"
cat <<EOF
    cd $REPO
    conda activate RoboDojo                      # 缺环境就先跑 bash scripts/install.sh
    ls Assets/Robots Assets/Eval_Layout          # 资产下完才有
    OMNI_KIT_ACCEPT_EULA=YES bash scripts/robodojo.sh doctor --skip-isaac --skip-conda --skip-policy
    # 冒烟(1 条,先确认渲染正常):
    OMNI_KIT_ACCEPT_EULA=YES bash scripts/collect_data_parallel.sh <task> --save_dir /tmp/smoke ...
EOF
