#!/usr/bin/env bash
# 把本仓库完整镜像到共享盘(/mnt/data),让挂载同一共享盘的所有服务器都能直接读取。
#
# 与 sync_to_server.sh 的区别:那个只推代码到远端 SSH 机器;这个走共享文件系统,
# 默认包含 data/、Assets、XPolicyLab checkpoints 和 .git(完整副本)。
# 增量同步:重复运行只传变化的文件;不加 --delete 时不会删除目标端多出来的文件。
# 同步后会把指向本机源路径的绝对符号链接改写为共享副本内的相对链接(否则其他服务器读到断链)。
#
# 用法:
#   bash scripts/internal/sync_to_shared.sh [--dest <共享路径>] [--dry-run] [--delete] [--skip-caches]
#
# 例:
#   bash scripts/internal/sync_to_shared.sh                 # 增量同步到默认共享路径
#   bash scripts/internal/sync_to_shared.sh --dry-run       # 先看会传什么
#   bash scripts/internal/sync_to_shared.sh --delete        # 严格镜像(目标端多余文件会被删)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="/mnt/data/xuancun/RoboSafeJanus"
DRY_RUN=0
DELETE=0
SKIP_CACHES=0

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --delete) DELETE=1; shift ;;
    --skip-caches) SKIP_CACHES=1; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) error "未知参数: $1" ;;
  esac
done

[[ "$SRC" != "$DEST" ]] || error "源和目标相同($SRC),不会做任何事"
[[ "$DEST" != "/" ]] || error "--dest 不能是 /"
command -v rsync >/dev/null || error "找不到 rsync"

RSYNC_OPTS=(-aH --partial --human-readable)
# 共享副本对所有人可读写:同步时统一放开权限(否则 -a 会把权限改回源仓库的 600/664)。
# 注意:这包括 data/.cache/wandb/api_key 等凭据文件,所有能访问共享盘的人都能读到。
RSYNC_OPTS+=(--chmod=a+rwX)
# 交互终端显示逐文件进度,重定向到日志时只打汇总统计(否则日志会被进度刷新撑爆)
if [[ -t 1 ]]; then
  RSYNC_OPTS+=(--info=progress2)
else
  RSYNC_OPTS+=(--info=stats1)
fi
[[ "$DRY_RUN" == 1 ]] && RSYNC_OPTS+=(--dry-run)
[[ "$DELETE" == 1 ]] && RSYNC_OPTS+=(--delete)
# 本机缓存/字节码:换机器无意义,且会让每次同步都有噪声
if [[ "$SKIP_CACHES" == 1 ]]; then
  RSYNC_OPTS+=(--exclude='__pycache__/' --exclude='*.egg-info/' --exclude='.ruff_cache/'
               --exclude='.cache/' --exclude='data/.cache/')
fi

info "[1/3] 检查目标"
if [[ ! -d "$DEST" ]]; then
  mkdir -p "$DEST" 2>/dev/null || error "无法创建 $DEST(共享盘上需要先 mkdir + chown)"
fi
[[ -w "$DEST" ]] || error "$DEST 不可写;先让有权限的人执行:
    sudo mkdir -p $DEST && sudo chown $(id -un):$(id -gn) $DEST"
SRC_GIB=$(du -sx -BG "$SRC" 2>/dev/null | awk '{print $1}' | tr -d 'G')
AVAIL_GIB=$(df -BG --output=avail "$DEST" | tail -1 | tr -d ' G')
info "    源 $SRC (${SRC_GIB}G) → 目标 $DEST (共享盘可用 ${AVAIL_GIB}G)"
if [[ "$DRY_RUN" != 1 ]] && awk -v a="$AVAIL_GIB" -v s="$SRC_GIB" 'BEGIN{exit !(a <= s + 50)}'; then
  warn "    共享盘剩余空间与源大小接近,建议先确认清理"
fi

info "[2/3] 增量同步开始(完整仓库,含 .git / data / checkpoints)"
rsync "${RSYNC_OPTS[@]}" "$SRC/" "$DEST/"
[[ "$DRY_RUN" != 1 && "$DELETE" != 1 ]] && \
  warn "    未加 --delete:目标端已删除的文件不会被清理"

# 仓库里有指向本机源路径的绝对符号链接(checkpoints、import shim 等)。原样拷过去,
# 其他服务器上是断链;这里把它们改写成共享副本内的相对链接。-n 时只列出不改。
info "    改写指向源仓库的绝对符号链接"
LINK_COUNT=0
while IFS= read -r link; do
  target="$(readlink "$link")"
  shared="${target/$SRC/$DEST}"
  [[ -e "$shared" ]] || continue
  rel="$(realpath -m --relative-to="$(dirname "$link")" "$shared")"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "      would: ${link#$DEST/} -> $rel"
  else
    ln -sfn "$rel" "$link"
  fi
  LINK_COUNT=$((LINK_COUNT + 1))
done < <(find "$DEST" -type l -lname "$SRC/*" 2>/dev/null || true)
[[ "$LINK_COUNT" -gt 0 ]] && info "    共 $LINK_COUNT 个链接已相对化"

info "[3/3] 校验"
if [[ "$DRY_RUN" == 1 ]]; then
  warn "    dry-run 结束,未写入任何文件"
else
  SRC_FILES=$(find "$SRC" -path "$SRC/.cache" -prune -o -type f -print | wc -l)
  DEST_FILES=$(find "$DEST" -path "$DEST/.cache" -prune -o -type f -print | wc -l)
  info "    源文件数 $SRC_FILES / 目标文件数 $DEST_FILES"
  [[ "$DEST_FILES" -ge "$SRC_FILES" ]] || \
    warn "    目标文件少于源(源可能仍有写入,或未加 --delete 无法判断);稍后重跑一次"
  BAD_PERMS=$(find "$DEST" -maxdepth 2 \( \( -type d ! -perm -o+rx \) -o \( -type f ! -perm -o+r \) \) 2>/dev/null | head -5 || true)
  if [[ -n "$BAD_PERMS" ]]; then
    warn "    这些条目其他服务器可能读不到:"
    echo "$BAD_PERMS" | sed 's/^/      /'
  else
    info "    权限正常:目录 o+rx、文件 o+r"
  fi
  DANGLING=$(find "$DEST" -xtype l 2>/dev/null | head -5 || true)
  if [[ -n "$DANGLING" ]]; then
    warn "    断链(源端可能本来就是断的):"
    echo "$DANGLING" | sed "s|^$DEST/|      |"
  else
    info "    符号链接全部可解析"
  fi
fi
