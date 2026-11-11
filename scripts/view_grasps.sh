#!/usr/bin/env bash
# 在浏览器里交互查看 GraspGenX 抓取缓存(data/grasp_pose,由 scripts/gen_grasps.sh 生成),基于 viser。
#   bash scripts/view_grasps.sh [port] [src/data_collect/view_grasps.py 的额外参数]
# 例:bash scripts/view_grasps.sh
#     bash scripts/view_grasps.sh 8766 --objects bread/0,mug/15
# 打开 http://<本机>:<port>;从别的机器看:ssh -L <port>:localhost:<port> <本机>。Ctrl-C 退出。
# 默认对比 data/grasp_pose_backup_pre_table_20261111(加桌面过滤前的备份,侧栏“缓存”切换);--backup_dir 可改。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

port=${1:-8765}
shift $(( $# < 1 ? $# : 1 ))

ggx_python=${GRASPGENX_PYTHON:-third_party/GraspGenX/.venv/bin/python}
if [[ ! -x "${ggx_python}" ]]; then
  echo "[view_grasps] GraspGenX python not found at ${ggx_python}; install GraspGenX (uv sync) or set GRASPGENX_PYTHON" >&2
  exit 1
fi

# 不需要 GPU;避免 import torch 时占显存
export CUDA_VISIBLE_DEVICES=""
exec "${ggx_python}" src/data_collect/view_grasps.py --port "${port}" "$@"
