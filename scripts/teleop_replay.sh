#!/usr/bin/env bash
# 带相机重放已录制的遥操作 episode,写成 XPolicyLab HDF5。
#   bash scripts/teleop_replay.sh <task_name> [gpu_id] [传给 src/data_collect/teleop_replay.py 的额外参数]
# 例: bash scripts/teleop_replay.sh stack_bowls 0
#
# 与 collect_data.sh 一样,反复加载场景时渲染会泄漏显存(每个 episode 约 0.5 GB),因此每
# `ROBODOJO_REPLAY_CHUNK` 条(默认 10)换一个新进程,新进程跳过 replay_info.json 中已有的 episode;
# 卡住的块在 `ROBODOJO_REPLAY_CHUNK_TIMEOUT` 秒后被强杀。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"

task_name=${1:?usage: teleop_replay.sh <task_name> [gpu_id] [extra args]}
gpu_id=${2:-0}
shift $(( $# < 2 ? $# : 2 ))

if ! command -v python >/dev/null 2>&1; then
  echo "[teleop_replay] 'python' not found; activate the sim env first: conda activate RoboDojo" >&2
  exit 1
fi

chunk=${ROBODOJO_REPLAY_CHUNK:-10}
chunk_timeout=${ROBODOJO_REPLAY_CHUNK_TIMEOUT:-$(( chunk * 180 + 300 ))}

# 从额外参数中解析输出目录和配置名,用于定位 <config>_teleop 目录
save_dir="data/RoboDojo"
env_cfg_type="arx_x5"
args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
  case "${args[i]}" in
    --save_dir) save_dir="${args[i + 1]}" ;;
    --env_cfg_type) env_cfg_type="${args[i + 1]}" ;;
    --redo) echo "[teleop_replay] --redo re-renders everything; run it in one process" >&2 ;;
  esac
done
config_name=$(awk '/^config_name:/ {print $2}' "env_cfg/${env_cfg_type}.yml")
out_dir="${save_dir}/${task_name}/${config_name:-${env_cfg_type}}_teleop"

# raw/ 中既没有 HDF5、也不在 replay_info.json 里的 episode 数(即待重放条数)。
pending() {
  python - "${out_dir}" <<'PY'
import json, os, sys
d = sys.argv[1]
raw = os.path.join(d, "raw")
names = [f[:-4] for f in os.listdir(raw) if f.endswith(".npz")] if os.path.isdir(raw) else []
info_path = os.path.join(d, "replay_info.json")
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
print(sum(1 for n in names if n not in info and not os.path.exists(os.path.join(d, "data", n + ".hdf5"))))
PY
}

export CUDA_VISIBLE_DEVICES=${gpu_id}
# 循环直到没有待处理的 episode;若一个进程结束后待处理数没有减少,说明卡在同一条上,停止以免死循环
prev=-1
while :; do
  left=$(pending)
  (( left > 0 )) || break
  if (( left == prev )); then
    echo "[teleop_replay] no progress in the last process (${left} left); stopping" >&2
    exit 1
  fi
  prev=${left}
  echo "[teleop_replay] ${left} episodes left; new process for up to ${chunk}"
  PYTHONWARNINGS=ignore::UserWarning timeout --kill-after=30 "${chunk_timeout}" python -u src/data_collect/teleop_replay.py \
    --task_name "${task_name}" \
    --enable_cameras \
    --headless \
    "$@" \
    --max_episodes "${chunk}" \
    || {
      status=$?
      # 126/127:命令本身无法执行,重试也没用,直接退出。
      if (( status == 126 || status == 127 )); then
        echo "[teleop_replay] could not start python (status ${status}); stopping" >&2
        exit "${status}"
      fi
      echo "[teleop_replay] process exited with status ${status} (timeout/crash); resuming in a fresh process"
    }
done
echo "[teleop_replay] done: all episodes in ${out_dir}/raw replayed (see replay_info.json)"
