#!/usr/bin/env bash
# 用 cuRobo 脚本专家采集轨迹。
#   bash scripts/collect_data.sh <task_name> [episode_num] [gpu_id] [传给 src/data_collect/expert_collection.py 的额外参数]
# 例: bash scripts/collect_data.sh stack_bowls 50 0 --env_cfg_type arx_x5
#
# Isaac Sim 在反复加载场景时会泄漏渲染资源;单个进程跑几十个 episode 后会耗尽 RTX descriptor set
# ("Unable to allocate descriptor sets")并卡死。因此采集按每 `ROBODOJO_COLLECT_CHUNK` 次尝试(默认 15)分块,
# 每块在一个新进程中从 `seed.txt` / `next_seed.txt` 续采;每块超过 `ROBODOJO_COLLECT_CHUNK_TIMEOUT` 秒
# 仍未结束则强杀,防止卡死。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"

task_name=${1:?usage: collect_data.sh <task_name> [episode_num] [gpu_id] [extra args]}
episode_num=${2:-50}
gpu_id=${3:-0}
shift $(( $# < 3 ? $# : 3 ))

if ! command -v python >/dev/null 2>&1; then
  echo "[collect_data] 'python' not found; activate the sim env first: conda activate RoboDojo" >&2
  exit 1
fi

# 分块大小与单块超时(秒)
chunk=${ROBODOJO_COLLECT_CHUNK:-15}
chunk_timeout=${ROBODOJO_COLLECT_CHUNK_TIMEOUT:-$(( chunk * 120 + 300 ))}

# 从额外参数中解析出输出目录、配置名和总尝试上限,用于定位 seed.txt 和控制循环
save_dir="data/RoboDojo"
env_cfg_type="arx_x5"
max_tries=$(( episode_num * 10 ))
args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
  case "${args[i]}" in
    --save_dir) save_dir="${args[i + 1]}" ;;
    --env_cfg_type) env_cfg_type="${args[i + 1]}" ;;
    --max_tries) max_tries="${args[i + 1]}" ;;
  esac
done
config_name=$(awk '/^config_name:/ {print $2}' "env_cfg/${env_cfg_type}.yml")
seed_file="${save_dir}/${task_name}/${config_name:-${env_cfg_type}}/seed.txt"

# 已保存的成功 episode 数 = seed.txt 中的 seed 个数
saved() { [[ -f "${seed_file}" ]] && wc -w < "${seed_file}" || echo 0; }

export CUDA_VISIBLE_DEVICES=${gpu_id}
tries=0
# 每轮起一个新进程,最多尝试 n 次;脚本自己追加的 --max_tries 放在最后,覆盖用户传入的值(用户值只决定总上限)
while (( $(saved) < episode_num && tries < max_tries )); do
  n=$(( max_tries - tries < chunk ? max_tries - tries : chunk ))
  echo "[collect_data] $(saved)/${episode_num} saved; new process for up to ${n} attempts"
  PYTHONWARNINGS=ignore::UserWarning timeout --kill-after=30 "${chunk_timeout}" python -u src/data_collect/expert_collection.py \
    --task_name "${task_name}" \
    --episode_num "${episode_num}" \
    --enable_cameras \
    --headless \
    "$@" \
    --max_tries "${n}" \
    || {
      status=$?
      # 126/127:命令本身无法执行,重试也没用,直接退出。
      if (( status == 126 || status == 127 )); then
        echo "[collect_data] could not start python (status ${status}); stopping" >&2
        exit "${status}"
      fi
      echo "[collect_data] process exited with status ${status} (timeout/crash); resuming in a fresh process"
    }
  tries=$(( tries + n ))
done
saved_num=$(saved)
if (( saved_num < episode_num )); then
  echo "[collect_data] incomplete: ${saved_num}/${episode_num} episodes in ${seed_file%/seed.txt}; attempt limit exhausted" >&2
  exit 1
fi
echo "[collect_data] done: ${saved_num}/${episode_num} episodes in ${seed_file%/seed.txt}"
