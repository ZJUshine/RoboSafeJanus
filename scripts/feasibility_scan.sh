#!/usr/bin/env bash
# 离线可行性扫描(src/data_collect/feasibility_scan.py):不跑物理执行,统计专家规划在一批布局上各段的通过率。
#   bash scripts/feasibility_scan.sh <task_name> [num_seeds=50] [gpu_id=0] [传给 feasibility_scan.py 的额外参数]
# 例: bash scripts/feasibility_scan.sh heat_sandwich_microwave 50 0 --funnel_grasps 20
#
# 结果写到 data/debug/feasibility/<task>/scan.jsonl(可续跑)和 summary.json。Isaac 同一进程反复载入场景约 20 次后
# 会卡死,所以每 ROBODOJO_SCAN_CHUNK(默认 15)个 seed 换一个新进程,单块超过 ROBODOJO_SCAN_CHUNK_TIMEOUT 秒强杀。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES

task_name=${1:?usage: feasibility_scan.sh <task_name> [num_seeds] [gpu_id] [extra args]}
num_seeds=${2:-50}
gpu_id=${3:-0}
shift $(( $# < 3 ? $# : 3 ))

if ! command -v python >/dev/null 2>&1; then
  echo "[feasibility_scan] 'python' not found; activate the sim env first: conda activate RoboDojo" >&2
  exit 1
fi

chunk=${ROBODOJO_SCAN_CHUNK:-15}
chunk_timeout=${ROBODOJO_SCAN_CHUNK_TIMEOUT:-$(( chunk * 300 + 300 ))}

out_dir="data/debug/feasibility/${task_name}"
args=("$@")
for (( i = 0; i < ${#args[@]}; i++ )); do
  [[ "${args[i]}" == "--out_dir" ]] && out_dir="${args[i + 1]}"
done
jsonl="${out_dir}/scan.jsonl"
scanned() { [[ -f "${jsonl}" ]] && wc -l < "${jsonl}" || echo 0; }

export CUDA_VISIBLE_DEVICES=${gpu_id}
stalls=0
while (( $(scanned) < num_seeds )); do
  before=$(scanned)
  echo "[feasibility_scan] ${before}/${num_seeds} seeds scanned; new process for up to ${chunk}"
  PYTHONWARNINGS=ignore::UserWarning timeout --kill-after=30 "${chunk_timeout}" python -u src/data_collect/feasibility_scan.py \
    --task_name "${task_name}" \
    --num_seeds "${num_seeds}" \
    --enable_cameras \
    --headless \
    "$@" \
    --max_seeds_this_run "${chunk}" \
    || echo "[feasibility_scan] process exited with status $? (timeout/crash); resuming in a fresh process"
  if (( $(scanned) == before )); then
    stalls=$(( stalls + 1 ))
    if (( stalls >= 2 )); then
      echo "[feasibility_scan] no progress in two processes; stopping" >&2
      exit 1
    fi
  else
    stalls=0
  fi
done
python src/data_collect/feasibility_scan.py --task_name "${task_name}" --summarize "$@"
