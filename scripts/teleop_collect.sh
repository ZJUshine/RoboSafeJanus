#!/usr/bin/env bash
# 通过 XRoboToolkit 进行 PICO 遥操作(只录关节指令;操作者看着 Isaac Sim 窗口操作)。
#   bash scripts/teleop_collect.sh <task_name> [episode_num] [gpu_id] [传给 src/data_collect/teleop_collection.py 的额外参数]
# 例: bash scripts/teleop_collect.sh stack_bowls 20 0
#     bash scripts/teleop_collect.sh stack_bowls 1 0 --input scripted --headless   # 无头显,测试整条管线
# 需要本机运行 XRoboToolkit PC 服务(/opt/apps/roboticsservice/runService.sh),且头显上的 XRoboToolkit App
# 已连接到本机。录制完成后用 scripts/teleop_replay.sh 渲染观测。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"

task_name=${1:?usage: teleop_collect.sh <task_name> [episode_num] [gpu_id] [extra args]}
episode_num=${2:-50}
gpu_id=${3:-0}
shift $(( $# < 3 ? $# : 3 ))

# 遥操作是交互式的单进程会话,不分块;默认不加 --headless,需要显示 Isaac 窗口
export CUDA_VISIBLE_DEVICES=${gpu_id}
PYTHONWARNINGS=ignore::UserWarning python -u src/data_collect/teleop_collection.py \
  --task_name "${task_name}" \
  --episode_num "${episode_num}" \
  "$@"
