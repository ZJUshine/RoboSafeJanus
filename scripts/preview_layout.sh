#!/usr/bin/env bash
# 通过 Isaac Sim WebRTC livestream 在本地浏览器里预览任务布局,并选定物体的中心位置。
#   bash scripts/preview_layout.sh <task> <layout.json> [gpu_id] [preview_layout.py 额外参数]
# 例: bash scripts/preview_layout.sh heat_sandwich_microwave /tmp/heat_sandwich/nominal.json 3
#     LIVESTREAM=0 bash scripts/preview_layout.sh heat_sandwich_microwave /tmp/x.json 3   # 只在服务器上截图
#
# 启动后用浏览器打开打印出来的地址(同一内网时直接连;否则先 ssh -L 8011:localhost:8011 转发)。
# 浏览器里可以拖动微波炉 / 物体,右侧面板实时显示坐标;点 “Save layout” 写回 json。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"

task_name=${1:?usage: preview_layout.sh <task_name> <layout.json> [gpu_id] [extra args]}
layout=${2:?usage: preview_layout.sh <task_name> <layout.json> [gpu_id] [extra args]}
gpu_id=${3:-0}
shift $(( $# < 3 ? $# : 3 ))

export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMNI_KIT_ACCEPT_EULA=YES
export LIVESTREAM=${LIVESTREAM:-2}

if [[ "${LIVESTREAM}" != "0" ]]; then
  # Isaac Sim 5.1 的 WebRTC 服务端(omni.kit.livestream.webrtc)不再自带浏览器页面,
  # 客户端用 NVIDIA 的 "Isaac Sim WebRTC Streaming Client" 连服务器的 49100(信令)+ 47998(UDP 媒体)。
  echo "[preview] WebRTC server on $(hostname -I | awk '{print $1}')  signal 49100/tcp  media 47998/udp"
  echo "[preview] connect with the Isaac Sim WebRTC Streaming Client (server address = 上面这个 IP)"
fi

python -u src/data_collect/preview_layout.py \
  --task_name "${task_name}" \
  --layout "${layout}" \
  --enable_cameras \
  "$@"
