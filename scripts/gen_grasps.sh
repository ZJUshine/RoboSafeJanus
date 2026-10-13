#!/usr/bin/env bash
# 用 GraspGenX 为任务用到的 Rigid 物体资产离线生成抓取缓存(CollectEnv.grasp_actor 读取)。
#   bash scripts/gen_grasps.sh <task_name>[,<task_name>...] [gpu_id] [src/data_collect/grasp_gen.py 的额外参数]
# 例:bash scripts/gen_grasps.sh pour_liquid_into_cup 0
#     bash scripts/gen_grasps.sh pour_liquid_into_cup 0 --overwrite
#
# 两步:先在当前 python(RoboDojo 环境,需要 yaml)里从 task/RoboDojo/config/<task>.yml 列出 Rigid 段的
# <category>/<index>,再用 GraspGenX 自己的 venv(third_party/GraspGenX/.venv,可用 GRASPGENX_PYTHON 覆盖)
# 运行 grasp_gen.py。已有缓存的资产会跳过(缺可视化图时只补画图);输出在
# data/grasp_pose/<gripper>/<category>/<index>.npz,每个物体另有同名 .png 可视化。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

tasks=${1:?usage: gen_grasps.sh <task_name>[,<task_name>...] [gpu_id] [extra args]}
gpu_id=${2:-0}
shift $(( $# < 2 ? $# : 2 ))

ggx_python=${GRASPGENX_PYTHON:-third_party/GraspGenX/.venv/bin/python}
if [[ ! -x "${ggx_python}" ]]; then
  echo "[gen_grasps] GraspGenX python not found at ${ggx_python}; install GraspGenX (uv sync) or set GRASPGENX_PYTHON" >&2
  exit 1
fi

# 任务 YAML 中 Rigid 段的所有 <category>/<index>,去重后以逗号连接
objects=$(python - "${tasks}" <<'PY'
import sys
import yaml

objs = []
for task in sys.argv[1].split(","):
    cfg = yaml.safe_load(open(f"task/RoboDojo/config/{task}.yml"))
    for group in cfg.get("Rigid") or []:
        for cat in group.get("category", []):
            for idx in cat.get("index", [0]):
                item = f"{cat['name']}/{int(idx)}"
                if item not in objs:
                    objs.append(item)
print(",".join(objs))
PY
)
if [[ -z "${objects}" ]]; then
  echo "[gen_grasps] no Rigid objects in ${tasks}" >&2
  exit 1
fi
echo "[gen_grasps] ${tasks}: ${objects}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
"${ggx_python}" src/data_collect/grasp_gen.py --objects "${objects}" "$@"
