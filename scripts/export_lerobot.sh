#!/usr/bin/env bash
# 把 data/RoboSafeJanus-Benchmark 按任务转换成 LeRobot,写到与原数据同样分组的目录:
#   data/RoboSafeJanus-Benchmark-Lerobot/v30/<group>/<safe_tasks|risky_tasks>/<task>/  (v3.0)
#   data/RoboSafeJanus-Benchmark-Lerobot/v21/<group>/<safe_tasks|risky_tasks>/<task>/  (v2.1)
#   bash scripts/export_lerobot.sh [task_glob] [--format v30|v21|both] [传给转换器的额外参数]
# 例:
#   bash scripts/export_lerobot.sh                                   # 全部已采集任务,两种格式
#   bash scripts/export_lerobot.sh 'pour_water_*' --format v21       # 只导 v2.1
#   bash scripts/export_lerobot.sh 'pour_water_*' --format v30 --max_episode 50
#
# 两种格式分别在独立的 conda 环境里运行(环境变量可覆盖):
#   ROBOSAFEJANUS_LEROBOT_V30_ENV (默认 xpolicy_lerobot_v30,兼容旧的 ROBOSAFEJANUS_LEROBOT_ENV)
#   ROBOSAFEJANUS_LEROBOT_V21_ENV (默认 xpolicy_lerobot_v21)
# 创建方法见 docs/data_collection.md。转换器只认 data/<bench>/<task>/<config>/data 三层结构,
# 所以从 data/RoboSafeJanus-Benchmark-flat(按任务的软链接视图,脚本会先重建)读取。
# 已存在的任务输出会被覆盖,<task>_failed 不导出。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

format="both"
task_glob=""
extra_args=()

# 第一个非选项参数是任务 glob,其余参数原样透传给转换器。
if (( $# > 0 )) && [[ "$1" != -* ]]; then
  task_glob=$1
  shift
fi
task_glob=${task_glob:-*}

while (( $# > 0 )); do
  case "$1" in
    --format)
      if (( $# < 2 )); then
        echo "[export_lerobot] --format 需要一个值: v30 | v21 | both" >&2
        exit 2
      fi
      format=$2
      shift 2
      ;;
    --format=*)
      format=${1#*=}
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

case "${format}" in
  v30|v3|v3.0) formats=(v30) ;;
  v21|v2|v2.1) formats=(v21) ;;
  both|all) formats=(v30 v21) ;;
  *)
    echo "[export_lerobot] 无效的 --format: ${format} (可选 v30 | v21 | both)" >&2
    exit 2
    ;;
esac

env_name_v30=${ROBOSAFEJANUS_LEROBOT_V30_ENV:-${ROBOSAFEJANUS_LEROBOT_ENV:-xpolicy_lerobot_v30}}
env_name_v21=${ROBOSAFEJANUS_LEROBOT_V21_ENV:-xpolicy_lerobot_v21}
env_cfg_type=${ROBOSAFEJANUS_LEROBOT_ENV_CFG:-arx_x5}
lerobot_root=${PROJECT_ROOT}/data/RoboSafeJanus-Benchmark-Lerobot
out_root_v30=${ROBOSAFEJANUS_LEROBOT_V30_ROOT:-${lerobot_root}/v30}
out_root_v21=${ROBOSAFEJANUS_LEROBOT_V21_ROOT:-${lerobot_root}/v21}
flat_bench="RoboSafeJanus-Benchmark-flat"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

for fmt in "${formats[@]}"; do
  env_name=${env_name_v30}
  [[ "${fmt}" == "v21" ]] && env_name=${env_name_v21}
  if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "[export_lerobot] conda 环境不存在: ${env_name} (格式 ${fmt}),创建方法见 docs/data_collection.md" >&2
    exit 1
  fi
done

first_env=${env_name_v30}
[[ "${formats[0]}" == "v21" ]] && first_env=${env_name_v21}
conda activate "${first_env}"
python src/data_collect/benchmark_layout.py --flat_view

failed=()
for task_link in "data/${flat_bench}"/${task_glob}; do
  task_name=$(basename "${task_link}")
  [[ "${task_name}" == *_failed || ! -d "${task_link}/${env_cfg_type}/data" ]] && continue
  # 输出放在 <out_root>/<group>/<level>/ 下,repo_id 为任务名
  task_dir=$(python src/data_collect/benchmark_layout.py data/RoboSafeJanus-Benchmark "${task_name}")
  group_rel=$(dirname "${task_dir#data/RoboSafeJanus-Benchmark/}")

  for fmt in "${formats[@]}"; do
    if [[ "${fmt}" == "v30" ]]; then
      env_name=${env_name_v30}
      out_root=${out_root_v30}
      converter="XPolicyLab/scripts/transform_lerobot_v30_format.py"
    else
      env_name=${env_name_v21}
      out_root=${out_root_v21}
      converter="XPolicyLab/scripts/transform_lerobot_v21_format.py"
    fi
    group_dir="${out_root}/${group_rel}"
    mkdir -p "${group_dir}"
    echo "[export_lerobot] ${fmt} ${task_name} -> ${group_dir}/${task_name}"
    conda activate "${env_name}"
    if ! HF_LEROBOT_HOME="${group_dir}" python "${converter}" \
        "${flat_bench}.${task_name}.${env_cfg_type}" --repo_id "${task_name}" "${extra_args[@]}"; then
      failed+=("${fmt}:${task_name}")
    fi
  done
done

if (( ${#failed[@]} )); then
  echo "[export_lerobot] failed: ${failed[*]}" >&2
  exit 1
fi
