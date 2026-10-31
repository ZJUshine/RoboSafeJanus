#!/usr/bin/env bash
# 把 data/RoboSafeJanus-Benchmark 按任务转换成 LeRobot,写到与原数据同样分组的目录:
#   data/RoboSafeJanus-Benchmark-Lerobot/v30/<group>/<safe_tasks|risky_tasks>/<task>/  (v3.0)
#   data/RoboSafeJanus-Benchmark-Lerobot/v21/<group>/<safe_tasks|risky_tasks>/<task>/  (v2.1)
#   bash scripts/export_lerobot.sh [task_glob] [--format v30|v21|both] [传给 v3.0 转换器的额外参数]
# 例:
#   bash scripts/export_lerobot.sh                                   # 全部已采集任务,两种格式
#   bash scripts/export_lerobot.sh 'pour_water_*' --format v21       # 只从已有 v3.0 派生 v2.1
#   bash scripts/export_lerobot.sh 'pour_water_*' --format v30 --max_episode 50
#
# v3.0 从 HDF5 导出(XPolicyLab/scripts/transform_lerobot_v30_format.py);v2.1 不再从 HDF5
# 重新导出,而是由 src/data_collect/lerobot_v30_to_v21.py 从同一任务的 v3.0 派生
# (按帧切片 + CRF 30 重编码,比直接导出快一个数量级),所以 --format v21 要求 v3.0 已导出。
# 额外参数只传给 v3.0 转换器。
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

if [[ "${formats[*]}" == "v21" ]] && (( ${#extra_args[@]} )); then
  echo "[export_lerobot] --format v21 从 v3.0 派生,忽略额外参数: ${extra_args[*]}" >&2
fi

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

  v30_dir="${out_root_v30}/${group_rel}/${task_name}"
  for fmt in "${formats[@]}"; do
    if [[ "${fmt}" == "v30" ]]; then
      mkdir -p "$(dirname "${v30_dir}")"
      echo "[export_lerobot] v30 ${task_name} -> ${v30_dir}"
      conda activate "${env_name_v30}"
      if ! HF_LEROBOT_HOME="$(dirname "${v30_dir}")" python XPolicyLab/scripts/transform_lerobot_v30_format.py \
          "${flat_bench}.${task_name}.${env_cfg_type}" --repo_id "${task_name}" "${extra_args[@]}"; then
        failed+=("v30:${task_name}")
        # v3.0 失败时不能用残缺的 v3.0 派生 v2.1
        break
      fi
    else
      v21_dir="${out_root_v21}/${group_rel}/${task_name}"
      if [[ ! -f "${v30_dir}/meta/info.json" ]]; then
        echo "[export_lerobot] v21 ${task_name}: 没有 v3.0 导出 ${v30_dir},先导 v3.0" >&2
        failed+=("v21:${task_name}")
        continue
      fi
      mkdir -p "$(dirname "${v21_dir}")"
      echo "[export_lerobot] v21 ${task_name} <- ${v30_dir}"
      conda activate "${env_name_v21}"
      if ! python src/data_collect/lerobot_v30_to_v21.py "${v30_dir}" "${v21_dir}"; then
        failed+=("v21:${task_name}")
      fi
    fi
  done
done

if (( ${#failed[@]} )); then
  echo "[export_lerobot] failed: ${failed[*]}" >&2
  exit 1
fi
