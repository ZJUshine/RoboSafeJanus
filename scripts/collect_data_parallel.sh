#!/usr/bin/env bash
# 每张 GPU 跑多个 worker 并行采集脚本专家轨迹,结束后合并成一个数据集。
#   bash scripts/collect_data_parallel.sh <task_name> <episode_num> <gpu_ids> [传给 expert_collection.py 的额外参数]
# 例: bash scripts/collect_data_parallel.sh pour_liquid_into_cup 100 5,6,7 --keep_failed
#
# 单个采集进程约占 9 GB 显存、半张卡(瓶颈在自身的 Python/物理循环),所以每张 GPU 由
# `ROBODOJO_PER_GPU` 个 worker(默认 3)共享。worker k 运行 scripts/collect_data.sh,seed 从
# start_seed + k * `ROBODOJO_SEED_STRIDE`(默认 1000)开始,写到 <save_dir>/_shards/<task>/w<k>/;
# 全部结束后由 src/data_collect/merge_shards.py 合并到 <save_dir>/<task>/<config>/,采集和合并成功后删掉整个分片目录
# (``<save_dir>/_shards/<task>``,含各 worker 的 episode / seed.txt / logs);加 --keep_shards 可保留分片。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

task_name=${1:?usage: collect_data_parallel.sh <task_name> <episode_num> <gpu_ids> [extra args]}
episode_num=${2:?episode_num required}
IFS=, read -r -a gpus <<< "${3:?gpu_ids required, e.g. 5,6,7}"
shift 3

per_gpu=${ROBODOJO_PER_GPU:-3}
stride=${ROBODOJO_SEED_STRIDE:-1000}
save_dir="data/RoboDojo"
start_seed=100000
env_cfg_type="arx_x5"
keep_shards=0
extra=()
# --save_dir / --start_seed 由本脚本按 worker 重新分配,不透传;--keep_shards 由本脚本处理,也不透传;
# --env_cfg_type 既要用于取 config_name,也要透传
while (( $# )); do
  case "$1" in
    --save_dir) save_dir="$2"; shift 2 ;;
    --start_seed) start_seed="$2"; shift 2 ;;
    --keep_shards) keep_shards=1; shift ;;
    --env_cfg_type) env_cfg_type="$2"; extra+=("$1" "$2"); shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
config_name=$(awk '/^config_name:/ {print $2}' "env_cfg/${env_cfg_type}.yml")
shard_root="${save_dir}/_shards/${task_name}"

# worker 数不超过要采的条数
workers=$(( ${#gpus[@]} * per_gpu ))
(( workers > episode_num )) && workers=${episode_num}
# 每个 worker 最多尝试 10 × 分到的条数(collect_data.sh 的默认值),必须落在自己的 seed 段内,否则会与下一个 worker 重叠。
if (( 10 * ((episode_num + workers - 1) / workers) > stride )); then
  echo "[collect_parallel] per-worker tries exceed seed ranges of ${stride}; raise ROBODOJO_SEED_STRIDE" >&2
  exit 1
fi
log_dir="${shard_root}/logs"
mkdir -p "${log_dir}"
echo "[collect_parallel] ${task_name}: ${episode_num} episodes, ${workers} workers on GPUs ${gpus[*]}"

pids=()
for (( k = 0; k < workers; k++ )); do
  # 均分 episode,余数分给前几个 worker;GPU 轮流分配
  n=$(( episode_num / workers + (k < episode_num % workers ? 1 : 0) ))
  gpu=${gpus[$(( k % ${#gpus[@]} ))]}
  bash scripts/collect_data.sh "${task_name}" "${n}" "${gpu}" "${extra[@]}" \
    --save_dir "${shard_root}/w${k}" --start_seed $(( start_seed + k * stride )) \
    > "${log_dir}/w${k}.log" 2>&1 &
  pids+=($!)
  echo "[collect_parallel] worker ${k}: ${n} episodes, GPU ${gpu}, seeds from $(( start_seed + k * stride )), log ${log_dir}/w${k}.log"
  sleep 5  # 错开 Isaac 启动
done

# 等所有 worker 结束(任一失败则最终返回 1),然后合并分片;worker 失败时保留数据和日志供续采。
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
merge_args=(--task_name "${task_name}" --shard_root "${shard_root}" --save_dir "${save_dir}" --config_name "${config_name}")
if (( keep_shards || status )); then
  merge_args+=(--keep_shards)
  if (( status )); then
    echo "[collect_parallel] some workers failed; retaining shards and logs at ${shard_root}" >&2
  fi
else
  merge_args+=(--remove_shards)
fi
python src/data_collect/merge_shards.py "${merge_args[@]}"
echo "[collect_parallel] merged into ${save_dir}/${task_name}/${config_name}"
exit "${status}"
