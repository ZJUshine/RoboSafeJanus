# 服务器与运行环境

本文记录 RoboSafeJanus 的数据收集环境和评测环境。评测命令与结果口径见 [`docs/model_evaluation.md`](model_evaluation.md),采集流程见 [`docs/data_collection.md`](data_collection.md)。

---

## 0. 总览

| 机器 | 用途 | 地址 | GPU |
| :-- | :-- | :-- | :-- |
| 本地 `hierophant` | 数据收集、评测、开发 | 内网 `10.14.101.148`;IB `192.168.32.105` | 8× RTX 5880 Ada 48GB |
| 远端 gpuhome | 数据收集 | `root@hz01-ssh.gpuhome.cc -p 30160` | 1× RTX 6000D 85GB |
| 远端 8 卡 pod | 评测 | `u12310077@10.202.94.52 -p 21195` | 8× RTX 5090 |

---

## 1. 数据收集环境

### 1.1 本地 `hierophant`

| 项 | 值 |
| :-- | :-- |
| 系统 | Ubuntu 24.04.2 LTS |
| CPU / 内存 | Intel Xeon Gold 6526Y,64 线程 / 755 GiB |
| GPU | 8× NVIDIA RTX 5880 Ada Generation,49,140 MiB(约 48GB)/卡 |
| 驱动 | 580.159.04,CUDA 13.0 |
| 仓库 | `/mnt/hierophant-1/xuancun/RoboSafeJanus` |
| 数据盘 | `/mnt/hierophant-1`,3.6TB,可用约 2.3TB |
| 资产 | `Assets/` 软链接到 `.cache/robodojo_assets_repo/Assets`(39GB) |
| conda 根 | `/home/xuancun/miniconda3` |
| 采集环境 | `conda activate RoboDojo`(Python 3.11.17 / PyTorch 2.7.0+cu128) |
| 采集输出 | `data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/<config>/` |

本地采集使用脚本专家 seed 段 `100000+`,PICO 遥操作用 `200000+`,不要与评测布局的 `0..99999` 重叠。

常用命令:

```bash
conda activate RoboDojo
cd /mnt/hierophant-1/xuancun/RoboSafeJanus

# 冒烟
ROBODOJO_COLLECT_LOG=INFO ROBODOJO_COLLECT_CHUNK=5 \
  bash scripts/collect_data.sh <task> 1 0 --max_tries 5 --save_fail_video

# 批量
bash scripts/collect_data.sh <task> 100 0

# 多 worker
ROBODOJO_PER_GPU=3 bash scripts/collect_data_parallel.sh <task> 100 0
```

### 1.2 远端 gpuhome

| 项 | 值 |
| :-- | :-- |
| 地址 | `root@hz01-ssh.gpuhome.cc -p 30160` |
| 系统 | Ubuntu 24.04.4,内核 5.15 |
| 驱动 | 580.119.02 |
| GPU | 1× RTX 6000D 85GB |
| CPU / 内存 | 112 核 / 503GB |
| 仓库 | `/root/rivermind-data/RoboJanus` |
| conda | `/root/rivermind-fs/miniconda3`,软链 `/root/miniconda3` |
| 采集环境 | `conda activate RoboDojo`(Python 3.11 / Isaac Sim 5.1.0 / IsaacLab 0.54.3) |
| 采集输出 | `$REPO/data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/<config>/` |

部署时资产已从本地 `rsync -L` 同步,并把绝对路径改写为 `/root/rivermind-data/RoboJanus/`。

采集前注意:

- 系统盘 `/` 约剩 3.3GB,数据盘 `/root/rivermind-data` 约剩 40GB;
- 单条 episode HDF5 约 100MB,先规划空间;
- `third_party/GraspGenX` 未同步;换物体/新任务前需要先安装并执行 `bash scripts/gen_grasps.sh <task> 0`;
- `third_party/RoboTwin` 未同步,采集不需要。

常用命令:

```bash
ssh root@hz01-ssh.gpuhome.cc -p 30160
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
cd /root/rivermind-data/RoboJanus

bash scripts/robodojo.sh doctor
ROBODOJO_COLLECT_LOG=INFO bash scripts/collect_data.sh \
  <task> 1 0 --max_tries 5 --save_fail_video --save_dir /tmp/smoke
```

采集完成后把数据拉回本地,不要拉 `_shards/`、`_stream/` 或缓存目录:

```bash
rsync -a --partial -e "ssh -p 30160" \
  root@hz01-ssh.gpuhome.cc:/root/rivermind-data/RoboJanus/data/RoboSafeJanus-Benchmark/<group>/ \
  data/RoboSafeJanus-Benchmark/<group>/
```

---

## 2. 评测环境

### 2.1 本地 `hierophant`

| 项 | 值 |
| :-- | :-- |
| GPU | 8× RTX 5880 Ada 48GB |
| 评测客户端环境 | `conda activate RoboDojo` |
| 策略环境 | Pi_0 / Pi_05 使用 `uv`(OpenPI `.venv`);其他策略按各自 README |
| 仓库 | `/mnt/hierophant-1/xuancun/RoboSafeJanus` |
| 评测布局 | `Assets/Eval_Layout/RoboDojo/arx_x5/0/<task>_<0..49>.json` |
| 模型镜像 | `data/RoboSafeJanus-Model/<POLICY>/<task_or_group>/<run>/` |
| checkpoint 软链接 | `XPolicyLab/policy/<POLICY>/checkpoints/<run>` |
| 评测结果 | `eval_result/RoboDojo/<task>/<policy>/<config>/...` |

建议最多同时跑 3 条流水线,每条 = 1 张 policy GPU + 1 张 sim GPU。

```bash
cd /mnt/hierophant-1/xuancun/RoboSafeJanus
export OMNI_KIT_ACCEPT_EULA=YES

bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task pour_water_bowl \
  --ckpt RoboSafeJanus-pour_water_X_safe_tasks-arx_x5-joint-0 \
  --policy-env uv --action-type joint \
  --eval-num 50 --policy-gpu 0 --env-gpu 1 --seed 0
```

### 2.2 远端 8× RTX 5090 pod

| 项 | 值 |
| :-- | :-- |
| 地址 | `u12310077@10.202.94.52 -p 21195` |
| GPU | 8× RTX 5090 |
| 仓库 | `~/Reseach/RoboSafeJanus`(目录名是 `Reseach`) |
| 模型镜像 | `~/Reseach/RoboSafeJanus/data/RoboSafeJanus-Model/` |
| 日志 | `~/eval_logs/` |
| 评测结果 | `~/Reseach/RoboSafeJanus/eval_result/` |

建议 4 条流水线,GPU 配对 `0/1`、`2/3`、`4/5`、`6/7`。

pod 镜像缺少 Isaac/Kit 运行所需系统库,实例重建后先执行:

```bash
sudo apt-get update -q
sudo apt-get install -y -q rsync libx11-6 libxext6 libxrender1 \
  libgl1 libglu1-mesa libxmu6 libxi6 ffmpeg
```

远端评测命令:

```bash
cd ~/Reseach/RoboSafeJanus
export OMNI_KIT_ACCEPT_EULA=YES

bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task pour_water_bowl \
  --ckpt RoboSafeJanus-pour_water_X_safe_tasks-arx_x5-joint-0 \
  --policy-env uv --action-type joint \
  --eval-num 50 --policy-gpu 0 --env-gpu 1 --seed 0
```

评测结束后把结果拉回本地:

```bash
cd /mnt/hierophant-1/xuancun/RoboSafeJanus
rsync -a -e "ssh -p 21195" \
  u12310077@10.202.94.52:~/Reseach/RoboSafeJanus/eval_result/RoboDojo/<task>/<policy>/ \
  eval_result/RoboDojo/<task>/<policy>/
```

拉回后核对 `_result.json` 的条数、`eval_time` 和视频数量与远端一致。
