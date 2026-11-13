# 数据采集与模型评测服务器

本文记录 RoboSafeJanus 的数据收集环境和模型测环境。评测命令与结果口径见 [`docs/model_evaluation.md`](model_evaluation.md),采集流程见 [`docs/data_collection.md`](data_collection.md)。

---

## 0. 总览

| 机器 | 地址 | GPU |
| :-- | :-- | :-- |
| 本地 `hierophant` | xuancun@10.14.101.148 -p 31415 | 8× RTX 5880 Ada 48GB |
| CHPC| u12310077@10.202.94.52 -p 21195| 8× RTX 5090 |

## 1. 具体

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