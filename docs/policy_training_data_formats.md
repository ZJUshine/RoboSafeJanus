# XPolicyLab 各策略训练数据格式统计

统计对象：`XPolicyLab/policy/` 下全部 59 个策略目录（HEAD `de9b445`）。
目标：回答“每个策略训练时到底需要 **HDF5**、**LeRobot v2.1** 还是 **LeRobot v3.0**”。

证据来源优先级：

1. `policy/<POLICY>/README.md` 的 `Data Processing` / `Training` 章节；
2. `process_data.sh` 与其调用的 Python 转换脚本（`codebase_version`、`LeRobotDataset.create`、`h5py` 等实际写盘行为）；
3. 上游 vendored 代码里的 loader 断言（如 `KinRT/full35_assets.py` 校验 `codebase_version == "v3.0"`）；
4. 本机已安装依赖（`Pi_0/openpi/.venv` 内 `lerobot 0.4.4`，`CODEBASE_VERSION = "v3.0"`）。

> 注意：README 自述的“LeRobot v2.1”并不等于“可以用官方 `scripts/transform_lerobot_v21_format.py` 直接产出”。是否需要策略自有 converter 见“关键差异”列。

---

## 1. 汇总

| 训练数据格式 | 数量 | 策略 |
| :-- | :-: | :-- |
| LeRobot **v3.0** | 9 | SmolVLA、StarVLA、InternVLA_A1、DreamZero、OpenDM、VPP2、KinRT、G05、GalaxeaVLA |
| LeRobot **v2.1** | 19 | GO1、Mem_0、A1、Abot_M0、AHA_WAM、Being_H05、EventVLA、Evo_1、FastWAM、GigaWorldPolicy、GR00T_N17、InternW0_delta、LDA_1B、LingBot_VA、LingBot_VLA、RISE、Spatial_Forcing、WorldScape_Policy_2、Discrete_Forcing |
| **HDF5**（直接进训练） | 8 | ACT、TinyVLA、RDT_1B、H_RDT、Hy_Embodied_05_VLA、OpenWAM、PatchWAM、X_VLA |
| HDF5 → 其他私有格式 | 3 | DP（zarr）、MoPA（NPZ）、OpenVLA_OFT（ALOHA → TFDS） |
| 其他原生格式（非 HDF5、非 LeRobot） | 6 | Dexbotic_DM0、Xiaomi_Robotics_0、Xiaomi_Robotics_1、X_WAM、Spirit_v15、OLA_SEM |
| **待确认** | 3 | Pi_0、Pi_05、Pi_0_Fast |
| 仅评测 / 无训练管线 | 11 | Awomo05、GPT_6_Astra_Direct_EEF、Liber_0_lite、Meituan_Robotics_0、MolmoAct2、physicalRSI、Simate_beta、SimpleMemVLA、demo_policy、Dexora_1B、InternVLA_A1_5 |

合计 9 + 19 + 8 + 3 + 6 + 3 + 11 = **59**，与目录数一致。

---

## 2. LeRobot v3.0（9）

| 策略 | 数据来源 / 转换入口 | 关键差异 | 证据 |
| :-- | :-- | :-- | :-- |
| SmolVLA | 直接读 `HF_LEROBOT_HOME/<repo_id>`，无 `process_data.sh` | 官方 converter keys；`--rename_map` 只在 load 时改成 `camera1/2/3` | `policy/SmolVLA/README.md:21` |
| StarVLA | 自带 converter | 官方 keys + `meta/modality.json`（GR00T 式 loader 需要） | `policy/StarVLA/README.md:46` |
| InternVLA_A1 | 用 `repo_id` 指定数据集 | 上游 trainer 硬校验 `CODEBASE_VERSION="v3.0"`；依赖 converter 写的 `robot_type: unified_robot` 选列 | `policy/InternVLA_A1/README.md:19,21` |
| DreamZero | 共享 `RoboDojo_sim_arx-x5_v30`；`process_data.sh` 另走 HDF5→AgiBot 私有布局 | 私有布局与官方导出**不可互换**（`top_head/hand_left/hand_right`，state 20D / action 22D） | `policy/DreamZero/README.md:32,36,38` |
| OpenDM | v3.0 视频导出 → 转成 OpenDM dexdataset JSONL | 也支持 v2.1（`data-v21`）；明确要求 joint+gripper，禁用 `RoboDojo_ee_*` | `policy/OpenDM/README.md:57,65` |
| VPP2 | `RoboDojo_ee_lerobot_v30_video` | **EE16** 绝对位姿导出，非官方 joint14 布局；不支持原始 HDF5 | `policy/VPP2/README.md:34` |
| KinRT | `${HF_LEROBOT_HOME}/RoboDojo_lerobot_v30_video` | 断言 `codebase_version == "v3.0"`；重训需完整 data+videos+episode metadata | `policy/KinRT/README.md:30`、`KinRT/full35_assets.py:166` |
| G05 | 直接读 `ROBODOJO_LEROBOT_V30_ROOT` | 不需要 `meta/modality.json`，norm stats 训练时算 | `policy/G05/README.md:124,126` |
| GalaxeaVLA | 直接读 `GALAXEA_DATASET_DIR`（含 `meta/tasks.parquet`） | 官方 keys，flat 14 维 state/action 由 cfg 切片 | `policy/GalaxeaVLA/README.md:21,23` |

---

## 3. LeRobot v2.1（19）

| 策略 | 数据来源 / 转换入口 | 关键差异 | 证据 |
| :-- | :-- | :-- | :-- |
| RISE | `RISE_RAW_DATASET` → symlink 到 `data/<tag>-lerobot/` | 官方 v2.1 keys，无 HDF5 转换 | `policy/RISE/README.md:21` |
| AHA_WAM | `AHA_WAM_TRAIN_DATASET_DIR` 指向 prepared v2.1 | 官方 v2.1 keys | `policy/AHA_WAM/README.md:19` |
| FastWAM | `data/<dataset_id>/lerobot/` + `dataset_stats.json` + T5 embeds | 相机 key 从 metadata 自动发现 | `policy/FastWAM/README.md:21,33` |
| Evo_1 | 官方 `transform_lerobot_v21_format.py` | 官方 keys | `policy/Evo_1/README.md:28` |
| Being_H05 | `LEROBOT_DATA_PATH`（默认共享 v21）→ link | 官方 keys，仅 link | `policy/Being_H05/README.md:21` |
| Abot_M0 | `RoboDojo_sim_v21_video_abot` | v2.1 + `meta/modality.json`（`left_joints`/`right_joints`/gripper 切片）；AV1 编码 | `policy/Abot_M0/README.md:21,23` |
| LDA_1B | 复用已有 v2.1，生成 GR00T 式 `modality.json` 视图 | `data/`、`videos/` symlink 回源，不改源数据 | `policy/LDA_1B/README.md:21,33` |
| Mem_0 | 自带 `xpolicylab_to_lerobot.py`（HDF5 → LeRobot） | 单 `observation.image.head_camera`、16 维 state、Mn 子任务标注；不吃官方导出 | `policy/Mem_0/README.md:21,23` |
| GO1 | 自带 `process_data.py`（HDF5 → LeRobot） | 相机名 `cam_head`/`cam_hand_left`/`cam_hand_right`；`lerobot.common` 时代 API | `policy/GO1/process_data.py:19`、`policy/GO1/README.md:30,32` |
| A1 | 自带 `process_data.sh`（HDF5 → v2.1 video） | 头部相机 `cam_head` 而非 `cam_high` | `policy/A1/README.md:19` |
| EventVLA | 下载 `RoboDojo_lerobot_v21_video`（`KailunSu/niantian`） | 用 GR00T 式 loader；`expert_data_num` 不生效 | `policy/EventVLA/README.md:19` |
| InternW0_delta | 自带 converter（HDF5 → v2.1，parquet/集） | 官方 keys、14D、480×640；转换需要装有旧 lerobot 的 Python | `policy/InternW0_delta/README.md:40,48` |
| LingBot_VLA | 上游 `train.sh` 默认 `RoboDojo_sim_arx-x5_v30` 标签 | 但 install pin 的是 **pre-v3.0** `lerobot.common` API，必须按安装版本对齐导出 | `policy/LingBot_VLA/README.md:32` |
| LingBot_VA | v2.1 为输入 → `process_data.sh` 生成 Wan2.2 VAE latents | 训练必需 `latents/` + `empty_emb.pt`，不能直接喂纯 LeRobot | `policy/LingBot_VA/README.md:19` |
| GigaWorldPolicy | `process_data.sh`：HDF5 → v2.1（或 link 现成 v2.1） | converter 内写死 `"codebase_version": "v2.1"`；但 `train.sh` 默认 repo id 叫 `*_v30`（见第 6 节） | `policy/GigaWorldPolicy/process_data.sh:45`、`giga_world_policy/scripts/convert_xpolicylab_hdf5_to_lerobot.py:328` |
| GR00T_N17 | 输入共享 v3.0，`process_data.sh` 用上游工具**降级成 v2.1** | 训练侧是 v2.1；其他 robot 必须显式给 `GR00T_SRC_DATASET` | `policy/GR00T_N17/README.md:33` |
| Spatial_Forcing | 上游 open_sf，`repo_id=RoboDojo_lerobot_v21_video` | 无顶层 `process_data.sh`/`train.sh`，走上游 recipe | `policy/Spatial_Forcing/open_sf/README.md:15`、`policy/Spatial_Forcing/README.md:19` |
| WorldScape_Policy_2 | 上游 LeRobot v2 原生布局，`process_data.sh` 只 link + 写原生 metadata | 不是官方 XPolicyLab 导出，需按上游 data guide 生成 | `policy/WorldScape_Policy_2/README.md:56` |
| Discrete_Forcing | `DF_DATA_ROOT` 下 RoboTwin clean `Clean/<task>/` | 接受 **v2.0/v2.1**，需上游 `meta/modality.json`，只校验+link | `policy/Discrete_Forcing/README.md:29,31` |

---

## 4. HDF5（11）

### 4.1 直接以 HDF5 训练（8）

| 策略 | 实际读取内容 | 证据 |
| :-- | :-- | :-- |
| ACT | `episode_*.hdf5`：`action`、`observations/qpos`、`observations/images/cam_head|cam_left_wrist|cam_right_wrist` | `policy/ACT/detr/process_data.py`（h5py 写出） |
| TinyVLA | 合并后的 `episode_XXXXXXX.hdf5`（`hdf5_image_compression=lzf`） | `policy/TinyVLA/process_data.py:209,299` |
| RDT_1B | 原始 HDF5 symlink 到 `data/<4-tuple>/` + `lang_embeds/` | `policy/RDT_1B/README.md:21` |
| H_RDT | 直接读 `HRDT_SOURCE_ROOT` 下 RoboDojo HDF5，无 `process_data.sh` | `policy/H_RDT/README.md:17` |
| Hy_Embodied_05_VLA | HDF5 manifest（`HDF5_DIR`）；`process_data.sh` 只算 `norm_stats.pkl` | `policy/Hy_Embodied_05_VLA/README.md:19` |
| OpenWAM | 原生 RoboDojo HDF5（`<task>/<embodiment>/data/episode_*.hdf5`），明确 “No LeRobot conversion” | `policy/OpenWAM/README.md:26` |
| PatchWAM | RoboTwin 2.0 HDF5 导出，`process_data.sh` 只校验+link，不转码 | `policy/PatchWAM/README.md:30` |
| X_VLA | 上游 HDF5 + `language_instruction` 键，`meta.json` 列 datalist | `policy/X_VLA/xvla/README.md`（Training → Data Preparation） |

### 4.2 HDF5 再转私有格式（3）

| 策略 | 最终训练产物 | 证据 |
| :-- | :-- | :-- |
| DP | HDF5 → `data/<4-tuple>.zarr` | `policy/DP/README.md:19` |
| MoPA | HDF5 → NPZ episodes + statistics（“does not use the LeRobot format”） | `policy/MoPA/README.md:61-63` |
| OpenVLA_OFT | HDF5 → ALOHA 布局 → TFDS `aloha_<run_id>` | `policy/OpenVLA_OFT/README.md:17` |

---

## 5. 其他原生格式（6）

| 策略 | 训练产物 | 证据 |
| :-- | :-- | :-- |
| Dexbotic_DM0 | Dexdata（`jsonl` + `video/`），按 `robodojo_<tag>.py` 注册数据源 | `policy/Dexbotic_DM0/README.md:19`、`dexbotic/docs/Data.md` |
| Xiaomi_Robotics_0 | XR-0 JSON/video 格式 | `policy/Xiaomi_Robotics_0/README.md:21` |
| Xiaomi_Robotics_1 | xr1 JSON/MP4（`data/chunk-XXXX/episode_*.json` + `videos/`）+ Hydra data cfg | `policy/Xiaomi_Robotics_1/README.md:19`（输出表） |
| X_WAM | metadata + `data/` + `video/`（`transform_robodojo_to_xwam.py`） | `policy/X_WAM/README.md:13,15` |
| Spirit_v15 | `task_desc.json` + `meta/task_info.json` + `data/episode_*/{meta,states,videos}` | `policy/Spirit_v15/spirit_v15/dataset/dataset.py:30-50` |
| OLA_SEM | 上游 RoboTwin download+convert（含 `umt5_wan/` 文本嵌入） | `policy/OLA_SEM/README.md:38` |

---

## 6. 待确认与已知不一致

### 6.1 Pi_0 / Pi_05 / Pi_0_Fast —— 版本缺失，且转换脚本与 pin 冲突

- 三个 README 只写 “the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version”，**没有写具体版本**（`policy/Pi_0/README.md:21`、`Pi_05`:21、`Pi_0_Fast`:21），违反 `XPolicyLab/README.md:467` 的约定（训练用 LeRobot 的策略必须声明数据集版本）。
- `openpi/pyproject.toml:48` pin `lerobot==0.4.4`，本机 `.venv` 内该版本 `CODEBASE_VERSION = "v3.0"`（`lerobot/datasets/lerobot_dataset.py:83`），即 `LeRobotDataset.create` 默认落 v3.0。
- 但 `openpi/scripts/process_data.py:16` 仍是 `from lerobot.common.datasets.lerobot_dataset import LeRobotDataset`。实测：

  ```bash
  cd policy/Pi_0/openpi && ./.venv/bin/python -c "import lerobot.common"
  # ModuleNotFoundError: No module named 'lerobot.common'
  ```

- 结论：按 pin 的 0.4.4，目标格式应为 **v3.0**；当前脚本在新环境跑不通，需要先决定“改 import 走 v3.0”还是“降 lerobot 到 v2.1”。确认后才能真正定版。

### 6.2 GigaWorldPolicy 命名误导

`process_data.sh` 与 converter 都产出 v2.1（`convert_xpolicylab_hdf5_to_lerobot.py:328` 写 `"codebase_version": "v2.1"`，stage1/stage2 脚本也要求 v2.1），但 `train.sh` 默认 repo id 是 `XPolicyLab_sim_arx-x5_v30`（`policy/GigaWorldPolicy/train.sh:26`）。只看名字容易误判为 v3.0。

### 6.3 “v2.1” ≠ “官方 converter 输出可用”

以下策略虽是 v2.1 语义，但 key / 布局偏离官方 `scripts/transform_lerobot_v21_format.py`，必须走自己的转换：

- A1（`cam_head`）、GO1（`cam_head`/`cam_hand_*`）、Mem_0（单相机 + 16 维 state + 子任务）、Abot_M0 / LDA_1B（额外 `meta/modality.json`）、InternW0_delta（自带 converter）；
- v3.0 侧同样有例外：VPP2（EE16 而非 joint14）、DreamZero（AgiBot 私有布局）、StarVLA（额外 `modality.json`）、WorldScape_Policy_2（上游原生 v2）。

---

## 7. 需要跨格式时的官方工具

都在 `XPolicyLab/scripts/` 下，输出同一套 keys（`observation.state`、`action`、`observation.images.cam_high|cam_left_wrist|cam_right_wrist`）：

| 方向 | 脚本 |
| :-- | :-- |
| 轨迹 HDF5 → LeRobot v2.1 | `scripts/transform_lerobot_v21_format.py` |
| 轨迹 HDF5 → LeRobot v3.0 | `scripts/transform_lerobot_v30_format.py` |
| v3.0 → v2.1（下载已发布导出） | `scripts/RoboDojo/download_robodojo_data.sh lerobot_v2.1 / lerobot_v3.0`；GR00T_N17 走上游 lerobot 工具 |

两者限制（`XPolicyLab/README.md#official-lerobot-conversion`）：joint-space 双臂专用（pose / 单臂轨迹会失败）、`D` 按匹配目标补零、三路相机 key 恒定存在（缺的填黑帧、RGB 由 `decode_image_bit` 保证）、默认 AV1 编码。

---

## 8. 复核命令

```bash
cd XPolicyLab/policy

# 各策略 Data Processing 章节所在行（空 = 无该章节）
for p in */; do printf "%-24s %s\n" "${p%/}" \
  "$(grep -n '^## Data Processing' "$p/README.md" 2>/dev/null | cut -d: -f1)"; done

# 版本自述：能把 v2.1 / v3.0 说清楚的只有一部分策略
grep -rniE "lerobot v2\.1|lerobot v3\.0|lerobot v2\.0" --include='README.md' .

# 转换脚本真实写盘版本
grep -rn "codebase_version\|CODEBASE_VERSION" --include='*.py' . | grep -v "/\."

# HDF5 直连证据（h5py 读写）
grep -rln "h5py.File" --include='*.py' .

# openpi 三兄弟的 pin 与 import 是否自洽
for p in Pi_0 Pi_05 Pi_0_Fast; do
  printf "%-10s pin:%s import:%s\n" "$p" \
    "$(grep -n 'lerobot==' "$p/openpi/pyproject.toml" | cut -d: -f1)" \
    "$(grep -n 'lerobot.common' "$p/openpi/scripts/process_data.py" | cut -d: -f1)"
done
```
