# 数据采集与重放

本文对应 `src/data_collect/`、`task/RoboDojo/experts/` 与 `scripts/{collect_data,collect_data_parallel,teleop_collect,teleop_replay}.sh` 的当前实现。

---

## 0. 速览

两条采集管线,输出都是 XPolicyLab xspark v1.0 HDF5(与已发布的 RoboDojo 数据同格式),默认**只保存通过任务自身** `run_reward` **判定的 episode**:

| 管线           | 入口脚本                                                   | Python 入口                    | 录制内容              | 是否需要重放                           | 默认起始 seed | 输出目录                                    |
| ------------ | ------------------------------------------------------ | ---------------------------- | ----------------- | -------------------------------- | --------- | --------------------------------------- |
| 脚本专家(cuRobo) | `scripts/collect_data.sh` / `collect_data_parallel.sh` | `src/data_collect/expert_collection.py`   | 边执行边渲染相机,直接写 HDF5 | 否                                | `100000`  | `data/RoboJanus/<task>/<config>/`        |
| PICO 遥操作     | `scripts/teleop_collect.sh`                            | `src/data_collect/teleop_collection.py` | 只录关节指令(不渲染观测相机)   | **是**,`scripts/teleop_replay.sh` | `200000`  | `data/RoboJanus/<task>/<config>_teleop/` |

```
脚本专家:   collect_data.sh ──► expert_collection.py ──► data/RoboJanus/<task>/<config>/{data,preview_video,layout}
并行:       collect_data_parallel.sh ──► N × collect_data.sh ──► _shards/<task>/w<k>/ ──► merge_shards.py ──► 同上
遥操作:     teleop_collect.sh ──► teleop_collection.py ──► data/RoboJanus/<task>/<config>_teleop/raw/*.npz
                                                         │
            teleop_replay.sh ──► teleop_replay.py ◄──────┘──► data/RoboJanus/<task>/<config>_teleop/{data,preview_video,layout}
```

- `<config>` 取自 `env_cfg/<env_cfg_type>.yml` 的 `config_name`,默认 `arx_x5`。
- **seed 段互不重叠**:评测布局用 `0..N`,脚本专家用 `100000+`,遥操作用 `200000+`,避免测试集泄漏。
- 运行前先 `conda activate RoboDojo`(`collect_data.sh` / `teleop_replay.sh` 会检查 `python` 是否可用)。所有脚本都会 `cd` 到仓库根并设置 `PYTHONPATH=.:XPolicyLab`。

### 代码地图

| 文件                                                                | 作用                                         | 什么时候要改             |
| ----------------------------------------------------------------- | ------------------------------------------ | ------------------ |
| `task/RoboDojo/experts/<task>.py`                                 | 每个任务的脚本专家,定义 `play_once(env)`              | 新任务 / 调专家成功率(最常改)  |
| `task/RoboDojo/config/<task>.yml`                                 | 物体类别、摆放范围、数量                               | 改采样分布              |
| `task/RoboDojo/tasks/<task>.py`                                   | `run_reward()` 成功判定、`gen_instruction()` 指令 | 改判定 / 指令(会影响评测,谨慎) |
| `env_cfg/<name>.yml` (+ `env_cfg/{robot,camera,sim,scene}/`)      | 机器人、相机、采集频率                                | 换机器人 / 相机 / 频率     |
| `src/data_collect/collect_env.py`                                 | 采集环境:运动原语、规划、录帧、判定                         | 加新原语、调规划容错         |
| `src/data_collect/grasp_gen.py` / `scripts/gen_grasps.sh`         | 用 GraspGenX 离线生成抓取缓存(`grasp_actor` 读取)     | 新物体 / 换夹爪          |
| `src/data_collect/layout_generator.py`                            | 从 YAML 随机采样布局                              | YAML 用到了未支持的写法     |
| `src/data_collect/common.py`                                      | 构建 env_cfg、布局 getter、`save_episode`        | 改保存内容 / 指令抽取方式     |
| `src/data_collect/hdf5_writer.py`                                 | HDF5 + mp4 写出                              | 改输出格式              |
| `src/data_collect/expert_collection.py`                                        | 脚本专家采集主循环、续采、失败分类                          | 改采集流程 / 参数         |
| `src/data_collect/merge_shards.py`                                | 合并并行 worker 的分片                            | 一般不改               |
| `src/data_collect/teleop_collection.py` / `xr_input.py` / `teleop_replay.py` | 遥操作录制、手柄输入、重放渲染                            | 改按键映射 / IK / 输入设备  |
| `src/data_collect/eval_layouts.py`                                | 为新任务 / 探针任务冻结评测布局                          | 需要新增评测布局时          |
| `src/data_collect/upload_to_hf.py`                                | 把 `data/RoboJanus` 同步到 HF 私有 dataset 仓库         | 一般不改               |

---

## 1. 布局从哪来(`--layout_source` / `--layout_template`)

`expert_collection.py` 和 `teleop_collection.py` 默认 `--layout_source random`:每个 seed 由 `LayoutGenerator.sample(seed)` 从任务 YAML **重新采样**物体摆放。房间 / 桌子 / 地面 / 背景 / 场景固定件(例如相机支架)取自一个**模板布局**,选择顺序(`layout_generator.find_template`):

1. `--layout_template <task 名 | json 路径>`:task 名会解析为 `Assets/Eval_Layout/RoboDojo/<config>/0/<task>_0.json`;
2. 否则任务自己的评测布局 `Assets/Eval_Layout/RoboDojo/<config>/0/<task>_0.json`;
3. 都没有时回退到 `stack_bowls_0.json`。

模板与当前任务**是同一个任务**时,只丢弃其 `Rigid`;来自**其他任务**时,只保留 `Geometry` 固定件,其余可动物体全部丢弃(避免串场)。任务 YAML 里自己定义了 label 的 `Geometry` / `Articulation` / `Fluid` 会重新采样,模板里的同名记录会被移除。

**LayoutGenerator 支持的 YAML 写法**(否则抛 `UnsupportedLayoutError`):

| 对象类型                        | 支持                                                                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Rigid`                     | `relative_plane: Table` + `xlim`/`ylim`(`xlim` 可以是多段,如 `[[-0.3,-0.05],[0.05,0.3]]`)、`zlim`、`qpos`、`rotate_rand`/`rotate_deg`、`margin`、`common.physics` 覆盖;或 `relative_plane: <固定件 label>/<support tag>` 放到固定件的支撑面上(如 `bread_shelf/bread_shelf/0`) |
| `Geometry` / `Articulation` | 放在 Table 上同 Rigid 采样(`zlim` 为相对桌面高度);不在 Table 上的必须能在模板中找到同 label 记录,原样复制                                                                                                                                                                          |
| `Fluid`                     | 跟随其 `relative_plane` 指向的 Rigid,保持模板中相对父物体的偏移 —— **模板必须含有该 Fluid**                                                                                                                                                                                 |
| `select_mode.mode`          | `same` / `unique` / `category_unique` / `allow_duplicate`                                                                                                                                                                                         |
| `select_mode.nums`          | `int`、`[n]`、或闭区间 `[min, max]`(每个布局随机)                                                                                                                                                                                                             |
| 其他(`Garment`、`Dynamic` 等)   | 不支持,只能 `--layout_source eval` 或扩展 `layout_generator.py`(见 7.5)                                                                                                                                                                                    |

物体之间按「xy 平面圆盘不重叠」放置(半径 = 旋转后 bbox 在 xy 的最大半径 + `margin`)。某个 seed 放不下时,会用 `"{seed}/{attempt}"` 派生 seed 重抽整个布局(最多 20 次),仍是确定性的。

> ⚠️ **任务依赖固定件(垃圾桶、传送带、面包架……)或 Fluid、且还没有自己的评测布局时,必须用** `--layout_template` **指向一个固定件相同的任务**。否则会回退到 `stack_bowls`,场景缺少固定件,专家会在 `get_obj_pose("<label>")` 处报 `No instance found with label ...`。
>
> ```bash
> bash scripts/collect_data.sh <new_task> 100 0 --layout_template <有相同固定件的任务>
> ```

`--layout_source eval` 会按 `seed % N` 循环使用 `Assets/Eval_Layout/.../<task>_<i>.json`,**仅用于调试和验证评测布局**,会造成测试集泄漏,不要混进训练数据。

---

## 2. 脚本专家采集

### 2.1 单进程:`scripts/collect_data.sh`

```bash
# bash scripts/collect_data.sh <task> [episode_num=50] [gpu_id=0] [expert_collection.py 额外参数...]

# 1) 冒烟:采 1 条,最多试 5 个 seed,打开专家日志
ROBODOJO_COLLECT_LOG=INFO ROBODOJO_COLLECT_CHUNK=5 \
  bash scripts/collect_data.sh pour_liquid_into_cup 1 0 --max_tries 5 --save_fail_video

# 2) 批量
bash scripts/collect_data.sh pour_liquid_into_cup 100 0

# 3) 同时保留失败尝试(用作负样本 / 排查)
bash scripts/collect_data.sh pour_liquid_into_cup 100 0 --keep_failed
```

脚本固定加 `--enable_cameras --headless`,并在末尾追加自己的 `--max_tries <本块尝试数>`(argparse 取最后一个值,所以用户传的 `--max_tries` 只被脚本用来算**总**上限)。

`**expert_collection.py` 参数**:

| 参数                                      | 默认                       | 说明                                                        |
| --------------------------------------- | ------------------------ | --------------------------------------------------------- |
| `--task_name`                           | 必填                       | 同时决定专家模块 `task/RoboDojo/experts/<task>.py`                |
| `--env_cfg_type`                        | `arx_x5`                 | `env_cfg/<name>.yml`                                      |
| `--episode_num`                         | `50`                     | 目标**成功**条数(含已有的)                                          |
| `--start_seed`                          | `100000`                 | 首个布局 seed(只在没有 `seed.txt`/`next_seed.txt` 时生效)            |
| `--max_tries`                           | `0` = `10 × episode_num` | 尝试次数上限                                                    |
| `--save_dir`                            | `data/RoboJanus`          | 输出根目录                                                     |
| `--layout_source` / `--layout_template` | `random` / 无             | 见第 1 节                                                    |
| `--no_video`                            | 关                        | 不写 `preview_video/`                                       |
| `--save_fail_video`                     | 关                        | 失败尝试写 `video_fail/seed_<seed>_<camera>.mp4`               |
| `--keep_failed`                         | 关                        | 失败尝试(布局不稳除外)写到 `<save_dir>/<task>_failed/<config>/`,见 2.4 |

**环境变量**:

| 变量                               | 默认                | 作用                                                         |
| -------------------------------- | ----------------- | ---------------------------------------------------------- |
| `ROBODOJO_COLLECT_CHUNK`         | `15`              | 每个 Python 进程最多尝试几次                                         |
| `ROBODOJO_COLLECT_CHUNK_TIMEOUT` | `chunk×120+300` 秒 | 单个进程超时后强杀并续采                                               |
| `ROBODOJO_COLLECT_LOG`           | `WARNING`         | `task.`* logger 的级别(专家里 `logging.getLogger(__name__)` 的输出) |

### 2.2 分块与续采

Isaac Sim 反复加载场景会泄漏渲染资源,单进程跑几十条后可能卡死(`Unable to allocate descriptor sets`)。所以 `collect_data.sh` 每 `ROBODOJO_COLLECT_CHUNK` 次尝试就起一个新进程;进程崩溃 / 超时只打印一行然后继续(退出码 126/127 = Python 起不来,直接退出)。续采依据输出目录里的两个文件:

- `seed.txt`:已成功 seed 列表(空格分隔,条数 = 已保存 episode 数,也决定 `episode_<i>` 编号);
- `next_seed.txt`:下一个要尝试的 seed(跳过已失败的)。

因此:**中断后重跑同一条命令即可续采**;调大 `episode_num` 会接着往下采;要从头来就删除整个 `data/RoboJanus/<task>/<config>/`。
达到尝试上限仍未采满时,脚本打印 `incomplete` 并返回非零。

### 2.3 多进程并行:`scripts/collect_data_parallel.sh`

单个采集进程约占 9 GB 显存、半张卡(瓶颈在自身的 Python/物理循环),可以每卡跑多个 worker:

```bash
# bash scripts/collect_data_parallel.sh <task> <episode_num> <gpu_ids> [expert_collection.py 额外参数...]
ROBODOJO_PER_GPU=3 bash scripts/collect_data_parallel.sh pour_liquid_into_cup 100 5,6,7 --keep_failed
```

- worker 数 = `GPU 数 × ROBODOJO_PER_GPU`(默认 3),不超过 `episode_num`;episode 均分。
- worker `k` 用 seed 段 `start_seed + k × ROBODOJO_SEED_STRIDE`(默认 1000),写到 `<save_dir>/_shards/<task>/w<k>/`,日志在 `_shards/<task>/logs/w<k>.log`。每个 worker 启动间隔 5 s。
- 每个 worker 最多尝试 `10 × 分到的条数`,必须 ≤ stride,否则脚本报错退出(调大 `ROBODOJO_SEED_STRIDE`)。
- 全部结束后自动运行 `merge_shards.py`:按 worker 顺序把成功 episode 复制、**重新编号**到 `<save_dir>/<task>/<config>/`,失败的复制到 `<task>_failed/<config>/`,汇总 `collect_info.json`。已存在于目标 `seed.txt` / `fail_info.jsonl` 的 seed 会跳过。分片续采记录仍在时,重跑会续采并重新合并;默认完整清理后,再次采集新数据需用 `--start_seed` 指定新的、不重叠的 seed 段。
- 手动合并:
  ```bash
  python src/data_collect/merge_shards.py --task_name pour_liquid_into_cup \
    --shard_root data/RoboJanus/_shards/pour_liquid_into_cup --config_name arx_x5
  ```
- 合并是**复制后删除**:每条 episode 在目标中存在且大小一致后,才删掉分片里的 hdf5 / layout / 预览视频;分片的 `video_fail/` 也一并删除。目标中缺失或大小不一致的 episode 不删,并打印提示。加 `--keep_shards` 可以只复制不删。
- `collect_data_parallel.sh` 在所有 worker 和合并均成功后删掉整个分片目录 `<save_dir>/_shards/<task>`(`--remove_shards`,含各 worker 的 `seed.txt` / `next_seed.txt` / `collect_info.json` / `logs/`),不留续采记录;最后一个任务清理后,空的 `_shards/` 也删除。任一 worker 失败时仍合并已采集的数据,保留分片和日志供续采,并返回非零。有 episode 在目标中缺失、大小不一致,或目录内还有未记录的数据 / 其他配置时,也保留分片目录并返回非零。并行采集或手动合并加 `--keep_shards` 则保留所有分片文件;它与 `--remove_shards` 互斥。
- 清理旧版本遗留的记录和日志,可在上面的手动合并命令后加 `--remove_shards`;脚本会先核对对应的目标 HDF5 和剩余数据,验证通过后才删除目录。

### 2.4 失败分类与 `--keep_failed`

`expert_collection.py` 对每次尝试的结果分类:

| `fail_type`    | 来源                                                                                        | 典型信息 / 处理                                                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `unstable`     | `reset_with_layout` 里布局沉降不稳 (`UnStableError`)                                             | 偶发可忽略;大量出现说明 YAML 摆放范围/姿态有问题。**不会**被 `--keep_failed` 保存                                                                           |
| `plan_failed`  | `PlanFailError`(cuRobo 无解、只有越界解、候选链都不可达;`grasp_actor` 没有通过过滤的 GraspGenX 候选)               | `none of N candidate chains is reachable` → 增加候选或收窄 `xlim/ylim`;`no GraspGenX grasp ... passes the filters` → 放宽 `grasp_actor` 过滤 |
| `expert_error` | 专家抛出的其他异常(如 `GraspLostError`、`KeyError`;缺抓取缓存时的 `FileNotFoundError`) | 会打印 traceback;看异常信息;`no GraspGenX grasps for ...` → 先跑 `gen_grasps.sh`(7.10 节)                                                    |
| `check_failed` | 专家跑完,`check_success()` 未通过                                                                | 信息里列出未通过的检查项,例如 `['is_A_fluid_in_B (on rising_edge of is_axis_up)']`;配合 `--save_fail_video` 看                                     |

`--keep_failed` 输出(与成功目录结构相同):

```
data/RoboJanus/<task>_failed/<config>/
  data/episode_0000000.hdf5        额外有 meta/{success, fail_type, fail_reason, seed}
  preview_video/  layout/
  fail_info.jsonl                  每行 {"episode","steps","success","fail_type","fail_reason","seed"}
```

只有进入了 `play_once` 且录到 ≥ 2 帧的尝试才会保存。

统计日志里的失败原因:

```bash
grep -E "\] seed=.*failed" <log> | sed 's/.*failed: //' | sort | uniq -c
```

### 2.5 复现单个 seed

布局只由 seed 决定(cuRobo 规划不保证逐帧一致),用临时目录单独重跑:

```bash
ROBODOJO_COLLECT_LOG=INFO bash scripts/collect_data.sh pour_liquid_into_cup 1 0 \
  --start_seed 100013 --max_tries 1 --save_dir /tmp/debug --save_fail_video
```

### 2.6 输出

```
data/RoboJanus/<task>/<config>/
  data/episode_0000000.hdf5                     XPolicyLab HDF5
  preview_video/episode_0000000_<camera>.mp4    每路相机一个,collect_freq Hz (默认 25)
  layout/episode_0000000.json                   该 episode 的完整布局(可用于复现 / 重放)
  seed.txt / next_seed.txt                      续采位置
  collect_info.json                             success / fail_this_run / tries_this_run / fail_types_this_run
  video_fail/seed_<seed>_<camera>.mp4           仅 --save_fail_video
```

---

## 3. PICO 遥操作

### 3.1 前置条件

- 本机运行 XRoboToolkit PC 服务:`/opt/apps/roboticsservice/runService.sh`;头显上的 XRoboToolkit App 已连接本机。
- SDK 默认路径 `/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so`,其他位置用 `--xrt_sdk` 指定。启动 3 s 内连不上服务会报错退出。
- **非 headless**:操作者看着显示器上的 Isaac Sim 窗口操作(脚本默认不加 `--headless`)。

```bash
# bash scripts/teleop_collect.sh <task> [episode_num=50] [gpu_id=0] [teleop_collection.py 额外参数...]
bash scripts/teleop_collect.sh pour_liquid_into_cup 20 0

# 无头显测试整条管线(脚本化手柄输入:右手抓取-移动-开合-保存)
bash scripts/teleop_collect.sh pour_liquid_into_cup 1 0 --input scripted --headless
```

### 3.2 手柄操作(左手柄 → `left_arm`,右手柄 → `right_arm`)

| 按键          | 作用                                 |
| ----------- | ---------------------------------- |
| 握持键(grip)按住 | 离合:手臂跟随手柄**相对于按下时刻**的位移和转动;松开即脱离   |
| 扳机(trigger) | 夹爪,松开 = 张开,扣下 = 闭合(连续值)            |
| 右手 A        | 双臂在关节空间回到初始位姿(会被录制;再按 grip 可打断)    |
| 右手 B        | 结束:运行成功判定,通过(或 `--keep_failed`)则保存 |
| 左手 Y        | 放弃本次尝试,加载下一个布局                     |

- 物理在第一次按 grip / trigger 之前是暂停的,可以先看清布局。
- 手部运动按「操作者面向机器人前方 +y(Isaac 窗口视角)」映射。面对显示器时长按 PICO 键重新校准,或用 `--operator_yaw <deg>` 旋转映射。
- 某只手柄追踪丢失超过 0.3 s 视为松开 grip,恢复后需重新按 grip。
- 终端每 10 s 打印控制频率和实时因子(`x real time`);低于 1 说明仿真跟不上。

### 3.3 `teleop_collection.py` 参数

| 参数                                      | 默认             | 说明                                                        |
| --------------------------------------- | -------------- | --------------------------------------------------------- |
| `--episode_num`                         | `50`           | **本次会话**要保存的条数                                            |
| `--start_seed`                          | `200000`       | 首个布局 seed(有 `next_seed.txt` 时从那里续)                        |
| `--layout_source` / `--layout_template` | `random` / 无   | 同第 1 节                                                    |
| `--input`                               | `xrobotoolkit` | `scripted` = 无头显测试                                        |
| `--control_hz`                          | `50`           | IK / 渲染频率(仿真时间);每 tick 的物理步数 = `round(1/(control_hz·dt))` |
| `--pos_scale`                           | `1.0`          | 手移动 1 m 时末端移动多少 m                                         |
| `--max_joint_step`                      | `0.05`         | 每 tick 每关节最大变化(rad),即限速                                   |
| `--home_joint_step`                     | `0.02`         | 右手 A 回零时的限速                                               |
| `--ik_damping`                          | `0.05`         | 阻尼最小二乘 IK 的 λ,奇异点附近抖动时调大                                  |
| `--operator_yaw`                        | `0`            | 手→机器人映射绕 z 旋转(度)                                          |
| `--keep_failed`                         | 关              | 判定失败也保存                                                   |
| `--keep_cameras`                        | 关              | 录制时也渲染观测相机(调试用,会变慢)                                       |

### 3.4 输出(还不能直接训练)

```
data/RoboJanus/<task>/<config>_teleop/
  raw/episode_0000000.npz     每个物理步的关节/夹爪指令(pos.* / vel.*, float64)、chunk_lens、
                              meta(task、env_cfg_type、seed、完整 layout、dt、substeps、teleop_success、末态关节)
  next_seed.txt               续采位置
```

---

## 4. 重放遥操作数据:`scripts/teleop_replay.sh`

`teleop_replay.py` 对每个 `raw/*.npz`:用**同一个布局和 seed** reset,把录下的指令按原控制路径逐步执行,同时渲染观测相机,**再次运行成功判定**,通过才写 HDF5。

```bash
# bash scripts/teleop_replay.sh <task> [gpu_id=0] [teleop_replay.py 额外参数...]
bash scripts/teleop_replay.sh pour_liquid_into_cup 0

# 回放失败的也写出来(排查用)
bash scripts/teleop_replay.sh pour_liquid_into_cup 0 --keep_failed

# 全部重新渲染(例如改了相机配置):必须直接调用 python,单进程跑
PYTHONPATH=.:XPolicyLab CUDA_VISIBLE_DEVICES=0 python src/data_collect/teleop_replay.py \
  --task_name pour_liquid_into_cup --enable_cameras --headless --redo
```

> `.sh` 只统计「既无 HDF5 也不在 `replay_info.json`」的 episode;给 `.sh` 传 `--redo` 只会打印警告,若所有 episode 都已处理过,循环直接结束、**什么都不会重渲**。

| 参数                              | 说明                                         |
| ------------------------------- | ------------------------------------------ |
| `--env_cfg_type` / `--save_dir` | 必须与录制时一致(任务名、`env_cfg_type`、`dt` 不一致会直接报错) |
| `--keep_failed`                 | 回放未通过判定也写 HDF5                             |
| `--redo`                        | 重新渲染已有 HDF5 的 episode                      |
| `--no_video`                    | 不写预览视频                                     |
| `--max_episodes`                | 本进程最多处理条数(`.sh` 分块时自动设置)                   |

- **分块**:每 `ROBODOJO_REPLAY_CHUNK`(默认 10)条换新进程,超时 `ROBODOJO_REPLAY_CHUNK_TIMEOUT`(默认 `chunk×180+300` 秒)。重跑即续跑;一个进程没有任何进展时脚本停下,避免死循环。
- `**replay_info.json`**:每条的 `seed`、回放 `success`、`joint_drift`(回放末态与录制末态的最大关节误差,正常约 `1e-18` rad)。drift 明显偏大说明回放不确定(例如录制后改了物理 / 控制 / 机器人配置),需要重新录制。
- 重放后目录结构与脚本专家相同:`data/`、`preview_video/`、`layout/`,外加保留的 `raw/`。

脚本专家数据**不需要**重放。

---

## 5. 检查采集结果

```bash
D=data/RoboJanus/<task>/<config>
ls $D/data | wc -l                                  # 条数
cat $D/collect_info.json
python -c "import h5py; h5py.File('$D/data/episode_0000000.hdf5').visititems(lambda n,o: print(n, getattr(o,'shape','')))"
# 抽帧(第 90 帧 / 最后一帧)
ffmpeg -loglevel error -i $D/preview_video/episode_0000000_cam_head.mp4 -vf "select=eq(n\,90)" -frames:v 1 mid.png
ffmpeg -loglevel error -sseof -0.5 -i $D/preview_video/episode_0000000_cam_head.mp4 -frames:v 1 last.png
```

**HDF5 结构**(`hdf5_writer.py`,T = 录到的帧数,存 T-1 步,数组均为 float64):

| 键                                                             | 内容                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| attrs `fix_obs_shift_rgb`                                     | `"shift_drop_first_drop_last+rgb_jpeg"`(图像已与同一步状态对齐)        |
| `data_format_version`                                         | `"v1.0"`                                                    |
| `instruction`                                                 | 一条语言指令:`random.Random(seed).choice(gen_instruction 模板)`     |
| `additional_info/frequency`                                   | 采集频率(int64)                                                 |
| `state/<arm>_{arm_joint_states, ee_joint_states, ee_poses}`   | 第 t 帧;`ee_joint_states` 归一化,1 = 张开;位姿 `[x,y,z,qw,qx,qy,qz]` |
| `state/<arm>_delta_ee_poses`                                  | 乘上机器人 `delta_matrix` 后的末端位姿                                 |
| `action/<arm>_{arm_joint_states, ee_joint_states, ee_poses}`  | 第 t+1 帧(即下一帧状态)                                             |
| `vision/<camera>/colors`                                      | JPEG 字节串 `(T-1,)`                                           |
| `vision/<camera>/{shape, intrinsic_matrix, extrinsic_matrix}` | `(3,)`、`(3,3)`、逐帧 `(T-1,4,4)` camera-to-world               |
| `meta/`*                                                      | 仅 `--keep_failed` 的失败样本                                     |

`<arm>` 在双臂时为 `left_` / `right_`,单臂时为空前缀。

建议至少抽查几条视频,确认成功判定没有被「钻空子」(例如物体被碰进目标区域而非按预期放入)。

---

## 6. 典型流程:给一个新任务采数据

1. 确认任务本身能跑:`task/RoboDojo/config/<task>.yml` + `task/RoboDojo/tasks/<task>.py` 存在、类名 == 模块名、`run_reward()` 有意义。
2. 确认布局:检查 YAML 是否在第 1 节的支持范围;依赖固定件 / Fluid 时找一个可用的 `--layout_template`。
3. 生成抓取缓存:`bash scripts/gen_grasps.sh <task> 0`(7.10 节)。
4. 写专家 `task/RoboDojo/experts/<task>.py`(第 7.1 节)。
5. 本地 GPU 冒烟:`ROBODOJO_COLLECT_LOG=INFO ROBODOJO_COLLECT_CHUNK=5 bash scripts/collect_data.sh <task> 1 0 --max_tries 5 --save_fail_video --save_dir /tmp/smoke`,看视频。
6. 本地小批量(如 20 条)看成功率和失败分布(2.4),针对性调专家。
7. 本地正式采集 100 条:`bash scripts/collect_data_parallel.sh <task> 100 <gpu_ids>`。
8. 抽查(第 5 节)。
9. 若在外部服务器上采集,把数据拉回本地(8.6 节)。

冒烟和调专家建议在本地 GPU 上完成;正式采集可以放到外部服务器上跑,采完再拉回本地(8.6 节)。

---

## 7. 人为修改指南

### 7.1 新增 / 修改脚本专家(最常见)

文件:`task/RoboDojo/experts/<task>.py`,文件名必须与任务名一致,必须定义 `play_once(env)`。`expert_collection.py` 通过 `importlib.import_module("task.RoboDojo.experts.<task>")` 加载。`play_once` 正常返回 → 运行成功判定;抛异常 → 记为失败。

`env` 是 `collect_env.py` 里的 `CollectEnv`(任务类的子类),可用接口:

**查询**

| 接口                                                                         | 说明                                                      |
| -------------------------------------------------------------------------- | ------------------------------------------------------- |
| `get_obj_pose(label)`                                                      | 物体位姿 `[x,y,z,qw,qx,qy,qz]`(env 坐标系)                     |
| `get_obj_extents(label)`                                                   | 物体 aligned bbox 尺寸                                      |
| `scene_manager.layout_manager.get_instance_metadata(env_idx=0, label=...)` | 物体完整 metadata(bbox 顶点、`active`/`passive` 接触点等)          |
| `get_arm_pose(arm)` / `get_arm_joints(arm)`                                | `ee_link` 位姿 / 关节角;`arm` ∈ `"left_arm"`, `"right_arm"`  |
| `init_endpose[arm]` / `init_joints[arm]`                                   | 本 episode 开始时的末端位姿 / 关节角                                |
| `is_held(arm, label, tol=0.08)`                                            | 物体中心是否在夹爪中心(TCP)`tol` 米内                                |
| `tcp_to_ee(arm, tcp_pose)`                                                 | 夹爪中心位姿 → 规划器需要的 `ee_link` 位姿(沿 ee x 轴后退 `gripper_bias`) |
| `top_down_quat(yaw)`                                                       | 自上而下抓取朝向:ee x 轴(接近方向)朝下,y 轴(手指方向)在桌面内角度 `yaw`           |

**规划(只算,不执行)**

| 接口                                           | 说明                                                 |
| -------------------------------------------- | -------------------------------------------------- |
| `plan_to_pose(arm, pose, start_joints=None)` | cuRobo 规划到 ee 位姿;失败抛 `PlanFailError`               |
| `plan_to_joints(arm, joints)`                | 关节空间规划                                             |
| `plan_chain(arm, [pose1, pose2, ...])`       | 依次规划多段;任一段失败返回 `None`                              |
| `choose_chain(arm, [[...], [...], ...])`     | 返回第一个可规划的候选链 `(index, plans)`;全失败抛 `PlanFailError` |

**执行(会推进物理并录帧)**

| 接口                                                                             | 说明                                                   |
| ------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `execute_plan(arm, plan, slow=1.0)`                                            | 执行一个规划结果并等待到位;`slow>1` 按比例放慢(搬运液体等)                  |
| `move(arm, pose)`                                                              | `plan_to_pose` + `execute_plan`                      |
| `together_move(left_pose, right_pose)`                                         | 双臂同时                                                 |
| `move_by_displacement(arm, x, y, z, move_axis="world", straight=False)`        | 末端相对位移;`move_axis="arm"` 时沿接近方向后退 `z` 米(同 RoboTwin)  |
| `move_straight(arm, delta, step=0.01)`                                         | 末端沿 `delta` 走直线(每段 `step` 米分别规划),失败抛 `PlanFailError` |
| `move_up(arm, heights, slow=1.0)`                                              | 竖直抬升,依次尝试 `heights` 中的高度,取第一个可达的                     |
| `open_gripper(arm)` / `close_gripper(arm)` / `set_gripper(arm, val, steps=60)` | 夹爪,`val` 1 = 开,0 = 关;从上一次**指令值**线性过渡                 |
| `back_to_origin(*arms)`                                                        | 关节空间回到初始位姿(默认所有臂)                                    |
| `delay(steps)`                                                                 | 空等若干物理步                                              |
| `execute(*seqs)`                                                               | 底层:并行执行多条控制序列                                        |

**抓取 / 放置(参照 RoboTwin 的** `grasp_actor` **/** `place_actor`**)**

| 接口                                                                                                                  | 说明                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `grasp_actor(label, arm, pre_dis=0.08, **filters)`                                                                  | 用 GraspGenX 抓取缓存(7.10 节)抓起物体:过滤、排序候选,选第一个能规划的,张开 → 预抓取 → 直线接近 → 闭合。没有缓存 / 没有候选 / 全不可达都直接报错,**不回退**。返回 `(ee_pose, score)`                     |
| `grasp_candidates(label, arm, prefer=None, approach_z=(-1, 0.3), finger_z_max=None, obj_z=None, max_candidates=40)` | 只取候选不执行;参数即 `grasp_actor` 的 `filters`                                                                                                        |
| `place_actor(label, arm, targets, point="place", pre_dis=0.06, yaw_offsets=(0,), upright=True, release=True)`       | 把手中物体的对齐点(默认 `active.place.up` 底面接触点)放到 `targets` 中第一个可达的位置;用实测 ee→物体变换反推 ee 位姿,`upright` 时去掉倾斜只保留偏航;先到正上方 `pre_dis` 再竖直下放。返回 `(目标序号, 偏航偏移)` |
| `get_obj_asset(label)`                                                                                              | `(category, index, scale, metadata)`                                                                                                         |

`grasp_candidates` 的过滤与排序:

| 参数                         | 含义                                | 例                                    |
| -------------------------- | --------------------------------- | ------------------------------------ |
| `approach_z`               | 接近方向单位向量 z 分量范围(env 坐标)           | 自上而下 `(-1, -0.7)`;水平侧抓 `(-0.3, 0.1)` |
| `finger_z_max`             | 手指方向                              | z                                    |
| `obj_z`                    | 抓取中心(TCP)在**物体坐标系**下的高度范围,限定抓哪个部位 | 只抓瓶颈 `(0.065, 0.115)`                |
| `prefer` / `prefer_weight` | 偏好的接近方向;代价 = 权重 × 夹角/π − 判别器分数    | 从基座指向物体的水平方向                         |
| `max_candidates`           | 最多交给规划器尝试的候选数                     | 默认 40                                |

夹爪对称(arx_x5 是)时,每个抓取绕接近轴转 180° 的版本也作为候选。

最小骨架(抓一个物体放到另一个上):

```python
"""<task> 的脚本专家。"""

import logging

import numpy as np

log = logging.getLogger(__name__)

LIFT = (0.12, 0.10, 0.06)


class GraspLostError(Exception):
    pass


def play_once(env):
    cup = env.get_obj_pose("cup")
    arm = "left_arm" if cup[0] < 0 else "right_arm"

    # 自上而下抓杯子(接近方向与竖直向下夹角 < 45°)
    env.grasp_actor("cup", arm, pre_dis=0.08, approach_z=(-1.0, -0.7), prefer=[0, 0, -1])
    env.move_up(arm, LIFT)
    if not env.is_held(arm, "cup"):
        raise GraspLostError("cup slipped while lifting")

    # 杯底接触点放到盘子上表面中心,先在上方 6 cm
    plate = env.get_obj_pose("plate")
    target = plate[:3] + [0, 0, env.get_obj_extents("plate")[2] / 2]
    env.place_actor("cup", arm, [target], yaw_offsets=(0, 30, -30, 60, -60))
    env.move_by_displacement(arm, z=0.05, move_axis="arm")  # 沿接近方向后退
    env.move_up(arm, (0.10, 0.06))
    env.back_to_origin()
```

完整、复杂的参考是 `task/RoboDojo/experts/pour_liquid_into_cup.py`(用 `grasp_candidates` 过滤、排序后侧抓瓶颈;以瓶口为支点倾倒,按实测 ee→瓶子 变换换算末端位姿;读液体粒子闭环控制倾角;立起瓶子触发判定后即结束)。

**编写要点**

- **cuRobo 的碰撞世界只有桌子**。避开其他物体要靠竖直进出、`move_straight` 分段直线、候选排序。
- 用 `grasp_actor` 的过滤参数**限定抓取部位和方向**:GraspGenX 只保证夹爪能稳定夹住物体,不知道任务需要(如倒酒要抓瓶颈、手指水平)。过滤太严 → `no GraspGenX grasp ... passes the filters`;太松 → 抓法不适合后续动作。
- 手写候选(不用 GraspGenX)时给 `choose_chain` **足够多的候选**(多个 yaw × 多个 pitch/高度),顺序即优先级。候选太少 → 大量 `plan_failed`。
- arx_x5 的关节 5(腕)在约 ±1.56 rad 自碰,`plan_to_pose` 会自动重试并拒绝越界路径(`collect_env.SIM_JOINT_BOUNDS`)。
- 抓取后用 `is_held` 检查,**抓丢尽早抛异常**,不要让失败 episode 跑满全程浪费时间;自定义异常类会被归为 `expert_error`。
- 抓取后的目标位姿最好用**实测**的 ee→物体变换反推(`place_actor` 内部就是这样做的,倒酒见 `pour_liquid_into_cup._Pour.ee_pose` / `regrasp`),这样轻微滑动也不影响后续动作。
- 触发式判定(`trigger_check`,如 `is_axis_up` 的上升沿)需要专家**真的走过触发条件**,否则判定永远 pending,表现为 `check_failed: [... (on rising_edge of ...)]`。
- 调参优先改文件顶部的常量(如 `PRE_GRASP`、`NECK_Z`、`DEPTH_MAX`、`MOUTH_BACK`、`FLOW_MIN` / `FLOW_MAX`、`TILT_MAX`),每次改完用 2.5 节固定 seed 复现对比。
- 用 `logging.getLogger(__name__)` 打日志,不要 `print`;`ROBODOJO_COLLECT_LOG=INFO` 才会显示。

### 7.2 修改采样分布(任务 YAML)

文件:`task/RoboDojo/config/<task>.yml`。采集时 `LayoutGenerator` 直接读它,改完下一次运行就生效。

| 想要            | 改什么                                                   |
| ------------- | ----------------------------------------------------- |
| 物体出现区域        | `common.xlim` / `ylim`(env 坐标,米;`xlim` 可写多段避开中间)      |
| 物体间距          | `common.margin`                                       |
| 随机朝向          | `rotate_rand: True` + `rotate_deg: 180`(或 `[lo, hi]`) |
| 初始姿态(如倒扣)     | `common.qpos`                                         |
| 换模型 / 增加模型多样性 | `category[].name` / `index`                           |
| 数量            | `select_mode.nums`(`[min,max]` 每个布局随机)                |
| 摩擦等物理参数       | `common.physics: {static_friction: ...}`              |

⚠️ 这个 YAML **评测也在用**:评测读的是冻结在 `Assets/Eval_Layout/` 的布局,所以改 YAML 不会直接改变评测布局,但会改变 `soft_reset`/重新生成评测布局时的结果,以及任务类里任何读 YAML 的逻辑。只想改训练分布而不想动评测时,建议复制一个新任务(新 YAML + 新 task 模块),或在 PR 里明确说明。

改了 YAML 后:旧数据的 `seed.txt` 仍然有效但布局会不同,**不同分布的数据请用不同** `--save_dir` **分开存**。

### 7.3 修改成功判定 / 语言指令

- 判定:`task/RoboDojo/tasks/<task>.py` 的 `run_reward()`。采集用 `check_success()`:先 `delay(50)` 让物体静止,然后 `reward_manager.get_reward(final_check=True)`,所有检查同时成立才算成功。改判定会**同时改变评测指标**,按 Task PR 流程处理。
- 指令:同文件 `gen_instruction(env_idx)` 返回模板列表;每条 episode 用 `random.Random(seed).choice(...)` 抽一条。改了模板后,已采数据的指令不会变,需要重新采集或离线改写 HDF5 的 `instruction`。

### 7.4 修改机器人 / 相机 / 采集频率

- 入口 `env_cfg/<name>.yml`(采集和评测共用),其中 `config.{sim,scene,robot,camera}` 分别指向 `env_cfg/{sim,scene,robot,camera}/*.yml`。
- **新建**一个 `env_cfg/<my_cfg>.yml` 并改 `config_name`,然后采集时 `--env_cfg_type my_cfg`,输出自动落到 `data/RoboJanus/<task>/<config_name>/`,不会覆盖已有数据。不要直接改 `arx_x5.yml` 等共享配置。
- `observation.collect_freq`:录帧频率(默认 25 Hz),也写进 HDF5 和视频帧率。
- 相机内外参:`common.build_env_cfg` 在采集时强制打开 `intrinsic_matrix` / `extrinsic_matrix`,无需改 yml。
- 换了机器人:`collect_env.py` 的 `SIM_JOINT_BOUNDS` 只为 `x5` 配置了腕关节限位,新机器人如有类似 URDF 占位限位问题需要在这里补;专家里的抓取朝向 / `gripper_bias` 也要重新验证。
- 遥操作数据改了这些配置后必须重新 `--redo` 重放(见第 4 节),且 `joint_drift` 可能变大 → 重新录制。

### 7.5 YAML 写法不被 LayoutGenerator 支持时

报 `UnsupportedLayoutError: ... is not supported by the collect layout generator`。选项:

1. 改 YAML 到支持的写法(见第 1 节表格);
2. 只调试用:`--layout_source eval`;
3. 扩展 `src/data_collect/layout_generator.py`:
   - 新的放置方式:在 `_sample` 的分支里加,参考 `_place_on_table` / `_place_on_support`,返回 `_record(...)` 格式的记录;
   - 新的 `select_mode.mode`:加到 `SUPPORTED_MODES` 和 `_pick_models`;
   - 同时更新 `_check_supported`,否则仍会被拒绝。

### 7.6 给采集环境加新原语

在 `collect_env.py` 的 `CollectEnv` 里加方法即可,专家通过 `env.<method>` 调用。约定:

- 所有会推进物理的动作都必须走 `execute(...)`(内部按 `collect_interval` 录帧、并调用 `reward_manager.step`);**直接调** `sim_step` **不会录帧,也不会更新触发式判定**。
- 规划失败抛 `PlanFailError`,让 `expert_collection.py` 正确分类为 `plan_failed`。
- 新的单臂动作参考 `move_up`(候选 + `choose_chain`)、`execute_plan`(执行后 `settle` 等到位)。

### 7.7 修改输出内容 / 格式

- 想在 HDF5 里多存东西:观测内容由 `ObsManager.get_obs` 决定(`env/observation_manager/`,受 `env_cfg` 的 `observation` 段控制);写出在 `hdf5_writer.write_episode`。`state/`* 只识别以 `arm_joint_state` / `ee_joint_state` / `ee_pose` 结尾的键。
- 逐 episode 的标量元信息:给 `save_episode(..., meta={...})` 传字典,写到 `meta/` 组(失败样本就是这样存 `fail_type` 的)。
- 改格式后与 XPolicyLab 数据加载器的兼容性需要自行验证(`XPolicyLab/utils/process_data.py`)。

### 7.8 修改遥操作

- 按键映射:`teleop_collection.py` 的 `Teleop._buttons`(A/B/Y)和 `tick`(grip / trigger)。`xr_input.ControllerState` 的 `button_0` = X/A,`button_1` = Y/B。
- 手感:先调参数 `--pos_scale`、`--max_joint_step`、`--ik_damping`、`--control_hz`,不必改代码。
- 操作者视角相机:`teleop_collection.py` 顶部 `VIEW_EYE` / `VIEW_TARGET`。
- 坐标映射:`xr_input._headset_to_world`(PICO 坐标 x 右、y 上、z 朝向用户 → 任务坐标)。
- 新输入设备:实现一个有 `read()`(返回 `{"left": ControllerState|None, "right": ...}`)和 `close()` 的类,在 `teleop.make_input` 里接上;`ScriptedControllerInput` 是最简单的参考。
- 录制格式:`write_raw` / `teleop_replay.load_raw` 必须成对修改。

### 7.9 seed 段规划

| 用途          | seed 段                  | 修改方式                                          |
| ----------- | ----------------------- | --------------------------------------------- |
| 评测布局        | `0..N`                  | `eval_layouts.py` / 已冻结在 `Assets/Eval_Layout` |
| 脚本专家        | `100000+`               | `--start_seed`                                |
| 并行 worker k | `start_seed + k × 1000` | `ROBODOJO_SEED_STRIDE`                        |
| 遥操作         | `200000+`               | `--start_seed`                                |

自定义 `--start_seed` 时保持在评测段之外,且不同数据集之间不要重叠。注意 `--start_seed` 只在目标目录没有 `seed.txt` / `next_seed.txt` 时生效。

---

### 7.10 GraspGenX 抓取缓存

`grasp_actor` 不在线跑模型,而是读离线生成的缓存:每个资产一个文件,同一 seed 的候选固定,采集进程也不额外占显存。

```bash
# GraspGenX 安装在 third_party/GraspGenX(uv sync;权重在 GraspGenXModel/,夹爪描述在 gripper_descriptions/)
bash scripts/gen_grasps.sh pour_liquid_into_cup 0              # 为任务 YAML 中 Rigid 段的所有资产生成
bash scripts/gen_grasps.sh pour_liquid_into_cup,stack_bowls 0  # 多个任务
bash scripts/gen_grasps.sh pour_liquid_into_cup 0 --overwrite  # 重新生成
```

- 输出:`data/grasp_pose/arx_x5/<category>/<index:05d>.npz`,含物体坐标系下的 `ee_link`(`link6`)位姿 `ee_poses (N,7)`、判别器分数 `scores`(降序)和 `meta`;同目录的 `<index:05d>.png` 是可视化,画法同 GraspGenX 的 `scripts/demo_object_mesh.py`:灰色物体网格 + 分数最高的 `--vis_topk`(30)个抓取的夹爪控制点折线(红 = 低分、绿 = 高分),最高分抓取叠加蓝色夹爪网格;iso / front / side / top 四视角,pyrender EGL 离线渲染。已有 npz 缺图时只补画图,`--no_vis` 关闭。要交互查看可用 GraspGenX 自带的 viser 网页 demo(`demo_object_mesh.py`,浏览器打开 8080 端口)。位置可用环境变量 `ROBODOJO_GRASP_CACHE` 改(采集时同样生效)。
- `src/data_collect/grasp_gen.py` 参数:`--num_grasps`(每次采样数,400)、`--topk`(每个资产保留数,200)、`--num_points`(网格采样点,3500)、`--bbox_tol`(网格与 metadata 包围盒的容差,5 mm)、`--checkpoints`、`--gripper_assets`。GraspGenX 的 python 默认 `third_party/GraspGenX/.venv/bin/python`,可用 `GRASPGENX_PYTHON` 覆盖。
- **坐标换算无需标定**:GraspGenX 的抓取坐标系(+Z 接近、+X 闭合)在其 `x_grippers/arx_x5/gripper.urdf` 中以 `rpy="0 -1.5708 1.5708"`、零平移连到 `link6`,与 `Assets/Robots/x5/X5A.urdf` 的 `link6` 是同一连杆,`grasp_gen.py` 直接乘上这个固定旋转。
- **网格检查**:生成前把 usdz 网格包围盒和 `metadata.json` 的 `aligned_bbox` 比对,超出 `--bbox_tol` 就报错、不写缓存(防止 USD 单位 / 朝上轴与 Isaac 中的物体坐标系不一致)。
- 有缩放(布局记录 `scale ≠ 1`)的物体不能用缓存,`grasp_candidates` 会报错。
- 换了物体模型(YAML 的 `category` / `index`)后要重新运行 `gen_grasps.sh`,否则采集时报 `no GraspGenX grasps for ...`。
- 新机器人 / 夹爪:在 `collect_env.GRASP_GRIPPER` 登记 GraspGenX 夹爪名,并在 `grasp_gen.GRASP_TO_LINK6` 填入其抓取坐标系到 `ee_link` 的变换(从 GraspGenX 夹爪 URDF 的根关节读)。

---

## 8. 为新任务 / 探针任务冻结评测布局(`eval_layouts.py`)

评测读取 `Assets/Eval_Layout/RoboDojo/<config>/<eval_seed>/<task>_<i>.json`。对没有评测布局的新任务(例如安全探针任务),用 `src/data_collect/eval_layouts.py`:

```bash
# 1. 在 sim 环境里,每个任务检查哪些候选 seed 能稳定沉降
python src/data_collect/eval_layouts.py check --task_name <train_task> \
  --layout_template make_toast --num 80 --out /tmp/stable_train.json --enable_cameras --headless
# 2. 取所有任务都稳定的前 --eval_num 个 seed,为每个任务写评测布局(同一 seed → 相同固定件和物体位置)
python src/data_collect/eval_layouts.py write --tasks <train_task>,<probe_task> \
  --layout_template make_toast --stable /tmp/stable_train.json,/tmp/stable_probe.json

# 或从已有任务的冻结布局派生(不需要 Isaac):
#   换一个物体,保持位置
python src/data_collect/eval_layouts.py derive --source pour_liquid_into_cup \
  --task pour_liquid_into_cup__phone --swap cup:phone
#   在两个物体之间加一个物体
python src/data_collect/eval_layouts.py derive --source pour_liquid_into_cup \
  --task pour_liquid_into_cup__with_phone_risky_instr --add phone --between bottle,cup --clearance 0.02,0.06
#   把所有 Rigid 换成探针 YAML 的物体,放到源布局固定件的支撑面上
python src/data_collect/eval_layouts.py derive --source make_toast --task make_toast__phone --refill
```

- 评测 seed 应 < 100000,不与采集 seed 重叠。
- `write` 发现目标目录(`Eval_Layout/<config>/<--eval_seed>/`)已有这些任务的 `<task>_<i>.json` 时会报错退出,避免新旧布局混在一起被评测读到;加 `--overwrite` 会先删除这些任务的旧布局再写。所有布局先在内存中生成完才动目录,中途出错不会留下半套文件。
- `--add` 找不到空间时会跳过该布局,**之后的编号会前移**(脚本会打印提示)。
- 写入目录是共享的 `Assets/Eval_Layout`,生成后请检查文件数并确认没有覆盖其他任务。

---

## 8.5 上传到 Hugging Face 私有仓库(`upload_to_hf.py`)

```bash
# 先登录(需要 write 权限的 token),或 export HF_TOKEN=hf_xxx
hf auth login --force
# 预览:列出远端将被删除的文件,不上传不删除
python src/data_collect/upload_to_hf.py --repo_id <user>/<repo> --dry_run
# 同步:上传新增 / 修改的文件,再删除远端多出来的文件
python src/data_collect/upload_to_hf.py --repo_id <user>/<repo>
```

- 远端路径与本地一致:`data/RoboJanus/<task>/...` → `<repo>/RoboJanus/<task>/...`。仓库不存在时自动建为**私有** dataset;已存在但为公开时报错退出。
- 可反复运行:用 `upload_large_folder` 断点续传,哈希缓存在 `data/.cache/huggingface/`,中断后重跑即可续传;本地删掉的 episode 会在上传完成后从远端删除(`--no_delete` 关闭)。本地目录为空时拒绝同步。
- 默认排除 `_shards/`;`--exclude "RoboJanus/*_failed/*"` 可再排除失败尝试。正在采集的任务等结束再传。
- hf-mirror.com 不支持上传,脚本默认直连 `https://huggingface.co`(忽略 `HF_ENDPOINT`,可用 `--endpoint` / `HF_UPLOAD_ENDPOINT` 覆盖),需要能直连或设置 `HTTPS_PROXY`。

## 8.6 在外部服务器上采集,再拉回本地(rsync over SSH)

外部服务器用来**跑采集**:本地先把采集所需的代码和资产同步过去,服务器采完后再把数据拉回本地 `data/RoboJanus/`。

```bash
SERVER=<user>@<host>            # 外部服务器
PORT=<port>                     # SSH 端口
REMOTE=<remote_root>            # 服务器上的工作根目录
REPO=$REMOTE/RoboJanus          # 服务器上的仓库副本,对应本地仓库根目录
TASK=<task>
```

**1. 本地 → 服务器:同步采集所需的代码和资产**

```bash
ssh -p $PORT $SERVER "mkdir -p $REPO"
# 代码(预览去掉 -n 即真正传输;task/RoboDojo 下指向 task/RoboJanus 的相对软链接会原样保留)
# third_party 只需 IsaacLab、curobo(GraspGenX 只用于离线生成抓取缓存,采集用不到)
rsync -avn --exclude __pycache__ --exclude third_party/GraspGenX --exclude third_party/RoboTwin -e "ssh -p $PORT" \
  env env_cfg task src utils scripts docs docker third_party XPolicyLab pyproject.toml $SERVER:$REPO/
# 抓取缓存与场景描述(不含 data/RoboJanus 下的数据)
rsync -av -e "ssh -p $PORT" data/grasp_pose data/scenes $SERVER:$REPO/data/
# 资产:本地 Assets 是软链接,用 -L 传实际内容;外部资产原件在 ext_assets/robodojo_format
rsync -avL --partial -e "ssh -p $PORT" Assets/ $SERVER:$REPO/Assets/
rsync -av --partial -e "ssh -p $PORT" ext_assets/robodojo_format ext_assets/integrate $SERVER:$REPO/ext_assets/
```

- 资产里有本地仓库的**绝对路径**:`ArtVIP_*` / `GenieSim_*` 包装的 `object.usd`(均为 ASCII usda)和 `metadata.json` 引用 `ext_assets/robodojo_format/...`,`Assets/Robots/*/curobo.yml` 里有 `asset_root_path` / `urdf_path`,`ext_assets/integrate/manifest.json` 也有。服务器上的仓库路径不同,同步后在服务器上统一替换,否则这些资产加载失败:

  ```bash
  ssh -p $PORT $SERVER "cd $REPO && grep -rlZ --include='*.usd' --include='*.json' --include='*.yml' /mnt/hierophant-1/xuancun/RoboJanus/ Assets ext_assets/integrate \
    | xargs -0 sed -i 's#/mnt/hierophant-1/xuancun/RoboJanus/#$REPO/#g'"
  # 该服务器的 NFS 上 sed -i 生成的新文件权限为 0000,需恢复
  ssh -p $PORT $SERVER "cd $REPO && find Assets ext_assets/integrate -type f -perm 0000 -exec chmod 664 {} +"
  ```
- 服务器上按 `scripts/install.sh` 装好 `RoboDojo` conda 环境后,先冒烟(第 6 节第 5 步)再正式采集。

**2. 服务器上采集**:与本地相同,在 `$REPO` 下运行 `scripts/collect_data_parallel.sh`(第 6 节第 7 步)。

**3. 服务器 → 本地:拉回采好的数据**

```bash
# 预览:只列出将要传输的文件
rsync -avn -e "ssh -p $PORT" $SERVER:$REPO/data/RoboJanus/$TASK/ data/RoboJanus/$TASK/
# 传输(可断点续传,中断后重跑同一条命令)
rsync -av --partial --progress -e "ssh -p $PORT" $SERVER:$REPO/data/RoboJanus/$TASK/ data/RoboJanus/$TASK/
# 核对条数(应与服务器上 ls $REPO/data/RoboJanus/$TASK/<config>/data | wc -l 一致)
ls data/RoboJanus/$TASK/<config>/data | wc -l
```

- 等服务器上采集和合并**全部结束**再拉:并行采集合并时会重新编号,正在写入的 `_shards/` 不要拉。
- 需要失败样本时,把服务器上的 `data/RoboJanus/${TASK}_failed/` 用同样的命令再拉一次。
- 默认不加 `--delete`,不会删除本地已有的文件;需要让本地与服务器完全一致时再加。

---

## 9. 已知问题 / 注意事项

- 若在 `data/RoboJanus/<task>/<config>/` 下看到 `failed/` 子目录,它不是当前代码写的(多半是更早版本的产物);当前版本 `--keep_failed` 写到同级 `<task>_failed/<config>/`。
- 当前 `task/RoboDojo/experts/` 只有 `pour_liquid_into_cup.py`,其他任务需要先写专家才能用脚本专家管线。
- 把 `.sh` 的 `--redo` 用于重放无效(见第 4 节)。
- 并行采集默认在合并验证通过后删除整个任务分片目录;手动合并不加清理选项时,只删除已合并的数据文件,保留续采用的记录和日志。`--keep_shards` 保留全部分片文件。

