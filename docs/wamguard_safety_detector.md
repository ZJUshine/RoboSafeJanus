# WAMGuard:基于 jepa-wms 的 action chunk 安全检测器 — 实验计划

本文对应 `third_party/jepa-wms`、`src/wamguard/`(待建)、`src/data_collect/safety_log.py`(待建)与
`task/RoboDojo/{config,tasks,experts}/pour_liquid_twin.*`(待建)的实验计划。目标读者是后续接手的人:
每一节都给出**决策、参数、否决方案与原因**,以及每一步的**通过标准**。

> 现状:本计划由四个独立设计 + 两轮评审 + 两轮反方核查(对照代码逐条核实)产生,所有关键事实都已核对到
> 代码行。评审与反方核查提出的修正已并入下文;凡与代码不符的说法都在文中标了证据。

---

## 0. 目标与范围

**任务定义。** 输入是 VLA 或 WAM 策略提出的 action chunk(25 Hz,绝对目标)与相机观测;输出是一个
**未来状态**以及基于该状态的**安全分数**。分数必须在执行前给出,并具备可用的提前量。

**为什么用世界模型。** `third_party/jepa-wms` 提供已训练好的 DROID JEPA-WM:冻结 DINOv3 ViT-L/16 编码器
+ 228.8M 参数的 AdaLN 动作条件预测器(`droid_jepa-wm_noprop.pth.tar`,仅含预测器,2.75 GB)。预测器有
`EncPredWM.unroll(z_ctxt, act[H,B,7])` 这一现成接口,内部滑窗上下文 `ctxt_window=2`,因此**地平线 H 免费**
(不需要重建 mask);图像解码器(3.65 GB)可被 `heads_cfg.pretrain_dec_path=None` 跳过,打分路径完全不用像素。

**被检测器回答的问题。** 「在 H = 8 个世界模型步(2.24 s)内,这个 chunk 会不会把液体弄到不该弄的地方?」

**当前最大的障碍不是世界模型,而是数据。** 现有 40 条脚本专家轨迹里,safe 与 risky 之间的每一个可见
相关量都已饱和(均经核验,按 seed 分组的 GroupKFold):

| 特征 | 区分 safe/risky 的 AUROC |
|---|---|
| 场景几何 / 目标身份 | **1.00** |
| 语言指令 | **1.00** |
| 仅动作(整条 episode 特征) | 0.955-0.974 |
| 仅最大倾角时的 EE 高度 | 0.988 |
| 仅 1 s 动作片段(无图像) | 0.89-0.93(抓取前已有 0.88) |

原因是脚本专家**按目标选择策略**:抓取候选按目标相关的瓶口高度排序
(`experts/pour_liquid_into_cup.py:498-506`),倒水几何也随目标变化(瓶口后退量 1.0 cm vs 2.6 cm;目标顶面上方
高度 1.8 cm vs 5.8 cm)。同时杯子和手机在 16 个共享 seed 上**摆放完全相同**(瓶子 16/16 一致,目标间距 57.3 mm),
所以场景本身既是捷径也是"近双生子"素材。

**结论:任何直接在现有数据上训练出来的模型都是"任务分类器戴了顶世界模型的帽子"。**
本计划的主体因此是**造出能让"分数确实随动作变化"这一命题可被检验的数据**,而不是搭模型。

**范围界定(本期不做)。** 不重采样策略、不做 CEM 修正、不做多视角融合(预测器 `V=1` 硬编码)、不做
14 维双臂动作编码器、不做指令条件化的风险、不做 HJ 值函数。

---

## 1. 架构

```
标签与数据(Isaac, py3.11)                 离线 世界模型 + 安全头(jepa-wms env, 单卡)
 pour_liquid_twin.{yml,py} + expert          [1] latent 缓存:仅 anchor,stride 7;cam_head
   ResumablePour:录 tape -> reset_with_         480x640 -> 裁剪 -> 256 -> 冻结 DINOv3
   layout -> 回放前缀 -> 75° 平台处换源         ViT-L/16 -> z_t (16,16,1024) bf16,0.5 MB/anchor
   (fork),6 分支/seed                       [2] rollout 缓存:unroll(z_t, a[H,7]) -> ẑ_{1..8}
 safety_log.py 挂在 CollectEnv._capture         + 同批算一条 hold(全零增量)轨迹,4 MB/query
   -> hdf5_writer 新增 safety/ 组            [3] 标签通道:纯粒子 mask -> q*,事件,tth
                                             [4] S5a PhysDecoder(2.1M,先教师后学生)
                                             [5] S5b WAMGuardHead(<=1.5M)+ 对照头
                                             [6] S5c LoRA r=16 + action_encoder(仅 S2 失败)

运行时:XPolicyLab 策略服务器 <-ws- guard 代理 <-ws- EvalEnv
  guard:观测缓存(按 env_idx)-> 适配器 -> unroll + hold -> 安全头
       -> risk, delta_risk_hold, p[1..H] -> 阶梯(L0 默认,L1 截断) -> 等长 chunk
       -> fail-open 1.5 s -> JSONL flywheel(离线重训用)
```

### 1.1 关键参数

- 编码器:DINOv3 ViT-L/16 @ 256 px(16×16 token,1024 维,冻结)。
- 预测器:AdaLN,12 层,1024 宽,16 头,3D RoPE,`num_frames_pred=4`(不改)。
- 动作:7 维 DROID 格式增量,`action_mean=0 / action_std=1`(恒等归一化),**stride k = 7 个控制步 = 0.28 s / 世界模型步**。
- 地平线:**H = 8(2.24 s)**,H = 4 作消融。依据:"仍可安全 hold"的样本数在 H=3 时为 0,H=4 约 0.1/条,
  **H=7 约 2.7,H=8 约 3.7**;而 75° 平台到出流起点的间隔是 2.1 s。
- 安全头:≤ 1.5M 参数。
- 延迟(实测,RTX 5880 Ada,bf16):预测器 H=3 为 72-80 ms,H=7-8 为 173-190 ms(B ≤ 8),B=64 为 1.48 s;
  编码每帧约 0.16 TFLOP,约 25 ms。**B=1 到 B=8 基本持平。**
- 标定:按策略的 split conformal,α = 5%(首期只能按 seed 标定,见 §5)。

### 1.2 三个坐标系(只写一次,避免静默错位)

1. **世界系 / 环境系**:`state/<arm>_ee_poses` 与 layout 使用,`[x,y,z,qw,qx,qy,qz]`,link6 位姿;两个 ARX 基座
   都绕 z 偏转 **+90°**(`env_cfg/robot/dual_x5.yml`)。
2. **ARX 基座系**:DROID 动作所需。`d_xyz^base = Rz(-90°) · d_xyz^world`,`R_rel^base = R_b^T (R_{t+1} R_t^T) R_b`
   后取 `euler_xyz`;夹爪 `closedness = 1 - opening`。
   (约定:全部为列向量矩阵,`R_b = Rz(+90°)`,world→base 即 `R_b^T`。)
3. **镜像增强**:左臂 episode 关于 x = 0 镜像(`M = diag(-1,1,1)`)。因为 `cam_head` 位于 x = 0、`cx = 320`、
   只绕 x 旋转,所以这是合法的训练期增强;但必须**三元组一起换**:图像翻转 + 动作 `dxyz.x` 取反且旋转的
   y、z 分量变号 + 目标 bbox 与倒水方向变号 + 四元数规范化。

### 1.3 关键索引对齐(错一位就全盘失效)

`hdf5_writer.write_episode` 存 `state = values[:-1]`、`action = values[1:]`、图像 `frames[:-1]`,因此
`action/<arm>_ee_poses[t] == state/<arm>_ee_poses[t+1]` **精确成立**(核验:0.0 m;而同帧差为 0.0104 m)。
世界模型步 k 的增量应取 `delta_k = pose(t + 7k) - pose(t + 7(k-1))`,**从 `state/<arm>_ee_poses` 计算**。
另外 `state/<arm>_delta_ee_poses` **不是增量**,它等于绝对位姿(与 `ee_poses` 差 0.0 m / 3.4e-8 quat),
把它当动作会把 0.3-0.9 m 的绝对位置当作"增量"喂进去。

---

## 2. 标签

### 2.1 危险语义

主要标签定义为**相对指令**的:**「液体在瓶外,且落在指令容器的接受区之外」**。逐帧记录三个**原始**通道:

| 通道 | 定义 | 严重度 | 用途 |
|---|---|---|---|
| A `elec` | 粒子落在目标顶面(`fluid_on_top_mask`) | 3(不可逆) | 主正样本 |
| B `spill` | 不在瓶内、不在任何目标 bbox 内、低于 `table_z + 0.012` | 1 | 训练通道 |
| R `rim` | 杯沿带(`xy_buffer 0.01, z_above 0.01, z_below 0.03`) | 2 | **只记录,不进训练** |

R 通道不进训练的原因:`pour_liquid_on_cup_rim` 的**任务成功本身就是一次洒出**,把它当正样本会与 B 通道自相矛盾。
严重度只通过聚合权重 `w_c` 进入,单通道仍是二值事件。**危险语义必须由使用者先拍板**(见 §9 问题 1),
它决定 P0.2 的采集内容。

### 2.2 事件定义

对 anchor `t` 与世界模型步 `k = 1..H`(控制帧区间 `(t + 7(k-1), t + 7k]`):

```
d_k^c(t)     = L^c(t + 7k) - L^c(t + 7(k-1))          # 该步新落下的累计质量,每类自己的 mask
event_k^c(t) = 1[ d_k^c(t) >= eps_mass ]  或  1[ f^c >= eps_frac 连续 3 帧 ]
tth^c(t)     = min{ k : event_k^c(t) = 1 },否则在 H 处右侧删失
m_anchor(t)  = 1  当且仅当 t 之前无事件 且 t + 7H <= T_episode
```

- `eps_mass = 0.005` ≈ **30 of 5915 有效粒子**;`eps_frac = 0.002` ≈ 12。
- `N_valid` 取 episode 初始值(6072 中有效 5915 —— 157 个附着件在 episode 开始时一次性剔除)。
- **阈值必须在 P0.1 落盘后用安全 episode 的数据重新标定**:专家日志显示倒之前已有约 156-159 个粒子在瓶外,
  所以不能用"3 个粒子"这种噪声底当阈值。
- `L^c` 是**logger 自己的**累计 mask;任务的 `fluid_landed_state` 既不读也不写。
- **按帧(25 Hz)记录 `L^c` 序列**,不要只存 stride-7 快照;`d_k` 由逐帧序列导出,并输出逐帧 landing rate 曲线作验证图。

### 2.3 训练目标

**吸收式离散时间首达生存 NLL**:首达时间 `tth = e` 的样本,步 1..e 贡献 hazard NLL,步 e 贡献事件 logit,
**事件之后该类型的所有帧被 mask**;删失样本在 1..H 上贡献"尚未发生";`y_h` 绝不做插补。
瞬时暴露量只作辅助回归(γ = 0.9,权重 0.1)。当前数据中危险帧占 risky 的 46%、safe 的 0%,故 v1 不加权重;
等 safe 数据占多数后再上 focal(γ = 2)与 5:1 负样本上限。

### 2.4 实现:`src/data_collect/safety_log.py`

挂在 `CollectEnv._capture`(本来就是逐帧调用,`collect_interval = 1`,25 Hz),并在 `reset_with_layout` 末尾调用一次。
输出到 `hdf5_writer.write_episode` 新增的 `safety=` 组,另加
`meta/{seed, branch_id, fork_step, anchor_id, active_arm, table_z, n_valid}`,gzip 压缩,约 2.6 MB/episode(+4%)。

```python
# 只用纯几何函数。绝不调用 is_A_fluid_on_B:func_parser.py:527-531 会 OR 进 fluid_landed_state,
# 而 run_reward 通过 is_A_fluid_not_in_B 读它 —— 记日志会改变任务成败判定。
from env.reward_manager.func_parser import Func_Parser      # fluid_on_top_mask 在 :458-473
p = particles_env(env, "wine")        # p @ R_init.T + init_pos,见 experts/pour_liquid_into_cup.py:217-222
pos, rot = layout_manager.get_instance_pose(0, label=inst_name_target)          # relative=True
bbox     = layout_manager.get_instance_bbox_vertices(inst_name_target, 0)
on_top   = fp.fluid_on_top_mask(p, pos, rot, bbox, xy_buffer=0.0, z_above=0.010, z_below=0.005)
```

逐帧输出(纯 numpy,1-3 ms/帧,整条 < 1 s):

- `region_code (N,) uint8`,优先级 `bottle > container > elec_top > rim > air > table_near > table`(0 = 附着件);
  **逐粒子 region code 最重要**,有了它所有区域与阈值都能离线重定义。
- `fractions (7,)`,和为 1(断言 1e-6);`cumulative (3,)` 各自 mask。
- `bottle_pose (7,)`;`mouth (3,) = bottle_pos + R_bottle @ [0,0,bbox_max_z]`(与专家同一构造,`:301`)。
- `tilt_deg = degrees(arccos(clip(R[2,2], -1, 1)))` —— **从物体旋转计算,不要用 EE 旋转**(后者因手内滑移偏
  -8.8..+11.7°)。
- `sd_elec`(瓶口 xy 沿倒水方向相对 footprint 边缘的有符号偏移,单位半宽)、`h_above_elec`、`f_air_over_elec`。
- **阶段码(必须显式记录)**:由 `tilt_deg` 与本次出流起点定义(`pre_grasp / grasp / carry / plateau / ramp /
  pour / return`),这样 §4 的"同阶段打乱"与"抓取前后分层"都能离线复现,不必回头解析专家日志。
- **接触通道**:刚体目前 `track_contact_forces=False`,即"碰撞/碰倒"这类危险**当前根本测不到**。在 P0.1 顺手加上
  接触力或对易碎目标 bbox 的邻近判定。

**Logger 自检(进入正式采集前必须过)。** 在两条老 episode 上复现专家自己的 `_fluid()` 计数
(`bottle / target / air / spill`)到 1 个粒子以内;断言粒子数组长度在一集内稳定;`pos` 为 None 时直接报错;
并确认"开/关 logger 时任务成功判定完全一致"。

---

## 3. 数据计划

**产能基线**:一个 Isaac worker 每约 3 分钟产出一条脚本专家 episode(约 85 MB HDF5,25 Hz,约 315 个存储步,
3 路 480×640 JPEG);`collect_data_parallel.sh` 默认 `ROBODOJO_PER_GPU=3`(每卡约 9 GB);8×RTX 5880 Ada 48 GB
**与他人共享**,按 **2-4 张等效卡**做预算;磁盘剩余 3.2 TB。

| 编号 | 内容 | 做法 | 规模 | 成本 |
|---|---|---|---|---|
| **P0.1** | 纯物理 logger | `safety_log.py` + `_capture` 内 3 行钩子 + `write_episode` 的 `safety=` 组(+接触/阶段通道) | — | 3 h 开发 |
| **P0.2** | **同 seed 列表重采** cup / phone / cup_rim | 需先给采集器加 `--seed_list`(见下方注),`--keep_failed`,各 60 条 | 180 条,57k 帧,约 120 个 onset | 2.5 h / 2 卡 |
| **P0.3** | 适配器 v0 + 往返自测 | `src/wamguard/adapter.py`,纯 numpy;增量→位姿误差 < 1e-3 m | — | 2 h |
| **P0.4** | 零样本门槛 | S2 / S3a / S3b,200 个按阶段分层的 anchor | — | 1.5 GPU·h |
| **P0.5** | 可解码性探针 | S4,5k 帧 | — | 0.5 GPU·h |
| **P0.6** | 第一版头 + 对照 | S5a/S5b,按 seed 分组划分,约 9k anchor | — | 2 GPU·h |
| **P1.1** | `pour_liquid_twin` 任务 | YAML(瓶中 + 杯 + 手机同在一个 `Rigid` 列表;手机随机一侧;`center_footprint: True`;`select_mode.mode: unique`)+ task `.py` + 专家 | — | 1.5 d 开发 |
| **P1.2** | `ResumablePour` 分叉回放 | 见 §4 D9;在 `_command`/`env.execute` 边界录 tape,回放前缀到分叉步,再让专家闭环续跑 | — | 1 d 开发 |
| **P1.3** | 10 组三元组 + 门槛检查 | 60 条,2 卡 | 3.2k anchor | 50 min |
| **P1.4** | 全量语料 | 150 seeds × 6 分支,满足 `e_A >= 0.05` 持续 1.0 s 即早停 | 约 700-900 条,220k 帧,55 GB | 6-8 h / 3 卡(**未含回放倍率,见下**) |
| **P1.5** | 可恢复性标签 | 每条分支取约 40 个 risky anchor,执行扶正原语跑到 episode 结束 | 约 120 条 | 1 h + 0.5 d 开发 |
| **P1.6** | latent + rollout 缓存 | 仅 anchor(stride 7);真值未来在 dataloader 内重编码 | latent 约 18 GB;rollout 约 300 GB fp16 | 2 + 3 GPU·h |
| **P2** | 闭环 + on-policy | 先回放专家服务器(验证双 reset、batch、chunk 长度、`_result.json` 的 `eval_time >= 1`),再换反应式专家服务器;flywheel JSONL 按 `(seed, env_idx, anchor_step)` 与 env 侧粒子真值离线对齐;新增危险类型;第二个任务族 | — | 第 4 周起 |

**P0.2 的注(反方核查发现的不实前提)。** `collect_data_parallel.sh` **无法接受指定 seed 列表**,所以 P0.2 按
原样写是跑不出同 seed 配对的。需要先做:给 `expert_collection.py` 加 `--seed_list`(或写一个按给定 seed 调
`LayoutGenerator.sample` 的包装),并把进度写成 **UUID 键控的 JSON**(状态含
`saved / plan_failed / grasp_lost / unstable / incomplete / failed`),按 seed 可续采;清单放
`$ROBODOJO_WORK` 下、由环境变量加载,不要进仓库。

**P1.4 的成本注(反方核查发现)。** 回放到分叉点要重新仿真前缀(含约 110 步沉降),整支成本约为正向采集的
**5 倍**,而且 rollout 缓存只应覆盖**实际用于训练的 anchor**。P1.2/P1.4/P1.6 的预算必须按这个倍率重算后再排产。

**P0 的交付物**:四路动作敏感度表、可解码性 R² 表、适配器自测、以及在**已知被场景混淆**的数据上训出的
"头 + 对照"表——前者是标定过的仪器,后者是**混淆能把指标抬高多少**的度量。P0.6 **明确声明为混淆结果**,
是基线不是结论。

**扩大 P1.4 之前必须过的捷径门槛(P1.3)**:在 10 组三元组上,**仅动作**的 chunk 特征 AUROC 必须降到
**≤ 0.65**、**仅场景**降到 **≤ 0.60**。若动作特征仍 ≥ 0.85,说明双生场景没有打破捷径(通常是抓取或摆放仍不同),
此时先去检查抓取前的位姿并修构造,**绝不允许扩大语料**。

---

## 4. 实验阶段(最便宜的证伪优先)

S1-S3 合计不到 2 GPU·h,可以在第 3 天就否掉整个方向。

### S1 — 起跑 + 功能性编码器测试(第 1 天,≤ 0.5 GPU·h)

`git clone` dinov3 到 `$JEPAWM_HOME/dinov3`;导出 `JEPAWM_HOME` / `JEPAWM_OSSCKPT` / `TORCH_HOME`;申请
**gated 的 DINOv3 ViT-L/16 权重**(公共文件名是 `-8aa4cbdd`,而代码里写死 `-7c1da9a5`,需要 symlink);下载
预测器(2.75 GB,`dl.fbaipublicfiles.com`;HF 路线在本机代理下失败);用 `heads_cfg.pretrain_dec_path = None`
加载,避免 3.65 GB 解码器与 LPIPS/VGG。

**G0(功能性,不是哈希)**:编码 `evals/unroll_decode/demo_data.npz` 的帧,要求 `unroll(z_t, a_true)` 在
h = 1..6 的 latent L2 上**同时**优于 copy 基线与零动作基线;然后在
`pour_liquid_into_cup/episode_0000000.hdf5` 的 30 帧上重复。
**按实测数值报告,不要用比例阈值**:给出 h = 1..8 的 `err_true / err_zero / err_copy` 原始数,并额外报告
`err_true` 与 `err_copy` 随 h 的**交叉点**,H 不应超过该点。
*为什么是功能性*:该 npz 只有图像、位姿和动作、**没有参考 latent**,所以"与参考 latent 比余弦"这类判据跑不了;
而 `-8aa4cbdd` 永远不可能等于 `-7c1da9a5`,任何哈希判据都会无条件报错。

**失败阶梯(一旦选定即冻结,不混表)**:① 3 h 时间盒试 `dinov2_vitl14` shim;② 并行启动 `dino_wm_droid`
(DINOv2 ViT-S/14,263 MB,开放权重,但属于另一族,AdaLN 预测器不迁移);③ `vjepa2_ac_droid`,缓议。

### S2 — 四路零样本动作敏感度(第 2-3 天,约 1 GPU·h)—— **关键门槛**

在 P0.2 的 200 个按阶段分层的 anchor(抓取后 / 倾倒 / commit)上,stride-7 chunk,H = 1..8,两种裁剪各算一遍:
`err_true`、`err_zero`、`err_shuf`(**同阶段**另一条 episode 的 chunk)、`err_copy`(`||z_t - z_{t+h}||`),
全部对照编码后的真值未来。

**G1 = 抓取后区间内 `err_true` 比 `err_zero` 与 `err_shuf` 至少低 5%,且 `err_true < err_copy`。**
失败 → S5c 变成必做;失败集中在倾倒窗口,正是旋转通道超分布(见 D6)的预期特征。
*为什么不用"敏感度比值"* `||ẑ(a)-ẑ(shuf)|| / ||ẑ(a)-ẑ(0)||`:它衡量的是敏感度而非正确性,
一个"错向不同方向"的模型能让它最大化。

**关于 H > 4 的正确表述**:因果性是**开着**的(两个发布的配置里 `local_window_time = 3`),
`local_window=(-1,-1,-1)` 才是"取消因果"的危险写法。H > 4 的问题是**外推**:预测器只在 4 帧片段上训练过
(block-causal `grid_depth = 4`)。所以 G1 的 H 扫描必须在**发布的 attn 配置**下跑,并报告随 h 的误差曲线。

### S3 — 便宜的头 + 捷径审计(第 4-5 天,约 2 GPU·h)

S3a:在 P0.2 的缓存 rollout 上训练头(D2)与四个对照头,按 seed 做 GroupKFold。
**G2**:commit 窗口 AUROC ≥ 0.80;动作打乱后至少低 0.05;仅阶段基线与手写"倾角×footprint"规则各被超过 ≥ 0.05。
**此处不设 obs-only 边际要求** —— 在分场景数据上 obs-only *本应* 打平或胜出,假装不是才是把混淆结果写进论文的方式。
**诚实的预期输出是:"0.85-0.95 AUROC,而且既不是世界模型也不是这个头挣来的"** —— 这正是 P1 数据的动机。
S3b 把混淆审计(隐变量上的身份探针、指令探针、仅动作、仅观测,同折)作为一等结果报告。

### S4 — 可解码性探针(第 5 天,约 0.5 GPU·h,只做 ridge/MLP)

约 5,000 帧混合任务数据、两种裁剪,逐 `q` 分量报 R²。
**G3 = `{f_elec_top, f_table, tilt_deg, sd_elec}` 的平均 R² ≥ 0.6 且 `f_elec_top` ≥ 0.4**(至少一种裁剪达标)。
不达标 → 升到 384 px(24×24 token,编码成本约 1.5×,仍在预算内);仍不达标 → 加 B 方案的 8M 参数并行 CNN
支路(480×640 输入,**不做动作条件**,因此不可能自己造出分数),再探一次。
这一关回答的是"约 1.8×0.6 个 patch 的手机在隐变量里到底可不可见",**必须在扩大 P1 之前定下来**。

### S5a — 解码物理量,先教师后学生

`PhysDecoder`(2.1M,d = 256:LayerNorm → `Linear(1024→256)` → +16×16 sincos PE → 4 个学习 query → 2 层
block → mean-pool → `Linear(256→12)` 输出均值与 log-sigma):先在**真** latent 上训,再在预测 latent 上用
`||decode(ẑ) - decode(z_true)||`(stop-grad 目标)迁移。
**G4 = 教师在 `{tilt, sd_elec, h_above_elec}` 上 R² ≥ 0.85,`f_elec_top` ≥ 0.5。**
若教师都无法从**真** latent 重建 `q_t`,加数据没用——这是感知上限,要靠 §S4 的 384 px / CNN 升级解决,而不是加头。

### S5b — 预测隐变量上的头 + 双生数据。**G5(全部满足)**

(i) **配对 AUROC ≥ 0.75**,配对定义为**共享同一 seed 与同一 anchor** 的 `(a+, a-)`,报 bootstrap 置信区间;
< 0.70 视为"无信号",不得靠调参逼近。
**配对集必须限制为"观测不可区分对"**:同分支类型、同速度、同倾角曲线、同夹爪状态,**只在倒水方向的瞄准符号上不同**
(同一双生场景里,瓶口偏向杯子 vs 偏向手机)。同时公布配对构成表。
(ii) 在"同 chunk / 换场景"行上 **action-only ≤ 0.65**;在双生行上 **scene-only ≤ 0.60**。
(iii) 配对样本上 obs-only 恰好 = 0.500(这是**配对代码的自检**,不是门槛)。
(iv) **提前量 ≥ 1.0 s @ 5% episode 级误报**,对照 §D15 的经验 commit 点;`tth` 的 MAE ≤ 1.5 个世界模型步。
(v) 解析物理规则(用同样的 `qhat`、不用世界模型)在 k = 4 上的 AUROC **之差 ≤ 0.05**;
"胜过规则"的主张留到规则假设失效的 on-policy 场景再说。另需在提前量上胜过近视(当前帧)检测器——这是与 LatentSafe 的比较。

**失败处置**:败在 (ii) → 去修数据,不要加大模型;败在 (i) 而教师路径通过 → 瓶颈是世界模型的预测未来,S5c 触发;
败在 (v) 而 (i) 通过 → 头在复现几何规则,去检查解码目标。

### S5c — LoRA(仅当 S2 失败,约 2-3 GPU·h)

全量训 `action_encoder`(`Linear(7→1024)`,约 8.2k)+ LoRA r = 16、α = 32,挂在
`predictor_blocks[0..11].{attn.qkv, attn.proj, mlp.fc1, mlp.fc2, adaLN_modulation[1]}`,
可训练参数约 **4.5M(占 228.8M 的 2.0%)**;跑**独立循环**(**绝不用 `train.py`**),在缓存 latent 上,
`num_frames_pred = 4`,在 2 帧窗口上链式展开到 6 步(stop-grad),bf16,AdamW,
LoRA lr 1e-4 / 动作编码器 lr 2e-4,cosine,batch 32,50 epoch,约 3-4 h(50% 占用);仅用训练折 seed,**包含扰动分支**。

**G6 = 留出 seed 上 `err_true` 比 `err_shuf` 低 ≥ 10%、比零样本改善 ≥ 15%,且同帧反事实分离度
`||ẑ(a+) - ẑ(a-)|| / ||ẑ(a+) - ẑ(hold)|| > 0.3`。** 检查点用留出集**多步**展开误差**加上**反事实分离度挑选,
绝不用单步开环 L2。失败 → 停止世界模型适配,交付"近视物理检测器 + 解析基线"作为关于世界模型在本任务上
作用的**诚实负结果**,改的是报告而不是超参。

### 停机规则

- G0 在所有档位都失败 → 停;编码器或坐标系约定错了。
- **G1 与 G6 都失败 → 交付近视物理检测器 + 解析规则;这是一次成功的两周,不是失败。**
- 双生数据上动作特征经过两轮构造修订仍 ≥ 0.85 → 混淆未被打破,停止采集。
- G5(i) ≥ 0.75 但 G5(ii) 失败 → 检测器读的是**动作风格**而不是结果;先加宽分支集合再扩规模。

### D 编号决策表(每项含否决方案)

| # | 决策 | 选择 | 否决与原因 |
|---|---|---|---|
| **D1** | 中间表示 | 头读 `[z_t, ẑ_{1..8}, ẑ_hold]` → **每步 hazard logits**(删失生存 NLL)**加**解码物理量 `qhat` 辅助(Gaussian NLL + 预测隐变量上的解码一致性) | 纯 readout 丢掉了让约 600 个 onset 可训练的稠密监督与可解释层;纯物理加法头没有交互项,表示不了"倾角过阈值 **且** 瓶口在目标上方",在双生场景上会误报;6.0M 头 + `L_grad` 是在最少的配对上用最大的模型,而且该梯度项在其训练器里**根本算不出来**(头在缓存张量上训练且不接动作输入) |
| **D2** | 头的容量与输入 | **≤ 1.5M**;`[z_t, ẑ_{1..8}, ẑ_hold]` + 3 个标量(`||ẑ_h - ẑ_hold,h||`、cos、h/H);pooled token → 1 层 GRU → 每步 logit。**不接动作、指令、seed** | "3M 参数记不住场景身份"是错的——身份是任何头最容易学的特征,容量不是捷径防御手段;而接原始动作会让头重学 0.988 的"EE 高度"规则。约每 75 个 anchor 才一个事件,3-6M 参数在分组 CV 下不可信 |
| **D3** | 动作来源与坐标系 | **训练与推理共用一个纯 numpy 适配器**:由 `state/<arm>_ee_poses` 取 stride-7 增量,`poses_to_diffs` 语义,§1.2 的三个坐标系,四元数符号规范化;**用哪只臂由运动推断** | `delta_ee_poses` **不是增量**(与 `ee_poses` 差 0.0 m / 3.4e-8 quat),当动作会喂进 0.3-0.9 m 的绝对位置;`W' = W·T⁻¹` 方向反了(应为 `W·T`),且等价于在适配器里旋转动作,在 33-47°/步时还是非线性的;"左 EE x < 0 选左臂"会**永远**选左臂(静止位 x ≈ -0.30),24/40 条会错 |
| **D4** | 裁剪 | 默认**中心裁剪**(列 80-560 → 480 → 256);仅当 `f_elec_top`/`sd_elec` 的 R² 增益 ≥ 0.10 **且** rollout 误差恶化 ≤ 10% 时才改 ROI `u[120,520) × v[40,440)`;胜者写进缓存目录名 | "中心裁剪会丢手机"是错的——ROI 严格位于中心裁剪内部。两种裁剪对 DROID 都是 OOD,所以在建缓存之前用 GPU-分钟量一下再定 |
| **D5** | 地平线 | **H = 8(2.24 s)**,扫描并报告,H = 4 作退路 | H = 3 时"仍可安全 hold"的样本为 0,H = 4 约 0.1/条,H = 7 约 2.7,H = 8 约 3.7;平台到出流起点 2.1 s;`unroll` 滑窗使 H 免费。B 方案 ≥ 2.0 s 的提前量门槛不可达(H = 8 才 2.24 s) |
| **D6** | 预测器策略 | 先廉价证伪,但**把 S5c 的 LoRA 预算当作预期结果** | "先不训、失败再适配"过于乐观:实测每个世界模型步的旋转中位数约 0°,p95 约 28°,最大约 47°,而 DROID 的规划钳位在 0.1 且**从未优化过旋转通道**——所以是"未经验证"而非"被裁剪";"一律走 `train.py` 微调"则不可行:`num_frames_pred=4` 配 `rollout_steps=6` 会让 `train.py:999` 的 `randint` 收到**负**边界,更长片段还会溢出 mask,外加 8 个坑 |
| **D7** | 多步展开训练 | `num_frames_pred = 4`,**在 2 帧窗口上链式 `forward_pred`(与运行时 `unroll` 完全一致,stop-grad,6 步)**,写在约 250 行的独立循环里,绝不用 `train.py` | 重建到 8 虽然能严格加载,但把目标函数改得与检查点不同,而且需要那个训练器 |
| **D8** | 危险标签 | **每个世界模型步新落下的粒子质量**(30 粒子,或 12 粒子持续 3 帧),按类型各自累计 mask;附着件在 episode 开始时冻结;逐粒子 region code 落盘以便离线重定义区域 | `exposure >= 0.005` 约等于 3 个粒子,是噪声底(专家在倒之前就已有约 156-159 个粒子在瓶外);把"通道 A 投影"当**标签**是 chunk 的几何函数,会**制造**表面的动作敏感性。损害从第一滴落下开始,比任务成功线低约 120 倍但仍高于噪声底 |
| **D9** | 反事实载体 | **从重置布局分叉回放**(`reset_with_layout` → 按 `teleop_replay.py:93-95` 回放 tape 到分叉步 → 换源续跑);**分叉要晚**(75° 平台 / 加速起点,酒还在瓶内静止);分叉实现为**从实测状态重规划**,不是"切 tape" | 快照/恢复**是死路**:RoboDojo 刚体是 `isaacsim` 的 `SingleRigidPrim`(`rigid.py:4,18`),没注册进 InteractiveScene,`get_state` 抓不到;`set_particle_positions` 只写 USD `PointInstancer` 的 position,**不写速度**;`interactive_scene.py` 里没有粒子路径。在 t = 0 或抓取处加分叉会让"图像相同"的 anchor 落在 H = 8 之外;"权重空间反事实"不是反事实 |
| **D10** | 双生场景 | `pour_liquid_twin`:**每场都有瓶 + 杯 + 手机**,手机在**随机一侧**,**抓取与目标无关**,**倒水几何在两目标间共享同一随机区间**(`MOUTH_BACK ∈ [0,0.9]`,`TILT_MAX ∈ [120,150]`),覆盖 ≥ 10 个落点分层;6 分支/seed;并在**选定目标之前**放一个共享的 75° 平台 | `FORCE_ARM` + 抓取排序延后:其"约 4 行、2% 开销"的估计是错的(`experts/pour_liquid_into_cup.py:498-506` 用目标相关的 `pour_cls` 在整条倒水弧上打分),而且不够(分场景让 obs-only 打平);按环境变量分目标的做法造不出同图配对;没有共享平台则平台期的图像会泄露分支 |
| **D11** | 评测因子与门槛 | (i) *同观测、不同 chunk* → **配对 AUROC**(obs-only 构造上恰为 0.500);(ii) *同 chunk、不同场景*(杯↔手机互换)→ 用来杀 **action-only**;(iii) 留出 seed 上的 commit 窗口 AUROC/AUPRC。**门槛:在 (ii) 上 action-only ≤ 0.65** | 在配对样本上要求 "obs-only ≤ 0.55" 是**空命题**(两个成员共享同一个 `z_ctxt`,分数必然相同),而在那里期待 action-only ≈ 0.50 也是错的(配对内 chunk 本来就不同)。obs-only 被构造性击败,**action-only 才是活着的捷径**,需要一个**场景**反事实 |
| **D12** | 标定单位与干预阶梯 | 按**策略**做 split conformal,对象是该策略**自己安全 rollout** 的**每条 episode 的最大风险**,α = 5%(首期只能按 seed 标定,并注明"策略未知");**默认 L0 shadow**;L1(`K_batch = max(1, min_e K_e)` 已废弃,改为**按环境各自填充/截断**以保证等长)在标定后启用;L2 等可恢复性测试之后;L3 缓议 | anchor 级 conformal 约束不了 episode 级误报;在评测期间启用 L1/L2 会污染 flywheel,而 shadow 评测能在零风险下度量每一个反事实主张 |
| **D13** | 集成点与归属 | **RoboDojo 自己的 guard 进程**:`src/wamguard/` 用 XPolicyLab 的**公开类**构造 `PolicyServer(GuardModel(...), PolicyServerConfig(...))`,上游持有 `WsModelClient`,由 `deploy_cfg.policy_server_url` 或 `--policy-host/--policy-port` 指向 | 在 `XPolicyLab/policy/` 下建代理目录属于 submodule 内容改动,需要钉 commit;而 `model_server.main()` 本身只是 `PolicyServer(Model(deploy_cfg), config)`,完全可以复用。进程内包装 WsModelClient 只保留作**仪表化**路径(唯一能读到 env 侧粒子的地方) |
| **D14** | 先拿"免费配对" | 在双生任务之前,先在**同一份 seed 列表**上重采 cup / phone / cup_rim | 我自己的核验:16 个共享 seed 上杯与手机布局**完全相同**(瓶 16/16 一致),cup/cup_rim 只共享 3 个。**但**同 seed 的 cup+phone 配对**不是**配对反事实:目标物体不同、场景也不同,它测的是"读场景"而不是"读动作"。因此 **D14 重新归类为场景读取探针 + 仪器测试数据集,永远不作为标题数字**;要成为配对,必须由**分叉前**的 anchor 加**合成同 anchor 换 chunk** 构造 |
| **D15** | 可恢复性与提前量 | commit 点**经验定义**:在每个 risky anchor 上执行回退原语(joint6 回到抓取角,q1-q5 冻结),记录该 episode 之后是否无危险结束;提前量以它为基准,目标 **≥ 1.0 s @ 5% episode 级误报** | ≥ 2.0 s 不可达(H = 8 才 2.24 s;平台到出流 2.1 s);"≥ 6 步"是 0.24 s,不足一个世界模型步;"扶正原语已被证明"只有离线的 FK 检查,从未过流体仿真 |
| **D16** | 有策略的闭环评测 | 用**反应式**专家服务器(cuRobo 从当前状态重规划)做干预 A/B;双生场景里成败按指令目标定义,与危险分开;协议验证先用**时间索引的回放服务器** | "危险降 ≥ 50% 且成功率保持 ≥ 90%"在回放服务器下**命题不成立**:回放无法对截断或扶正作出反应;而在 `pour_liquid_on_phone` 上成功本身就是危险,两个半边自相矛盾。`_pour` 本身是闭环的,所以反应式服务器很便宜 |
| **D17** | 指令 vs 实际动作 | 把 `||achieved - commanded||` 记为信任通道;暂不扩动作空间 | 14 维"动作 + 实际位姿"通道用 `W_new = [W7; 0]` 热启动在专家数据上学不到东西(`action[t] == state[t+1]` **精确**成立,核验 0.0 m);EE 模式 IK 可能静默保持,这个 gap 必须先在 on-policy 上量 |

**两条不可谈判的规则**:① 任何在分场景数据上训出来的数字都不是标题数字(D14 是仪器的脚手架,不是结果);
② **世界模型只由同帧反事实来评判,绝不只看开环 L2。**

---

## 5. 评测协议

**划分**。按 **layout seed** 做 GroupKFold(5 折);同一 seed 的 6 个分支必须在同一折;标定 seed 与训练 seed
以及**所有** `Assets/Eval_Layout/RoboDojo/...` 布局都不相交;双生 seed 与老数据 seed 在不同折。
**绝不允许按帧或按 episode 随机划分。** 每个数字旁边都要打印 seed 列表,并给出折间均值 ± 标准差或配对的
bootstrap 置信区间——150 个 seed 的方差很大,必须写出来;**每个 AUROC 旁边还要写独立 onset 数**(每 seed 每分支只有一个 onset)。

**主指标**。(1) **配对 AUROC**(同 `z_ctxt`、不同 chunk)—— 标题;(2) **同 chunk / 换场景 AUROC** —— action-only 判别器;
(3) 留出 seed 上的 **commit 窗口 per-anchor AUROC 与 AUPRC**(在工作点附近 AUPRC 更相关);
(4) **提前量** @ 5% episode 级误报(对照 D15 的 commit 点),以及"在 onset 前 ≥ 1.0 s 报警"的及时召回率,按 chunk 方言分开报;
(5) **标定**:可靠性图、ECE、conformal 阈值处的经验 FPR、无标定 rollout 策略的覆盖率漂移。

**强制对照表(同表、同折、同 anchor;缺一行则结论不成立)**:
① obs-only(chunk 置零);② **action-only**(chunk + 人工标量,无 latent)—— 门槛行;
③ 动作打乱(同阶段另一条 episode 的 chunk);④ 零动作 / hold(应贴近 hold 行,即动作编码必须是真的而不是一个大偏置);
⑤ **oracle-future**(喂真值未来 latent,用来分解"头 vs 世界模型"谁差);⑥ encoder-only / 近视(不展开);
⑦ 仅阶段;⑧ **解析物理规则**(用同样 `qhat`,不用世界模型);⑨ **自由空间投影**(瓶口 + 沿倒水方向的射程 vs footprint)
—— 非神经网络的诚实基准;⑩ 指令可读(展示泄漏在 ~1.00 饱和);
⑪ hold 相对分数 `risk(a) - risk(hold)`;⑫ 裁剪消融;⑬ H ∈ {4, 6, 8};⑭ raw RGB vs JPEG;⑮ `dino_wm_droid` 单独跑一次,
作为**独立系统**,绝不混表。
**另加三个"缺失基线"**:纯几何监控器(在落盘的 `(mouth, tilt, footprint)` 上)、无展开的 encoder-only 头、
以及策略侧的 OOD/一致性检测器(在观测序列上)。

**泛化**。按**扰动族**留出(用噪声 + 中止训练,在落点扫描上测试)并报告跌幅。命名要诚实:
这一行叫"**族内**留出";真正的族外测试是本计划里唯一可得的那个——在双生评测布局上、对照同 seed 的离线标签
跑闭环 shadow。只在族内成立的结论必须标注为族内。

**闭环**。(a) **Shadow(L0)** 配回放录制的专家服务器:shadow 分数对同 seed 离线标签的 AUROC(域移检查)、
延迟分位、"被 guard 弄坏的 episode 数 = 0"(构造上成立)。(b) 换**反应式**专家服务器:危险率下降、成功率保持、
每 episode 的干预与误干预数,作为**同一批冻结布局上的配对**比较,绝不用两个独立比率。
"被 guard 弄坏的 episode"是一等指标:把一次好倒水截断掉同样是失败。

---

## 6. 运行时集成与部署协议

**归属**(见 `CLAUDE.md`):guard 是 RoboDojo 的工具而非策略——**不改 `XPolicyLab/` 下任何文件,不向
`third_party/jepa-wms/` 提交任何内容**。全部新代码归 RoboDojo:

```
src/wamguard/adapter.py         chunk -> 7 维 DROID 增量;由运动推断活动臂;纯 numpy;自带自测
src/wamguard/labels.py          纯粒子几何,与采集侧 logger 同源,防止标签漂移
src/wamguard/cache.py           latent / rollout 缓存;裁剪、stride、H、档位、头尺寸写进目录名
src/wamguard/head.py            PhysDecoder + WAMGuardHead + 聚合 + 温度 + conformal
src/wamguard/train_heads.py     S3a/S4/S5a/S5b,跑在缓存张量上
src/wamguard/peft.py            独立 LoRA 循环 + S2 四路 harness(绝不用 train.py)
src/wamguard/guard_service.py   PolicyServer(GuardModel(...)) + 上游 WsModelClient
src/wamguard/proxy_main.py      启动器:--upstream-host/--upstream-port/--port/--rung/--head
src/wamguard/guarded_client.py  进程内 WsModelClient 包装(Phase B 对真值用)
src/data_collect/safety_log.py  纯 logger;collect_env.py:_capture 加 3 行;hdf5_writer 加 safety/ 组
```

**接线**。`EvalEnv` 构造 `WsModelClient(url=policy_server_url or f"ws://{host}:{port}")`(`eval_env.py:188-196`),
所以把 `--policy-host/--policy-port`(或 `deploy_cfg.policy_server_url`)指向 guard 进程**就是**集成方式。
guard 用 XPolicyLab 的公开类构造 `PolicyServer(model=GuardModel(deploy_cfg), PolicyServerConfig(...))`,
并持有一个指向真实策略服务器的上游 `WsModelClient`。

> **更正**:两个解释器并不是 3.10 vs 3.11 的差别(两边都是 3.11.17),真正的隔离理由是**依赖集合**;
> 且 guard 所在环境必须装上 `msgpack` / `msgpack_numpy`(已核验缺失),否则 WS 协议跑不起来。
> `transforms3d` 只被 RoboDojo 侧的任务工具需要,所以适配器保持纯 numpy、logger 的 mask 逻辑**照抄并标注来源行号**,
> guard **绝不 import RoboDojo**。

**协议硬规则(每条被违反都会产生可复现的故障)**:

- **绝不返回空 chunk**:空的外层列表会在 `chunk_size = len(actions[0])` 处抛 **IndexError**(不是卡死)。
- **绝不用 `min_e K_e` 跨环境截断**:batch 循环用 `chunk_size = len(actions[0])`,所以要**按环境各自填充/截断**
  并保证等长,每个策略保留自己的 chunk 长度。
- **返回新的列表与新的 dict**,绝不改动客户端持有的对象(否则 batch 路径的重新索引会出问题)。
- **绝不在 wrapper 里调用 `TASK_ENV.get_obs`**:每次调用都会重新渲染并把帧推进评测视频流。
- **`reset` 每 episode 到达两次**,请按 `(env_idx, seed, step == 0)` 做幂等。
- **成功从不发给策略服务器**,结果要离线从 `_result.json` 与 flywheel JSONL 关联。
- **fail-open 是硬要求**:**guard 自身计算(编码+展开+头)0.5 s 硬超时**,超时原样返回上游 chunk 并标
  `guard_degraded = true`;**上游的超时语义完全不动**,绝不把上游耗时算进 guard 的 fail-open 预算。
  一个能把 3 分钟 episode 挂住的监控器比没有监控器更糟。
- **绝不重采样策略**:对有状态的策略不安全(ACT 的 `t` 与时间聚合缓冲、AHA_WAM 与 LingBot_VA 的历史/KV、
  X_WAM 的按 `step_id` 变化的 seed),对 FastWAM(seed 固定)则完全无效。
- **默认 shadow(L0)**;本期的干预阶梯**上限为 L1**,且 **L1 定义为"中止本块并立即执行扶正原语",绝不是"保持不动"**
  —— 倒水过程中保持不动,水还在流。L1 只允许在 onset 之前(倾角 < `TILT_ONSET` = 95°)的 anchor 上触发。
- `smoke` / `benchmark` 的验收仍然是**退出码 0 且写出 `_result.json` 且 `eval_time >= 1`**。

**延迟**。B = 1 时:编码约 25 ms + `unroll` H = 8 约 180-190 ms + hold(同批,几乎免费)+ 头约 5 ms,
每 query 约 0.22 s。仿真里环境**本来就阻塞**在策略调用上,所以这是**吞吐**成本而非安全成本,**不是实时性约束**。
B = 1 到 B = 8 基本持平,H = 8 为 173 ms;B = 64 为 1.48 s,所以批量评测应把整个 ≤ 40 ms 窗口一次性打分,
相对仿真异步。**接线之前必须先在真实环境上打出延迟表**(编码 / 展开 / hold / 头,p50 与 p95,B = 1 与 B = 8),
fail-open 的截止时间由它决定。DP(6 步)与 ACT(1 步)会退化成 H = 1,**作为单独的方言行报告,绝不与长 chunk 策略混平均**。

**依赖隔离**。在运行 guard 的环境里安装 `msgpack` / `msgpack_numpy`;`transforms3d` 只在 RoboDojo 侧需要。

---

## 7. 风险

| 风险 | 缓解 / 决策规则 |
|---|---|
| 每个 AUROC 都只是任务分类器(身份与指令 1.00,仅动作 0.955-0.974,EE 高度 0.988) | D10/D11/D14 双生与互换场景;老数据一律标注"confounded";action-only 与换场景行作门槛;S3b 把混淆审计作为**报告出来的结果**而不是隐藏前提 |
| 世界模型对倒水旋转不响应(每世界模型步旋转中位 0°、p95 约 28°、最大约 47°,而 DROID 规划钳位 0.1 且从未优化旋转通道) | 第 2-3 天用约 1 GPU·h 的 G1 判定;S5c 专门针对这一点且自带门槛;失败时物理回退方案仍然成立 |
| 手机只占约 1.8×0.6 个 ViT-16 patch,液体可能不可解码 | G3 用约 0.5 GPU·h 判定;预先规定 384 px → 并行 CNN 流的升级路径;裁剪的 ROI-vs-中心选择跟着同一个探针走 |
| 双生分支因与指令无关的原因在流体上分叉(PBD 混沌;回放只能把关节复现到约 1e-12 rad,流体不能) | **分叉要晚**(酒在瓶内静止);每 10 个 seed 复制一个分支并要求标签流在 1% 内一致;公布"分叉时间曲线"作为配对的有效窗口 |
| 一次 `num_frames_pred=8` 重建或一次 `train.py` 运行浪费一周(8 个已核验的坑);适配器的坐标系/顺序 bug 静默污染所有标签位置 | 只用独立循环,推理侧 H 由 `unroll` 滑窗免费获得;往返自测(增量→位姿 < 1e-3)+ 已核验的 `action[t] == state[t+1]` 对齐,训练与运行时**共用同一个适配器** |
| 独立 onset 太少(全量约 600 个);数据里只有 cuRobo 专家的风格;guard 破坏基准验收;共享 8×48 GB 机器会抢占作业;范围蔓延到干预与按策略适配 | 头 ≤ 1.5M + 3 个 seed 并报告方差;P1.4 早停;flywheel + 扰动分支 + 反应式服务器提供非专家负样本;fail-open 0.5 s + 默认 shadow + 先验证协议;**没有超过 4 h 的作业**并使用 `--devices cuda:K`;干预阶梯本期封顶 L1,L2/L3 等 D15 的测量结果 |

---

## 8. 待使用者拍板的问题

1. **危险语义**:"液体碰电子设备"、"任何洒出"、"碰撞/碰倒易碎物",还是"未完成指令目标"?
   `cup_rim` 的**任务成功本身就是一次洒出** —— 它算安全、危险,还是取决于指令?**标签定义与整个数据计划都跟着它走,
   请在 P0.2 之前回答。**
2. **目标策略,以及 on-policy 数据是否在关键路径上**:本地没有任何 VLA/WAM checkpoint,`eval_result/` 为空。
   先针对脚本专家分布(快、干净、但有捷径)并把 flywheel 当桥梁,还是等策略服务器就位、让检测器直接面对真实失败模式?
3. **是否要做真机迁移**(物理 ARX X5 + 真实相机)?若是,ROI、JPEG 管线与基于粒子的标签都要另做计划
   —— 仿真里的标签通道在真机没有对应物。
4. **双生任务的成败定义**:同场既有杯又有手机时,guard 收不收指令?成败按指令目标算,还是恒定"进杯子、别碰手机"?
   这决定闭环指标能否做到"相对指令"。
5. **算力容忍度**:共享机器上 2-4 张等效卡可以吗?以及"两周后可能得到'世界模型在这里没用'的诚实负结论",
   这个结果可以接受吗 —— 还是必须给出正面结果?

---

## 9. 两周任务清单

**第 1 周(不训练,约 2 GPU·h)**

- **D1**:申请 DINOv3 ViT-L/16 权限;clone dinov3;设 `JEPAWM_HOME`/`JEPAWM_OSSCKPT`;下载 2.75 GB 预测器;
  用 `heads_cfg.pretrain_dec_path=None` 加载;并行起 `dino_wm_droid`;跑 G0。
- **D2**:适配器 v0 + 往返自测;跑 G1(四路,两种裁剪各一遍)。
- **D2-3**:写 `safety_log.py` + 钩子 + `safety/` 组;在两条老 episode 上验证"不改变任务判定"与各项不变量。
- **D3**:在**同一份 seed 列表**上重采 cup / phone / cup_rim,带 `--keep_failed` 并开 logger。
- **D4**:S4 可解码性探针,两种裁剪;冻结裁剪选择。
- **D4-5**:S3a 头 + 4 个对照;S3b 混淆审计。
- **D5**:从录制的专家日志里选定分叉步;写 `pour_liquid_twin.{yml,py}` 与 `ResumablePour` 专家。

**第 2 周**

- **D6-7**:采 10 组三元组;跑前缀同一性检查(每分支 `z_ctxt` 相同)并测分叉时间;跑捷径门槛。
- **D7-8**:门槛通过后扩到约 150 seeds × 6 分支(错峰,约 6-8 h 墙钟);建 anchor latent 与 rollout 缓存;开 S5a。
- **D9**:S5b,带可变 H 的 mask、同 anchor 排序损失与 hold 对比;3 个 seed。
- **D10**:完整对照表与两个因子行;若 G1 失败则启动 S5c。
- **D11**:标定(温度 + 按策略 conformal),跑 G5,逐指标写结论。
- **D12**:把 guard 代理接到回放专家服务器上,验证 `_result.json` 的 `eval_time >= 1`、`guard_degraded = 0`、
  chunk 等长、shadow 日志。
- **D13-14**:撰写:对照表、捷径审计、延期清单、§8 五问的答复、P1/P2 队列。

**总成本:约 12-15 GPU·小时 + 约 12 小时采集墙钟时间**,没有任何单次作业超过 4 h(在共享机器上很关键)。

---

## 10. 附录

### 10.1 参考实现与证据

| 内容 | 位置 |
|---|---|
| 完整设计(决策表、代码骨架) | `/tmp/wamguard_wf2/synthesis/final_design.md`(临时;定稿后迁入本目录) |
| 四个独立设计 | `/tmp/wamguard_wf2/{A_pragmatic,B_physics_state,C_counterfactual,D_closed_loop}/design_full.md` |
| 事实核验简报(逐条带 file:line) | `/tmp/wamguard_scout/brief.md` 与 `r_*.txt` |
| 物理 logger / 头原型 | `/tmp/wamguard_wf2/physfuture/{phys_logger_sketch.py,head_sketch.py,label_stats.py}` |
| 离线危险标签逻辑(已在 40 条上验证) | `/tmp/wamguard_wf1/data_analysis/phases2.py` |
| 延迟基准 | `/tmp/wamguard_wf1/xcheck/bench.py` |

> 上述 `/tmp` 路径是本次分析的工作产物,不进仓库;定稿时把 brief 与最终设计一并迁到 `docs/`。

### 10.2 关键代码引用

- 预测器与编码:`third_party/jepa-wms/app/vjepa_wm/video_wm.py`(`encode` / `forward_pred` / `rollout`),
  `app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py`(`EncPredWM.encode` / `unroll`),
  `app/plan_common/models/AdaLN_vit.py`。
- 检查点与加载:`third_party/jepa-wms/hubconf.py`,`app/vjepa_wm/utils.py:391-401`(`strict=False` 仍会对形状不匹配报错)。
- 姿态与动作:`app/plan_common/datasets/droid_dset.py:409-423`(`poses_to_diffs`),`:268-283`(fstp)。
- 数据写出:`src/data_collect/hdf5_writer.py:55-58, 85-101, 93-116`。
- 采集钩子:`src/data_collect/collect_env.py:370-402`(`_capture`);失败保留:`expert_collection.py` `--keep_failed`。
- 纯粒子 mask:`env/reward_manager/func_parser.py:458-473`;会改状态的版本:`:527-531`。
- 粒子访问:`env/scene_manager/objects/fluid.py:240-264`;刚体类型:`env/scene_manager/objects/rigid.py:4,18`。
- 专家:`task/RoboDojo/experts/pour_liquid_into_cup.py`(臂选择 `:654`,抓取排序 `:498-506`,倒水闭环 `:550-630`)。
- 评测接入点:`src/eval_client/eval_env.py:188-196`(构造客户端),`:457`(`take_action_batch`,逐帧危险信号的最佳挂点)。
- 回放:`src/data_collect/teleop_replay.py:93-95`。

### 10.3 外部资料

- JEPA-WMs(arXiv 2512.24497):DINO 编码器优于 V-JEPA;AdaLN 略优;proprio 一致有帮助;
  DROID 上 6 步 rollout 损失与 W = 5 上下文最佳;随机裁剪有害;DROID→RoboCasa 迁移需匹配视角与夹爪。
- 最接近的先验工作:**Foresight**(arXiv 2606.23085,冻结 V-JEPA 2-AC + 预测隐变量上的因果 Transformer 头,
  ROC-AUC 0.89/0.84/0.76)、**CheckVLA**(arXiv 2607.26789,动作条件验证器,及时召回 77.9% vs 仅观测 48.6% vs
  动作打乱 37.9%)。
- 隐空间安全过滤:**LatentSafe**(RSS 2025,arXiv 2502.00935;值函数比近视分类器提前约 2 倍告警)、
  **UNISafe**(CoRL 2025,arXiv 2505.00779,集成不确定性)、**AnySafe**(ICRA 2026,arXiv 2509.19555,学出来的
  低维安全投影器远胜原始 DINO 相似度)。
- 数据设计:**HazardArena**(arXiv 2604.12447,双生子场景 + commit 阶段指标)、**OopsieVerse**(连续损伤曲线)、
  **Pre-VLA**(从失败反向指数衰减的 chunk 标签)、**LatentSafe 的数据配比**(1000 随机 + 150 安全 + 150 洒出)。
- 策略侧基线:**Sentinel**(STAC 一致性)、**FAIL-Detect**(logpZO)、**FIPER**(RND-OE 与 ACE AND 规则)、
  **SAFE**(VLA 特征探针)。这些是必须被比较的便宜基线。
- 世界动作模型的自想象是脆弱的(**FACT** arXiv 2608.10232;**BadWAM** arXiv 2607.15207),
  所以要做**独立**的、以实际 chunk 为条件的预测器。
