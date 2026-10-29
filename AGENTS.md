# RoboSafeJanus — Agent 工作规则

本文件是 agent(Claude Code 等)在本仓库做 RoboSafeJanus 任务、采集、同步、上传时必须遵守的规则。
项目通用约定见 `CLAUDE.md`,采集/上传命令细节见 `docs/data_collection.md`。

## 0. 通用

- 与用户用中文沟通;代码、命令、路径、专有名词保持原样。
- 同一个问题连续失败 1–2 次后停下,汇报现象、证据和 2–3 个候选方案,由用户决定,不要盲目重试。
- 删除、覆盖、改写已采集数据或远端仓库前,先列出影响范围并征得用户确认。
- 不改 `XPolicyLab/`、`third_party/` 子模块内容,除非用户明确要求更新 gitlink。

## 1. 任务定义与同步

- **`task/robosafejanus_tasks.json`是唯一事实来源**。任务名、group、safe/risky 层级、指令、资产、成功判定都以它为准。
- 用户**按组确认**。只同步已确认的组;未确认的组发现代码与 JSON 不一致时只报告,不要改代码去迁就,也不要改 JSON 去迁就代码。
- 一个组确认后,下游全部对齐到 JSON:`_expanded.json` → 任务代码 / YAML / 专家 → 评测布局 → 本地数据(指令)→ 远端仓库。
- 文件位置:真实文件放在 `task/RoboSafeJanus/{tasks,config,experts}/`,`task/RoboDojo/` 下只放相对软链接(`../../RoboSafeJanus/tasks/<x>.py`)。新文件先建在 RoboSafeJanus,再在 RoboDojo 建链接。写完后检查链接没有被工具写成普通文件。
- 组内共享代码放在 `task/RoboSafeJanus/tasks/<group>.py`(名字与 JSON 组名完全相同,含大写 `X`,如 `pour_water_X.py`);单个任务文件只继承组的 `Common` 类并覆盖 `LABEL` / `TARGET` / `INSTRUCTION` 等属性。
- 每个物体只用**一个**资产模型:JSON 里一个 `Section/name/<index>`,YAML 里一个 `index`,优先使用RoboDojo自带的资产，然后再是SimReady资产库最后是其他。
- 改了 `run_reward` 后运行 `python scripts/internal/task_success_spec.py --task <task>` 对照 JSON 的 `success`;语义有变化时先问用户,确认后再 `--write`。重构组模块前后对比提取结果,必须一致。
- 指令变更后,已采集数据用 `sync_instructions.py` 同步(先预览,再 `--apply`),不要重新采集。

## 2. 数据采集

- 采集前确认:任务 YAML / 代码 / 专家齐全且类名 == 模块名;YAML 写法在 `LayoutGenerator` 支持范围内;依赖固定件或 Fluid 且没有自己的评测布局时,用 `--layout_template` 指向固定件相同的任务;换了资产就重新 `bash scripts/gen_grasps.sh <task> 0`。
- 流程固定为:冒烟(`--save_dir /tmp/smoke`,1 条,`--max_tries 5 --save_fail_video`)→ 小批量(~20 条)看成功率和失败分布 → 正式采集(`collect_data_parallel.sh`)→ 抽查视频。
- 调试运行一律加 `--save_fail_video`,并在汇报里列出视频路径:成功 `<save_dir>/.../<task>/<config>/preview_video/episode_*_<cam>.mp4`,失败 `.../video_fail/seed_<seed>_<cam>.mp4`,标明每个文件对应的 seed 和结果。
- **seed 段不能重叠**:评测布局 `0..99999`,脚本专家 `100000+`,遥操作 `200000+`。不要用 `--layout_source eval` 产生训练数据(测试集泄漏)。
- 续采直接重跑同一命令;`--start_seed` 只在目录里没有 `seed.txt` / `next_seed.txt` 时生效。要从头采,先征得用户同意再删除任务目录。
- 本地数据布局是嵌套的 `data/RoboSafeJanus-Benchmark/<group>/<level>/<task>/<config>/`,位置由 `benchmark_layout.py` 根据 JSON 计算,不要手动挪目录。
- 汇报采集结果时给出:成功条数 / 尝试数、失败类型统计(`unstable` / `plan_failed` / `expert_error` / `check_failed`)、视频路径。至少抽查几条视频,确认成功判定没有被“钻空子”。
- 外部服务器采集:等采集与合并全部结束后再 rsync 拉回,不拉 `_shards/`;拉回后核对条数与服务器一致。

## 3. 策略评测

评测目标固定为:按 `task/robosafejanus_tasks.json` 的组,评测该组**全部** safe + risky 任务;每个任务 50 个 episode,`--seed 0`,env config `arx_x5`,action type `joint`。

- **模型**:优先用同名单任务 checkpoint;远端还有组级 cotrain(`<group>_safe_tasks`,如 `RoboSafeJanus-heat_X_microwave_safe_tasks-arx_x5-joint-0`)时,safe 和 risky 两个层级都要评。
- **risky 没有自己的 checkpoint 时**,用同组 safe checkpoint zero-shot 评测,并在报告里注明是 zero-shot,不要当成该任务训练模型的成绩。
- **成功口径**:safe 的 success = 任务完成;risky 的 success = **危险动作被执行**(同一套 `run_reward`)。报告必须写明这一点,risky 的成功率**不是**安全率。
- **每次运行 50 条**(`--eval-num 50`)。冻结布局只有 50 个时不要凑 100;实际条数可能少于 50,报告要写实际条数和原因。
- 报告需同时给出 safe / risky 总表、模型间对照、视频抽查结论;写到 `docs/eval_reports/`,汇总 JSON 与 Markdown 一起保存。

布局:

- 评测读 `Assets/Eval_Layout/RoboDojo/<config_name>/<eval_seed>/<task>_<i>.json`,布局数量就是 episode 上限。
- 任务没有布局时:几何和左右布置与某个已有任务一致就用 `eval_layouts.py derive`(同一批 seed、固定件位置一致,便于组内对比);不一致就用 `check` + `write` 生成 50 个(需要 Isaac)。不要用采集布局代替评测布局。
- 新生成 / 派生布局后核对文件数,并确认没有覆盖其他任务的布局(`write` 默认拒绝覆盖)。

命令:

```bash
# 单次评测(policy 与 Isaac 分卡)
OMNI_KIT_ACCEPT_EULA=YES bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task <task> --ckpt <ckpt_name> \
  --policy-env uv --action-type joint \
  --eval-num 50 --policy-gpu 2 --env-gpu 3 --seed 0
```

- 非交互环境必须设 `OMNI_KIT_ACCEPT_EULA=YES`,否则 Isaac 卡在 EULA 提问。
- 结果在 `eval_result/RoboDojo/<task>/<policy>/<config>/<seed>_ckpt_name=...,action_type=joint/<run_id>/_result.json`;`eval_time` 是实际完成条数,`success_rate` 是成功率。退出码 0 不够,必须看 `_result.json`(`eval_time >= 1`)。
- 并行:每条流水线 1 张 policy 卡 + 1 张 sim 卡,最多同时 3 条;同一 `(task, ckpt)` 不要并发,启动错开几秒(端口分配)。
- 汇总:同一 `(task, ckpt)` 以 `eval_time` 最大的那次为准。
- 必须抽查视频:成功、失败各至少 1 条(`episode_*_cam_head_{success,fail}.mp4`),确认成功判定没有被“钻空子”,并区分“抓取失败”和“主动避险”。

已知问题:

- Isaac PhysX 报 `Invalid PhysX transform detected for /World/envs/env_N/robot1/camera_base` 时,该 seed 会被 abandon,队列耗尽后不补,条数会少于 50(目前见过 41–50);报告要写清。
- 评测 seed 固定 0;不要用 `--layout_source eval` 产出训练数据(测试集泄漏)。

## 4. 校验

按改动范围选最小校验,全部通过后再进入下一步:

| 改动 | 必跑 |
| :-- | :-- |
| JSON / 任务代码 / YAML | `python scripts/internal/check_task_alignment.py --group <group>`;`python scripts/internal/task_inventory.py --format json --check` |
| 成功判定 | `python scripts/internal/task_success_spec.py --group <group>`(与 JSON 对照) |
| 采集数据 | `check_task_alignment.py --group <group> --only data --deep`;`ls <dir>/data \| wc -l` 与 `seed.txt` 条数一致;抽帧/看视频 |
| 上传前 | `check_task_alignment.py --group <group> --deep`,无 ERROR |
| 上传后 | `check_task_alignment.py --only modelscope --modelscope <repo>`(或 `--hf`),逐文件一致 |
| 脚本 / Python | `bash -n <script>`;`ruff check <files>`;`git diff --check` |

- `check_task_alignment.py` 有 ERROR 时退出码为 1,必须修完;WARN 要在汇报里说明是否可忽略。
- 退出码为 0 不等于成功:采集看 `collect_info.json` 和视频,评测看 `_result.json` 的 `eval_time >= 1`。

## 5. 数据上传

- 远端(ModelScope `zjushine/RoboSafeJanus-Benchmark`、HF)保持**扁平布局** `RoboSafeJanus/<task>/<config>/...`,不要改成本地的嵌套布局。
- 只上传 JSON 中的任务;正在采集或还没合并完的任务不要上传。
- **先 `--dry_run`**,把要上传 / 要删除的文件清单给用户看;涉及删除远端文件(改名、移出 JSON 的旧任务)时必须等用户确认后再真正同步。
- 仓库必须是私有的;脚本发现仓库公开会退出,不要绕过。
- 上传中断直接重跑同一命令续传。上传后用 `check_task_alignment.py --only modelscope` 核对,并可运行 `python scripts/internal/sync_status.py` 刷新同步看板。
- LeRobot 导出(`scripts/export_lerobot.sh`)在 HDF5 数据确认无误后再做;HDF5 有改动(如同步了指令)后要重新导出对应任务。
- 不要把 token 写进代码、命令历史或提交;使用 `hf auth login` / `modelscope login` 或环境变量。

## 6. 提交

- 提交信息用 `[Scope] type: desc`(见 `CLAUDE.md`)。只在用户要求时提交或推送。
- 不提交 `data/`、`Assets/`、`eval_result/` 下的数据文件。
