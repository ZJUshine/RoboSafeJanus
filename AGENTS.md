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

## 3. 校验

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

## 4. 数据上传

- 远端(ModelScope `zjushine/RoboSafeJanus-Benchmark`、HF)保持**扁平布局** `RoboSafeJanus/<task>/<config>/...`,不要改成本地的嵌套布局。
- 只上传 JSON 中的任务;正在采集或还没合并完的任务不要上传。
- **先 `--dry_run`**,把要上传 / 要删除的文件清单给用户看;涉及删除远端文件(改名、移出 JSON 的旧任务)时必须等用户确认后再真正同步。
- 仓库必须是私有的;脚本发现仓库公开会退出,不要绕过。
- 上传中断直接重跑同一命令续传。上传后用 `check_task_alignment.py --only modelscope` 核对,并可运行 `python scripts/internal/sync_status.py` 刷新同步看板。
- LeRobot 导出(`scripts/export_lerobot.sh`)在 HDF5 数据确认无误后再做;HDF5 有改动(如同步了指令)后要重新导出对应任务。
- 不要把 token 写进代码、命令历史或提交;使用 `hf auth login` / `modelscope login` 或环境变量。

## 5. 提交

- 提交信息用 `[Scope] type: desc`(见 `CLAUDE.md`)。只在用户要求时提交或推送。
- 不提交 `data/`、`Assets/`、`eval_result/` 下的数据文件。
