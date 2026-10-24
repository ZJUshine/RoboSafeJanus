"""基于 cuRobo 的脚本专家数据采集,参照 RoboTwin 的 collect_data.py。

对每个 seed:采样一个布局,运行该任务的专家(``task/RoboDojo/experts/<task>.py`` 中的
``play_once(env)``),只有通过任务自身的 ``run_reward`` 判定才保留该 episode,
并写成 XPolicyLab HDF5 + mp4 预览视频。

输出::

    <save_dir>/<task>/<config>/
        data/episode_0000000.hdf5
        preview_video/episode_0000000_<camera>.mp4   每路相机一个(cam_head、cam_left_wrist 等)
        layout/episode_0000000.json     本条使用的布局,用于重放 / 调试
        seed.txt                        已成功的 seed(续采位置)
        collect_info.json               成功 / 失败次数统计

使用 --keep_failed 时,失败的尝试写到同级的 ``<task>_failed`` 目录,结构相同::

    <save_dir>/<task>_failed/<config>/
        data/ preview_video/ layout/    HDF5 额外带 meta/{success,fail_type,fail_reason,seed}
        fail_info.jsonl                 每条保留的失败尝试一行
"""

import argparse
from collections import Counter
import glob
import importlib
import json
import logging
import os
import sys
import traceback

from isaaclab.app import AppLauncher

# 命令行参数;AppLauncher 的参数(--headless、--enable_cameras 等)也一并注册
parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--expert", type=str, default=None, help="experts/<name>.py to run (default: task_name)")
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="env_cfg/<name>.yml (robot/camera/obs)")
parser.add_argument("--episode_num", type=int, default=50, help="successful episodes to collect")
parser.add_argument("--start_seed", type=int, default=100000, help="first layout seed; kept away from eval seeds")
parser.add_argument("--max_tries", type=int, default=0, help="stop after this many attempts (0 = 10 x episode_num)")
parser.add_argument("--save_dir", type=str, default="data/RoboSafeJanus")
parser.add_argument(
    "--layout_source",
    choices=("random", "eval"),
    default="random",
    help="random: sample new layouts from the task YAML; eval: replay Assets/Eval_Layout (debug only, test leakage)",
)
parser.add_argument("--layout_template", type=str, default=None, help="task name or json whose fixtures to reuse")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--save_fail_video", action="store_true", help="also save videos of failed attempts (debugging)")
parser.add_argument(
    "--keep_failed",
    action="store_true",
    help="also save failed attempts (except unstable layouts) under <task>_failed/, with fail_type / fail_reason",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402
from env.global_configs import BENCHMARK  # noqa: E402

# 必须先启动 Isaac Sim 应用,之后才能导入依赖它的模块
add_zero_delay_kit_args(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from src.data_collect.collect_env import PlanFailError, create_collect_env  # noqa: E402
from src.data_collect.common import build_env_cfg, make_layout_getter, save_episode  # noqa: E402
from src.data_collect.hdf5_writer import write_video  # noqa: E402
from utils.cluttered_generator import UnStableError  # noqa: E402


def load_expert(task_name):
    """导入 ``task/<BENCHMARK>/experts/<task_name>.py`` 并返回其中的 ``play_once``。"""
    module = importlib.import_module(f"task.{BENCHMARK}.experts.{task_name}")
    if not hasattr(module, "play_once"):
        raise AttributeError(f"task/{BENCHMARK}/experts/{task_name}.py must define play_once(env)")
    return module.play_once


def main():
    # Isaac 已经配置了 root logger,所以只给专家所在的 ``task`` 命名空间单独挂 handler。
    expert_log = logging.getLogger("task")
    expert_log.setLevel(os.environ.get("ROBODOJO_COLLECT_LOG", "WARNING"))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    expert_log.addHandler(handler)
    expert_log.propagate = False
    task_name, cfg_name = args_cli.task_name, args_cli.env_cfg_type
    play_once = load_expert(args_cli.expert or task_name)
    env_cfg, collect_cfg = build_env_cfg(task_name, cfg_name)
    freq = int(collect_cfg["observation"].get("collect_freq", 25))

    get_layout = make_layout_getter(
        env_cfg, collect_cfg, task_name, args_cli.layout_source, template=args_cli.layout_template
    )

    # 续采:seed.txt 记录已成功的 seed,从最大成功 seed 的下一个开始
    save_path = os.path.join(args_cli.save_dir, task_name, collect_cfg["config_name"])
    os.makedirs(save_path, exist_ok=True)
    seed_file = os.path.join(save_path, "seed.txt")
    seed_list = []
    if os.path.exists(seed_file):
        seed_list = [int(s) for s in open(seed_file).read().split()]
    seed = max(seed_list) + 1 if seed_list else args_cli.start_seed
    # 失败的 seed 不会出现在 seed.txt 中;分块重启时借助 next_seed.txt 把它们也跳过。
    next_seed_file = os.path.join(save_path, "next_seed.txt")
    if os.path.exists(next_seed_file):
        seed = max(seed, int(open(next_seed_file).read()))
    suc_num, fail_num, tries = len(seed_list), 0, 0
    max_tries = args_cli.max_tries or 10 * args_cli.episode_num
    fail_path = os.path.join(args_cli.save_dir, f"{task_name}_failed", collect_cfg["config_name"])
    fail_saved, fail_types = 0, Counter()
    print(f"[collect] task={task_name} target={args_cli.episode_num} resume_from={suc_num} seed={seed}")

    env = create_collect_env(env_cfg, simulation_app)
    while suc_num < args_cli.episode_num and tries < max_tries:
        tries += 1
        # 失败分类:unstable(布局不稳)/ plan_failed(规划失败)/ expert_error(专家异常)/ check_failed(判定未过)
        reason, fail_type, started = "", "", False
        try:
            layout = get_layout(seed)
            env.reset_with_layout(layout, seed)
            started = True
            play_once(env)
            ok = env.check_success()
            if not ok:
                fail_type, reason = "check_failed", f"success check failed: {env.failed_checks()}"
        except UnStableError as e:
            ok, fail_type, reason = False, "unstable", f"{type(e).__name__}: {e}"
        except PlanFailError as e:
            ok, fail_type, reason = False, "plan_failed", f"{type(e).__name__}: {e}"
        except Exception as e:
            ok, fail_type, reason = False, "expert_error", f"{type(e).__name__}: {e}"
            traceback.print_exc()

        if ok:
            # 成功:episode 编号 = 已成功条数,并立即更新 seed.txt,保证随时中断都能续采
            name = f"episode_{suc_num:07d}"
            n = save_episode(env, layout, save_path, name, freq, seed, video=not args_cli.no_video)
            seed_list.append(seed)
            with open(seed_file, "w") as fp:
                fp.write(" ".join(map(str, seed_list)))
            suc_num += 1
            print(f"\033[92m[collect] episode {suc_num - 1} saved (seed={seed}, {n} steps)\033[0m")
        else:
            fail_num += 1
            fail_types[fail_type] += 1
            # 若本次尝试没能通过 reset,env.frames 里还是上一次尝试的帧,所以要检查 started。
            if args_cli.save_fail_video and started and env.frames:
                for cam in env.frames[0]["vision"]:
                    path = os.path.join(save_path, "video_fail", f"seed_{seed}_{cam}.mp4")
                    write_video(env.frames, path, freq, camera=cam)
            print(f"\033[91m[collect] seed={seed} failed: {reason}\033[0m")
            if args_cli.keep_failed and started and fail_type != "unstable" and len(env.frames) >= 2:
                name = f"episode_{len(glob.glob(os.path.join(fail_path, 'data', '*.hdf5'))):07d}"
                meta = {"success": False, "fail_type": fail_type, "fail_reason": reason, "seed": seed}
                try:
                    n = save_episode(env, layout, fail_path, name, freq, seed, video=not args_cli.no_video, meta=meta)
                except Exception as e:
                    print(f"[collect] could not save failed seed={seed}: {type(e).__name__}: {e}")
                else:
                    with open(os.path.join(fail_path, "fail_info.jsonl"), "a") as fp:
                        fp.write(json.dumps({"episode": name, "steps": n, **meta}) + "\n")
                    fail_saved += 1
                    print(f"[collect] failed attempt kept as {task_name}_failed/{name} ({fail_type}, {n} steps)")
        # 无论成败都推进 seed 并写 next_seed.txt
        seed += 1
        with open(next_seed_file, "w") as fp:
            fp.write(str(seed))

    info = {"success": suc_num, "fail_this_run": fail_num, "tries_this_run": tries, "fail_types_this_run": fail_types}
    if args_cli.keep_failed:
        info["failed_saved_this_run"] = fail_saved
        info["failed_saved_total"] = len(glob.glob(os.path.join(fail_path, "data", "*.hdf5")))
    with open(os.path.join(save_path, "collect_info.json"), "w") as fp:
        json.dump(info, fp, indent=2)
    print(f"[collect] done: {suc_num}/{args_cli.episode_num} episodes, {fail_num} failures in {tries} tries")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
