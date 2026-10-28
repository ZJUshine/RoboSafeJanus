"""带相机重放录制好的遥操作指令,并写出 XPolicyLab HDF5。

``teleop_collection.py`` 只录关节指令,不渲染观测相机。本脚本用相同的布局 + seed reset 环境,
把录下的指令按同一控制路径重新执行,像脚本专家采集一样录制观测,
并且只有任务成功判定再次通过时才保存该 episode。

输出(与 ``raw/`` 同级)::

    <save_dir>/<task>/<config>_teleop/
        data/episode_0000000.hdf5
        preview_video/episode_0000000_<camera>.mp4   每路相机一个(cam_head、cam_left_wrist 等)
        layout/episode_0000000.json
        replay_info.json            每条 episode 的回放成功与否 + 相对录制时的关节漂移
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--env_cfg_type", type=str, default="arx_x5")
parser.add_argument("--save_dir", type=str, default="data/RoboSafeJanus-Benchmark")
parser.add_argument("--redo", action="store_true", help="re-render episodes that already have an HDF5")
parser.add_argument("--keep_failed", action="store_true", help="write HDF5 even if the replay fails the success check")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--max_episodes", type=int, default=0, help="render at most this many episodes (0 = all)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from env.camera_manager.capture.render_sync import add_zero_delay_kit_args  # noqa: E402

add_zero_delay_kit_args(args_cli)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

from src.data_collect.benchmark_layout import task_dir  # noqa: E402
from src.data_collect.collect_env import create_collect_env  # noqa: E402
from src.data_collect.common import build_env_cfg, save_episode  # noqa: E402


def load_raw(path):
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        keys = [k[len("pos."):] for k in f.files if k.startswith("pos.")]
        pos = {k: f[f"pos.{k}"] for k in keys}
        vel = {k: f[f"vel.{k}"] for k in keys}
        lens = f["chunk_lens"]
    flat = [
        {k: {"position": pos[k][i].tolist(), "velocity": vel[k][i].tolist()} for k in keys} for i in range(lens.sum())
    ]
    # 按 chunk_lens 把展平的指令重新切回逐 tick 的控制块
    bounds = np.concatenate([[0], np.cumsum(lens)])
    return meta, [flat[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def main():
    task_name, cfg_name = args_cli.task_name, args_cli.env_cfg_type
    env_cfg, collect_cfg = build_env_cfg(task_name, cfg_name)
    freq = int(collect_cfg["observation"].get("collect_freq", 25))
    save_path = os.path.join(task_dir(args_cli.save_dir, task_name), collect_cfg["config_name"] + "_teleop")
    raw_dir = os.path.join(save_path, "raw")
    raws = sorted(f for f in os.listdir(raw_dir) if f.endswith(".npz")) if os.path.isdir(raw_dir) else []
    info_path = os.path.join(save_path, "replay_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    # 已记录在 replay_info.json 中的 episode 之前重放过(包括失败的);同样跳过,
    # 保证分块运行(teleop_replay.sh)每次都有进展。
    done = lambda name: name in info or os.path.exists(os.path.join(save_path, "data", name + ".hdf5"))  # noqa: E731
    todo = [f for f in raws if args_cli.redo or not done(f[: -len(".npz")])]
    print(f"[replay] {len(todo)}/{len(raws)} episodes to render from {raw_dir}")
    if args_cli.max_episodes > 0:
        todo = todo[: args_cli.max_episodes]
    env = create_collect_env(env_cfg, simulation_app) if todo else None
    for fname in todo:
        name = fname[: -len(".npz")]
        meta, chunks = load_raw(os.path.join(raw_dir, fname))
        if meta["task_name"] != task_name or meta["env_cfg_type"] != cfg_name:
            raise ValueError(f"{fname} was recorded for {meta['task_name']}/{meta['env_cfg_type']}")
        if abs(meta["dt"] - float(env.dt)) > 1e-12:
            raise ValueError(f"{fname} was recorded with dt={meta['dt']}, env has dt={env.dt}")

        env.reset_with_layout(meta["layout"], meta["seed"], record=True)
        for chunk in chunks:
            env.execute(chunk)
        # 回放末态与录制末态关节角的最大偏差;正常应接近 0,偏大说明回放不确定
        drift = max(
            float(np.abs(env.get_arm_joints(arm) - np.asarray(q)).max()) for arm, q in meta["final_arm_joints"].items()
        )
        ok = env.check_success()
        info[name] = {"seed": meta["seed"], "success": bool(ok), "joint_drift": drift}
        if ok or args_cli.keep_failed:
            n = save_episode(env, meta["layout"], save_path, name, freq, meta["seed"], video=not args_cli.no_video)
            print(f"\033[92m[replay] {name} written ({n} frames, success={ok}, joint drift {drift:.2e} rad)\033[0m")
        else:
            print(
                f"\033[91m[replay] {name} failed on replay (joint drift {drift:.2e} rad): {env.failed_checks()}\033[0m"
            )
        with open(info_path, "w") as fp:
            json.dump(info, fp, indent=2)

    if env is not None:
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
