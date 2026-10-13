"""脚本专家(expert_collection.py)与遥操作采集入口共用的辅助函数。

必须在 ``AppLauncher`` 启动应用之后再导入。
"""

import importlib
import json
import os
import random

from omegaconf import OmegaConf

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from src.data_collect.hdf5_writer import write_episode
from src.data_collect.layout_generator import LayoutGenerator, list_eval_layouts
from utils.load_file import load_json, load_yaml
from utils.pipeline_utils import process_config, process_randomization

BENCHMARK_PATH = os.path.join(ROOT_DIR, "task", BENCHMARK)
task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")


def build_env_cfg(task_name, cfg_name):
    """读取 ``env_cfg/<cfg_name>.yml`` 及其引用的 sim / scene / camera / robot 配置和任务 YAML,
    拼成单环境(num_envs=1)的采集配置。返回 ``(env_cfg, collect_cfg)``。"""
    collect_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, cfg_name + ".yml"))
    collect_cfg["task_name"] = task_name
    collect_cfg["num_envs"] = 1
    # 已发布的数据带相机内外参;评测(同一个 yml)不需要,所以只在采集时强制打开。
    vision = collect_cfg["observation"].setdefault("vision", {})
    vision["intrinsic_matrix"] = vision["extrinsic_matrix"] = True
    sub = collect_cfg["config"]
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", sub["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", sub["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", sub["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", sub["robot"] + ".yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(BENCHMARK_PATH, "config"), task_name)),
            "collect_cfg": collect_cfg,
            # utils.pipeline_utils 从 eval_cfg 读取 domain_randomization。
            "eval_cfg": collect_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task_name)
    # 相机渲染频率与采集频率一致
    freq = collect_cfg["observation"].get("collect_freq", 0)
    OmegaConf.update(env_cfg, "camera.default_frequency", freq, force_add=True)
    env_cfg.sim.seed = [0]
    return env_cfg, collect_cfg


def make_layout_getter(env_cfg, collect_cfg, task_name, source="random", template=None):
    """返回 ``seed -> layout`` 函数:从任务 YAML 随机采样,或循环使用评测布局(仅调试用)。"""
    if source == "eval":
        eval_files = list_eval_layouts(task_name, collect_cfg["config_name"])
        return lambda seed: load_json(eval_files[seed % len(eval_files)])
    generator = LayoutGenerator(
        OmegaConf.to_container(env_cfg.task_env, resolve=True), task_name, collect_cfg["config_name"],
        template=template,
    )
    return generator.sample


def _delta_matrices(env):
    """每个目标机械臂的状态键前缀 -> 其 delta_matrix(见 RobotManager.get_delta_endpose)。"""
    robots = env._target_robots()
    return {(r.arm_name.split("_")[0] + "_" if len(robots) > 1 else ""): r.delta_matrix for r in robots}


def save_episode(env, layout, save_path, name, frequency, seed, video=True, meta=None):
    """把 ``env.frames`` 写成 ``data/<name>.hdf5``(以及 ``preview_video/<name>_<camera>.mp4``),
    并把布局写成 ``layout/<name>.json``。

    每个 episode 一条指令,用布局 seed 从任务的指令模板中抽取。
    ``meta`` 原样传给 ``write_episode``(例如失败尝试的失败原因)。返回写入的步数。
    """
    n = write_episode(
        env.frames,
        os.path.join(save_path, "data", f"{name}.hdf5"),
        instruction=random.Random(seed).choice(env.instructions()),
        frequency=frequency,
        delta_matrices=_delta_matrices(env),
        video_dir=os.path.join(save_path, "preview_video") if video else None,
        meta=meta,
    )
    os.makedirs(os.path.join(save_path, "layout"), exist_ok=True)
    with open(os.path.join(save_path, "layout", f"{name}.json"), "w") as fp:
        json.dump(layout, fp, indent=1)
    return n
