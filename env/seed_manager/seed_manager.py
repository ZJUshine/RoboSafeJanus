from collections.abc import Iterable, Mapping
from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any, Dict, List

from env.global_configs import ASSETS_PATH, BENCHMARK
from utils.load_file import *


class SeedManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config: Mapping[str, Any] = config
        self.num_envs: int = int(self.config["num_envs"])

        # config fields used for directory layout
        self.task_name: str = str(self.config["task_name"])
        self.config_name: str = str(self.config["config_name"])

        self.st_idx: int
        self.ed_idx: int
        self.type: str

        self._current_batch_seeds: List[int] | None = None

    def init_eval(
        self,
        completed_layout_ids: Iterable[int] | None = None,
        abandoned_layout_ids: Iterable[int] | None = None,
    ):
        self.eval_seed = self.config.get("seed", 0)
        layout_dir = Path(ASSETS_PATH, "Eval_Layout", BENCHMARK, self.config_name, str(self.eval_seed))
        # S1 变体评测用 <task>__<variant>_<i>.json(src/eval_client/variants.py),默认是任务名
        layout_name = str(self.config.get("layout_name") or self.task_name)
        pattern = re.compile(rf"{re.escape(layout_name)}_\d+\.json")
        matching_files = sorted(
            [p for p in layout_dir.iterdir() if pattern.fullmatch(p.name)],
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
        )

        # layout_id 取文件名里的编号 <i>:编号连续时(普通任务)和按顺序数一样;
        # 变体布局跳过了放不下障碍物的编号,这样 layout_id 仍和派生它的 S1.0 布局一一对应。
        self.seed_info = {}
        for file_path in matching_files:
            self.seed_info[int(file_path.stem.rsplit("_", 1)[-1])] = {"scene_layout": str(file_path)}

        all_layout_ids = sorted(self.seed_info)
        excluded = set(int(s) for s in (completed_layout_ids or [])) | set(int(s) for s in (abandoned_layout_ids or []))
        if excluded:
            self.seed_list: List[int] = [s for s in all_layout_ids if s not in excluded]
            print(
                f"[SeedManager] init_eval resume filter: excluded={len(excluded)} "
                f"remaining={len(self.seed_list)}/{len(all_layout_ids)}"
            )
        else:
            self.seed_list = all_layout_ids
        self.st_idx = 0
        self.ed_idx = len(self.seed_list)

        self.type = "eval"
        self.idx = 0
        self._current_batch_seeds = None

    def get_seeds(self, max_count: int | None = None) -> List[int] | None:
        """Return a list of seeds for the next `reset()` call.

        Returns None when enough episodes have been successfully collected.
        """

        if self.idx >= self.ed_idx:
            return None
        if max_count is not None:
            batch_size = min(self.num_envs, max(0, int(max_count)))
            if batch_size == 0:
                return None
            batch = self.seed_list[self.idx : min(self.idx + batch_size, self.ed_idx)]
            self.idx += len(batch)
            self._current_batch_seeds = batch
            return batch
        if self.idx + self.num_envs > self.ed_idx:
            batch = self.seed_list[self.idx : self.ed_idx]
            result = deepcopy(batch)
            for _ in range(self.num_envs - len(result)):
                batch.append(self.seed_list[self.ed_idx - 1])  # pad with last seed if not enough remaining
        else:
            batch = self.seed_list[self.idx : self.idx + self.num_envs]
        self.idx += self.num_envs
        self._current_batch_seeds = batch
        return batch

    def requeue(self, seed: int) -> None:
        """把一个因 PhysX 崩溃被放弃的 seed 排到队尾,再给它一次机会。

        只在评测循环里用(``src/eval_client/main.py``):崩掉的那条重试一两次通常能跑出来;
        重试次数有上限,连续崩掉的最后还是算放弃。"""
        self.seed_list.append(int(seed))
        self.ed_idx = len(self.seed_list)

    def get_seed_scene_info(self, seed: int) -> Dict[str, Any]:
        seed_info = self.seed_info.get(seed)
        if seed_info is None:
            raise ValueError(f"Seed {seed} not found in seed list.")
        file_path = seed_info.get("scene_layout")
        if file_path is None or not os.path.exists(file_path):
            raise ValueError(f"Scene layout file not found for seed {seed} at expected path {file_path}.")
        data = load_json(file_path)
        return data

    def eval_step(self):
        self._current_batch_seeds = None
