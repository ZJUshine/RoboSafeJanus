"""为数据采集采样全新的场景布局。

评测回放的是 ``Assets/Eval_Layout`` 下冻结的布局;在这些布局上采训练数据会泄漏测试集。
本模块把任务 YAML 的 ``Rigid`` 段转成一个同结构的新布局字典,
房间 / 桌子 / 地面 / 背景 / 几何固定件则沿用模板布局。

支持的 ``Rigid`` 写法:``relative_plane: Table``、``xlim`` / ``ylim`` / ``zlim``、
``qpos``、``rotate_rand`` / ``rotate_deg``、``margin``、``center_footprint``(范围和占地圆以 bbox 中心为准),
以及 ``select_mode.mode`` 为 ``same`` / ``unique`` / ``category_unique`` / ``allow_duplicate``;``select_mode.nums``
可以是整数、``[n]`` 或闭区间 ``[min, max]``(每个布局随机取一个数)。Rigid 组也可以放在
已采样固定件的支撑面上(``relative_plane: <固定件 label>/<support tag>``,例如面包放在
``bread_shelf/bread_shelf/0``)。支撑面上的组可以设 ``common.mirror: <名字>``:每个布局对每个名字抛一次硬币,
同名的组一起把支撑面坐标系里的 y 偏移取反(整组左右对调,例如货架上的两个商品区)。
``common.physics`` 覆盖资产自带的物理参数(如摩擦)。
放在 Table 上的 ``Geometry`` / ``Articulation`` 组用同样方式采样(其 ``zlim`` 是相对桌面的高度);
其他固定件(如场景里的相机支架)直接从模板复制。``Fluid`` 组跟随其父 Rigid label,
保持模板中相对父物体的偏移。其余写法一律抛 ``UnsupportedLayoutError``,
调用方可以改用 ``--layout-source eval``。
"""

from copy import deepcopy
import glob
import os
import random
import re

import numpy as np
import transforms3d as t3d

from env.global_configs import ASSETS_PATH, BENCHMARK, OBJECTS_PATH
from utils.load_file import load_json

OBJECT_TYPES = ["Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"]
SUPPORTED_MODES = {"same", "unique", "category_unique", "allow_duplicate"}


class UnsupportedLayoutError(Exception):
    """任务 YAML 用了生成器不支持的写法,或某个布局无法放下所有物体。"""

    pass


def eval_layout_dir(config_name, eval_seed=0):
    """评测布局目录 ``Assets/Eval_Layout/<benchmark>/<config>/<eval_seed>``。"""
    return os.path.join(ASSETS_PATH, "Eval_Layout", BENCHMARK, config_name, str(eval_seed))


def find_template(task_name, config_name, template=None):
    """模板布局路径:``template``(task 名或 json 路径)> 任务自己的 ``<task>_0.json`` > ``stack_bowls_0.json``。"""
    layout_dir = eval_layout_dir(config_name)
    if template is not None:
        path = template if os.path.isfile(template) else os.path.join(layout_dir, f"{template}_0.json")
    else:
        path = os.path.join(layout_dir, f"{task_name}_0.json")
        if not os.path.isfile(path):
            path = os.path.join(layout_dir, "stack_bowls_0.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Layout template not found: {path}")
    return path


def list_eval_layouts(task_name, config_name, eval_seed=0):
    """``eval_seed`` 目录下该任务的评测布局,按编号 ``i`` 排序。"""
    # 精确匹配 <task>_<i>.json:直接 glob 会把 <task>_random_<i>.json 和探针变体也算进来。
    pattern = re.compile(rf"{re.escape(task_name)}_\d+\.json")
    files = [
        p
        for p in glob.glob(os.path.join(eval_layout_dir(config_name, eval_seed), "*.json"))
        if pattern.fullmatch(os.path.basename(p))
    ]
    return sorted(files, key=lambda p: int(p.rsplit("_", 1)[-1].split(".")[0]))


class LayoutGenerator:
    """由任务 YAML + 模板布局,按 seed 确定性地生成新布局(见模块说明)。"""

    SAMPLED_FIXTURES = ("Geometry", "Articulation")

    def __init__(self, task_cfg, task_name, config_name, template=None):
        self.task_cfg = task_cfg
        self.task_name = task_name
        self.template_path = find_template(task_name, config_name, template)
        self.template = load_json(self.template_path)
        same_task = os.path.basename(self.template_path).startswith(f"{task_name}_")
        # 来自其他任务的模板只借用其固定件,否则它的物体会混进本任务的场景。
        drop = {"Rigid"} if same_task else set(OBJECT_TYPES) - {"Geometry"}
        self.base = {k: v for k, v in self.template.items() if k not in drop}
        # 任务 YAML 自己定义的固定件和流体 label 会重新采样,从模板中去掉;
        # 相机支架这类场景固定件保持模板中的样子。
        for obj_type in (*self.SAMPLED_FIXTURES, "Fluid"):
            own = {lb for g in self._groups(obj_type) for lb in g.get("select_mode", {}).get("label", [])}
            if obj_type in self.base and own:
                kept = {
                    cat: [r for r in recs if r.get("label") not in own] for cat, recs in self.base[obj_type].items()
                }
                self.base[obj_type] = {cat: recs for cat, recs in kept.items() if recs}
        # 桌面高度 = 桌子中心 z + 一半厚度
        table = self.template["Table"]
        self.table_top = float(table["default_pos"][2]) + float(table["scale"][2]) / 2.0
        self._check_supported()

    def _groups(self, obj_type):
        """任务 YAML 中某类对象的组列表(没有则为空)。"""
        return self.task_cfg.get(obj_type, []) or []

    def _template_record(self, obj_type, label):
        """模板布局中 ``label`` 对应的记录,没有则返回 None。"""
        for inst in self.template.get(obj_type, {}).values():
            for rec in inst:
                if rec.get("label") == label:
                    return rec
        return None

    def _check_supported(self):
        """提前检查 YAML 写法是否受支持,避免采集到一半才失败。"""
        for obj_type in ("Rigid", *self.SAMPLED_FIXTURES):
            for group in self._groups(obj_type):
                common = group.get("common", {})
                mode = group.get("select_mode", {}).get("mode", "allow_duplicate")
                plane = str(common.get("relative_plane", "Table"))
                on_support = obj_type == "Rigid" and "/" in plane
                on_table = plane.lower() == "table" and "xlim" in common
                fixed = obj_type != "Rigid" and all(
                    self._template_record(obj_type, lb) for lb in group.get("select_mode", {}).get("label", [])
                )
                if mode not in SUPPORTED_MODES or not (on_table or on_support or fixed):
                    raise UnsupportedLayoutError(
                        f"{self.task_name}: {obj_type} group (mode={mode}, relative_plane={plane}) is not supported "
                        "by the collect layout generator; use --layout-source eval or extend layout_generator.py."
                    )
        for group in self._groups("Fluid"):
            for label in group.get("select_mode", {}).get("label", []):
                if self._template_record("Fluid", label) is None:
                    raise UnsupportedLayoutError(
                        f"{self.task_name}: Fluid {label} needs a template layout that has it (--layout_template)"
                    )

    @staticmethod
    def _metadata(category, index, obj_type="Rigid"):
        """资产的 ``metadata.json``(bbox、支撑面、物理参数等)。"""
        return load_json(os.path.join(OBJECTS_PATH, obj_type, category, f"{index:05d}", "metadata.json"))

    def _pick_models(self, group, rng):
        """按 ``select_mode`` 选出模型,返回 ``[((category, index), label), ...]``。"""
        cats = group["category"]
        select = group.get("select_mode", {})
        labels = list(select.get("label", []))
        nums = select.get("nums", len(labels) or 1)
        if isinstance(nums, (list, tuple)):
            # [n] 为固定数量,[min, max] 在每个布局里随机取一个(含两端)
            nums = int(nums[0]) if len(nums) == 1 else rng.randint(int(nums[0]), int(nums[1]))
        nums = int(nums)
        mode = select.get("mode", "allow_duplicate")
        pool = [(c["name"], int(i)) for c in cats for i in c.get("index", [0])]
        if mode == "same":
            picks = [rng.choice(pool)] * nums
        elif mode == "unique":
            if len(pool) < nums:
                raise UnsupportedLayoutError(f"{self.task_name}: need {nums} unique models, pool has {len(pool)}")
            picks = rng.sample(pool, nums)
        elif mode == "category_unique":
            chosen = rng.sample(cats, nums)
            picks = [(c["name"], int(rng.choice(c.get("index", [0])))) for c in chosen]
        else:
            picks = [rng.choice(pool) for _ in range(nums)]
        # label 不够时用 "<类别名><序号>" 补齐
        if len(labels) < nums:
            labels += [f"{picks[i][0]}{i}" for i in range(len(labels), nums)]
        return list(zip(picks, labels[:nums]))

    @staticmethod
    def _physics(obj_type, meta, group=None):
        """物体的物理参数;组内 ``common.physics``(如 ``static_friction``)覆盖资产自带的值。"""
        if obj_type == "Articulation":
            return {"type": "articulation"}
        override = (group or {}).get("common", {}).get("physics", {}) or {}
        return {**meta.get("physics", {}), **override, "type": obj_type.lower()}

    @staticmethod
    def _record(cat, idx, label, plane, pos, ori, physics):
        """布局 json 中一个物体的记录,字段与评测布局一致。"""
        return {
            "category": cat,
            "category_idx": idx,
            "group": None,
            "relative_plane": plane,
            "need_check_stable": True,
            "label": label,
            "default_pos": [float(v) for v in pos],
            "default_ori": [float(v) for v in ori],
            "scale": [1.0, 1.0, 1.0],
            "physics": physics,
            "visual": {},
        }

    def _place_on_table(self, obj_type, group, cat, idx, label, meta, rng, placed, max_tries):
        """在桌面 ``xlim`` / ``ylim`` 内随机放置,与已放物体在 xy 平面上按圆盘互不重叠。"""
        common = group.get("common", {})
        margin = float(common.get("margin", 0.01))
        verts = np.asarray(meta["geometry"]["aligned_bbox"]["vertices"], dtype=float)
        qpos = list(common.get("qpos", [1, 0, 0, 0]))
        rot_verts = verts @ t3d.quaternions.quat2mat(qpos).T
        # center_footprint:xlim / ylim 和占地圆都以旋转后 bbox 的水平中心为准,而不是物体原点
        # (原点不在物体中心时用,例如原点在一端、平放的手机)
        center = (rot_verts.min(axis=0) + rot_verts.max(axis=0))[:2] / 2 if common.get("center_footprint") else 0.0
        # 占地半径:按 qpos 旋转后的 bbox 在 xy 平面的最大半径 + margin
        radius = float(np.max(np.linalg.norm(rot_verts[:, :2] - center, axis=1))) + margin
        zlim = common.get("zlim")
        # 默认让物体底面刚好落在桌面上方 2 mm;Rigid 的 zlim 是绝对高度,固定件的 zlim 是相对桌面的高度
        if obj_type == "Rigid":
            z = float(zlim[0]) if zlim else self.table_top - float(rot_verts[:, 2].min()) + 0.002
        else:
            z = self.table_top + (float(zlim[0]) if zlim else 0.002 - float(rot_verts[:, 2].min()))
        xlim = common["xlim"]
        # xlim 可以是多段可选区间,例如 [[-0.3, -0.05], [0.05, 0.3]]
        xranges = xlim if isinstance(xlim[0], (list, tuple)) else [xlim]
        for _ in range(max_tries):
            x = rng.uniform(*rng.choice(xranges))
            y = rng.uniform(*common["ylim"])
            if all((x - px) ** 2 + (y - py) ** 2 > (radius + pr) ** 2 for px, py, pr in placed):
                break
        else:
            raise UnsupportedLayoutError(f"{self.task_name}: could not place {label} without overlap")
        placed.append((x, y, radius))
        deg = 0.0
        if common.get("rotate_rand", False):
            rd = common.get("rotate_deg", 180)
            deg = rng.uniform(-rd, rd) if np.isscalar(rd) else rng.uniform(*rd)
        yaw = t3d.euler.euler2quat(0, 0, np.deg2rad(deg))
        if common.get("center_footprint"):
            # 采样到的 (x, y) 是 bbox 中心,换算回物体原点
            x, y = np.array([x, y]) - t3d.quaternions.quat2mat(yaw)[:2, :2] @ center
        ori = t3d.quaternions.qmult(yaw, qpos)
        return self._record(cat, idx, label, "Table", [x, y, z], ori, self._physics(obj_type, meta, group))

    def _place_on_support(self, group, cat, idx, label, meta, records, mirror=False):
        """把 ``label`` 放到某个已采样固定件的支撑面上,例如把面包放进 bread_shelf 的槽位。

        与场景加载器的做法一致:物体的 ``qpos`` 在支撑面坐标系中生效,``xlim`` / ``ylim``
        (取第一个值)是该坐标系内的偏移,物体底面贴在支撑面上。``mirror`` 为真时 y 偏移取反(``common.mirror``)。
        """
        common = group.get("common", {})
        parent_label, tag = str(common["relative_plane"]).split("/", 1)
        parent = records.get(parent_label)
        if parent is None:
            raise UnsupportedLayoutError(f"{self.task_name}: {label} sits on unknown fixture {parent_label}")
        obj_type, parent_rec = parent
        support = self._metadata(parent_rec["category"], parent_rec["category_idx"], obj_type)["passive"]["support"]
        center = np.asarray(support[tag]["center"][0], dtype=float)
        qpos = list(common.get("qpos", [1, 0, 0, 0]))
        scale = [float(v) for v in common.get("scale", [1.0, 1.0, 1.0])]
        verts = np.asarray(meta["geometry"]["aligned_bbox"]["vertices"], dtype=float) * scale
        rest = -float((verts @ t3d.quaternions.quat2mat(qpos).T)[:, 2].min())
        y = float(np.ravel(common.get("ylim", [0.0]))[0])
        offset = [float(np.ravel(common.get("xlim", [0.0]))[0]), -y if mirror else y, rest]
        R_parent = t3d.quaternions.quat2mat(parent_rec["default_ori"])
        local = center[:3] + t3d.quaternions.quat2mat(center[3:]) @ np.asarray(offset)
        pos = np.asarray(parent_rec["default_pos"]) + R_parent @ local
        ori = t3d.quaternions.qmult(parent_rec["default_ori"], t3d.quaternions.qmult(center[3:], qpos))
        rec = self._record(cat, idx, label, common["relative_plane"], pos, ori, self._physics("Rigid", meta, group))
        rec["scale"] = scale
        return rec

    def sample(self, seed, max_tries=2000, redraws=20):
        """``seed`` 对应的布局。若有物体放不下,就用由 ``seed`` 派生的 seed 重抽整个布局
        (仍是确定性的;第一次就能放下的 seed 不受影响)。"""
        for attempt in range(redraws):
            try:
                return self._sample(random.Random(seed if attempt == 0 else f"{seed}/{attempt}"), max_tries)
            except UnsupportedLayoutError as e:
                if "without overlap" not in str(e):
                    raise
        raise UnsupportedLayoutError(f"{self.task_name}: no placeable layout for seed {seed} in {redraws} draws")

    def _sample(self, rng, max_tries):
        """抽一次布局:先放固定件,再放 Rigid(可能放在固定件上),最后让 Fluid 跟随其父物体。"""
        layout = deepcopy(self.base)
        placed = []  # 已放物体的 (x, y, 半径)
        records = {}  # label -> (对象类型, 记录)
        mirrors = {}  # common.mirror 名字 -> 本布局是否左右对调
        for obj_type in (*self.SAMPLED_FIXTURES, "Rigid"):
            out = {}
            for group in self._groups(obj_type):
                plane = str(group.get("common", {}).get("relative_plane", "Table"))
                # 不在桌面上的固定件直接沿用模板中的记录
                if obj_type != "Rigid" and plane.lower() != "table":
                    for label in group.get("select_mode", {}).get("label", []):
                        rec = deepcopy(self._template_record(obj_type, label))
                        out.setdefault(rec["category"], []).append(rec)
                        records[label] = (obj_type, rec)
                    continue
                for (cat, idx), label in self._pick_models(group, rng):
                    meta = self._metadata(cat, idx, obj_type)
                    if obj_type == "Rigid" and "/" in plane:
                        name = group.get("common", {}).get("mirror")
                        if name is not None and name not in mirrors:
                            mirrors[name] = rng.random() < 0.5
                        rec = self._place_on_support(group, cat, idx, label, meta, records, mirrors.get(name, False))
                    else:
                        rec = self._place_on_table(obj_type, group, cat, idx, label, meta, rng, placed, max_tries)
                    out.setdefault(cat, []).append(rec)
                    records[label] = (obj_type, rec)
            if out or obj_type == "Rigid":
                layout[obj_type] = {**layout.get(obj_type, {}), **out}
        # 流体:按父物体相对模板的位移整体平移
        fluid = {}
        for group in self._groups("Fluid"):
            for label in group.get("select_mode", {}).get("label", []):
                rec = deepcopy(self._template_record("Fluid", label))
                parent = str(rec.get("relative_plane"))
                old_parent = self._template_record("Rigid", parent)
                if parent not in records or old_parent is None:
                    raise UnsupportedLayoutError(f"{self.task_name}: Fluid {label} follows unknown label {parent}")
                shift = np.asarray(records[parent][1]["default_pos"]) - np.asarray(old_parent["default_pos"])
                rec["default_pos"] = [float(v) for v in np.asarray(rec["default_pos"]) + shift]
                fluid.setdefault(rec["category"], []).append(rec)
        if fluid:
            layout["Fluid"] = fluid
        return layout
