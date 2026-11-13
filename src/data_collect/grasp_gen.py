"""用 GraspGenX 为物体资产离线生成抓取位姿,缓存给 ``CollectEnv.grasp_actor`` 使用。

在 GraspGenX 自己的 venv 里运行(不依赖 Isaac,也不导入本仓库的其他模块)。一般通过
``scripts/gen_grasps.sh`` 调用,它会从任务 YAML 列出要生成的物体::

    third_party/GraspGenX/.venv/bin/python src/data_collect/grasp_gen.py \\
        --objects wuliangye/0,mug/15,mug/16,goblet/6

输出(每个资产一组文件)::

    <cache_dir>/<gripper>/<category>/<index:05d>.npz
        ee_poses   (N, 7)  物体坐标系下 ``link6``(规划器的 ``ee_link``)的位姿 [x, y, z, qw, qx, qy, qz]
        scores     (N,)    GraspGenX 判别器分数,已按从高到低排序
        meta       json    夹爪名、是否对称、网格/包围盒检查结果、生成参数

已有缓存的资产直接跳过(``--overwrite`` 重新生成)。要在浏览器里交互查看结果,用
``scripts/view_grasps.sh``(基于 viser,读同一批缓存)。

坐标约定:GraspGenX 的抓取坐标系 +Z 为接近方向、+X 为闭合方向,其夹爪描述
(``x_grippers/arx_x5/gripper.urdf``)用固定关节 ``rpy="0 -1.5708 1.5708"``、零平移把它连到
``link6``。这个 ``link6`` 与 ``Assets/Robots/x5/X5A.urdf`` 中的同一连杆一致,所以抓取乘上该固定变换
就直接是规划器要的 ``ee_link`` 位姿,无需额外标定;此时 ``link6`` 的 x 轴即接近方向、y 轴即手指方向,
与 ``CollectEnv`` 的约定相同。

桌面:GraspGenX 只看孤立物体的点云,不知道物体放在桌上,会生成从底下抓或手指 / 掌心插进桌面的抓取
(未过滤时 54 个资产平均约 58% 的缓存抓取穿桌,扁平物体接近 100%)。这里假设物体按自身坐标系直立放在
桌上(任务只做绕 z 的随机旋转,与偏航无关),在网格最低点 ``z_min`` 放一个虚拟桌面,丢弃夹爪碰撞网格
(``coll_mesh.obj``,张开状态)任一顶点低于 ``z_min + --table_clear`` 的抓取;每轮推理 ``--num_grasps``
个,最多 ``--max_passes`` 轮,直到攒够 ``--topk`` 个。

网格从 ``object.usdz`` 读取。为防止 USD 单位 / 朝上轴与 Isaac 中的物体坐标系不一致,生成前会把网格包围盒
与 ``metadata.json`` 的 ``aligned_bbox`` 比对,超出容差就报错跳过该资产,不会写入缓存。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

# pyrender 离线渲染走 EGL;须在任何模块(scene_synthesizer 等)导入 OpenGL 之前设置
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "true")

ROOT = Path(__file__).resolve().parents[2]
GGX_ROOT = ROOT / "third_party" / "GraspGenX"
DEFAULT_CACHE = ROOT / "data" / "grasp_pose"
OBJECTS_ROOT = ROOT / "Assets" / "Object" / "RoboDojo"

# gripper.urdf 中 world -> link6 的固定关节 rpy="0 -1.5708 1.5708"(URDF: R = Rz(yaw) Ry(pitch) Rx(roll))
_C, _S = np.cos(np.pi / 2), np.sin(np.pi / 2)
_RY = np.array([[_C, 0.0, -_S], [0.0, 1.0, 0.0], [_S, 0.0, _C]])  # Ry(-pi/2)
_RZ = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])  # Rz(pi/2)
GRASP_TO_LINK6 = {"arx_x5": _RZ @ _RY}

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True, help="comma-separated <category>/<index>, e.g. mug/15,wuliangye/0")
    parser.add_argument("--obj_type", default="Rigid")
    parser.add_argument("--gripper", default="arx_x5")
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--checkpoints", default=str(GGX_ROOT / "GraspGenXModel" / "release"))
    parser.add_argument(
        "--gripper_assets",
        default=str(GGX_ROOT / "gripper_descriptions" / "gripper_descriptions" / "assets"),
        help="directory holding x_grippers/",
    )
    parser.add_argument("--num_grasps", type=int, default=400, help="grasps sampled per inference pass")
    parser.add_argument("--topk", type=int, default=200, help="grasps kept per asset")
    parser.add_argument("--num_points", type=int, default=3500, help="points sampled from the mesh surface")
    parser.add_argument("--bbox_tol", type=float, default=0.005, help="max mesh vs metadata bbox mismatch (m)")
    parser.add_argument(
        "--table_clear",
        type=float,
        default=0.005,
        help="min height (m) of every gripper collision vertex above the virtual table at the mesh bottom; "
        "negative disables the table filter",
    )
    parser.add_argument(
        "--max_passes", type=int, default=10, help="inference passes to collect --topk table-free grasps"
    )
    parser.add_argument(
        "--min_keep", type=int, default=10, help="fail (keep any old cache) if fewer table-free grasps survive"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mat2quat(R):
    """旋转矩阵 -> [w, x, y, z](不依赖 transforms3d,GraspGenX 的 venv 里未必有)。"""
    import trimesh.transformations as tra

    m = np.eye(4)
    m[:3, :3] = R
    return tra.quaternion_from_matrix(m)  # trimesh 的顺序就是 [w, x, y, z]


def _load_mesh(usd_path, metadata, num_points, tol):
    """读取网格并采样表面点;包围盒与 metadata 不一致时抛 ValueError。"""
    import scene_synthesizer as synth
    import trimesh

    mesh = synth.Asset(str(usd_path)).mesh()
    verts = np.asarray(metadata["geometry"]["aligned_bbox"]["vertices"], dtype=float)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    err = float(np.max(np.abs(np.concatenate([mesh.bounds[0] - lo, mesh.bounds[1] - hi]))))
    if err > tol:
        raise ValueError(
            f"mesh bounds {np.round(mesh.bounds, 4).tolist()} differ from metadata bbox "
            f"{np.round([lo, hi], 4).tolist()} by {err:.4f} m (USD units / up axis?)"
        )
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(points, dtype=np.float32), err, mesh


def _quat2mat(q):
    import trimesh.transformations as tra

    return tra.quaternion_matrix(q)[:3, :3]


def _table_free(grasps, gripper_verts, table_z):
    """``grasps`` (K, 4, 4) 抓取坐标系位姿(物体坐标系)中,夹爪碰撞网格全部顶点都在 ``table_z`` 之上的掩码。"""
    # 只需世界 z:z_k,m = R_k[2, :] . v_m + t_k[2]
    z = np.einsum("kj,mj->km", grasps[:, 2, :3], gripper_verts) + grasps[:, 2, 3:4]
    return z.min(axis=1) >= table_z


def main():
    args = _parse()
    # 用本地已下载的权重与夹爪描述,避免 GraspGenX 在 import 时尝试自动克隆
    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(Path(args.checkpoints).parent))
    os.environ.setdefault("GRASPGENX_GRIPPER_CFG_DIR", str(Path(args.gripper_assets).parents[1]))

    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    import torch

    if args.gripper not in GRASP_TO_LINK6:
        raise SystemExit(f"[grasp_gen] no grasp -> ee_link transform for gripper {args.gripper}")
    to_link6 = GRASP_TO_LINK6[args.gripper]
    gripper_dir = Path(args.gripper_assets) / "x_grippers" / args.gripper

    model_cfg = load_model_cfg(os.path.join(args.checkpoints, "gen"), os.path.join(args.checkpoints, "dis"))
    sampler = GraspGenXSampler(model_cfg, args.gripper, assets_dir=args.gripper_assets)
    symmetric = bool(sampler.get_gripper_info().symmetric)
    import trimesh

    gripper_verts = np.asarray(trimesh.load(gripper_dir / "coll_mesh.obj", force="mesh").vertices, dtype=float)
    use_table = args.table_clear >= 0

    failed = []
    for item in args.objects.split(","):
        category, index = item.strip().split("/")
        index = int(index)
        out = Path(args.cache_dir) / args.gripper / category / f"{index:05d}.npz"
        cached = out.exists() and not args.overwrite
        if cached:
            with np.load(out) as f:
                old_clear = json.loads(str(f["meta"])).get("table_clear")
            note = "" if old_clear is not None else " [no table filter, regenerate with --overwrite]"
            print(f"[grasp_gen] {category}/{index}: cached, skipped ({out}){note}")
            continue
        obj_dir = OBJECTS_ROOT / args.obj_type / category / f"{index:05d}"
        usd = obj_dir / "object.usdz"
        if not usd.exists():
            usd = obj_dir / "object.usd"
        try:
            metadata = json.load(open(obj_dir / "metadata.json"))
            points, bbox_err, mesh = _load_mesh(usd, metadata, args.num_points, args.bbox_tol)
        except Exception as e:
            print(f"[grasp_gen] {category}/{index}: FAILED to load mesh: {type(e).__name__}: {e}")
            failed.append(item)
            continue
        table_z = float(mesh.bounds[0][2])  # 虚拟桌面:物体直立时网格最低点

        # 每个资产固定随机种子,重新生成得到同样的缓存
        seed = int(hashlib.md5(f"{category}/{index}".encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # 与 GraspGenX 的 demo 一致:点云减去均值后推理,再把抓取平移回物体坐标系
        center = points.mean(axis=0)
        kept_g, kept_s, sampled, passes = [], [], 0, 0
        while passes < max(1, args.max_passes) and sum(len(g) for g in kept_g) < args.topk:
            passes += 1
            g, s = GraspGenXSampler.run_inference(
                points - center,
                sampler,
                grasp_threshold=-1.0,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.num_grasps,  # 全部保留,过滤桌面后再统一取 topk
                min_grasps=1,
                max_tries=1,
                remove_outliers=False,
            )
            if len(g) == 0:
                continue
            g = g.cpu().numpy().astype(float)
            s = s.cpu().numpy().astype(float)
            g[:, :3, 3] += center
            sampled += len(g)
            if use_table:
                mask = _table_free(g, gripper_verts, table_z + args.table_clear)
                g, s = g[mask], s[mask]
            kept_g.append(g)
            kept_s.append(s)
            if not use_table:
                break
        n_kept = sum(len(g) for g in kept_g)
        if n_kept < max(1, args.min_keep):
            why = (
                f"only {n_kept} of {sampled} grasps clear the table (< --min_keep {args.min_keep})"
                if sampled
                else "GraspGenX returned no grasps"
            )
            print(f"[grasp_gen] {category}/{index}: FAILED, {why}")
            failed.append(item)
            continue
        grasps, scores = np.concatenate(kept_g), np.concatenate(kept_s)
        order = np.argsort(-scores)[: args.topk]
        grasps, scores = grasps[order], scores[order]

        ee_poses = []
        for g in grasps:
            R = g[:3, :3] @ to_link6
            ee_poses.append(np.concatenate([g[:3, 3], _mat2quat(R)]))  # 已加回 center
        meta = {
            "gripper": args.gripper,
            "symmetric": symmetric,
            "frame": "object frame; pose of ee_link (link6), x = approach, y = finger axis",
            "mesh": str(usd.relative_to(ROOT)),
            "bbox_error_m": bbox_err,
            "num_grasps": args.num_grasps,
            "topk": args.topk,
            "table_clear": args.table_clear if use_table else None,
            "table_z": table_z if use_table else None,
            "passes": passes,
            "sampled": sampled,
            "table_free": n_kept,
            "num_points": args.num_points,
            "seed": seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, ee_poses=np.asarray(ee_poses), scores=scores, meta=json.dumps(meta))
        print(
            f"[grasp_gen] {category}/{index}: {len(ee_poses)} grasps, scores "
            f"{scores.min():.3f}-{scores.max():.3f}, table-free {n_kept}/{sampled} in {passes} pass(es), "
            f"bbox err {bbox_err * 1000:.1f} mm -> {out}"
        )

    if failed:
        raise SystemExit(f"[grasp_gen] failed for {len(failed)} asset(s): {','.join(failed)}")


if __name__ == "__main__":
    main()
