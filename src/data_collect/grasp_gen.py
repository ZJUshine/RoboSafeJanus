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
    <cache_dir>/<gripper>/<category>/<index:05d>.png
        可视化(仿 GraspGenX demo_object_mesh.py):灰色物体网格 + 分数最高的 ``--vis_topk`` 个抓取的控制点折线
        (红 = 低分、绿 = 高分),最高分抓取叠加蓝色夹爪网格;iso / front / side / top 四个视角,pyrender EGL 离线渲染。
        已有缓存但缺图的资产只补画图,不重新推理。

坐标约定:GraspGenX 的抓取坐标系 +Z 为接近方向、+X 为闭合方向,其夹爪描述
(``x_grippers/arx_x5/gripper.urdf``)用固定关节 ``rpy="0 -1.5708 1.5708"``、零平移把它连到
``link6``。这个 ``link6`` 与 ``Assets/Robots/x5/X5A.urdf`` 中的同一连杆一致,所以抓取乘上该固定变换
就直接是规划器要的 ``ee_link`` 位姿,无需额外标定;此时 ``link6`` 的 x 轴即接近方向、y 轴即手指方向,
与 ``CollectEnv`` 的约定相同。

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
    parser.add_argument("--vis_topk", type=int, default=30, help="grasps drawn in the per-asset .png")
    parser.add_argument("--no_vis", action="store_true", help="do not write the .png visualizations")
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


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """相机位姿(OpenGL 约定:相机看向 -z)。"""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:  # 正上方俯视
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def _segment_mesh(a, b, radius):
    """a -> b 的细圆柱(pyrender 的 GL 线宽固定 1 px,用圆柱画线)。"""
    import trimesh

    return trimesh.creation.cylinder(radius=radius, segment=np.stack([a, b]), sections=8)


def _visualize(mesh, ee_poses, scores, to_link6, gripper_dir, out_png, title, topk, size=480):
    """按 GraspGenX ``scripts/demo_object_mesh.py`` 的画法离线渲染:灰色物体网格;前 ``topk`` 个抓取画成
    夹爪扫掠体的控制点折线,颜色由分数决定(红 = 低、绿 = 高,同 ``viser_utils.get_color_from_score``);
    分数最高的抓取叠加蓝色夹爪网格。四个视角拼成一张 png。``ee_poses`` 为 link6 位姿,先换回抓取坐标系。"""
    from PIL import Image, ImageDraw
    import pyrender
    import trimesh

    cfg = json.load(open(gripper_dir / "config.json"))
    sv = cfg["sweep_volume"]
    w, h, f = sv["extents"][0], sv["extents"][2], sv["offset"][2]
    # 同 viser_utils.generate_control_points_from_sweep_volume
    ctrl = np.array(
        [[w / 2, 0, h / 2 + f], [w / 2, 0, -h / 2 + f], [0, 0, -h / 2 + f], [0, 0, 0],
         [0, 0, -h / 2 + f], [-w / 2, 0, -h / 2 + f], [-w / 2, 0, h / 2 + f]]
    )  # fmt: skip
    gripper_mesh = trimesh.load(gripper_dir / "vis_mesh.obj", force="mesh")

    n = min(topk, len(ee_poses))
    grasps = []
    for pose in ee_poses[:n]:
        T = np.eye(4)
        T[:3, :3] = _quat2mat(pose[3:]) @ to_link6.T  # link6 -> 抓取坐标系(固定关节无平移)
        T[:3, 3] = pose[:3]
        grasps.append(T)

    def material(rgb, rough=0.6):
        return pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[*rgb, 1.0], metallicFactor=0.1, roughnessFactor=rough
        )

    scene = pyrender.Scene(ambient_light=[0.45, 0.45, 0.45, 1.0], bg_color=[1.0, 1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material([169 / 255] * 3)))
    radius = 0.0012
    for j, (T, score) in enumerate(zip(grasps, scores[:n])):
        pts = ctrl @ T[:3, :3].T + T[:3, 3]
        color = [0, 100 / 255, 1.0] if j == 0 else [1.0 - score, score, 0.0]
        segs = [_segment_mesh(pts[k], pts[k + 1], radius * (2 if j == 0 else 1)) for k in range(len(pts) - 1)]
        scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(segs), material=material(color)))
    if grasps:  # ee_poses 已按分数降序,第 0 个即最高分
        scene.add(pyrender.Mesh.from_trimesh(gripper_mesh, material=material([0, 100 / 255, 1.0], 0.4)), pose=grasps[0])

    # 视野按物体 + 所有抓取控制点取景
    all_pts = np.concatenate([np.asarray(mesh.vertices)] + [ctrl @ T[:3, :3].T + T[:3, 3] for T in grasps])
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    radius_view = np.linalg.norm(all_pts - center, axis=1).max()
    dist = radius_view / np.tan(np.pi / 8) * 1.05
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    views = {"iso": (1, -1, 0.8), "front": (0, -1, 0.15), "side": (1, 0, 0.15), "top": (0, -0.001, 1)}
    renderer = pyrender.OffscreenRenderer(size, size)
    tiles = []
    for name, d in views.items():
        d = np.asarray(d, dtype=float)
        pose = _look_at(center + dist * d / np.linalg.norm(d), center)
        cam_node = scene.add(cam, pose=pose)
        light = scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.5), pose=pose)
        color, _ = renderer.render(scene)
        scene.remove_node(cam_node)
        scene.remove_node(light)
        tile = Image.fromarray(color)
        ImageDraw.Draw(tile).text((8, 6), name, fill=(0, 0, 0))
        tiles.append(tile)
    renderer.delete()

    header = 26
    sheet = Image.new("RGB", (size * len(tiles), size + header), "white")
    for k, tile in enumerate(tiles):
        sheet.paste(tile, (k * size, header))
    lo, hi = (float(scores[:n].min()), float(scores[:n].max())) if n else (0.0, 0.0)
    ImageDraw.Draw(sheet).text(
        (8, 6),
        f"{title}: top {n} of {len(ee_poses)} grasps, score {lo:.3f}-{hi:.3f} "
        "(red = low, green = high; blue mesh = best grasp)",
        fill=(0, 0, 0),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


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

    failed = []
    for item in args.objects.split(","):
        category, index = item.strip().split("/")
        index = int(index)
        out = Path(args.cache_dir) / args.gripper / category / f"{index:05d}.npz"
        png = out.with_suffix(".png")
        cached = out.exists() and not args.overwrite
        if cached and (args.no_vis or png.exists()):
            print(f"[grasp_gen] {category}/{index}: cached, skipped ({out})")
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
        if cached:  # 只补画图
            data = np.load(out)
            name = f"{category}/{index}"
            _visualize(mesh, data["ee_poses"], data["scores"], to_link6, gripper_dir, png, name, args.vis_topk)
            print(f"[grasp_gen] {category}/{index}: cached, visualization written -> {png}")
            continue

        # 每个资产固定随机种子,重新生成得到同样的缓存
        seed = int(hashlib.md5(f"{category}/{index}".encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # 与 GraspGenX 的 demo 一致:点云减去均值后推理,再把抓取平移回物体坐标系
        center = points.mean(axis=0)
        grasps, scores = GraspGenXSampler.run_inference(
            points - center,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk,
            remove_outliers=False,
        )
        if len(grasps) == 0:
            print(f"[grasp_gen] {category}/{index}: FAILED, GraspGenX returned no grasps")
            failed.append(item)
            continue
        grasps = grasps.cpu().numpy().astype(float)
        scores = scores.cpu().numpy().astype(float)
        order = np.argsort(-scores)
        grasps, scores = grasps[order], scores[order]

        ee_poses = []
        for g in grasps:
            R = g[:3, :3] @ to_link6
            ee_poses.append(np.concatenate([g[:3, 3] + center, _mat2quat(R)]))
        meta = {
            "gripper": args.gripper,
            "symmetric": symmetric,
            "frame": "object frame; pose of ee_link (link6), x = approach, y = finger axis",
            "mesh": str(usd.relative_to(ROOT)),
            "bbox_error_m": bbox_err,
            "num_grasps": args.num_grasps,
            "topk": args.topk,
            "num_points": args.num_points,
            "seed": seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, ee_poses=np.asarray(ee_poses), scores=scores, meta=json.dumps(meta))
        if not args.no_vis:
            name = f"{category}/{index}"
            _visualize(mesh, np.asarray(ee_poses), scores, to_link6, gripper_dir, png, name, args.vis_topk)
        print(
            f"[grasp_gen] {category}/{index}: {len(ee_poses)} grasps, scores "
            f"{scores.min():.3f}-{scores.max():.3f}, bbox err {bbox_err * 1000:.1f} mm -> {out}"
        )

    if failed:
        raise SystemExit(f"[grasp_gen] failed for {len(failed)} asset(s): {','.join(failed)}")


if __name__ == "__main__":
    main()
