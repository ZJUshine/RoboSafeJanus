"""浏览器里交互查看 GraspGenX 抓取缓存(``src/data_collect/grasp_gen.py`` 的输出),基于 viser。

在 GraspGenX 自己的 venv 里运行,一般通过 ``scripts/view_grasps.sh`` 调用::

    third_party/GraspGenX/.venv/bin/python src/data_collect/view_grasps.py --port 8765

然后浏览器打开 ``http://<本机>:8765``(从别的机器看:``ssh -L 8765:localhost:8765 <本机>``)。

侧栏:
  - 物体下拉框:``<cache_dir>/<gripper>`` 下所有缓存的资产(``--objects`` 可限定);
  - 缓存:``new`` = ``--cache_dir``,``backup`` = ``--backup_dir``(例如加桌面过滤前的备份),用于对比;
  - 显示前 K 个 / 最低分数:按分数筛选画出的抓取(控制点折线);
  - 着色:``score``(红 = 低、绿 = 高,同 GraspGenX)或 ``table``(绿 = 离桌面 >= ``--table_clear``,红 = 穿桌);
  - 选中抓取:在当前显示的抓取中选一个,画出完整的张开夹爪网格,并列出分数 / 接近方向 / 离桌高度。
场景坐标为物体坐标系(z 向上),虚拟桌面在网格最低点,与 ``grasp_gen.py`` 的过滤一致。
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_gen as gg  # noqa: E402  (同目录;复用坐标变换、网格读取与穿桌判定)


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=str(gg.DEFAULT_CACHE))
    parser.add_argument(
        "--backup_dir",
        default=str(gg.ROOT / "data" / "grasp_pose_backup_pre_table_20261111"),
        help="second cache to compare against (ignored if missing)",
    )
    parser.add_argument("--gripper", default="arx_x5")
    parser.add_argument("--objects", default="", help="comma-separated <category>/<index>; default = all cached")
    parser.add_argument("--obj_type", default="Rigid")
    parser.add_argument(
        "--gripper_assets",
        default=str(gg.GGX_ROOT / "gripper_descriptions" / "gripper_descriptions" / "assets"),
    )
    parser.add_argument("--table_clear", type=float, default=0.005, help="clearance used for the 'table' colouring")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def _list_assets(cache_root):
    """``cache_root`` 下的 ``<category>/<index>``(按名字排序)。"""
    if not cache_root.is_dir():
        return []
    items = []
    for f in sorted(cache_root.glob("*/*.npz")):
        items.append(f"{f.parent.name}/{int(f.stem)}")
    return sorted(items, key=str.lower)


def main():
    args = _parse()
    import trimesh
    import viser

    gripper_dir = Path(args.gripper_assets) / "x_grippers" / args.gripper
    to_link6 = gg.GRASP_TO_LINK6[args.gripper]
    cfg = json.load(open(gripper_dir / "config.json"))
    sv = cfg["sweep_volume"]
    w, h, f = sv["extents"][0], sv["extents"][2], sv["offset"][2]
    ctrl = np.array(
        [
            [w / 2, 0, h / 2 + f],
            [w / 2, 0, -h / 2 + f],
            [0, 0, -h / 2 + f],
            [0, 0, 0],
            [0, 0, -h / 2 + f],
            [-w / 2, 0, -h / 2 + f],
            [-w / 2, 0, h / 2 + f],
        ]
    )  # 同 viser_utils.generate_control_points_from_sweep_volume
    seg_local = np.stack([ctrl[:-1], ctrl[1:]], axis=1)  # (6, 2, 3)
    gripper_vis = trimesh.load(gripper_dir / "vis_mesh.obj", force="mesh")
    gripper_coll = np.asarray(trimesh.load(gripper_dir / "coll_mesh.obj", force="mesh").vertices, dtype=float)

    caches = {"new": Path(args.cache_dir) / args.gripper}
    backup = Path(args.backup_dir) / args.gripper
    if backup.is_dir() and backup.resolve() != caches["new"].resolve():
        caches["backup"] = backup
    assets = _list_assets(caches["new"])
    if args.objects:
        wanted = [o.strip() for o in args.objects.split(",") if o.strip()]
        assets = [a for a in wanted if a in assets] or wanted
    if not assets:
        raise SystemExit(f"[view_grasps] no grasp caches under {caches['new']}")

    meshes = {}  # asset -> (trimesh, table_z)

    def load_mesh(asset):
        if asset not in meshes:
            category, index = asset.split("/")
            obj_dir = gg.OBJECTS_ROOT / args.obj_type / category / f"{int(index):05d}"
            usd = obj_dir / "object.usdz"
            if not usd.exists():
                usd = obj_dir / "object.usd"
            import scene_synthesizer as synth

            mesh = synth.Asset(str(usd)).mesh()
            meshes[asset] = (mesh, float(mesh.bounds[0][2]))
        return meshes[asset]

    def load_cache(which, asset):
        """抓取坐标系位姿 (N, 4, 4)、分数、meta、离桌高度(夹爪碰撞网格最低点 - 桌面);缓存缺失返回 None。"""
        category, index = asset.split("/")
        path = caches[which] / category / f"{int(index):05d}.npz"
        if not path.exists():
            return None
        with np.load(path) as d:
            ee, scores, meta = d["ee_poses"], d["scores"], json.loads(str(d["meta"]))
        T = np.tile(np.eye(4), (len(ee), 1, 1))
        for k, pose in enumerate(ee):
            T[k, :3, :3] = gg._quat2mat(pose[3:]) @ to_link6.T  # link6 -> 抓取坐标系(固定关节无平移)
            T[k, :3, 3] = pose[:3]
        _, table_z = load_mesh(asset)
        z = np.einsum("kj,mj->km", T[:, 2, :3], gripper_coll) + T[:, 2, 3:4]
        return T, np.asarray(scores, dtype=float), meta, z.min(axis=1) - table_z

    server = viser.ViserServer(host=args.host, port=args.port, label="GraspGenX grasps")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_width="large")

    gui_asset = server.gui.add_dropdown("物体", assets, initial_value=assets[0])
    gui_cache = server.gui.add_dropdown("缓存", list(caches), initial_value="new")
    gui_topk = server.gui.add_slider("显示前 K 个", min=1, max=200, step=1, initial_value=30)
    gui_min = server.gui.add_slider("最低分数", min=0.0, max=1.0, step=0.01, initial_value=0.0)
    gui_color = server.gui.add_dropdown("着色", ["score", "table"], initial_value="score")
    gui_table = server.gui.add_checkbox("显示桌面", True)
    gui_pick = server.gui.add_slider("选中抓取 (#)", min=0, max=29, step=1, initial_value=0)
    gui_gripper = server.gui.add_checkbox("显示夹爪网格", True)
    gui_info = server.gui.add_markdown("")
    gui_pick_info = server.gui.add_markdown("")

    state = {"data": None, "shown": np.zeros(0, dtype=int)}
    lock = threading.RLock()  # 回调里改其他控件的值会同步触发它们的回调,须可重入

    def redraw_pick():
        data, shown = state["data"], state["shown"]
        if data is None or len(shown) == 0:
            server.scene.add_frame("/pick", show_axes=False, visible=False)
            gui_pick_info.content = ""
            return
        T, scores, _, clear = data
        j = int(min(gui_pick.value, len(shown) - 1))
        k = shown[j]
        wxyz = viser.transforms.SO3.from_matrix(T[k, :3, :3]).wxyz
        server.scene.add_mesh_simple(
            "/pick/gripper",
            vertices=gripper_vis.vertices.astype(np.float32),
            faces=gripper_vis.faces.astype(np.uint32),
            color=(0, 100, 255),
            opacity=0.85,
            wxyz=wxyz,
            position=T[k, :3, 3],
            visible=gui_gripper.value,
        )
        pts = seg_local @ T[k, :3, :3].T + T[k, :3, 3]
        server.scene.add_line_segments("/pick/ctrl", pts, (0, 100, 255), thickness=5.0, thickness_units="screen")
        approach = T[k, :3, 2]
        gui_pick_info.content = (
            f"**#{j}**(缓存第 {k} 个)  分数 `{scores[k]:.3f}`  \n"
            f"接近方向 `{np.round(approach, 2).tolist()}`(z = {approach[2]:+.2f})  \n"
            f"离桌高度 `{clear[k] * 1000:+.1f} mm`  \n"
            f"TCP 位置 `{np.round(T[k, :3, 3], 3).tolist()}`"
        )

    def redraw_grasps():
        data = state["data"]
        if data is None:
            server.scene.add_frame("/grasps", show_axes=False, visible=False)
            state["shown"] = np.zeros(0, dtype=int)
            redraw_pick()
            return
        T, scores, _, clear = data
        idx = np.flatnonzero(scores >= gui_min.value)[: int(gui_topk.value)]  # 缓存已按分数降序
        state["shown"] = idx
        gui_pick.max = max(1, len(idx) - 1)
        if gui_pick.value > len(idx) - 1:
            gui_pick.value = 0
        if len(idx) == 0:
            server.scene.add_frame("/grasps", show_axes=False, visible=False)
        else:
            pts = np.einsum("kij,snj->ksni", T[idx, :3, :3], seg_local) + T[idx, None, None, :3, 3]
            if gui_color.value == "score":
                c = np.stack([1.0 - scores[idx], scores[idx], np.zeros(len(idx))], axis=1)
            else:
                ok = clear[idx] >= args.table_clear
                c = np.where(ok[:, None], [0.1, 0.8, 0.1], [0.9, 0.1, 0.1])
            colors = np.broadcast_to((np.clip(c, 0, 1) * 255).astype(np.uint8)[:, None, None, :], pts.shape)
            server.scene.add_line_segments(
                "/grasps",
                pts.reshape(-1, 2, 3).astype(np.float32),
                colors.reshape(-1, 2, 3),
                thickness=2.0,
                thickness_units="screen",
            )
        redraw_pick()

    def load_asset():
        asset = gui_asset.value
        mesh, table_z = load_mesh(asset)
        server.scene.add_mesh_simple(
            "/object",
            vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.uint32),
            color=(200, 200, 200),
        )
        lo, hi = mesh.bounds
        ext = max(hi[0] - lo[0], hi[1] - lo[1]) + 0.25
        state["table"] = server.scene.add_box(
            "/table",
            color=(217, 199, 166),
            dimensions=(ext, ext, 0.002),
            opacity=0.6,
            position=((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, table_z - 0.001),
            visible=gui_table.value,
        )
        data = load_cache(gui_cache.value, asset)
        state["data"] = data
        if data is None:
            gui_info.content = f"**{asset}**:`{gui_cache.value}` 缓存中没有这个物体"
        else:
            _, scores, meta, clear = data
            n = len(scores)
            bad = int((clear < 0).sum())
            gui_topk.max = max(2, n)
            table = meta.get("table_clear")
            gui_info.content = (
                f"**{asset}**({gui_cache.value})  \n"
                f"抓取 {n} 个,分数 {scores.min():.3f} – {scores.max():.3f}  \n"
                f"穿桌(离桌 < 0){bad} 个 = {bad / n:.0%};离桌 < {args.table_clear * 1000:.0f} mm "
                f"{int((clear < args.table_clear).sum())} 个  \n"
                f"物体高 {(hi[2] - lo[2]) * 100:.1f} cm;生成时桌面过滤:"
                + (
                    "无"
                    if table is None
                    else f"{table * 1000:.0f} mm,{meta.get('table_free')}/{meta.get('sampled')} "
                    f"个通过,{meta.get('passes')} 轮"
                )  # fmt: skip
            )
        redraw_grasps()

    def guarded(fn):
        def cb(_):
            with lock:
                fn()

        return cb

    gui_asset.on_update(guarded(load_asset))
    gui_cache.on_update(guarded(load_asset))
    for g in (gui_topk, gui_min, gui_color):
        g.on_update(guarded(redraw_grasps))
    gui_pick.on_update(guarded(redraw_pick))
    gui_gripper.on_update(guarded(redraw_pick))

    @gui_table.on_update
    def _(_):
        if state.get("table") is not None:
            state["table"].visible = gui_table.value

    with lock:
        load_asset()
    print(f"[view_grasps] {len(assets)} assets, caches {', '.join(f'{k}={v}' for k, v in caches.items())}")
    print(f"[view_grasps] open http://localhost:{args.port}  (remote: ssh -L {args.port}:localhost:{args.port} <host>)")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
