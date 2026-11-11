"""用三角网格射线求交探测一个关节物体(微波炉 / 冰箱 / 烤面包机)的腔体几何,给专家的碰撞盒用。

不依赖 PhysX:把 USD 里所有 Mesh 读成三角网格(trimesh),从炉口正面沿 +y 打射线 ——
每条射线先打到的是炉口平面,最后打到的点是腔体后壁;只有能打进去的 (x, z) 范围才是炉口。
再从上往下打,找腔体顶面的高度。

用法::

    python src/data_collect/probe_cavity.py \
        --usd Assets/Object/RoboDojo/Articulation/SimReady_microwave/00000/object.usd \
        --front -0.6 --axis y --span-x=-0.4:0.4 --span-z 0.0:0.35
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--front", type=float, default=-0.6, help="ray origin along the approach axis")
parser.add_argument("--axis", default="y", choices=("x", "y"), help="approach axis (rays go along +axis)")
parser.add_argument("--span-x", default="-0.4:0.4", help="lateral range at the opening")
parser.add_argument("--span-z", default="0.0:0.35")
parser.add_argument("--step", type=float, default=0.02)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import trimesh  # noqa: E402


def _span(text):
    lo, hi = (float(v) for v in text.split(":"))
    return lo, hi


def _mesh(stage, root_path):
    """``root_path`` 下所有 Mesh 合成一个世界坐标的三角网格。"""
    root = stage.GetPrimAtPath(root_path)
    verts, faces, offset = [], [], 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        mat = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float).T
        verts.append(points @ mat[:3, :3].T + mat[:3, 3])
        start = 0
        for n in counts:  # 扇形三角化
            for k in range(1, n - 1):
                faces.append([indices[start] + offset, indices[start + k] + offset, indices[start + k + 1] + offset])
            start += n
        offset += len(points)
    return trimesh.Trimesh(np.vstack(verts), np.asarray(faces, dtype=int), process=False)


def main():
    add_reference_to_stage(usd_path=args_cli.usd, prim_path="/World/probe")
    for _ in range(3):
        simulation_app.update()
    stage = Usd.Stage.Open(args_cli.usd)  # 直接在 USD 层面算,不需要仿真
    root = stage.GetDefaultPrim()
    mesh = _mesh(stage, str(root.GetPath()))
    print(f"[probe] mesh {len(mesh.vertices)} vertices {len(mesh.faces)} faces")
    axis = 0 if args_cli.axis == "x" else 1
    lateral = 1 - axis
    xs = np.arange(*_span(args_cli.span_x), args_cli.step)
    zs = np.arange(*_span(args_cli.span_z), args_cli.step)
    grid = [(u, z) for u in xs for z in zs]
    origins = np.zeros((len(grid), 3))
    origins[:, axis] = args_cli.front
    origins[:, lateral] = [g[0] for g in grid]
    origins[:, 2] = [g[1] for g in grid]
    directions = np.zeros((len(grid), 3))
    directions[:, axis] = 1.0
    hits, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    order = np.argsort(index_ray)
    hits, index_ray = hits[order], index_ray[order]
    starts = {}
    ends = {}
    for point, ray_id in zip(hits, index_ray):
        starts.setdefault(int(ray_id), point)
        ends[int(ray_id)] = point
    entering = [
        (grid[i][0], grid[i][1], starts[i][axis], ends[i][axis])
        for i in starts
        if ends[i][axis] - starts[i][axis] > 0.03  # 打进去 >= 3 cm 才算穿过炉口
    ]
    if not entering:
        print("[probe] no ray entered the cavity")
        return
    u = np.asarray([e[0] for e in entering])
    z = np.asarray([e[1] for e in entering])
    print(f"[probe] {len(entering)}/{len(grid)} rays enter the cavity")
    print(f"[probe] opening plane {args_cli.axis} = {np.median([e[2] for e in entering]):.4f}")
    print(f"[probe] cavity back   {args_cli.axis} = {max(e[3] for e in entering):.4f}")
    print(f"[probe] cavity span lateral = {u.min():.4f} .. {u.max():.4f}")
    print(f"[probe] cavity span z  = {z.min():.4f} .. {z.max():.4f}")
    # 从上往下打:腔体顶面(射线从 z=1 往下,取第一个交点)
    down = np.array([[0.0, 0.0, -1.0]])
    top_z = []
    for uu in np.linspace(u.min(), u.max(), 7):
        for yy in np.linspace(np.median([e[2] for e in entering]) + 0.03, max(e[3] for e in entering) - 0.03, 4):
            origin = np.array([[uu if axis == 0 else yy, yy if axis == 0 else uu, 1.0]])
            hit, _, _ = mesh.ray.intersects_location(origin, down, multiple_hits=False)
            if len(hit) and hit[0][2] > 0.02:
                top_z.append(hit[0][2])
    if top_z:
        print(f"[probe] first surface from above z = {min(top_z):.4f} (grid min)")
    # 在腔体内部朝 +z / -z / +x / -x 打,量顶面、底面、左右壁
    inner_y = np.median([e[2] for e in entering]) - 0.10
    ceiling, floor, walls_lo, walls_hi = [], [], [], []
    for uu in np.linspace(u.min() + 0.02, u.max() - 0.02, 9):
        for zz in np.linspace(0.09, 0.22, 4):  # 从转盘上方往上打,量腔体顶面
            origin = np.array([[uu if axis == 0 else inner_y, inner_y if axis == 0 else uu, zz]])
            hit, _, _ = mesh.ray.intersects_location(origin, np.array([[0.0, 0.0, 1.0]]), multiple_hits=False)
            if len(hit) and hit[0][2] > zz + 0.01:
                ceiling.append(hit[0][2])
            hit, _, _ = mesh.ray.intersects_location(origin, np.array([[0.0, 0.0, -1.0]]), multiple_hits=False)
            if len(hit) and hit[0][2] < zz - 0.01:
                floor.append(hit[0][2])
    for zz in np.linspace(0.05, 0.18, 6):
        origin = np.array([[0.0, inner_y, zz]])
        hit, _, _ = mesh.ray.intersects_location(origin, np.array([[1.0, 0.0, 0.0]]), multiple_hits=False)
        if len(hit):
            walls_hi.append(hit[0][0])
        hit, _, _ = mesh.ray.intersects_location(origin, np.array([[-1.0, 0.0, 0.0]]), multiple_hits=False)
        if len(hit):
            walls_lo.append(hit[0][0])
    print(f"[probe] cavity ceiling z = {min(ceiling) if ceiling else float('nan'):.4f} (min over grid)")
    print(f"[probe] cavity floor   z = {min(floor) if floor else float('nan'):.4f} (min over grid)")
    lo = min(walls_lo) if walls_lo else float("nan")
    hi = max(walls_hi) if walls_hi else float("nan")
    print(f"[probe] walls x = {lo:.4f} .. {hi:.4f}")
    simulation_app.close()


if __name__ == "__main__":
    main()
