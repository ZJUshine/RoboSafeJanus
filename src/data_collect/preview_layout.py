"""WebRTC 推流预览布局:把物体摆到桌面上,在浏览器里挪动 / 观察,选定“中心位置”。

重新设计一个任务的布局时,先用本脚本把物体按一个标称布局放到桌子上,再用 Isaac Sim 的
WebRTC livestream 把整个 Kit 界面推到远端客户端(见 ``scripts/preview_layout.sh``)。布局沉降后
默认暂停时间线,可以直接在 3D 视图里用移动工具拖动微波炉 / 物体;右侧面板也给了每个物体一组
x / y / yaw 输入框。拖完按 “Capture scene poses” 把场景里的实际位姿读回面板,再按 “Save layout”
把 x / y / z / yaw 写回布局 json。这些位姿就是后续写进任务 YAML 的中心,再按小范围扰动展开。

- 桌面上画了 5 cm 间隔的网格(灰线,每 10 cm 一条白线),``x = 0`` 是红线、``y = 0`` 是绿线,
  方便直接读出大概位置。
- 也可以不走界面:外部修改布局 json,脚本会监听文件修改并自动重新加载,适合用命令行迭代坐标。
- 保存写回 ``--layout`` 指向的文件(或用 ``--save_to`` 另存),内容是 ``Rigid`` / ``Articulation``
  的 ``default_pos`` / ``default_ori``(env 坐标系,与 ``Assets/Eval_Layout`` 里的布局一致)。

用法::

    bash scripts/preview_layout.sh heat_sandwich_microwave /tmp/heat_sandwich/nominal.json 3

然后在本地浏览器打开脚本打印的地址(默认 ``http://<host>:8011/streaming/webrtc-client``)。
额外参数原样传给本脚本,例如 ``--grid_step 0.1``。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description="preview / pick a layout through WebRTC livestream")
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="env_cfg/<name>.yml")
parser.add_argument("--layout", type=str, required=True, help="layout json to preview (written back on save)")
parser.add_argument("--seed", type=int, default=0, help="seed used for the reset")
parser.add_argument("--grid_step", type=float, default=0.05, help="table grid spacing (m)")
parser.add_argument("--no_grid", action="store_true", help="do not draw the table grid")
parser.add_argument(
    "--view_camera", type=str, default="cam_head", help="camera the viewport looks through ('' = keep default)"
)
parser.add_argument(
    "--no_pause", action="store_true", help="keep the timeline running after settling (physics keeps settling)"
)
parser.add_argument("--save_view", type=str, default=None, help="directory for one png per camera after settling")
parser.add_argument(
    "--save_to", type=str, default=None, help="save the picked layout here (default: overwrite --layout)"
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac 必须先启动,之后才能 import 依赖它的模块
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.timeline  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from src.data_collect.collect_env import create_collect_env  # noqa: E402
from src.data_collect.common import build_env_cfg  # noqa: E402
from utils.load_file import load_json  # noqa: E402

LAYOUT_PATH = os.path.abspath(args_cli.layout)
LIVE_PATH = os.path.splitext(LAYOUT_PATH)[0] + ".live.json"
SAVE_PATH = os.path.abspath(args_cli.save_to) if args_cli.save_to else LAYOUT_PATH
GRID_ROOT = "/World/PreviewGrid"
GRID_COLORS = {
    "minor": (0.25, 0.25, 0.25),
    "major": (0.92, 0.92, 0.92),
    "x_axis": (0.95, 0.15, 0.15),
    "y_axis": (0.15, 0.90, 0.15),
}
GRID_LINE_WIDTH = 0.008  # m
MOVABLE = ("Rigid", "Articulation")
UI_STATE = {"save": False, "reload": False, "capture": False}
POSES = {}  # (obj_type, category, index) -> [x, y, z, yaw];界面改的就是它,保存时写回布局


# ------------------------------------------------------------------ 布局读写


def movable_records(layout):
    """布局中的可移动物体 ``[(obj_type, category, index, record), ...]``。"""
    out = []
    for obj_type in MOVABLE:
        for category, records in (layout.get(obj_type) or {}).items():
            for i, rec in enumerate(records):
                out.append((obj_type, category, i, rec))
    return out


def table_top(layout):
    table = layout["Table"]
    return float(table["default_pos"][2]) + float(table["scale"][2]) / 2.0


# ------------------------------------------------------------------ 桌面网格


def _material(stage, name, color):
    path = f"{GRID_ROOT}/Materials/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    for attr in ("diffuseColor", "emissiveColor"):
        shader.CreateInput(attr, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _strip(stage, name, x0, y0, x1, y1, z, material):
    """桌面上一条平躺的细长盒子(比曲线稳妥,RTX 下任何视角都看得见)。"""
    prim = UsdGeom.Cube.Define(stage, f"{GRID_ROOT}/{name}")
    prim.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2.0, (y0 + y1) / 2.0, z))
    xform.AddScaleOp().Set(Gf.Vec3f(max(abs(x1 - x0), GRID_LINE_WIDTH), max(abs(y1 - y0), GRID_LINE_WIDTH), 0.0008))
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)


def draw_grid(layout, step):
    """在桌面上画坐标网格:x = 0 红色、y = 0 绿色,其余每 10 cm 一条白线。"""
    stage = omni.usd.get_context().get_stage()
    old = stage.GetPrimAtPath(GRID_ROOT)
    if old and old.IsValid():
        stage.RemovePrim(GRID_ROOT)
    UsdGeom.Xform.Define(stage, GRID_ROOT)
    z = table_top(layout) + 0.0025
    table = layout["Table"]
    half_x = float(table["scale"][0]) / 2.0 - 0.02
    half_y = float(table["scale"][1]) / 2.0 - 0.02
    center_x = float(table["default_pos"][0])
    center_y = float(table["default_pos"][1])
    materials = {k: _material(stage, k, c) for k, c in GRID_COLORS.items()}

    def kind_of(value, axis):
        if abs(value) < 1e-6:
            return f"{axis}_axis"
        return "major" if round(value / step) % 2 == 0 else "minor"

    for i in range(-int(round(half_x / step)), int(round(half_x / step)) + 1):
        x = center_x + i * step
        _strip(stage, f"x{i + 100}", x, center_y - half_y, x, center_y + half_y, z, materials[kind_of(x, "x")])
    for j in range(-int(round(half_y / step)), int(round(half_y / step)) + 1):
        y = center_y + j * step
        _strip(stage, f"y{j + 100}", center_x - half_x, y, center_x + half_x, y, z, materials[kind_of(y, "y")])


# ------------------------------------------------------------------ 位姿


def _scene_object(env, label):
    lm = env.scene_manager.layout_manager
    inst = lm.get_instance_name(0, label)
    return lm.get_scene_object(0, inst) if inst else None


def yaw_deg(quat):
    w, x, y, z = quat
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def read_poses(env, layout):
    """从 env 读当前位姿,填进 ``POSES``:``{(obj_type, category, index): [x, y, z, yaw]}``。"""
    POSES.clear()
    for obj_type, category, i, rec in movable_records(layout):
        pose = np.asarray(env.get_obj_pose(rec["label"]), dtype=float)
        POSES[(obj_type, category, i)] = [*pose[:3], yaw_deg(pose[3:7])]


def apply_pose(env, layout, key):
    """把 ``POSES[key]`` 应用到场景(物理继续跑,物体随后自己沉降)。"""
    rec = next(r[3] for r in movable_records(layout) if (r[0], r[1], r[2]) == key)
    obj = _scene_object(env, rec["label"])
    if obj is None:
        return
    x, y, z, yaw = POSES[key]
    half = math.radians(yaw) / 2.0
    # 保留物体原来的 roll / pitch(平躺),只改 yaw
    obj.set_local_pose(
        translation=np.array([x, y, z], dtype=float),
        orientation=np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float),
    )


def _usd_body_pose(env, label):
    """物体的 USD 刚体根世界坐标(与 fabric 位姿对照,判断 3D 拖动有没有落到物理上)。"""
    from pxr import UsdPhysics

    lm = env.scene_manager.layout_manager
    inst = lm.get_instance_name(0, label)
    obj = lm.get_scene_object(0, inst) if inst else None
    if obj is None:
        return None
    root = obj.prim
    body = root if root.HasAPI(UsdPhysics.RigidBodyAPI) else None
    if body is None:
        body = next((p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
    if body is None:
        return None
    mat = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    origin = env.scene_manager.env_origins[0]
    origin = origin.detach().cpu().numpy() if hasattr(origin, "detach") else np.asarray(origin)
    return [round(float(t[i] - origin[i]), 4) for i in range(3)]


def write_live(env, layout):
    """把场景里的实际位姿写到 ``<layout>.live.json``:拖动 / 改参数后不用点按钮,外部直接读文件。"""
    data = {"time": time.strftime("%H:%M:%S")}
    for obj_type, category, i, rec in movable_records(layout):
        pose = np.asarray(env.get_obj_pose(rec["label"]), dtype=float)
        data[rec["label"]] = {
            "sim": [round(float(v), 4) for v in (*pose[:3], yaw_deg(pose[3:7]))],
            "usd_body": _usd_body_pose(env, rec["label"]),
        }
    tmp = LIVE_PATH + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(data, fp, indent=1)
    os.replace(tmp, LIVE_PATH)


def print_poses(layout):
    print("[preview] current poses (env frame; these are what Save writes):")
    for obj_type, category, i, rec in movable_records(layout):
        x, y, z, yaw = POSES.get((obj_type, category, i), [np.nan] * 4)
        print(f"  {rec['label']:<16} x={x:+.4f} y={y:+.4f} z={z:+.4f} yaw={yaw:+7.2f}")
    sys.stdout.flush()


def save_layout(layout):
    """把 ``POSES`` 写回 ``layout`` 并落盘 ``SAVE_PATH``;返回写入的物体数。"""
    for obj_type, category, i, rec in movable_records(layout):
        x, y, z, yaw = POSES[(obj_type, category, i)]
        half = math.radians(yaw) / 2.0
        rec["default_pos"] = [round(float(v), 6) for v in (x, y, z)]
        rec["default_ori"] = [round(v, 6) for v in (math.cos(half), 0.0, 0.0, math.sin(half))]
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as fp:
        json.dump(layout, fp, indent=1)
    return len(POSES)


# ------------------------------------------------------------------ 场景


def save_view(env, out_dir):
    """把 reset 时抓到的那一帧观测按相机存成 png(没连浏览器时也能先看)。"""
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    if not env.frames:
        return
    for cam, data in env.frames[-1].get("vision", {}).items():
        path = os.path.join(out_dir, f"{cam}.png")
        imageio.imwrite(path, np.asarray(data["color"]))
        print(f"[preview] wrote {path}")


def look_through(env, camera_name):
    """把视口相机设成任务里的某一路(默认 ``cam_head``),推流连上后直接是机器人头部视角。"""
    if not camera_name:
        return
    from omni.kit.viewport.utility import get_active_viewport

    cm = env.camera_manager
    names = cm.camera_names[0]
    if camera_name not in names:
        print(f"[preview] camera {camera_name} not found, have {names}")
        return
    cam = cm.cameras[0][names.index(camera_name)]
    path = getattr(cam, "prim_path", None) or cam._prim_path
    vp = get_active_viewport()
    print(f"[preview] viewport camera -> {path} (viewport={vp})")
    if vp is not None:
        vp.camera_path = path


def load_scene(env, layout, seed):
    """加载布局、沉降,读出位姿;``--pause`` 时把时间线停下来。"""
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    env.reset_with_layout(layout, seed, record=bool(args_cli.save_view))
    if not args_cli.no_pause:
        timeline.pause()
    if not args_cli.no_grid:
        draw_grid(layout, args_cli.grid_step)
    if args_cli.save_view:
        save_view(env, args_cli.save_view)
    read_poses(env, layout)
    look_through(env, args_cli.view_camera)


# ------------------------------------------------------------------ 界面


def _build_ui(env, layout):
    """右侧面板:每个物体一组 x / y / yaw 输入框 + Save / Reload。"""
    window = ui.Window("Layout preview", width=520, height=460)
    rows = {}
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label("Grid 5 cm; red line x=0, green line y=0. Edit x / y / yaw, then Save layout.")
            ui.Line()
            for obj_type, category, i, rec in movable_records(layout):
                key = (obj_type, category, i)
                x, y, _z, yaw = POSES[key]
                with ui.HStack(height=24, spacing=6):
                    ui.Label(str(rec["label"]), width=100)
                    models = {
                        "x": ui.SimpleFloatModel(x),
                        "y": ui.SimpleFloatModel(y),
                        "yaw": ui.SimpleFloatModel(yaw),
                    }
                    for name, model in models.items():
                        ui.Label(name, width=26)
                        ui.FloatDrag(model=model, step=0.005, width=90)
                    rows[key] = models
                    for model in models.values():
                        model.add_value_changed_fn(lambda *_, k=key: _on_edit(env, layout, k, rows))
            ui.Line()
            with ui.HStack(spacing=8):
                ui.Button("Capture scene poses", clicked_fn=lambda: UI_STATE.__setitem__("capture", True))
                ui.Button("Save layout", clicked_fn=lambda: UI_STATE.__setitem__("save", True))
                ui.Button("Reload layout", clicked_fn=lambda: UI_STATE.__setitem__("reload", True))
    return {"window": window, "rows": rows}


def _on_edit(env, layout, key, rows):
    """输入框变化:x / y 直接改,yaw 改角度;写进 POSES 并应用到场景。"""
    models = rows[key]
    POSES[key][0] = float(models["x"].get_value_as_float())
    POSES[key][1] = float(models["y"].get_value_as_float())
    POSES[key][3] = float(models["yaw"].get_value_as_float())
    apply_pose(env, layout, key)


# ------------------------------------------------------------------ 主循环


def main():
    layout = load_json(LAYOUT_PATH)
    env_cfg, collect_cfg = build_env_cfg(args_cli.task_name, args_cli.env_cfg_type)
    env = create_collect_env(env_cfg, simulation_app)
    print(f"[preview] task={args_cli.task_name} env={collect_cfg['config_name']}")
    print(f"[preview] layout={LAYOUT_PATH} -> save={SAVE_PATH}")
    load_scene(env, layout, args_cli.seed)
    print_poses(layout)

    refs = _build_ui(env, layout)
    mtime = os.path.getmtime(LAYOUT_PATH)
    last_live = 0.0

    while simulation_app.is_running():
        simulation_app.update()
        if time.time() - last_live > 2.0:
            last_live = time.time()
            try:
                write_live(env, layout)
            except Exception as e:  # 拖动过程中偶尔读到空 prim,不影响主循环
                print(f"[preview] live write failed: {type(e).__name__}: {e}")

        if UI_STATE["capture"]:
            # 在 3D 视图里拖过物体之后按它:把场景里的实际位姿读回输入框,再 Save
            UI_STATE["capture"] = False
            read_poses(env, layout)
            _sync_ui(refs, layout)
            print("[preview] captured poses from the scene")
            print_poses(layout)

        if UI_STATE["save"]:
            UI_STATE["save"] = False
            n = save_layout(layout)
            mtime = os.path.getmtime(SAVE_PATH) if os.path.exists(SAVE_PATH) else mtime
            print(f"[preview] saved {n} objects to {SAVE_PATH}")
            print_poses(layout)

        changed = os.path.exists(LAYOUT_PATH) and os.path.getmtime(LAYOUT_PATH) != mtime
        if UI_STATE["reload"] or changed:
            UI_STATE["reload"] = False
            mtime = os.path.getmtime(LAYOUT_PATH)
            fresh = load_json(LAYOUT_PATH)
            layout.clear()
            layout.update(fresh)
            try:
                load_scene(env, layout, args_cli.seed)
            except Exception as e:  # 布局不稳 / 资产缺失:保留当前场景,只报错
                print(f"[preview] reload failed: {type(e).__name__}: {e}")
            else:
                _sync_ui(refs, layout)
                print_poses(layout)


def _sync_ui(refs, layout):
    """重新加载后把输入框拉回新布局的位姿。"""
    for obj_type, category, i, rec in movable_records(layout):
        models = refs["rows"].get((obj_type, category, i))
        if models is None:
            continue
        x, y, _z, yaw = POSES[(obj_type, category, i)]
        for name, value in (("x", x), ("y", y), ("yaw", yaw)):
            models[name].set_value(float(value))


if __name__ == "__main__":
    main()
