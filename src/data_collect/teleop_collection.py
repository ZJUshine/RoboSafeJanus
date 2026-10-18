"""通过 XRoboToolkit 用 PICO 手柄遥操作 RoboDojo 任务。

操作者看着显示器上的 Isaac Sim 窗口,用 PICO 手柄驱动机械臂(头显上的
XRoboToolkit App 把手柄数据推送给 PC 服务)。这里录制每个物理步的关节指令;
为保证控制循环流畅,不渲染观测相机,之后由 ``teleop_replay.py`` 以 headless
模式带相机重放这些指令,写出常规的 XPolicyLab HDF5。

按键(左手柄 -> ``left_arm``,右手柄 -> ``right_arm``):
    grip 按住      离合:手臂跟随手柄相对于按下 grip 时刻(手柄与手臂各自位姿)
                   的运动
    trigger        夹爪(松开 = 张开,扣下 = 闭合)
    右手 A         双臂回到初始位姿(会被录制;再次按下 grip 会取消回零,
                   把手臂交还给操作者)
    右手 B         结束:运行任务成功判定并保存
    左手 Y         放弃本次尝试,加载下一个布局

手部运动按"操作者面向机器人前方(+y,即 Isaac 窗口视角)"来映射;面对显示器时
长按 PICO 键重新校准头显,或用 ``--operator_yaw`` 旋转映射。
第一次按下 grip 或 trigger 之前,物理仿真不会推进。

输出::

    <save_dir>/<task>/<config>_teleop/
        raw/episode_0000000.npz     关节指令 + 布局 + seed(重放的输入)
        next_seed.txt               续采位置
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task_name", type=str, required=True)
parser.add_argument("--env_cfg_type", type=str, default="arx_x5", help="env_cfg/<name>.yml (robot/camera/obs)")
parser.add_argument("--episode_num", type=int, default=50, help="episodes to save in this session")
parser.add_argument("--start_seed", type=int, default=200000, help="first layout seed (scripted uses 100000+)")
parser.add_argument("--save_dir", type=str, default="data/RoboJanus")
parser.add_argument("--layout_source", choices=("random", "eval"), default="random")
parser.add_argument("--layout_template", type=str, default=None, help="task name or json whose fixtures to reuse")
parser.add_argument(
    "--input", choices=("xrobotoolkit", "scripted"), default="xrobotoolkit", help="scripted: no headset, for testing"
)
parser.add_argument("--xrt_sdk", type=str, default=None, help="libPXREARobotSDK.so (default: the PC service's copy)")
parser.add_argument("--control_hz", type=float, default=50.0, help="IK / render rate (in simulated time)")
parser.add_argument("--pos_scale", type=float, default=1.0, help="robot displacement per metre of hand motion")
parser.add_argument("--max_joint_step", type=float, default=0.05, help="rad per control tick (speed limit)")
parser.add_argument("--home_joint_step", type=float, default=0.02, help="rad per control tick while homing (right A)")
parser.add_argument("--ik_damping", type=float, default=0.05, help="damped-least-squares lambda near singularities")
parser.add_argument("--operator_yaw", type=float, default=0.0, help="rotate the hand->robot mapping about z (deg)")
parser.add_argument("--keep_failed", action="store_true", help="also save attempts that fail the success check")
parser.add_argument("--keep_cameras", action="store_true", help="keep observation cameras rendering (debug)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import transforms3d as t3d  # noqa: E402

from src.data_collect.collect_env import create_collect_env  # noqa: E402
from src.data_collect.common import build_env_cfg, make_layout_getter  # noqa: E402
from src.data_collect.xr_input import XRT_SDK_LIB, ScriptedControllerInput, XRoboToolkitInput  # noqa: E402
from utils.cluttered_generator import UnStableError  # noqa: E402

HAND_TO_ARM = {"left": "left_arm", "right": "right_arm"}
# 操作者视角(env 坐标系):位于机械臂基座后方,高度足以越过头部相机支架看到桌面。
VIEW_EYE, VIEW_TARGET = (0.0, -0.9, 1.95), (0.0, 0.15, 0.76)


def write_raw(path, chunks, meta):
    """把逐 tick 的控制块展平成 ``pos.<key>`` / ``vel.<key>`` 数组(float64,保证精确重放)。"""
    flat = [ctrl for chunk in chunks for ctrl in chunk]
    arrays = {"chunk_lens": np.asarray([len(c) for c in chunks], dtype=np.int64)}
    for key in flat[0]:
        arrays[f"pos.{key}"] = np.asarray([c[key]["position"] for c in flat], dtype=np.float64)
        arrays[f"vel.{key}"] = np.asarray([c[key]["velocity"] for c in flat], dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)


class Teleop:
    def __init__(self, env, source, args):
        self.env, self.source, self.args = env, source, args
        self.rm = env.robot_manager
        self.arms = [r.arm_name for r in env._target_robots()]
        self.hand_to_arm = {h: a for h, a in HAND_TO_ARM.items() if a in self.arms} or {"right": self.arms[0]}
        self.substeps = max(1, round(1.0 / (args.control_hz * env.dt)))
        self.tick_period = self.substeps * env.dt
        self.request = None
        self._pressed = {}
        self._ik = {}

    def begin(self, layout, seed):
        self.env.reset_with_layout(layout, seed, record=False)
        for arm in self.arms:  # 机器人句柄只有在第一次 reset 之后才存在
            robot = self.env.robot(arm)
            art = self.rm.robot_key[self.rm.robot_list.index(robot)]
            body = art.body_names.index(robot.ee_link_name)
            joints = list(robot.arm_joint_indices)
            limits = art.data.soft_joint_pos_limits[0, joints].cpu().numpy()
            # 固定基座的 articulation,PhysX 的雅可比里不含固定的根 link,因此 body 下标要减 1。
            self._ik[arm] = (art, body - 1 if art.is_fixed_base else body, joints, limits)
        self.q_cmd = {a: self.env.get_arm_joints(a) for a in self.arms}
        self.q_home = {a: q.copy() for a, q in self.q_cmd.items()}
        self.homing = False
        self.grip = {a: 1.0 for a in self.arms}
        self.clutch = {a: None for a in self.arms}
        self.chunks = []
        self.started = False
        self.request = None
        # env.reset 会把相机频率重新设为 Kit 主循环的限速(25 Hz),这会限制每次 render();
        # 遥操作循环自己控制节奏,所以这里关掉限速。
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", False)
        origin = self.env.env_origins[0]
        origin = origin.cpu().numpy() if hasattr(origin, "cpu") else np.asarray(origin)
        self.env.sim.sim.set_camera_view(origin + VIEW_EYE, origin + VIEW_TARGET)

    # ------------------------------------------------------------- 输入

    def _edge(self, name, value):
        pressed = value > 0.5
        rising = pressed and not self._pressed.get(name, False)
        self._pressed[name] = pressed
        return rising

    def _buttons(self, states):
        right, left = states.get("right"), states.get("left")
        if right is not None and self._edge("right_a", right.button_0) and self.started:
            self.homing = True
            print("[teleop] homing: arms returning to the start pose")
        if right is not None and self._edge("right_b", right.button_1):
            self.request = "save"
        if left is not None and self._edge("left_y", left.button_1):
            self.request = "discard"

    def _target_pose(self, arm, ctrl_pose):
        """把手柄自离合按下以来的运动叠加到离合时刻的末端位姿上(均在世界坐标系)。"""
        ctrl0, ee0 = self.clutch[arm]
        pos = ee0[:3] + self.args.pos_scale * (ctrl_pose[:3] - ctrl0[:3])
        d_rot = t3d.quaternions.quat2mat(ctrl_pose[3:]) @ t3d.quaternions.quat2mat(ctrl0[3:]).T
        rot = d_rot @ t3d.quaternions.quat2mat(ee0[3:])
        return np.concatenate([pos, t3d.quaternions.mat2quat(rot)])

    def _track(self, arm, ctrl_pose):
        """朝目标位姿走一步阻尼最小二乘(DLS)IK。

        微分 IK 会停留在当前关节分支上(全局 IK 求解器对相近的位姿可能给出相距很远的
        构型),且耗时 < 1 ms。
        """
        art, jac_body, joints, limits = self._ik[arm]
        target, ee = self._target_pose(arm, ctrl_pose), self.env.get_arm_pose(arm)
        rot_err = t3d.quaternions.quat2mat(target[3:]) @ t3d.quaternions.quat2mat(ee[3:]).T
        axis, angle = t3d.axangles.mat2axangle(rot_err)
        err = np.concatenate([target[:3] - ee[:3], axis * angle])
        jac = art.root_physx_view.get_jacobians()[0, jac_body][:, joints].cpu().numpy().astype(float)
        lam2 = self.args.ik_damping**2
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(6), err)
        step = self.args.max_joint_step
        q = self.env.get_arm_joints(arm) + np.clip(dq, -step, step)
        return np.clip(q, limits[:, 0], limits[:, 1])

    def _home_step(self):
        """每条臂在关节空间沿直线朝初始位姿走一步,单步最大变化为 ``home_joint_step``。"""
        targets = {}
        for arm in self.arms:
            delta = self.q_home[arm] - self.q_cmd[arm]
            largest = float(np.max(np.abs(delta)))
            targets[arm] = self.q_cmd[arm] + delta * min(1.0, self.args.home_joint_step / max(largest, 1e-9))
        if all(np.allclose(targets[a], self.q_home[a], atol=1e-4) for a in self.arms):
            self.homing = False
            print("[teleop] homing done")
        return targets

    # ------------------------------------------------------------ 控制

    def _chunk(self, targets):
        """生成 ``substeps`` 个控制指令,把每条臂从上一次指令线性插值到 ``targets``。"""
        grip_ctrl = {}
        for arm in self.arms:
            grip_ctrl.update(
                self.rm.plan_endeffector_joint(env_idx=0, arm_tag=arm, target_val=self.grip[arm], control_step_num=1)[0]
            )
        n, chunk = self.substeps, []
        for k in range(1, n + 1):
            ctrl = {}
            for arm in self.arms:
                q0, q1 = self.q_cmd[arm], targets[arm]
                ctrl[self.rm.process_name(arm)] = {
                    "position": (q0 + (q1 - q0) * k / n).tolist(),
                    "velocity": ((q1 - q0) / (n * self.env.dt)).tolist(),
                }
            ctrl.update(grip_ctrl)
            chunk.append(ctrl)
        return chunk

    def tick(self):
        states = self.source.read()
        self._buttons(states)
        targets = dict(self.q_cmd)
        for hand, arm in self.hand_to_arm.items():
            c = states.get(hand)
            if c is None:  # 追踪丢失:松开离合,恢复后需重新按 grip
                self.clutch[arm] = None
                continue
            self.grip[arm] = 1.0 - float(np.clip(c.trigger, 0.0, 1.0))
            if self._edge(f"{hand}_grip", c.squeeze) and self.homing:
                self.homing = False
                print("[teleop] homing cancelled")
            if c.squeeze > 0.5 and not self.homing:
                if self.clutch[arm] is None:
                    self.clutch[arm] = (np.asarray(c.pose, dtype=float), self.env.get_arm_pose(arm))
                targets[arm] = self._track(arm, np.asarray(c.pose, dtype=float))
            else:
                self.clutch[arm] = None
        if self.homing:
            targets = self._home_step()
        if not self.started:
            self.started = any(v is not None for v in self.clutch.values()) or any(g < 1.0 for g in self.grip.values())
        if not self.started:
            self.env.render()
            return
        chunk = self._chunk(targets)
        self.env.execute(chunk)
        self.chunks.append(chunk)
        self.q_cmd = targets
        self.env.render()

    def meta(self, layout, seed, collect_cfg, success):
        return {
            "task_name": self.env.task_name,
            "env_cfg_type": self.args.env_cfg_type,
            "config_name": collect_cfg["config_name"],
            "seed": seed,
            "layout": layout,
            "dt": float(self.env.dt),
            "substeps": self.substeps,
            "teleop_success": bool(success),
            "final_arm_joints": {a: self.env.get_arm_joints(a).tolist() for a in self.arms},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def make_input():
    if args_cli.input == "scripted":
        return ScriptedControllerInput()
    source = XRoboToolkitInput(operator_yaw=args_cli.operator_yaw, lib_path=args_cli.xrt_sdk or XRT_SDK_LIB)
    deadline = time.monotonic() + 3.0
    while not source.connected and time.monotonic() < deadline:
        time.sleep(0.05)
    if not source.connected:
        raise RuntimeError(
            "No connection to the XRoboToolkit PC service; start it with /opt/apps/roboticsservice/runService.sh"
        )
    return source


def main():
    task_name, cfg_name = args_cli.task_name, args_cli.env_cfg_type
    env_cfg, collect_cfg = build_env_cfg(task_name, cfg_name)
    get_layout = make_layout_getter(
        env_cfg, collect_cfg, task_name, args_cli.layout_source, template=args_cli.layout_template
    )
    save_path = os.path.join(args_cli.save_dir, task_name, collect_cfg["config_name"] + "_teleop")
    raw_dir = os.path.join(save_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    seed_file = os.path.join(save_path, "next_seed.txt")
    seed = int(open(seed_file).read()) if os.path.exists(seed_file) else args_cli.start_seed
    saved_before = len([f for f in os.listdir(raw_dir) if f.endswith(".npz")])

    env = create_collect_env(env_cfg, simulation_app)
    if not args_cli.keep_cameras:
        # 遥操作时不创建观测相机的渲染产物;由 teleop_replay.py 负责渲染。
        env.capture_manager.init_cameras = lambda: None

    source = make_input()
    teleop = Teleop(env, source, args_cli)
    print(
        f"[teleop] task={task_name} save={save_path} seed={seed} control={1 / teleop.tick_period:.0f}Hz "
        f"({teleop.substeps} physics steps/tick), input={args_cli.input}"
    )

    saved = 0
    while simulation_app.is_running() and saved < args_cli.episode_num:
        try:
            layout = get_layout(seed)
            teleop.begin(layout, seed)
        except UnStableError as e:
            print(f"[teleop] seed={seed} skipped: {e}")
            seed += 1
            continue
        print(f"[teleop] seed={seed} ready: {' / '.join(env.instructions()[:1])}")
        t_next, t_report, n_ticks = time.monotonic(), time.monotonic(), 0
        while simulation_app.is_running() and teleop.request is None:
            teleop.tick()
            n_ticks += 1
            t_next += teleop.tick_period
            now = time.monotonic()
            if now < t_next:
                time.sleep(t_next - now)
            else:
                t_next = now  # 比实时慢:不追赶落下的 tick
            if now - t_report > 10.0:
                if teleop.started:
                    rtf = n_ticks * teleop.tick_period / (now - t_report)
                    print(f"[teleop] {n_ticks / (now - t_report):.1f} ticks/s ({rtf:.2f}x real time)")
                elif not any(teleop.source.read().values()):
                    print("[teleop] waiting: no controller data (XRoboToolkit app running on the headset?)")
                else:
                    print("[teleop] waiting: squeeze a grip or pull a trigger to start (physics is paused until then)")
                t_report, n_ticks = now, 0

        if teleop.request == "save" and teleop.chunks:
            ok = env.check_success()
            name = f"episode_{saved_before + saved:07d}"
            if ok or args_cli.keep_failed:
                meta = teleop.meta(layout, seed, collect_cfg, ok)
                write_raw(os.path.join(raw_dir, f"{name}.npz"), teleop.chunks, meta)
                saved += 1
                colour = "92" if ok else "93"
                steps = sum(len(c) for c in teleop.chunks)
                print(f"\033[{colour}m[teleop] {name} saved (seed={seed}, success={ok}, {steps} steps)\033[0m")
            else:
                print(f"\033[91m[teleop] seed={seed} not saved, failed checks: {env.failed_checks()}\033[0m")
        else:
            print(f"[teleop] seed={seed} discarded")
        seed += 1
        with open(seed_file, "w") as fp:
            fp.write(str(seed))

    print(f"[teleop] done: {saved} episodes saved to {raw_dir}")
    source.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
