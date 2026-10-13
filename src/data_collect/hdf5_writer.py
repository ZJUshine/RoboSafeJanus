"""把采集到的帧写成 XPolicyLab xspark v1.0 HDF5,结构与已发布的 RoboDojo 数据一致。

结构(T = 录到的帧数,存储步数 = T - 1;除特别说明外数组均为 float64)::

    attrs: fix_obs_shift_rgb       "shift_drop_first_drop_last+rgb_jpeg"
    data_format_version            "v1.0"                              (ascii 字符串)
    instruction                    一条语言指令                        (ascii 字符串)
    additional_info/frequency      int64
    state/<arm>_arm_joint_states   (T-1, arm_dim)   第 t 帧
    state/<arm>_ee_joint_states    (T-1, ee_dim)    归一化夹爪开度,1 = 张开
    state/<arm>_ee_poses           (T-1, 7)         [x, y, z, qw, qx, qy, qz]
    state/<arm>_delta_ee_poses     (T-1, 7)         乘上机器人 delta_matrix 后的末端位姿
    action/<arm>_{arm_joint_states,ee_joint_states,ee_poses}   第 t + 1 帧
    vision/<camera>/colors             (T-1,) JPEG 字节串(XPolicyLab encode_image_bit)
    vision/<camera>/shape              (3,) int64
    vision/<camera>/intrinsic_matrix   (3, 3)
    vision/<camera>/extrinsic_matrix   (T-1, 4, 4)  相机到世界坐标的变换

预览视频(``write_episode(..., video_dir=...)``)沿用已发布数据的
``preview_video/<episode>_<camera>.mp4`` 布局:每路相机一个 H.264 mp4,帧率为采集频率,
每个存储步一帧(即 ``vision/<camera>/colors`` 中的图像)。

``fix_obs_shift_rgb`` 标记已发布数据中图像与同一步的状态是对齐的;
用 ``ObsManager.render_for_capture`` 采到的帧本身已经对齐,所以这里不做平移。
"""

import os

import h5py
import numpy as np
import transforms3d as t3d

from utils.save_file import VideoStreamWriter
from XPolicyLab.utils.process_data import images_encoding

OBS_SHIFT_ATTR = "shift_drop_first_drop_last+rgb_jpeg"
STATE_KEYS = {
    "arm_joint_state": "arm_joint_states",
    "ee_joint_state": "ee_joint_states",
    "ee_pose": "ee_poses",
}


def _state_series(frames):
    """把逐帧的 ``state`` 字典按 HDF5 键名(``<arm>_arm_joint_states`` 等)整理成 (T, dim) 数组。"""
    series = {}
    for key in frames[0]["state"]:
        for src, dst in STATE_KEYS.items():
            if key.endswith(src):
                prefix = key[: -len(src)]
                series[f"{prefix}{dst}"] = np.asarray([np.asarray(f["state"][key], dtype=np.float64) for f in frames])
    return series


def _delta_ee_poses(ee_poses, delta_matrix):
    """末端姿态右乘机器人的 ``delta_matrix``,位置不变。"""
    rot = [t3d.quaternions.mat2quat(t3d.quaternions.quat2mat(p[3:]) @ delta_matrix) for p in ee_poses]
    return np.concatenate([ee_poses[:, :3], np.asarray(rot)], axis=1)


def _string(text):
    """纯 ASCII 文本用 ascii 字符串类型(与已发布数据一致),否则退回 utf-8。"""
    try:
        text.encode("ascii")
        return h5py.string_dtype(encoding="ascii")
    except UnicodeEncodeError:
        return h5py.string_dtype(encoding="utf-8")


def write_episode(frames, hdf5_path, instruction, frequency, delta_matrices=None, video_dir=None, meta=None):
    """``delta_matrices``:``{手臂前缀 ("left_", "right_", ""): 3x3}``;未列出的手臂用单位阵。

    给定 ``video_dir`` 时,还会为每路相机写 ``<video_dir>/<hdf5 文件名>_<camera>.mp4``。
    ``meta``:可选的 ``{name: str | number | bool}``,以标量 dataset 写到 ``meta/`` 下。
    """
    if len(frames) < 2:
        raise ValueError("At least two frames are required to create state/action pairs")
    delta_matrices = delta_matrices or {}
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["fix_obs_shift_rgb"] = OBS_SHIFT_ATTR
        f.create_dataset("data_format_version", data="v1.0", dtype=_string("v1.0"))
        f.create_dataset("instruction", data=instruction, dtype=_string(instruction))
        f.create_group("additional_info").create_dataset("frequency", data=np.asarray(frequency, dtype=np.int64))
        if meta:
            group = f.create_group("meta")
            for key, value in meta.items():
                if isinstance(value, str):
                    group.create_dataset(key, data=value, dtype=_string(value))
                else:
                    group.create_dataset(key, data=np.asarray(value))

        # state 取第 t 帧,action 取第 t+1 帧(即下一帧的状态)
        state, action = f.create_group("state"), f.create_group("action")
        for name, values in sorted(_state_series(frames).items()):
            state.create_dataset(name, data=values[:-1])
            action.create_dataset(name, data=values[1:])
            if name.endswith("ee_poses"):
                prefix = name[: -len("ee_poses")]
                delta = _delta_ee_poses(values[:-1], np.asarray(delta_matrices.get(prefix, np.eye(3)), dtype=float))
                state.create_dataset(f"{prefix}delta_ee_poses", data=delta)

        # 图像只存前 T-1 帧,与 state 对齐;JPEG 编码后按最长字节串定长存储
        vision = f.create_group("vision")
        for cam in frames[0]["vision"]:
            first = frames[0]["vision"][cam]
            colors = np.asarray([fr["vision"][cam]["color"] for fr in frames[:-1]], dtype=np.uint8)
            encoded, max_len = images_encoding(colors)
            group = vision.create_group(cam)
            group.create_dataset("colors", data=encoded, dtype=f"S{max_len}")
            group.create_dataset("shape", data=np.asarray(colors[0].shape, dtype=np.int64))
            if "intrinsic_matrix" in first:
                group.create_dataset("intrinsic_matrix", data=np.asarray(first["intrinsic_matrix"], dtype=np.float64))
            if "extrinsic_matrix" in first:
                extrinsics = [np.asarray(fr["vision"][cam]["extrinsic_matrix"], dtype=np.float64) for fr in frames[:-1]]
                group.create_dataset("extrinsic_matrix", data=np.asarray(extrinsics))

    if video_dir is not None:
        stem = os.path.splitext(os.path.basename(hdf5_path))[0]
        for cam in frames[0]["vision"]:
            write_video(frames[:-1], os.path.join(video_dir, f"{stem}_{cam}.mp4"), frequency, cam)
    return len(frames) - 1


def write_video(frames, video_path, frequency, camera="cam_head"):
    """把 ``frames`` 中某一路相机的图像写成 mp4;该相机不存在时什么也不做。"""
    if not frames or camera not in frames[0]["vision"]:
        return
    first = np.asarray(frames[0]["vision"][camera]["color"])
    writer = VideoStreamWriter(video_path, first.shape[0], first.shape[1], 3, fps=frequency)
    for fr in frames:
        writer.append(np.asarray(fr["vision"][camera]["color"]))
    writer.close(announce=False)
