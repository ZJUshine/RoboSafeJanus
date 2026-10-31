"""把一个 LeRobot v3.0 导出派生成 v2.1,省掉从 HDF5 重新解码、逐帧写 PNG 再编码的过程。

    python src/data_collect/lerobot_v30_to_v21.py <v30 任务目录> <v21 任务目录> [--workers N]

输出布局与 ``XPolicyLab/scripts/transform_lerobot_v21_format.py`` 直接从 HDF5 导出的一致:
``info.json`` / ``tasks.jsonl`` / ``episodes.jsonl`` 相同,parquet 数据与 schema 元数据相同,
``episodes_stats.jsonl`` 按 LeRobot 0.3.3 的方式计算(图像统计用采样帧,不带分位数)。

视频按整数帧号从 v3.0 的分块 mp4 中切出每个 episode,再用 0.3.3 的编码参数
(libsvtav1, CRF 30, g=2, yuv420p)重编码,体积与直接导出持平。v3.0 是 CRF 18,
重编码后相对 HDF5 原图的 PSNR 比直接导出低约 0.5 dB。

必须在 ``xpolicy_lerobot_v21`` 环境中运行:parquet 的 ``huggingface`` schema 元数据要按
该环境的 datasets 版本写,0.3.3 才能读;v3.0 环境写出的 ``List`` 类型它不认识。
目标目录已存在时会被替换。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import sys

import av
from datasets import Features
from lerobot.datasets.compute_stats import (
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.utils import (
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    write_episode,
    write_episode_stats,
    write_info,
    write_task,
)
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

# 与 transform_lerobot_v21_format.py 的 libsvtav1 配置一致(LeRobot 0.3.3 encode_video_frames)
VIDEO_CODEC = "libsvtav1"
VIDEO_OPTIONS = {"g": "2", "crf": "30"}
VIDEO_PIX_FMT = "yuv420p"


def load_episode_records(src: Path) -> list[dict]:
    paths = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{src}/meta/episodes 下没有 parquet")
    records = [row for path in paths for row in pq.read_table(path).to_pylist()]
    return sorted(records, key=lambda row: int(row["episode_index"]))


def convert_info(info: dict, num_episodes: int, video_keys: list[str]) -> dict:
    info = dict(info)
    info["codebase_version"] = "v2.1"
    info["data_path"] = DEFAULT_PARQUET_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if video_keys else None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    for feature in info["features"].values():
        if feature.get("dtype") != "video":
            feature.pop("fps", None)
    chunks_size = info["chunks_size"]
    counts = {
        "total_videos": num_episodes * len(video_keys),
        "total_chunks": (num_episodes + chunks_size - 1) // chunks_size if num_episodes else 0,
    }
    # 键顺序与 0.3.3 写出的 info.json 一致:两个计数紧跟在 total_tasks 后面
    ordered = {}
    for key, value in info.items():
        if key not in counts:
            ordered[key] = value
        if key == "total_tasks":
            ordered.update(counts)
    ordered.update(counts)
    return ordered


def frame_index(frame: av.VideoFrame, fps: int) -> int:
    return round(float(frame.pts * frame.time_base) * fps)


def split_episode_video(src: str, dst: str, start: int, length: int, fps: int) -> dict:
    """把 src 中 [start, start + length) 帧重编码到 dst,返回 0.3.3 口径的图像统计。"""
    sampled = set(sample_indices(length))
    images = []
    written = 0
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with av.open(src) as source, av.open(dst, "w") as output:
        in_stream = source.streams.video[0]
        source.seek(int(start / fps / in_stream.time_base), stream=in_stream, backward=True)
        out_stream = output.add_stream(VIDEO_CODEC, fps, options=VIDEO_OPTIONS)
        out_stream.pix_fmt = VIDEO_PIX_FMT
        out_stream.width = in_stream.codec_context.width
        out_stream.height = in_stream.codec_context.height
        for frame in source.decode(in_stream):
            index = frame_index(frame, fps)
            if index < start:
                continue
            if index >= start + length:
                break
            if index != start + written:
                raise ValueError(f"{src}: 期望第 {start + written} 帧,解码得到第 {index} 帧")
            if written in sampled:
                rgb = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                images.append(auto_downsample_height_width(rgb))
            if frame.format.name != VIDEO_PIX_FMT:
                frame = frame.reformat(format=VIDEO_PIX_FMT)
            # 源视频 g=2 时一半是关键帧,清掉帧类型,让编码器按自己的 GOP 放关键帧
            frame.pict_type = av.video.frame.PictureType.NONE
            frame.pts = written
            frame.time_base = Fraction(1, fps)
            for packet in out_stream.encode(frame):
                output.mux(packet)
            written += 1
        for packet in out_stream.encode():
            output.mux(packet)
    if written != length:
        raise ValueError(f"{src}: 从第 {start} 帧起应有 {length} 帧,只解码到 {written} 帧")
    stats = get_feature_stats(np.stack(images), axis=(0, 2, 3), keepdims=True)
    return {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in stats.items()}


def numeric_episode_stats(table, features: dict, fps: int) -> dict:
    stats = {}
    for key, feature in features.items():
        if feature["dtype"] in ("video", "image", "string"):
            continue
        if key == "timestamp":
            # 0.3.3 的缓冲区里时间戳是 frame_index / fps 的 float64,不是 parquet 里的 float32
            data = table.column("frame_index").to_numpy() / fps
        else:
            column = table.column(key).to_pylist()
            data = np.asarray(column, dtype=np.dtype(feature["dtype"]))
        stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return stats


def convert(src: Path, dst: Path, workers: int) -> None:
    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{src} 的 codebase_version 是 {info.get('codebase_version')},不是 v3.0")
    fps = int(info["fps"])
    chunks_size = int(info["chunks_size"])
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    records = load_episode_records(src)

    tmp = dst.parent / f".{dst.name}.v21_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    write_info(convert_info(info, len(records), video_keys), tmp)
    tasks = pd.read_parquet(src / "meta" / "tasks.parquet").sort_values("task_index")
    for task, row in tasks.iterrows():
        write_task(int(row["task_index"]), task, tmp)

    # 数据:按 episode_index 从 v3.0 的分块 parquet 中切出,schema 元数据按本环境的 datasets 重写
    numeric_stats = {}
    tables = {}
    for record in records:
        episode = int(record["episode_index"])
        data_key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        if data_key not in tables:
            tables.clear()
            tables[data_key] = pq.read_table(
                src / info["data_path"].format(chunk_index=data_key[0], file_index=data_key[1])
            )
        table = tables[data_key]
        table = table.filter(pc.equal(table.column("episode_index"), episode))
        if table.num_rows != int(record["length"]):
            raise ValueError(f"episode {episode}: parquet 有 {table.num_rows} 行,元数据记录 {record['length']} 帧")
        hf_features = Features.from_arrow_schema(table.schema.remove_metadata())
        metadata = {b"huggingface": json.dumps({"info": {"features": hf_features.to_dict()}}).encode()}
        path = tmp / DEFAULT_PARQUET_PATH.format(episode_chunk=episode // chunks_size, episode_index=episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(metadata), path)
        numeric_stats[episode] = numeric_episode_stats(table, features, fps)

    # 视频:每个 (episode, camera) 一个任务,按整数帧号切片后重编码
    jobs = {}
    image_stats: dict[int, dict] = {int(r["episode_index"]): {} for r in records}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in records:
            episode = int(record["episode_index"])
            for key in video_keys:
                source = src / info["video_path"].format(
                    video_key=key,
                    chunk_index=int(record[f"videos/{key}/chunk_index"]),
                    file_index=int(record[f"videos/{key}/file_index"]),
                )
                start = round(float(record[f"videos/{key}/from_timestamp"]) * fps)
                target = tmp / DEFAULT_VIDEO_PATH.format(
                    episode_chunk=episode // chunks_size, video_key=key, episode_index=episode
                )
                future = pool.submit(
                    split_episode_video, str(source), str(target), start, int(record["length"]), fps
                )
                jobs[future] = (episode, key)
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="[v30_to_v21] videos"):
            episode, key = jobs[future]
            image_stats[episode][key] = future.result()

    for record in records:
        episode = int(record["episode_index"])
        write_episode(
            {"episode_index": episode, "tasks": list(record["tasks"]), "length": int(record["length"])},
            tmp,
        )
        merged = {**numeric_stats[episode], **image_stats[episode]}
        write_episode_stats(episode, {key: merged[key] for key in features if key in merged}, tmp)

    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"[v30_to_v21] {src} -> {dst}: {len(records)} episodes, {len(jobs)} videos")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("src", type=Path, help="v3.0 任务目录(含 meta/info.json)")
    parser.add_argument("dst", type=Path, help="v2.1 输出任务目录,已存在会被替换")
    parser.add_argument(
        "--workers", type=int, default=min(16, os.cpu_count() or 1),
        help="并行编码的 (episode, camera) 数",
    )
    args = parser.parse_args()
    # SVT-AV1 每开一个编码器就打印一屏 info,只保留 warning 及以上(子进程继承)
    os.environ.setdefault("SVT_LOG", "2")
    convert(args.src.resolve(), args.dst.resolve(), args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
