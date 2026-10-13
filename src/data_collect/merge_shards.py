"""把 ``scripts/collect_data_parallel.sh`` 各 worker 的分片合并成一个数据集。

每个 worker 用各自的 seed 段采集到 ``<shard_root>/w<k>/<task>/<config>/``。本脚本把它们的 episode 复制到
``<save_dir>/<task>/<config>/``(编号接在已有数据之后),把 seed 追加到 ``seed.txt``;对
``<task>_failed/<config>/`` 中的失败尝试(``--keep_failed``)做同样处理,并汇总各分片的 ``collect_info.json``。
目标中已有的 seed 会被跳过,因此重复合并是安全的。

合并后删除分片中已经确认复制到目标的 episode 文件(hdf5、layout、预览视频;目标中同名文件存在且大小一致才删),
以及失败视频 ``video_fail/``;``seed.txt`` / ``next_seed.txt`` / ``collect_info.json`` / ``fail_info.jsonl``
和日志保留,重跑 ``collect_data_parallel.sh`` 时各 worker 仍按它们续采。``--keep_shards`` 时不删除。

``--remove_shards`` 连分片目录一起删(``<shard_root>``,含各 worker 的 ``w<k>/`` 与 ``logs/``),适合合并成功后
不再续采的场合;有 episode 缺失或与目标不一致时只提示、不删,避免丢数据。

    python src/data_collect/merge_shards.py --task_name pour_liquid_into_cup \
        --shard_root data/RoboDojo/_shards/pour_liquid_into_cup
"""

import argparse
from collections import Counter
import glob
import json
import os
import shutil


def _seeds(path):
    """读取 ``<path>/seed.txt`` 中的成功 seed 列表(没有则为空)。"""
    f = os.path.join(path, "seed.txt")
    return [int(s) for s in open(f).read().split()] if os.path.exists(f) else []


def _copy_episode(src_dir, src_name, dst_dir, dst_name):
    """复制 ``data/<name>.hdf5``、``layout/<name>.json`` 和 ``preview_video/<name>_*.mp4``;没有 hdf5 时返回 False。"""
    hdf5 = os.path.join(src_dir, "data", f"{src_name}.hdf5")
    if not os.path.exists(hdf5):
        return False
    for sub, pattern, ext in (("data", src_name, ".hdf5"), ("layout", src_name, ".json")):
        src = os.path.join(src_dir, sub, pattern + ext)
        if os.path.exists(src):
            os.makedirs(os.path.join(dst_dir, sub), exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, sub, dst_name + ext))
    for video in glob.glob(os.path.join(src_dir, "preview_video", f"{src_name}_*.mp4")):
        os.makedirs(os.path.join(dst_dir, "preview_video"), exist_ok=True)
        cam = os.path.basename(video)[len(src_name) :]
        shutil.copy2(video, os.path.join(dst_dir, "preview_video", dst_name + cam))
    return True


def _episode_files(src_dir, src_name, dst_dir, dst_name):
    """一个 episode 在分片中的文件及其在目标中的对应路径:``[(src, dst), ...]``。"""
    pairs = [
        (os.path.join(src_dir, sub, src_name + ext), os.path.join(dst_dir, sub, dst_name + ext))
        for sub, ext in (("data", ".hdf5"), ("layout", ".json"))
    ]
    for video in glob.glob(os.path.join(src_dir, "preview_video", f"{src_name}_*.mp4")):
        cam = os.path.basename(video)[len(src_name) :]
        pairs.append((video, os.path.join(dst_dir, "preview_video", dst_name + cam)))
    return [(src, dst) for src, dst in pairs if os.path.exists(src)]


def _remove_merged(src_dir, src_name, dst_dir, dst_name):
    """目标中每个文件都存在且与分片中的大小一致时,删除分片中的这些文件;返回删除的字节数,不一致时返回 None。"""
    pairs = _episode_files(src_dir, src_name, dst_dir, dst_name)
    if any(not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src) for src, dst in pairs):
        return None
    size = sum(os.path.getsize(src) for src, _ in pairs)
    for src, _ in pairs:
        os.remove(src)
    return size


def clean(shard_dirs, target, fail_shard_dirs, fail_path):
    """删除分片中已合并的 episode 文件和失败视频;返回 ``(删除的字节数, 未删除的 episode 列表)``。"""
    index = {seed: i for i, seed in enumerate(_seeds(target))}
    fail_log = os.path.join(fail_path, "fail_info.jsonl")
    fail_index = {}
    if os.path.exists(fail_log):
        fail_index = {r["seed"]: r["episode"] for r in map(json.loads, filter(str.strip, open(fail_log)))}
    freed, kept = 0, []
    for shard, fail_shard in zip(shard_dirs, fail_shard_dirs):
        for i, seed in enumerate(_seeds(shard)):
            name = f"episode_{i:07d}"
            if not os.path.exists(os.path.join(shard, "data", f"{name}.hdf5")):
                continue  # 之前已经删过
            n = _remove_merged(shard, name, target, f"episode_{index[seed]:07d}") if seed in index else None
            if n is None:
                kept.append(os.path.join(shard, "data", f"{name}.hdf5"))
            else:
                freed += n
        shard_log = os.path.join(fail_shard, "fail_info.jsonl")
        if os.path.exists(shard_log):
            for rec in map(json.loads, filter(str.strip, open(shard_log))):
                if not os.path.exists(os.path.join(fail_shard, "data", f"{rec['episode']}.hdf5")):
                    continue
                n = None
                if rec["seed"] in fail_index:
                    n = _remove_merged(fail_shard, rec["episode"], fail_path, fail_index[rec["seed"]])
                if n is None:
                    kept.append(os.path.join(fail_shard, "data", f"{rec['episode']}.hdf5"))
                else:
                    freed += n
        # 失败视频(--save_fail_video)只是调试用,不合并
        for video in glob.glob(os.path.join(shard, "video_fail", "*.mp4")):
            freed += os.path.getsize(video)
            os.remove(video)
    return freed, kept


def merge(shard_dirs, target, fail_shard_dirs, fail_path):
    """``shard_dirs`` / ``fail_shard_dirs``:每个 worker 的成功 / 失败 ``<task>[_failed]/<config>`` 目录。"""
    os.makedirs(target, exist_ok=True)
    seeds = _seeds(target)
    known = set(seeds)
    fail_log = os.path.join(fail_path, "fail_info.jsonl")
    known_fail = set()
    if os.path.exists(fail_log):
        known_fail = {json.loads(line)["seed"] for line in open(fail_log) if line.strip()}
    n_fail = len(glob.glob(os.path.join(fail_path, "data", "*.hdf5")))
    info = Counter()
    fail_types = Counter()

    for shard, fail_shard in zip(shard_dirs, fail_shard_dirs):
        # 成功 episode 按 seed.txt 中的顺序编号(对应 expert_collection.py 的 episode_{index})。
        for i, seed in enumerate(_seeds(shard)):
            if seed in known:
                continue
            if _copy_episode(shard, f"episode_{i:07d}", target, f"episode_{len(seeds):07d}"):
                seeds.append(seed)
                known.add(seed)
        # 失败尝试按 fail_info.jsonl 逐条复制,以 seed 去重
        shard_log = os.path.join(fail_shard, "fail_info.jsonl")
        if os.path.exists(shard_log):
            for line in open(shard_log):
                rec = json.loads(line)
                if rec["seed"] in known_fail:
                    continue
                name = f"episode_{n_fail:07d}"
                if _copy_episode(fail_shard, rec["episode"], fail_path, name):
                    with open(fail_log, "a") as fp:
                        fp.write(json.dumps({**rec, "episode": name}) + "\n")
                    known_fail.add(rec["seed"])
                    n_fail += 1
        # 汇总各分片最后一次运行的失败 / 尝试统计
        info_file = os.path.join(shard, "collect_info.json")
        if os.path.exists(info_file):
            shard_info = json.load(open(info_file))
            for key in ("fail_this_run", "tries_this_run"):
                info[key] += shard_info.get(key, 0)
            fail_types.update(shard_info.get("fail_types_this_run", {}))

    with open(os.path.join(target, "seed.txt"), "w") as fp:
        fp.write(" ".join(map(str, seeds)))
    summary = {
        "success": len(seeds),
        "shards": len(shard_dirs),
        "fail_last_runs": info["fail_this_run"],
        "tries_last_runs": info["tries_this_run"],
        "fail_types_last_runs": dict(fail_types),
        "failed_saved_total": n_fail,
    }
    with open(os.path.join(target, "collect_info.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def _tree_size(path):
    """``path`` 下所有文件的字节数(用于报告删掉了多少)。"""
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--shard_root", required=True, help="directory holding w0/, w1/, ...")
    parser.add_argument("--save_dir", default="data/RoboDojo")
    parser.add_argument("--config_name", default="arx_x5")
    parser.add_argument("--keep_shards", action="store_true", help="do not delete merged episode files from shards")
    parser.add_argument(
        "--remove_shards",
        action="store_true",
        help="delete the whole shard root (worker dirs and logs) once every episode is merged",
    )
    args = parser.parse_args()
    # 按 worker 编号排序,保证合并顺序稳定
    shards = sorted(
        glob.glob(os.path.join(args.shard_root, "w*", args.task_name, args.config_name)),
        key=lambda p: int(p.split(os.sep)[-3][1:]),
    )
    if not shards:
        raise SystemExit(f"no shards under {args.shard_root}/w*/{args.task_name}/{args.config_name}")
    fail_shards = [os.path.join(os.path.dirname(os.path.dirname(p)), f"{args.task_name}_failed", args.config_name)
                   for p in shards]
    target = os.path.join(args.save_dir, args.task_name, args.config_name)
    fail_path = os.path.join(args.save_dir, f"{args.task_name}_failed", args.config_name)
    print(json.dumps(merge(shards, target, fail_shards, fail_path), indent=2))
    if args.keep_shards:
        return
    freed, kept = clean(shards, target, fail_shards, fail_path)
    print(f"[merge] removed {freed / 1e9:.2f} GB of merged files from {args.shard_root}")
    if kept:
        print(f"[merge] kept {len(kept)} shard episodes that are missing or differ in the target, e.g. {kept[0]}")


if __name__ == "__main__":
    main()
