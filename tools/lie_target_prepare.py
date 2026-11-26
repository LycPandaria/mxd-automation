"""从回放标注准备按整段视频划分的数据集，并打包独立 Windows 试训包。

两种打包：
  --package：数据集在 Mac 上生成好，连同数据集一起打包（几 GB）；
  --source-package：只打代码、回放标注和起点权重（十几 MB），数据集到 Windows 上从视频生成（--videos）。

--add-own：把自己的录像（--own-manifest）加进已有的数据集，每一折写一份 splits_<折名>.json（train.py --splits 用）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from lie_recording_labels import label_recording, preview

REPO = Path(__file__).resolve().parent.parent
VIDEO_SUFFIXES = (".avi", ".mp4")
VIDEO_CACHE = "dataset_videos"
"""从 tar 包解出的视频放在数据集旁边这个文件夹，几个数据集共用，不重复解"""


def split_videos(names, seed):
    names = sorted(names)
    if len(names) < 5 or len(names) != len(set(names)):
        raise ValueError("至少需要5段名称不同的视频，才能按视频划分训练、验证和测试集")
    random.Random(seed).shuffle(names)
    n_test = max(1, round(len(names) * .1))
    n_val = max(1, round(len(names) * .2))
    return {"test": names[:n_test], "val": names[n_test:n_test + n_val],
            "train": names[n_test + n_val:]}


def find_videos(source: Path, names, cache: Path):
    """按名字（cycle_00001）找源视频：source 是文件夹就递归找 <名字>.avi；是 .tar 包就只把要用的几段
    解到 cache（文件名只用视频名，不用包里的路径）。返回 {名字: 路径}，缺了报错。"""
    names, found = set(names), {}
    if source.is_dir():
        for path in source.rglob("*"):
            if path.suffix.lower() in VIDEO_SUFFIXES and path.stem in names and path.stem not in found:
                found[path.stem] = path
    elif source.is_file() and tarfile.is_tarfile(source):
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source) as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.suffix.lower() not in VIDEO_SUFFIXES or name.stem not in names \
                        or name.stem in found:
                    continue
                target = cache / f"{name.stem}{name.suffix.lower()}"
                if not target.is_file() or target.stat().st_size != member.size:
                    print(f"从 {source.name} 解出 {name.stem}", flush=True)
                    with tar.extractfile(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                found[name.stem] = target
                if len(found) == len(names):
                    break
    else:
        raise ValueError(f"找不到视频文件夹或 tar 包：{source}")
    missing = sorted(names - set(found))
    if missing:
        raise ValueError(f"{source} 里缺 {len(missing)} 段视频：{', '.join(missing[:5])}" + (" …" if len(missing) > 5 else ""))
    return found


def prepare(manifest_path, out, every=6, seed=20261027, two_frame=False, prior=False, videos=None, edge_every=0):
    if every < 1:
        raise ValueError("every必须大于0")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_root = Path(manifest["output"])
    if not label_root.is_absolute():
        label_root = REPO / label_root
    recordings = [label_root / Path(v).stem for v in manifest["inputs"]]
    sources = {}
    for recording in recordings:
        meta_path = recording / "meta.json"
        if not meta_path.is_file() or not (recording / "frames.csv").is_file():
            raise ValueError(f"标注未保存完整：{recording}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source") != "replay_label":
            raise ValueError(f"不是回放标注：{recording}")
        if videos is None:
            if not Path(meta["video"]).is_file():
                raise ValueError(f"缺少源视频：{meta['video']}（视频在别处时用 --videos 指定文件夹或 tar 包）")
            sources[recording.name] = Path(meta["video"])
    if videos is not None:
        sources = find_videos(videos, [r.name for r in recordings], out.parent / VIDEO_CACHE)
    groups = split_videos([r.name for r in recordings], seed)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"输出目录已有文件，请使用新目录：{out}")
    for folder in ("images", "labels"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "sources.csv").write_text("stem,source\n", encoding="utf-8")
    (out / "split_plan.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    reports = []
    rng = random.Random(seed)
    for n, recording in enumerate(recordings, 1):
        report = label_recording(recording, out, every, rng, two_frame, prior, sources[recording.name], edge_every)
        if report["images"] == 0:
            raise ValueError(f"没有生成训练图：{recording}")
        reports.append(report)
        print(f"[{n}/{len(recordings)}] {recording.name}: {report['images']}张，"
              f"模拟准星处理{report['synthetic_crosshair_frames']}张", flush=True)
    image_groups = {}
    images = sorted(p for p in (out / "images").iterdir() if p.suffix in (".jpg", ".png"))
    for split, names in groups.items():
        image_groups[split] = [p.relative_to(out).as_posix() for p in images
                               if p.stem.rsplit("_", 1)[0] in names]
        if not image_groups[split]:
            raise ValueError(f"{split}集没有图片")
    split_info = {"seed": seed, "videos": groups, "images": image_groups}
    (out / "splits.json").write_text(json.dumps(split_info, indent=2) + "\n", encoding="utf-8")
    kind = "two_frame_prior" if prior else "two_frame" if two_frame else "single"
    summary = {"kind": "pilot", "input_kind": kind,
               "recordings": reports, "every": every, "edge_every": edge_every,
               "split_by": "whole_video", "seed": seed,
               "counts": {k: len(v) for k, v in image_groups.items()},
               "videos": None if videos is None else str(Path(videos).resolve()),
               "preprocessing_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                                        for name in ("tools/lie_recording_labels.py", "src/perception/lie_shadow.py")},
               "notes": ["淡出后的标签来自人工回放轨迹，尚无独立逐帧真值。",
                         f"{len(recordings)}段回放标注，不能代表全部531段的泛化效果。"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview(out, out / "preview.jpg")
    return summary


OWN_IMAGES_PER_SECOND = 30
"""自己的录像太少（买来的每秒取 10 张），每秒多取一些，贴边的帧才占得上分量"""


def own_every(frames_csv: Path):
    with open(frames_csv, encoding="utf-8") as f:
        seconds = [float(r["seconds"]) for r in csv.DictReader(f)]
    return max(1, round(1 / float(np.median(np.diff(seconds))) / OWN_IMAGES_PER_SECOND))


def add_own(manifest_path: Path, dataset: Path, recordings: Path, seed=20261031):
    """把自己的录像加进已有的形状先验数据集（dataset/splits.json 不动）。每一折：留出的那段进 val（挑最好的一轮时看得到
    贴边，只有买来的视频做 val 时分数一开始就到顶，最好的那轮还是原模型），其余进 train；test 还是买来的留出视频。
    录像放在 recordings/<名字>/video.avi；答案默认用录像自己的 frames.csv，manifest 里给了回放标注目录的用回放标注。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    if summary.get("input_kind") != "two_frame_prior":
        raise ValueError(f"{dataset} 不是形状先验数据集（input_kind = {summary.get('input_kind')}）")
    missing = [n for n in manifest["recordings"] if not (recordings / n / "video.avi").is_file()]
    if missing:
        raise ValueError(f"{recordings} 下缺录像：{', '.join(missing)}")
    rng = random.Random(seed)
    reports, images = [], {}
    for n, (name, replay) in enumerate(manifest["recordings"].items(), 1):
        answer = recordings / name if replay is None else manifest_path.parent / replay
        report = label_recording(answer, dataset, own_every(answer / "frames.csv"), rng, prior=True,
                                 video=recordings / name / "video.avi", own=True)
        images[name] = sorted(f"images/{p.name}" for p in (dataset / "images").glob(f"{name}_*.png"))
        reports.append(report)
        print(f"[{n}/{len(manifest['recordings'])}] {name}: {report['images']} 张，补准星 {report['crosshair_cleaned']}，"
              f"准星离白色图形（淡出前 1 秒）{report['aim_error_while_visible_px']}", flush=True)
    base = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    counts = {}
    for fold, holdout in manifest["folds"].items():
        if holdout not in images:
            raise ValueError(f"{fold} 留出的 {holdout} 不在 recordings 里")
        train = [n for n in images if n != holdout]
        splits = {"seed": base["seed"], "videos": {**base["videos"], "train": base["videos"]["train"] + train,
                                                   "val": base["videos"]["val"] + [holdout]},
                  "images": {**base["images"], "train": base["images"]["train"] + [p for n in train for p in images[n]],
                             "val": base["images"]["val"] + images[holdout]}}
        (dataset / f"splits_{fold}.json").write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        counts[fold] = {k: len(v) for k, v in splits["images"].items()}
    (dataset / "own_summary.json").write_text(json.dumps(
        {"manifest": manifest, "recordings": reports, "counts": counts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    preview(dataset, dataset / "own_preview.jpg", prefixes=set(images))
    return counts


def package(dataset, destination, weights=None):
    if destination.exists():
        raise ValueError(f"训练包已存在：{destination}")
    if not (dataset / "splits.json").is_file():
        raise ValueError("数据集尚未生成完成")
    prefix = "mxd-lie-target-pilot"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                z.write(path, f"{prefix}/dataset/{path.relative_to(dataset).as_posix()}")
        for path in sorted((REPO / "train/windows_lie_target").iterdir()):
            if path.is_file():
                z.write(path, f"{prefix}/{path.name}")
        for name in ("best.pt", "checkpoint_info.json"):
            path = (weights or REPO / "models/lie_target") / name
            if path.is_file():
                z.write(path, f"{prefix}/models/lie_target/{name}")
        quality = REPO / "replay_labels/pilot_10_quality.json"
        if quality.is_file():
            z.write(quality, f"{prefix}/label_quality.json")


SOURCE_FILES = ("tools/lie_target_prepare.py", "tools/lie_recording_labels.py", "tools/lie_closed_loop_eval.py",
                "tools/lie_compare_video.py", "docs/LIE_TARGET.md",
                "src/perception/lie_shadow.py", "src/perception/yolo_detector.py")
"""Windows 上生成数据集、训练、闭环评估要用的代码（保持仓库里的相对位置，工具按 ../src/perception 找 lie_shadow）"""


def package_source(manifest_path, destination, weights=None, compare=(), own_manifest=None):
    """不带数据集的训练包：代码 + 回放标注（frames.csv / meta.json）+ 起点权重，数据集到 Windows 上从视频生成。
    compare：一起带上的回退 / 对比模型目录（best.pt、best.onnx 和两个说明文件，能接着训也能评估）。
    own_manifest：自己录像的清单和它用到的回放标注（录像本身已经在 Windows 上）。"""
    if destination.exists():
        raise ValueError(f"训练包已存在：{destination}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_root = Path(manifest["output"])
    if not label_root.is_absolute():
        label_root = REPO / label_root
    weights = weights or REPO / "models/lie_target"
    if not (weights / "best.pt").is_file():
        raise ValueError(f"缺少起点权重：{weights / 'best.pt'}")
    prefix = "mxd-lie-target"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted((REPO / "train/windows_lie_target").iterdir()):
            if path.is_file() and path.suffix != ".pyc":
                z.write(path, f"{prefix}/{path.name}")
        for name in SOURCE_FILES:
            z.write(REPO / name, f"{prefix}/{name}")
        z.write(manifest_path, f"{prefix}/replay_labels/{manifest_path.name}")
        for video in manifest["inputs"]:
            for name in ("frames.csv", "meta.json"):
                z.write(label_root / Path(video).stem / name, f"{prefix}/replay_labels/{Path(video).stem}/{name}")
        if own_manifest is not None:
            z.write(own_manifest, f"{prefix}/replay_labels/{own_manifest.name}")
            for replay in json.loads(own_manifest.read_text(encoding="utf-8"))["recordings"].values():
                for name in () if replay is None else ("frames.csv", "meta.json"):
                    z.write(own_manifest.parent / replay / name, f"{prefix}/replay_labels/{replay}/{name}")
        for name in ("best.pt", "best.onnx", "checkpoint_info.json", "model_info.json"):
            if (weights / name).is_file():
                z.write(weights / name, f"{prefix}/models/lie_target/{name}")
        for folder in compare:
            for name in ("best.pt", "best.onnx", "checkpoint_info.json", "model_info.json"):
                if (folder / name).is_file():
                    z.write(folder / name, f"{prefix}/models/{folder.name}/{name}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, help="买来视频的回放标注清单（replay_labels/batch_60.json）")
    p.add_argument("--own-manifest", type=Path, help="自己录像的清单（replay_labels/own_recordings.json）")
    p.add_argument("--add-own", type=Path, help="把自己的录像加进这个已有的数据集，录像目录用 --recordings")
    p.add_argument("--recordings", type=Path, help="放 <时间>/video.avi 的文件夹")
    p.add_argument("--output", type=Path, help="数据集输出目录（新目录）")
    p.add_argument("--every", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--package", type=Path, help="连同数据集一起打成 Windows 训练包")
    p.add_argument("--source-package", type=Path, help="只打代码、回放标注和起点权重，数据集到 Windows 上生成")
    p.add_argument("--weights", type=Path, help="打进包里当训练起点的权重目录（含 best.pt），默认 models/lie_target")
    p.add_argument("--compare", type=Path, nargs="*", default=[], help="--source-package 时一起带上的对比模型目录")
    p.add_argument("--two-frame", action="store_true", help="两帧对比模型的输入（lie_shadow.motion_input）")
    p.add_argument("--prior", action="store_true", help="两帧对比 + 形状先验（4 通道 PNG，lie_shadow.with_prior）")
    p.add_argument("--videos", type=Path, help="源视频所在文件夹或 .tar 包（不用标注里记的路径）")
    p.add_argument("--videos-from", type=Path, help="没给 --videos 时，用这个已生成数据集 summary.json 里记的视频位置")
    p.add_argument("--edge-every", type=int, default=0, help="答案贴着画板边的帧每这么多帧取一张（0 = 不多取）")
    args = p.parse_args()
    if args.add_own:
        if args.own_manifest is None or args.recordings is None:
            p.error("--add-own 需要 --own-manifest 和 --recordings")
        print(json.dumps(add_own(args.own_manifest, args.add_own, args.recordings), ensure_ascii=False))
        return
    if args.manifest is None:
        p.error("需要 --manifest")
    if args.source_package:
        package_source(args.manifest, args.source_package, args.weights, args.compare, args.own_manifest)
        print("训练包：", args.source_package)
        return
    if args.output is None:
        p.error("生成数据集需要 --output")
    if args.videos is None and args.videos_from is not None:
        recorded = json.loads((args.videos_from / "summary.json").read_text(encoding="utf-8")).get("videos")
        if not recorded:
            p.error(f"{args.videos_from} 的 summary.json 里没有记视频位置，请用 --videos")
        args.videos = Path(recorded)
    summary = prepare(args.manifest, args.output, args.every, args.seed, args.two_frame, args.prior, args.videos,
                      args.edge_every)
    if args.package:
        package(args.output, args.package, args.weights)
    print(json.dumps(summary["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
