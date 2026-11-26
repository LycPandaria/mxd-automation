"""测谎目标模型闭环评估：按挂机程序的办法一步步跑（lie_shadow.TargetFinder，每秒 15 步），几个模型 A/B 对比；
不发送键盘或鼠标输入。

例如（Windows 训练包里，05_eval.bat 就是这一行）：
    python tools/lie_closed_loop_eval.py --model old=models/lie_target/best.onnx --results results \
        --own D:/mxd-automation/lie_recordings/20261024_090628 --output eval.json

形状先验模型的先验来自它自己上一步算的位置，只能闭环评估；数据集里的图先验来自答案，拿来评估会泄题。
三类视频：
  - 买来的留出视频（数据集 splits.json 的 test，答案是回放标注）：准星画在模型上一步算的位置，
    像程序拿着鼠标在跟（准星总落后目标一步）；
  - 自己的录像（--own，真准星，答案是玩家准星：玩家通过了测谎，但准星只要在图形上就算，离中心几十 px 也正常）。
    准星正套在答案上，学会「找准星」的模型这里分数反而最高，不能单看；
  - 自己的录像，程序拿鼠标（同样的 --own）：真准星先补掉（clean_crosshairs），准星画在模型上一步算的位置，答案还是玩家准星。
    这才是接管时的样子；「自己的录像」分高、这里分低，就是模型在找准星。也不算「成功！」字出来之后的时间。
指标只算淡出以后：
  within_30 / within_60  程序位置离答案 30 / 60px 以内的步数比例
  longest_lost_s         离答案 60px 以上连续最长多少秒
  no_box                 模型一个框都没出的步数比例
  nothing_near           模型出的框没有一个在答案 80px 以内（漏检，或者只框到了别处）
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = load("lie_shadow", REPO / "src" / "perception" / "lie_shadow.py")
OnnxDetector = load("yolo_detector", REPO / "src" / "perception" / "yolo_detector.py").OnnxDetector
from lie_recording_labels import RESULT_CUT_S, add_training_crosshair, clean_crosshairs  # noqa: E402
from lie_target_prepare import VIDEO_CACHE, find_videos  # noqa: E402

NEAR_PX = 80


def newest_runs(results: Path, tags=None):
    """results 下每种训练（model_info.json 的 tag；没有 tag 的按形状先验 on / off）最新的一次，返回 [(名字, best.onnx)]。
    tags：只要这几种。"""
    runs = {}
    for info in sorted(results.glob("*/model_info.json")):
        meta = json.loads(info.read_text(encoding="utf-8"))
        key = meta.get("tag") or f"prior_{meta.get('prior') or 'none'}"
        if (info.parent / "best.onnx").is_file() and (not tags or key in tags):
            runs[key] = (info.parent.name, info.parent / "best.onnx")
    return [(f"{key}({name})", path) for key, (name, path) in sorted(runs.items())]


def stats(errors, boxes):
    e = np.array(errors)
    if not len(e):
        return None
    run = longest = 0
    for v in e:
        run = run + 1 if v >= 60 else 0
        longest = max(longest, run)
    return {"steps": len(e), "within_30": round(float(np.mean(e < 30)), 3),
            "within_60": round(float(np.mean(e < 60)), 3), "median_px": round(float(np.median(e)), 1),
            "longest_lost_s": round(longest / shadow.RATE, 2),
            "no_box": round(float(np.mean([not has_box for has_box, _ in boxes])), 3),
            "nothing_near": round(float(np.mean([far for _, far in boxes])), 3)}


def frame_rows(frames_csv: Path):
    with open(frames_csv, encoding="utf-8") as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    seconds = [float(rows[i]["seconds"]) for i in sorted(rows)]
    fps = 1 / float(np.median(np.diff(seconds))) if len(seconds) > 1 else 30.
    return rows, max(1, round(fps / shadow.RATE))


SHAPE_NAME = "形状匹配（接管实际用的）"


def make_finder(model, conf, shape):
    """shape：接管实际用的 ShapeFinder（形状匹配，model 只在没记到图形时才用）；否则只看这个模型。"""
    finder = shadow.TargetFinder(OnnxDetector(str(model), conf=conf), shadow.model_input_kind(str(model)),
                                 random.Random(0))
    return shadow.ShapeFinder(finder, random.Random(0)) if shape else finder


def run_bought(model, video: Path, labels: Path, conf, shape=False):
    """买来的视频：准星画在模型上一步算的位置（程序拿着鼠标），答案是回放标注。"""
    rows, step = frame_rows(labels / "frames.csv")
    truth = {i: np.array([float(r["video_x"]), float(r["video_y"])]) for i, r in rows.items()
             if r["video_x"] and r.get("skip") != "1"}
    finder = make_finder(model, conf, shape)
    cap = cv2.VideoCapture(str(video))
    pred, seen, errors, boxes, i = None, False, [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0 and i in rows:
            board = np.ascontiguousarray(frame[:shadow.BOARD_SIZE[1], :shadow.BOARD_SIZE[0]])
            visible = shadow.white_target(board) is not None
            seen |= visible
            spot = None if pred is None else pred.copy()
            drawn = board if spot is None else add_training_crosshair(board, spot)
            pred = finder.step(drawn, None if spot is None else shadow.find_crosshair(drawn, spot),
                               float(rows[i]["seconds"]))
            t = truth.get(i)
            if seen and not visible and t is not None and pred is not None:
                errors.append(float(np.linalg.norm(pred - t)))
                boxes.append((bool(finder.boxes), all(np.linalg.norm(b[1] - t) > NEAR_PX for b in finder.boxes)))
        i += 1
    cap.release()
    return stats(errors, boxes)


def run_own(model, recording: Path, conf, shape=False):
    """自己的录像：真准星（玩家拿着鼠标），答案是玩家准星。"""
    rows, step = frame_rows(recording / "frames.csv")
    finder = make_finder(model, conf, shape)
    cap = cv2.VideoCapture(str(recording / "video.avi"))
    seen, errors, boxes, i = False, [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0 and i in rows:
            origin = shadow.board_box(frame)
            if origin is None:
                finder.reset()
                seen = False
            else:
                bx, by = origin
                board = np.ascontiguousarray(frame[by:by + shadow.BOARD_SIZE[1], bx:bx + shadow.BOARD_SIZE[0]])
                r = rows[i]
                cursor = None if not r["video_x"] else (int(r["video_x"]) - bx, int(r["video_y"]) - by)
                aim = None if cursor is None else shadow.find_crosshair(board, cursor)
                visible = shadow.white_target(board) is not None
                seen |= visible
                pred = finder.step(board, aim, float(r["seconds"]))
                if seen and not visible and aim is not None and pred is not None:
                    errors.append(float(np.linalg.norm(pred - aim)))
                    boxes.append((bool(finder.boxes),
                                  all(np.linalg.norm(b[1] - aim) > NEAR_PX for b in finder.boxes)))
        i += 1
    cap.release()
    return stats(errors, boxes)


def clean_recording(recording: Path):
    """读整段录像，记下每帧真准星的位置，再把准星补掉。返回 run_own_moved 要的东西（几个模型共用，只补一遍）。"""
    rows, step = frame_rows(recording / "frames.csv")
    cap = cv2.VideoCapture(str(recording / "video.avi"))
    raw = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw.append((frame, shadow.board_box(frame)))
    cap.release()
    bx, by = (int(v) for v in np.median([b for _, b in raw if b is not None], axis=0))
    boards, aims = [], {}
    for i, (frame, box) in enumerate(raw):
        board = None if box is None else frame[by:by + shadow.BOARD_SIZE[1], bx:bx + shadow.BOARD_SIZE[0]].copy()
        boards.append(board)
        r = rows.get(i)
        if board is not None and r and r["video_x"]:
            aims[i] = shadow.find_crosshair(board, (int(r["video_x"]) - bx, int(r["video_y"]) - by))
    del raw
    fps = shadow.RATE * step
    clean_crosshairs(boards, fps)
    last = max(i for i, b in enumerate(boards) if b is not None) - round(RESULT_CUT_S * fps)
    return boards, aims, rows, step, last


def run_own_moved(model, cleaned, conf, shape=False):
    """自己的录像，程序拿鼠标：准星画在模型上一步算的位置，答案是玩家当时的准星。"""
    boards, aims, rows, step, last = cleaned
    finder = make_finder(model, conf, shape)
    pred, seen, errors, box_stats = None, False, [], []
    for i in range(0, last + 1, step):
        if i not in rows:
            continue
        board = boards[i]
        if board is None:
            finder.reset()
            pred, seen = None, False
            continue
        visible = shadow.white_target(board) is not None
        seen |= visible
        spot = None if pred is None else pred.copy()
        drawn = board if spot is None else add_training_crosshair(board, spot)
        pred = finder.step(drawn, None if spot is None else shadow.find_crosshair(drawn, spot), float(rows[i]["seconds"]))
        aim = aims.get(i)
        if seen and not visible and aim is not None and pred is not None:
            errors.append(float(np.linalg.norm(pred - aim)))
            box_stats.append((bool(finder.boxes), all(np.linalg.norm(b[1] - aim) > NEAR_PX for b in finder.boxes)))
    return stats(errors, box_stats)


def overall(per_video):
    """每段视频的指标取平均（每段权重一样），步数相加。"""
    parts = [v for v in per_video.values() if v]
    if not parts:
        return None
    out = {k: round(float(np.mean([p[k] for p in parts])), 3) for k in parts[0] if k != "steps"}
    out["steps"] = sum(p["steps"] for p in parts)
    out["longest_lost_s"] = max(p["longest_lost_s"] for p in parts)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", action="append", default=[], help="名字=模型.onnx（旁边要有 model_info.json），可多次")
    p.add_argument("--results", type=Path, help="训练结果目录：每种训练（tag）自动加上最新的一次")
    p.add_argument("--tags", nargs="*", help="--results 里只要这几种训练（model_info.json 的 tag）")
    p.add_argument("--dataset", type=Path, default=REPO / "dataset", help="取 splits.json 的测试视频和生成时的视频位置")
    p.add_argument("--videos", type=Path, help="源视频文件夹或 tar 包（默认用数据集 summary.json 里记的）")
    p.add_argument("--labels", type=Path, default=REPO / "replay_labels")
    p.add_argument("--own", type=Path, nargs="*", default=[], help="自己的录像目录（video.avi + frames.csv），缺的跳过")
    p.add_argument("--conf", type=float, default=.1, help="和挂机程序一样（lie_shadow_conf 默认 0.1）")
    p.add_argument("--shape", action="store_true", help="多比一行形状匹配（接管实际用的），第一个模型当它的后备")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    models = [tuple(m.split("=", 1)) for m in args.model]
    if args.results:
        models += newest_runs(args.results, args.tags)
    if not models:
        p.error("至少要一个 --model 或 --results")
    if args.shape:
        models.append((SHAPE_NAME, models[0][1]))
    for name, path in models:
        print(f"模型 {name}: {path}（{shadow.model_input_kind(str(path))}）", flush=True)

    bought = {}
    if (args.dataset / "splits.json").is_file():
        names = json.loads((args.dataset / "splits.json").read_text(encoding="utf-8"))["videos"]["test"]
        source = args.videos
        if source is None and (args.dataset / "summary.json").is_file():
            recorded = json.loads((args.dataset / "summary.json").read_text(encoding="utf-8")).get("videos")
            source = Path(recorded) if recorded else None
        if source is not None:
            bought = find_videos(source, names, args.dataset.parent / VIDEO_CACHE)
        else:
            print("没有视频位置（--videos），跳过买来的留出视频", flush=True)
    own = [r for r in args.own if (r / "video.avi").is_file() and (r / "frames.csv").is_file()]
    for r in args.own:
        if r not in own:
            print(f"跳过（没有 video.avi / frames.csv）：{r}", flush=True)

    result = {"conf": args.conf, "models": {n: str(m) for n, m in models}, "bought": {}, "own": {}, "own_moved": {}}
    cleaned = {}
    for recording in own:
        print(f"补准星：{recording.name}", flush=True)
        cleaned[recording.name] = clean_recording(recording)
    for name, model in models:
        use_shape = name == SHAPE_NAME
        for video_name, video in sorted(bought.items()):
            s = run_bought(model, video, args.labels / video_name, args.conf, use_shape)
            result["bought"].setdefault(video_name, {})[name] = s
            print(f"[买来] {video_name} {name}: {s}", flush=True)
        for recording in own:
            s = run_own(model, recording, args.conf, use_shape)
            result["own"].setdefault(recording.name, {})[name] = s
            print(f"[自己] {recording.name} {name}: {s}", flush=True)
            s = run_own_moved(model, cleaned[recording.name], args.conf, use_shape)
            result["own_moved"].setdefault(recording.name, {})[name] = s
            print(f"[自己，程序拿鼠标] {recording.name} {name}: {s}", flush=True)
    result["overall"] = {group: {name: overall({v: m.get(name) for v, m in result[group].items()})
                                 for name, _ in models} for group in ("bought", "own", "own_moved") if result[group]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    keys = ("within_30", "within_60", "longest_lost_s", "no_box", "nothing_near")
    for group, label in (("bought", "买来的留出视频（准星跟着程序走）"), ("own", "自己的录像（真准星）"),
                         ("own_moved", "自己的录像（准星跟着程序走，接管时的样子）")):
        if group not in result["overall"]:
            continue
        print(f"\n== {label}，{len(result[group])} 段，每段平均 ==")
        print(f"{'模型':<40}" + "".join(f"{k:>16}" for k in keys))
        for name, _ in models:
            s = result["overall"][group][name]
            print(f"{name:<40}" + "".join(f"{'-' if s is None else s[k]:>16}" for k in keys))
    print("\n明细：", args.output)


if __name__ == "__main__":
    main()
