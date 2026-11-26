"""几个测谎目标模型在同一段自己的录像上闭环跑（lie_shadow.TargetFinder，和挂机程序一样），并排出对比视频；
不发送键盘或鼠标输入。

例如（只看淡出后 3.5~7.5 秒，放慢一倍）：
    python tools/lie_compare_video.py lie_recordings/20261026_125240 \
        --model old=models/lie_target_win60_2r/best.onnx --model prior60=models/lie_target/best.onnx \
        --start 3.5 --end 7.5 --slow 2 --output lie_detector_trials/compare_125240.mp4

每个模型一格（2 个左右并排，3~4 个排成田字格）：彩色圈是程序算的位置，黄线是形状先验模型当时用的剪影，
画面里的绿色准星是录像时的鼠标。顶上的时间从淡出那一刻算起。模型都从录像开头跑，--start/--end 只决定画哪几秒。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = load("lie_shadow", REPO / "src" / "perception" / "lie_shadow.py")
OnnxDetector = load("yolo_detector", REPO / "src" / "perception" / "yolo_detector.py").OnnxDetector

COLORS = [(255, 140, 0), (0, 0, 255), (255, 0, 255), (0, 165, 255)]
"""每个模型的圈（BGR）：蓝、红、紫、橙；绿色留给录像里的准星"""
HEAD = 40


def panel(board, name, color, pred, prior, caption):
    im = board.copy()
    if prior is not None and prior.any():
        contours, _ = cv2.findContours(prior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, contours, -1, (0, 255, 255), 1)
    if pred is not None:
        cv2.circle(im, tuple(int(v) for v in pred), 32, color, 3)
        cv2.drawMarker(im, tuple(int(v) for v in pred), color, cv2.MARKER_CROSS, 14, 2)
    head = np.zeros((HEAD, im.shape[1], 3), np.uint8)
    cv2.putText(head, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
    cv2.putText(head, caption, (380, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
    return np.vstack([head, im])


def grid(panels):
    if len(panels) <= 2:
        return np.hstack([p if k == 0 else np.hstack([np.full((p.shape[0], 8, 3), 255, np.uint8), p])
                          for k, p in enumerate(panels)])
    blank = np.zeros_like(panels[0])
    cells = panels + [blank] * (4 - len(panels))
    gap_v = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    rows = [np.hstack([cells[k], gap_v, cells[k + 1]]) for k in (0, 2)]
    return np.vstack([rows[0], np.full((8, rows[0].shape[1], 3), 255, np.uint8), rows[1]])


def stats(errors):
    """errors：[(淡出后秒数, 离录像准星多远)]。"""
    if not errors:
        return "淡出后没有可比的步"
    t, e = np.array(errors).T
    run = longest = 0
    start = None
    for k, v in enumerate(e):
        run = run + 1 if v >= 60 else 0
        if run > longest:
            longest, start = run, t[k - run + 1]
    where = f"（淡出后 {start:.1f}~{start + longest / shadow.RATE:.1f} 秒）" if longest else ""
    return f"离录像准星 60px 内 {np.mean(e < 60):.3f}，连续 60px 以上最长 {longest / shadow.RATE:.2f} 秒{where}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recording", type=Path, help="lie_recordings/<时间>（video.avi + frames.csv）")
    p.add_argument("--model", action="append", required=True, help="名字=模型.onnx（旁边要有 model_info.json），2~4 个")
    p.add_argument("--start", type=float, help="从淡出后第几秒开始画（默认从画板出现）")
    p.add_argument("--end", type=float, help="画到淡出后第几秒（默认到最后）")
    p.add_argument("--slow", type=int, default=1, help="放慢几倍")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    models = [tuple(m.split("=", 1)) for m in args.model]
    if not 1 <= len(models) <= 4:
        p.error("模型要 1~4 个")

    with open(args.recording / "frames.csv", encoding="utf-8") as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    seconds = [float(rows[i]["seconds"]) for i in sorted(rows)]
    step = max(1, round(1 / float(np.median(np.diff(seconds))) / shadow.RATE))
    finders = [shadow.TargetFinder(OnnxDetector(path, conf=.1), shadow.model_input_kind(path), random.Random(0))
               for _, path in models]
    errors = [[] for _ in models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.recording / "video.avi"))
    writer, seen, faded, i = None, False, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step or i not in rows:
            i += 1
            continue
        origin = shadow.board_box(frame)
        if origin is None:
            for finder in finders:
                finder.reset()
            i += 1
            continue
        bx, by = origin
        board = np.ascontiguousarray(frame[by:by + shadow.BOARD_SIZE[1], bx:bx + shadow.BOARD_SIZE[0]])
        r = rows[i]
        t = float(r["seconds"])
        cursor = None if not r["video_x"] else (int(r["video_x"]) - bx, int(r["video_y"]) - by)
        aim = None if cursor is None else shadow.find_crosshair(board, cursor)
        visible = shadow.white_target(board) is not None
        if seen and not visible and faded is None:
            faded = t
        seen |= visible
        rel = None if faded is None else t - faded
        phase = "shape visible" if visible else ("after fade" if seen else "before")
        caption = f"t={rel:+.2f}s {phase}" if rel is not None else phase
        cells = []
        for k, ((name, _), finder) in enumerate(zip(models, finders)):
            last = None if finder.tracker is None else finder.tracker.pos.copy()
            pred = finder.step(board, aim, t)
            prior = None if finder.shape is None else finder.shape.render(last, t, board.shape[:2])
            if rel is not None and not visible and aim is not None and pred is not None:
                errors[k].append((rel, float(np.linalg.norm(pred - aim))))
            cells.append(panel(board, name, COLORS[k], pred, prior, caption))
        in_window = ((args.start is None or (rel is not None and rel >= args.start))
                     and (args.end is None or rel is None or rel <= args.end))
        if in_window:
            image = grid(cells)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"avc1"), shadow.RATE,
                                         (image.shape[1], image.shape[0]))
            for _ in range(args.slow):
                writer.write(image)
        i += 1
    cap.release()
    if writer is None:
        raise SystemExit("选的时间段里没有画面（--start / --end 是淡出后的秒数）")
    writer.release()
    for (name, _), e in zip(models, errors):
        print(f"{name}: {stats(e)}")
    print("视频：", args.output)


if __name__ == "__main__":
    main()
