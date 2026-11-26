"""用测谎录像（LieRecorder 录的 video.avi + frames.csv）评估三种跟踪器；不发送键盘或鼠标输入。

例如：
    python tools/lie_recording_eval.py lie_recordings/20261024_090628 \
        --model models/lie_detector/best.onnx --output lie_detector_trials/rec_090628

录像是整个游戏画面（裁掉四周），金色画板只占中间一块（750×500，和 cycle_00152 一样大）：
  1. 找画板位置，把每帧裁成画板写成 board.avi，帧号和 frames.csv 一一对应；
  2. 在 board.avi 上跑混合跟踪（YOLO + 轮廓）、纯轮廓、纯 YOLO（每帧取离上次最近的框）；
  3. 真值 = 玩家鼠标：游戏把光标画成绿色准星，玩家是拿准星中心去套目标的，
     所以真值 = 光标坐标 + 准星相对光标的偏移（每段录像自己量）。
玩家跟得也有延迟和误差，所以误差几十 px 以内都算跟上；只有通过测谎的录像才能当真值。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from lie_detector_hybrid_trial import load_onnx_detector, replay
from lie_detector_video_trial import board_present, white_target
from lie_recording_labels import crosshair_offset

BOARD_SIZE = (750, 500)


def board_box(frame):
    """金色画板的 (x, y, w, h)，没有画板返回 None。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gold = np.uint8((hsv[..., 0] >= 10) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 65))
    gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(gold)
    if count < 2:
        return None
    x, y, w, h, area = stats[1 + np.argmax(stats[1:, 4])]
    if abs(w - BOARD_SIZE[0]) > 20 or abs(h - BOARD_SIZE[1]) > 20 or area < .7 * w * h:
        return None
    return int(x), int(y), int(w), int(h)


def crop_board(video, out_path):
    cap = cv2.VideoCapture(str(video))
    frames, boxes = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        boxes.append(board_box(frame))
    cap.release()
    found = np.array([b for b in boxes if b is not None])
    if not len(found):
        raise ValueError(f"{video} 里没找到金色画板")
    x, y = (int(v) for v in np.median(found[:, :2], axis=0))
    w, h = BOARD_SIZE
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), 30., (w, h))
    for frame in frames:
        writer.write(np.ascontiguousarray(frame[y:y + h, x:x + w]))
    writer.release()
    return (x, y), [i for i, b in enumerate(boxes) if b is not None]


def yolo_only(board_video, model):
    """纯 YOLO：画板上每帧取离上一次位置最近的框；白色目标可见时直接用它定位。"""
    detector = load_onnx_detector()(model, conf=.05)
    cap = cv2.VideoCapture(str(board_video))
    track, last = {}, None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if board_present(frame):
            visible = white_target(frame, last)
            if visible is not None:
                last = visible[1].copy()
            else:
                boxes = [np.array([d.x + d.w / 2, d.y + d.h / 2]) for d in detector.detect(frame)]
                if boxes and last is not None:
                    last = min(boxes, key=lambda c: np.linalg.norm(c - last))
                track[i] = last
        else:
            last = None
        i += 1
    cap.release()
    return track


def stats(errors):
    e = np.array(errors)
    if not len(e):
        return None
    return {"frames": len(e), "median_px": round(float(np.median(e)), 1),
            "p90_px": round(float(np.percentile(e, 90)), 1), "max_px": round(float(e.max()), 1),
            "within_30px": round(float(np.mean(e < 30)), 3), "within_60px": round(float(np.mean(e < 60)), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path, help="lie_recordings/<时间> 目录")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-weight", type=float, default=.0009)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    board_video = args.output / "board.avi"
    (bx, by), board_frames = crop_board(args.recording / "video.avi", board_video)
    offset, offset_samples = crosshair_offset(args.recording / "frames.csv", args.recording / "video.avi")
    replay(board_video, args.model, args.output, args.motion_weight)
    yolo_track = yolo_only(board_video, args.model)

    truth = {}
    for r in csv.DictReader(open(args.recording / "frames.csv", encoding="utf-8")):
        if r["video_x"]:
            truth[int(r["frame"])] = np.array([int(r["video_x"]) - bx, int(r["video_y"]) - by]) + offset
    hybrid_rows = list(csv.DictReader(open(args.output / "hybrid_candidates.csv", encoding="utf-8")))
    after = [int(r["frame"]) for r in hybrid_rows if r["state"] == "UNVERIFIED"]

    def errors(get):
        out = []
        for f in after:
            p = get(f)
            if p is not None and f in truth:
                out.append(float(np.linalg.norm(np.asarray(p, float) - truth[f])))
        return out

    by_frame = {int(r["frame"]): r for r in hybrid_rows}
    point = lambda r, kx, ky: (float(r[kx]), float(r[ky])) if r[kx] else None
    per_frame = []
    for f in after:
        r = by_frame[f]
        y = yolo_track.get(f)
        per_frame.append({"frame": f, "truth_x": round(truth[f][0], 1), "truth_y": round(truth[f][1], 1),
                          "hybrid_x": r["x"], "hybrid_y": r["y"], "outline_x": r["baseline_x"],
                          "outline_y": r["baseline_y"],
                          "yolo_x": None if y is None else round(float(y[0]), 1),
                          "yolo_y": None if y is None else round(float(y[1]), 1)})
    with (args.output / "vs_cursor.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_frame[0]))
        wr.writeheader()
        wr.writerows(per_frame)

    summary = {
        "recording": str(args.recording), "board_origin_in_video": [bx, by],
        "board_frames": [board_frames[0], board_frames[-1]] if board_frames else None,
        "crosshair_offset_px": [round(float(v), 1) for v in offset], "offset_samples": offset_samples,
        "after_fade_frames": [after[0], after[-1]] if after else None,
        "error_vs_cursor": {
            "hybrid": stats(errors(lambda f: point(by_frame[f], "x", "y"))),
            "outline_only": stats(errors(lambda f: point(by_frame[f], "baseline_x", "baseline_y"))),
            "yolo_only": stats(errors(lambda f: yolo_track.get(f))),
        },
    }
    (args.output / "vs_cursor.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
