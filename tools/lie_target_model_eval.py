"""在没参与训练的那段录像上评估「只框真目标」的模型；不发送键盘或鼠标输入。

例如：
    python tools/lie_target_model_eval.py lie_detector_trials/target_dataset 20261024_090628 \
        --model lie_detector_trials/target_runs/trial1/weights/best.onnx \
        --compare models/lie_detector/best.onnx --output lie_detector_trials/target_eval_trial1

图和答案都来自 lie_recording_labels.py 生成的数据集（准星已经抹掉，答案框在 labels/ 里，
sources.csv 记着每张图的框从哪来：visible 白纸看得见 / pale 半透明 / cursor 完全淡出、用玩家准星）。
两种用法都统计：
  - 单帧：只看这一帧置信度最高的框，离答案多远（模型自己能不能认出真目标）；
  - 最近框跟踪：从第一帧的答案出发，每帧取离上一次位置最近的框，没框或跳太远就原地等；
  - 运动预测跟踪（PredictTracker）：没框时按速度往前推，只信离预测位置近的框。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from lie_detector_hybrid_trial import load_onnx_detector

MAX_JUMP = 120
"""最近框跟踪：一帧（数据集里隔 2 帧录像）最多跳多远，再远就当没看到、原地等"""

# 运动预测跟踪用挂机程序里影子跟踪的同一份代码（src/perception/lie_shadow.py），离线评估和实际运行不会对不上。
# 按文件路径加载，绕开 src.perception/__init__ 里只在 Windows 上有的模块。
_spec = importlib.util.spec_from_file_location(
    "lie_shadow", Path(__file__).resolve().parent.parent / "src" / "perception" / "lie_shadow.py")
_shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shadow)
PredictTracker = _shadow.PredictTracker


def load_frames(dataset: Path, recording: str):
    with open(dataset / "sources.csv", encoding="utf-8") as f:
        sources = {r["stem"]: r["source"] for r in csv.DictReader(f)}
    frames = []
    for stem in sorted(s for s in sources if s.startswith(recording + "_")):
        image = cv2.imread(str(dataset / "images" / f"{stem}.jpg"))
        h, w = image.shape[:2]
        _, cx, cy, bw, bh = map(float, (dataset / "labels" / f"{stem}.txt").read_text().split())
        frames.append({"stem": stem, "source": sources[stem], "image": image,
                       "truth": np.array([cx * w, cy * h]), "size": (bw * w, bh * h)})
    if not frames:
        raise ValueError(f"{dataset} 里没有 {recording} 的图")
    return frames


def run(model: str, frames, conf: float):
    detector = load_onnx_detector()(model, conf=conf)
    track = frames[0]["truth"].copy()
    h, w = frames[0]["image"].shape[:2]
    predict = PredictTracker(frames[0]["truth"], (w, h))
    rows = []
    for f in frames:
        boxes = [(d.confidence, np.array([d.x + d.w / 2, d.y + d.h / 2])) for d in detector.detect(f["image"])]
        top = max(boxes, key=lambda b: b[0]) if boxes else None
        if boxes:
            near = min(boxes, key=lambda b: np.linalg.norm(b[1] - track))
            if np.linalg.norm(near[1] - track) <= MAX_JUMP:
                track = near[1]
        pred = predict.update(boxes)
        rows.append({"stem": f["stem"], "source": f["source"], "boxes": len(boxes),
                     "top_conf": None if top is None else round(float(top[0]), 3),
                     "top_error": None if top is None else float(np.linalg.norm(top[1] - f["truth"])),
                     "track_error": float(np.linalg.norm(track - f["truth"])),
                     "track": track.copy(),
                     "predict_error": float(np.linalg.norm(pred - f["truth"])), "predict": pred})
    return rows


def stats(rows, key):
    e = np.array([r[key] for r in rows if r[key] is not None])
    missing = sum(r[key] is None for r in rows)
    if not len(e):
        return {"frames": len(rows), "no_box": missing}
    return {"frames": len(rows), "no_box": missing, "median_px": round(float(np.median(e)), 1),
            "p90_px": round(float(np.percentile(e, 90)), 1),
            "within_30px": round(float(np.sum(e < 30) / len(rows)), 3),
            "within_60px": round(float(np.sum(e < 60) / len(rows)), 3)}


def summarize(rows):
    out = {}
    for name, keep in (("all", lambda r: True), ("visible", lambda r: r["source"] == "visible"),
                       ("pale", lambda r: r["source"] == "pale"), ("after_fade", lambda r: r["source"] == "cursor")):
        part = [r for r in rows if keep(r)]
        if part:
            out[name] = {"single_frame_top_box": stats(part, "top_error"), "tracking": stats(part, "track_error"),
                         "tracking_predict": stats(part, "predict_error"),
                         "boxes_per_frame_median": float(np.median([r["boxes"] for r in part]))}
    after = [r for r in rows if r["source"] == "cursor"]
    for key in ("track_error", "predict_error"):
        longest = run_len = 0
        for r in after:
            run_len = run_len + 1 if r[key] > 60 else 0
            longest = max(longest, run_len)
        # 数据集隔 2 帧录像取一张，30 帧/秒
        out[f"{key}_longest_over_60px_seconds"] = round(longest * 2 / 30, 2)
    return out


def contact_sheet(frames, rows, path, count=12):
    after = [(f, r) for f, r in zip(frames, rows) if r["source"] == "cursor"]
    tiles = []
    for f, r in after[::max(1, len(after) // count)][:count]:
        im = f["image"].copy()
        tx, ty = f["truth"].astype(int)
        bw, bh = (int(v / 2) for v in f["size"])
        cv2.rectangle(im, (tx - bw, ty - bh), (tx + bw, ty + bh), (0, 0, 255), 2)
        cv2.circle(im, tuple(int(v) for v in r["predict"]), 12, (0, 255, 255), 3)
        cv2.putText(im, f"{f['stem'][-5:]} err={r['predict_error']:.0f}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (375, 250)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("recording", help="留出来测试的录像名，如 20261024_090628")
    parser.add_argument("--model", required=True, help="新训的 best.onnx")
    parser.add_argument("--compare", help="拿来对比的旧模型（卖家的 yolo26n12）")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = args.dataset / "summary.json"
    if summary.is_file() and json.loads(summary.read_text(encoding="utf-8")).get("input_kind") == "two_frame_prior":
        parser.error("形状先验数据集里的剪影是按答案画的，拿来评估等于泄题；用 tools/lie_closed_loop_eval.py 闭环评估")

    args.output.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.dataset, args.recording)
    result = {"recording": args.recording, "conf": args.conf, "images": len(frames)}
    models = {"new": args.model, **({"compare": args.compare} if args.compare else {})}
    for name, model in models.items():
        rows = run(model, frames, args.conf)
        result[name] = {"model": model, **summarize(rows)}
        with (args.output / f"{name}_frames.csv").open("w", encoding="utf-8", newline="") as f:
            skip = ("track", "predict")
            wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in skip])
            wr.writeheader()
            wr.writerows({k: v for k, v in r.items() if k not in skip} for r in rows)
        contact_sheet(frames, rows, args.output / f"{name}_after_fade.jpg")
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
