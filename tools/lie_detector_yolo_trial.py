"""使用现成 ONNX 模型做测谎视频离线回放，不发送键鼠输入。

    python tools/lie_detector_yolo_trial.py path/to/cycle_00152.avi

默认模型 models/lie_detector/best.onnx。视频须裁到金色小游戏画布。
先用白色图形选择身份，再关联所有候选。输出的 TRACKED 是推断而非真值。
支持 --detections-cache 读取先前回放的 detections.jsonl，避免重复模型推理；
缓存核对源视频和模型 SHA256，并严格逐帧读取，不向跟踪器提供未来检测。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.lie_detector_tracker import LieShapeTracker  # noqa: E402
from src.perception.yolo_detector import Detection, OnnxDetector  # noqa: E402
from tools.lie_detector_video_trial import board_present, white_target  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(frame, detections, tracker, result, visible, frame_index, fps, trail):
    h, w = frame.shape[:2]
    left, right = frame.copy(), frame.copy()
    for d in detections:
        if d.confidence >= .25:
            cv2.rectangle(left, (d.x, d.y), (d.x + d.w, d.y + d.h), (200, 160, 70), 1)
            cv2.putText(left, f'{d.confidence:.2f}', (d.x, max(12, d.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (200, 160, 70), 1)
    for track in tracker.tracks:
        if track.observed and track.id != result.target_id:
            x1, y1, x2, y2 = np.rint(track.box).astype(int)
            cv2.rectangle(right, (x1, y1), (x2, y2), (150, 130, 80), 1)
            cv2.putText(right, str(track.id), (x1, max(12, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (150, 130, 80), 1)
    colors = {"VISIBLE": (40, 240, 70), "TRACKED": (240, 220, 40),
              "AMBIGUOUS": (0, 170, 255), "PREDICTED": (0, 170, 255)}
    color = colors.get(result.state, (70, 70, 250))
    if result.box is not None:
        x1, y1, x2, y2 = np.rint(result.box).astype(int)
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 3)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.drawMarker(right, center, color, cv2.MARKER_CROSS, 22, 2)
        trail.append(center)
    else:
        trail.clear()
    if len(trail) > 1:
        cv2.polylines(right, [np.asarray(trail, np.int32)], False, color, 1)
    if visible is not None:
        mask, center, _ = visible
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, (40, 240, 70), 2)
    canvas = np.zeros((h + 100, 2 * w, 3), np.uint8)
    canvas[100:, :w], canvas[100:, w:] = left, right
    count = sum(d.confidence >= .25 for d in detections)
    cv2.putText(canvas, f'YOLO26: {count} candidates at conf >= 0.25', (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (235, 235, 235), 1)
    label = f'ID {result.target_id} | {result.state}' if result.target_id is not None else result.state
    cv2.putText(canvas, label, (w + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2)
    cv2.putText(canvas, f'frame {frame_index} | {frame_index / fps:.2f}s | no input commands',
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
    cv2.putText(canvas, 'Green: white evidence | cyan: inferred identity | amber: uncertain',
                (w + 12, 54), cv2.FONT_HERSHEY_SIMPLEX, .46, (220, 220, 220), 1)
    cv2.putText(canvas, 'Detection confidence and fixed ID are not tracking accuracy.',
                (w + 12, 80), cv2.FONT_HERSHEY_SIMPLEX, .46, (200, 200, 200), 1)
    return canvas


def replay(video, model, output, cache_path=None):
    if output.exists():
        raise FileExistsError(f"请指定不存在的新输出目录: {output}")
    signature = {"schema": 1, "video_sha256": sha256(video),
                 "model_sha256": sha256(model), "min_confidence": .1}
    cache_in, cache_out = None, None
    if cache_path:
        cache_in = cache_path.open(encoding="utf-8")
        if json.loads(cache_in.readline()) != signature:
            cache_in.close()
            raise ValueError("检测缓存与当前视频、模型或阈值不匹配")
    detector = None if cache_in else OnnxDetector(str(model), conf=.1, intra_op_threads=2)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        if cache_in:
            cache_in.close()
        raise ValueError("视频无法读取或缺少有效 FPS")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w % 2 or h % 2:
        cap.release()
        if cache_in:
            cache_in.close()
        raise ValueError("本回放编码器需要偶数宽高的视频")
    output.mkdir(parents=True)
    writer = cv2.VideoWriter(str(output / "replay.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps,
                             (2 * w, h + 100))
    if not writer.isOpened():
        cap.release()
        if cache_in:
            cache_in.close()
        raise RuntimeError("OpenCV H.264 编码器不可用")
    if not cache_in:
        cache_out = (output / "detections.jsonl").open("w", encoding="utf-8")
        cache_out.write(json.dumps(signature) + "\n")
    tracker, trail = LieShapeTracker(fps), deque(maxlen=max(2, round(fps)))
    sample_frames = {round(s * fps) for s in (1, 2.5, 4, 6, 8, 10, 12, 14)}
    rows, samples, counts, times, inference_times = [], [], Counter(), [], []
    first_frame_by_state = {}
    longest_gap, current_gap = 0, 0
    index, started = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            active = board_present(frame)
            if cache_in:
                cached = json.loads(cache_in.readline())
                if cached["frame"] != index:
                    raise ValueError("检测缓存时间轴不连续")
                detections = [Detection(**d) for d in cached["detections"]]
            else:
                tinfer = time.perf_counter()
                detections = detector.detect(frame) if active else []
                if active:
                    inference_times.append(time.perf_counter() - tinfer)
                cache_out.write(json.dumps({"frame": index, "detections": [asdict(d) for d in detections]},
                                            ensure_ascii=False) + "\n")
            visible = white_target(frame) if active else None
            visible_box = None
            if visible is not None:
                ys, xs = np.where(visible[0] > 0)
                visible_box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
            result = tracker.update(detections, visible_box, active)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            counts[result.state] += 1
            first_frame_by_state.setdefault(result.state, index)
            if result.state in ("PREDICTED", "AMBIGUOUS", "LOST"):
                current_gap += 1
                longest_gap = max(longest_gap, current_gap)
            else:
                current_gap = 0
            row = {"frame": index, "seconds": index / fps, **asdict(result),
                   "candidates": len(detections), "processing_ms": elapsed * 1000}
            box = row.pop("box")
            row.update({k: box[j] if box is not None else None
                        for j, k in enumerate(("x1", "y1", "x2", "y2"))})
            rows.append(row)
            canvas = render(frame, detections, tracker, result, visible, index, fps, trail)
            writer.write(canvas)
            if index in sample_frames:
                samples.append(cv2.resize(canvas, (900, round(canvas.shape[0] * 900 / (2 * w)))))
            index += 1
            if index % max(1, round(fps * 5)) == 0:
                print(f'processed {index} frames / {index / fps:.1f}s | {result.state}', flush=True)
        if cache_in and cache_in.readline().strip():
            raise ValueError("检测缓存比视频更长")
    finally:
        cap.release()
        writer.release()
        if cache_in:
            cache_in.close()
        if cache_out:
            cache_out.close()
    if not rows:
        raise ValueError("视频没有可解码帧")
    with (output / "tracking.csv").open("w", encoding="utf-8", newline="") as f:
        csvwriter = csv.DictWriter(f, fieldnames=list(rows[0]))
        csvwriter.writeheader()
        csvwriter.writerows(rows)
    if samples:
        kh, kw = samples[0].shape[:2]
        sheet = np.zeros((math.ceil(len(samples) / 2) * kh, 2 * kw, 3), np.uint8)
        for i, sample in enumerate(samples):
            sheet[(i // 2) * kh:(i // 2 + 1) * kh, (i % 2) * kw:(i % 2 + 1) * kw] = sample
        cv2.imwrite(str(output / "keyframes.jpg"), sheet)
    summary = {
        "video": str(video.resolve()), "model": str(model.resolve()), **signature,
        "frames_decoded": index, "fps_metadata": fps, "duration_seconds": index / fps,
        "frame_size": [w, h], "uses_future_detections": False, "uses_cached_detections": bool(cache_path),
        "input_commands_emitted": 0, "states": dict(counts),
        "first_seconds_by_state": {k: n / fps for k, n in first_frame_by_state.items()},
        "white_evidence_reanchors": tracker.reanchors,
        "longest_uncertain_interval_seconds": longest_gap / fps,
        "processing_ms_median": float(np.median(times) * 1000),
        "processing_ms_p95": float(np.percentile(times, 95) * 1000),
        "inference_ms_median": float(np.median(inference_times) * 1000) if inference_times else None,
        "wall_seconds_with_encoding": time.perf_counter() - started,
        "ground_truth_after_fade_available": False, "game_pass_confirmed": False,
        "limitations": ["TRACKED and persistent IDs are temporal inferences, not verified target identity.",
                        "Detection confidence does not measure association accuracy.",
                        "Uncertain matches do not update the target motion filter.",
                        "Scene changes, including exit dialogs, do not confirm success.",
                        "Only a cropped gold game board is supported."],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/lie_detector/best.onnx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detections-cache", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    output = args.output or ROOT / "lie_detector_trials" / (args.video.stem + time.strftime("_yolo_%Y%m%d_%H%M%S"))
    replay(args.video, args.model, output, args.detections_cache)


if __name__ == "__main__":
    main()
