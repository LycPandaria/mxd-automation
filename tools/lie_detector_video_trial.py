"""测谎游戏离线视觉试验；不发送键盘或鼠标输入。

输入必须是已裁剪到小游戏画布的视频。例如：
    python tools/lie_detector_video_trial.py path/to/cycle_00152.avi

输出对比视频、逐帧候选位置、关键帧和 JSON 摘要。左右分别为轮廓匹配、
背景运动补偿后的轮廓匹配。两者都只使用当前帧和历史帧，不使用 YOLO。
绿色表示本帧直接看见白色图形；橙色仅表示淡出后的候选，未经真值验证。
拟合分数不是正确率，也不代表游戏通过。场景切换不作成功判定。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Candidate:
    state: str = "SEARCHING"
    x: float | None = None
    y: float | None = None
    angle: float | None = None
    scale: float | None = None
    fit_cost: float | None = None
    background_compensated: bool = False


def comparison_disagreement(rows):
    """比较两个候选的分歧，不能用这种分歧替代目标真值或正确率。"""
    distances = []
    first_over_60 = None
    for a, b in zip(rows[::2], rows[1::2]):
        if a["state"] == b["state"] == "UNVERIFIED":
            distance = math.hypot(float(a["x"]) - float(b["x"]),
                                  float(a["y"]) - float(b["y"]))
            distances.append(distance)
            if first_over_60 is None and distance > 60:
                first_over_60 = float(a["seconds"])
    if not distances:
        return None
    return {"median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(max(distances)),
            "first_over_60px_seconds": first_over_60}


# 白色图形的判据和影子跟踪（形状先验记图形）共用 src/perception/lie_shadow.py 里的同一份；按文件路径加载，
# 绕开 src.perception/__init__ 里只在 Windows 上有的模块
_spec = importlib.util.spec_from_file_location(
    "lie_shadow", Path(__file__).resolve().parent.parent / "src" / "perception" / "lie_shadow.py")
_shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shadow)
white_target = _shadow.white_target


def board_present(frame: np.ndarray) -> bool:
    """本试验针对金色画布；切换回地图时清除候选，不能把退场当作成功。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gold = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] > 65)
    return float(gold.mean()) > .65


class OutlineTrial:
    def __init__(self, compensated: bool):
        self.compensated = compensated
        self.center = None
        self.velocity = np.zeros(2)
        self.angle = 0.
        self.angular_velocity = 0.
        self.initial_area = None
        self.size = None
        self.canonical = None
        self.bank = {}
        self.visible_angles = deque(maxlen=120)
        self.angle_anchor = None
        self.omega = None
        self.gray_history = deque(maxlen=7)
        self.stopped = False
        self.result = Candidate()

    def _initialize(self, mask, center, area):
        ys, xs = np.where(mask > 0)
        radius = float(np.max(np.hypot(xs - center[0], ys - center[1])))
        self.size = int(math.ceil((2 * radius * 1.1 + 32) / 2) * 2)
        self.center = center.copy()
        self.initial_area = area
        offset = np.float32([[1, 0, self.size / 2 - center[0]],
                             [0, 1, self.size / 2 - center[1]]])
        self.canonical = cv2.warpAffine(mask, offset, (self.size, self.size),
                                        flags=cv2.INTER_NEAREST)

    def _kernel(self, angle, scale):
        key = (angle, scale)
        if key not in self.bank:
            r = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), angle, scale)
            mask = cv2.warpAffine(self.canonical, r, (self.size, self.size),
                                  flags=cv2.INTER_NEAREST)
            edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            self.bank[key] = np.float32(edge > 0)
        return self.bank[key]

    def _distance(self, gray, visible_mask):
        edge = cv2.Canny(visible_mask if visible_mask is not None else gray, 65, 130)
        distance = cv2.distanceTransform(255 - edge, cv2.DIST_L2, 3)
        applied = False
        if self.compensated and visible_mask is None and len(self.gray_history) == 7:
            before = self.gray_history[0]
            bg = np.full_like(gray, 255)
            cv2.circle(bg, tuple(self.center.astype(int)), int(self.size * .6), 0, -1)
            pts = cv2.goodFeaturesToTrack(before, 400, .02, 10, mask=bg)
            if pts is not None and len(pts) >= 12:
                next_pts, good, _ = cv2.calcOpticalFlowPyrLK(
                    before, gray, pts, None, winSize=(25, 25), maxLevel=3)
                flows = (next_pts - pts)[good[:, 0] > 0, 0]
                if len(flows) >= 12:
                    shift = np.median(flows, axis=0)
                    inliers = np.linalg.norm(flows - shift, axis=1) < 2.
                    if inliers.sum() >= 12:
                        shift = np.median(flows[inliers], axis=0)
                        warped = cv2.warpAffine(before, np.float32(
                            [[1, 0, shift[0]], [0, 1, shift[1]]]),
                            (gray.shape[1], gray.shape[0]))
                        residual = cv2.absdiff(warped, gray)
                        moving = cv2.dilate(np.uint8(residual > 28), np.ones((7, 7), np.uint8))
                        foreground_edges = edge * moving
                        fg_distance = cv2.distanceTransform(255 - foreground_edges, cv2.DIST_L2, 3)
                        distance = .35 * distance + .65 * np.minimum(fg_distance, 12)
                        applied = True
        return distance, applied

    def update(self, frame, index, visible, is_board):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray_history.append(gray)
        if not is_board:
            if self.center is not None:
                self.stopped = True
            self.result = Candidate("SCENE_CHANGED" if self.stopped else "SEARCHING")
            return self.result
        if self.stopped:
            self.result = Candidate("SCENE_CHANGED")
            return self.result
        if self.center is None:
            if visible is None:
                return self.result
            self._initialize(*visible)
        if visible is not None and not .7 < visible[2] / self.initial_area < 1.3:
            visible = None

        predicted = visible[1] if visible is not None else self.center + self.velocity
        h, w = gray.shape
        half = self.size / 2
        x0 = max(0, int(predicted[0] - half - 12))
        y0 = max(0, int(predicted[1] - half - 12))
        x1 = min(w, int(predicted[0] + half + 13))
        y1 = min(h, int(predicted[1] + half + 13))
        if y1 - y0 < self.size or x1 - x0 < self.size:
            self.result = Candidate("OUT_OF_SEARCH_AREA")
            return self.result
        distance, compensated = self._distance(gray, visible[0] if visible is not None else None)
        roi = distance[y0:y1, x0:x1]
        expected = self.angle + self.angular_velocity
        if visible is None and self.omega is not None:
            anchor_frame, anchor_angle = self.angle_anchor
            expected = anchor_angle + self.omega * (index - anchor_frame)
        candidates = []
        for delta in (-6, -4, -2, 0, 2, 4, 6):
            angle = int(round((expected + delta) / 2) * 2) % 360
            for scale in ((1.,) if visible is not None else (1., 1.08)):
                kernel = self._kernel(angle, scale)
                cost = cv2.matchTemplate(roi, kernel, cv2.TM_CCORR) / kernel.sum()
                yy, xx = np.mgrid[:cost.shape[0], :cost.shape[1]]
                cost += .0009 * ((xx + x0 + half - predicted[0]) ** 2 +
                                 (yy + y0 + half - predicted[1]) ** 2)
                if visible is None and self.omega is not None:
                    cost += .01 * delta ** 2
                iy, ix = np.unravel_index(cost.argmin(), cost.shape)
                pos = np.array([ix + x0 + half, iy + y0 + half])
                candidates.append((float(cost[iy, ix]), angle, scale, pos))
        value, angle, scale, pos = min(candidates, key=lambda c: c[0])
        if visible is not None:
            pos = visible[1].copy()
            self.visible_angles.append((index, angle))
            if len(self.visible_angles) >= 20:
                history = np.array(self.visible_angles)
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(history[:, 1])))
                self.omega = float(np.polyfit(history[:, 0], unwrapped, 1)[0])
                self.angle_anchor = (index, float(angle))
        self.velocity = .5 * self.velocity + .5 * (pos - self.center)
        da = (angle - self.angle + 180) % 360 - 180
        self.angular_velocity = .5 * self.angular_velocity + .5 * da
        self.center, self.angle = pos, float(angle)
        self.result = Candidate("VISIBLE" if visible is not None else "UNVERIFIED",
                                float(pos[0]), float(pos[1]), float(angle), scale,
                                value, compensated)
        return self.result

    def render(self, frame, title, index, fps):
        h, w = frame.shape[:2]
        image = np.zeros((h + 72, w, 3), np.uint8)
        image[72:] = frame
        result = self.result
        color = (60, 230, 80) if result.state == "VISIBLE" else (0, 185, 255)
        if result.x is not None:
            kernel = self._kernel(int(result.angle), result.scale)
            transform = np.float32([[1, 0, result.x - self.size / 2],
                                    [0, 1, result.y - self.size / 2]])
            edge = cv2.warpAffine(kernel, transform, (w, h))
            image[72:][edge > .5] = color
            cv2.drawMarker(image, (round(result.x), round(result.y) + 72), color,
                           cv2.MARKER_CROSS, 18, 2)
        cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
        text = f'{index / fps:.2f}s  {result.state}'
        if result.fit_cost is not None:
            text += f'  cost={result.fit_cost:.2f} (not accuracy)'
        cv2.putText(image, text, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        return image


def replay(video: Path, output: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"无法读取视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("视频缺少有效 FPS，无法确定时间轴")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w < 100 or h < 100:
        cap.release()
        raise ValueError("视频画幅无效")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.mp4", "candidates.csv", "summary.json", "keyframes.jpg"):
        if (output / name).exists():
            cap.release()
            raise FileExistsError(f"输出已存在，请指定新的 --output 目录: {output / name}")
    writer = cv2.VideoWriter(str(output / "comparison.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (2 * w, h + 72))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("当前 OpenCV 无 H.264 编码器，无法生成回放")
    trackers = [OutlineTrial(False), OutlineTrial(True)]
    names = ["Outline + rotation prior", "Background compensation + outline"]
    rows, timings, keyframes = [], [], []
    counters = [Counter(), Counter()]
    sample_frames = {round(s * fps) for s in (1, 2.5, 4, 6, 8, 10, 12, 14)}
    index, started = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            is_board = board_present(frame)
            previous = trackers[0].center
            visible = white_target(frame, previous) if is_board else None
            for name, tracker, counts in zip(names, trackers, counters):
                candidate = tracker.update(frame, index, visible, is_board)
                counts[candidate.state] += 1
                rows.append({"frame": index, "seconds": index / fps, "method": name,
                             **asdict(candidate)})
            timings.append(time.perf_counter() - t0)
            comparison = np.hstack([tracker.render(frame, name, index, fps)
                                     for tracker, name in zip(trackers, names)])
            writer.write(comparison)
            if index in sample_frames:
                keyframes.append(cv2.resize(comparison, (900, round((h + 72) * 900 / (2 * w)))))
            index += 1
            if index % max(1, round(fps * 5)) == 0:
                print(f"processed {index} frames / {index / fps:.1f}s", flush=True)
    finally:
        writer.release()
        cap.release()
    if not rows:
        raise ValueError("视频没有可解码的帧")
    with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    if keyframes:
        kh, kw = keyframes[0].shape[:2]
        sheet = np.zeros((math.ceil(len(keyframes) / 2) * kh, 2 * kw, 3), np.uint8)
        for i, frame in enumerate(keyframes):
            sheet[(i // 2) * kh:(i // 2 + 1) * kh, (i % 2) * kw:(i % 2 + 1) * kw] = frame
        cv2.imwrite(str(output / "keyframes.jpg"), sheet)
    summary = {
        "video": str(video.resolve()), "frames_decoded": index, "fps_metadata": fps,
        "duration_seconds": index / fps, "frame_size": [w, h],
        "coordinates": "video-local pixels; no input commands emitted",
        "uses_future_frames": False, "uses_yolo": False,
        "methods": {name: dict(counts) for name, counts in zip(names, counters)},
        "after_fade_method_disagreement_px": comparison_disagreement(rows),
        "processing_ms_median_both_methods": float(np.median(timings) * 1000),
        "processing_ms_p95_both_methods": float(np.percentile(timings, 95) * 1000),
        "total_wall_seconds_with_decode_render_encode": time.perf_counter() - started,
        "ground_truth_after_fade_available": False, "game_pass_confirmed": False,
        "limitations": ["Fit cost is not confidence or accuracy.",
                        "Orange candidates after the white target fades are unverified.",
                        "Rotation prior extrapolates the rate observed while the target is white.",
                        "Only cropped gold-board footage is supported; scene changes are not success."],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("lie_detector_trials") / (args.video.stem + time.strftime("_%Y%m%d_%H%M%S"))
    cv2.setNumThreads(2)
    replay(args.video, output)


if __name__ == "__main__":
    main()
