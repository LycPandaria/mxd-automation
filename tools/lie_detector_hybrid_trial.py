"""测谎游戏离线混合跟踪试验：YOLO 提候选 + 背景补偿轮廓匹配做判别；不发送键盘或鼠标输入。

例如：
    python tools/lie_detector_hybrid_trial.py cycle_00152.avi --model best.onnx \
        --output lie_detector_trials/cycle_00152_hybrid

白色目标可见时沿用 OutlineTrial（背景补偿）学习目标轮廓和旋转速度；淡出后每帧：
  1. 候选 = YOLO 框中心（只取预测位置附近、大小与目标相近的框）+ 纯轮廓在预测点附近的搜索；
  2. 每个候选在小窗口内用旋转后的目标轮廓做匹配，得到轮廓代价（越低越像目标）；
  3. 总代价 = 轮廓代价 + 运动权重 × 离预测点距离²，取最小者。
同时记录每帧"最优与次优候选（相距 >30px）"的代价差，差越小越可能认错。
没有淡出后的真值：代价不是正确率，结果不代表游戏通过。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lie_detector_video_trial import OutlineTrial, board_present, white_target

REPO = Path(__file__).resolve().parent.parent


def load_onnx_detector():
    """按文件路径加载，绕开 src.perception/__init__ 里只在 Windows 上有的 win32gui。"""
    spec = importlib.util.spec_from_file_location(
        "yolo_detector", REPO / "src" / "perception" / "yolo_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OnnxDetector


@dataclass
class HybridCandidate:
    state: str = "SEARCHING"
    x: float | None = None
    y: float | None = None
    angle: float | None = None
    fit_cost: float | None = None
    source: str | None = None          # "visible" / "yolo" / "outline"
    yolo_conf: float | None = None
    proposals: int = 0
    runner_up_margin: float | None = None
    outline_fit: float | None = None       # 纯轮廓搜索点的轮廓代价（不含运动惩罚）
    best_yolo_fit: float | None = None     # 所有 YOLO 候选里最低的轮廓代价
    best_yolo_dist: float | None = None    # 该 YOLO 候选离预测点的距离


class HybridTrial:
    def __init__(self, motion_weight=.0009, min_conf=.05, reacquire_radius=150.,
                 refine=14):
        self.outline = OutlineTrial(compensated=True)
        self.motion_weight = motion_weight
        self.min_conf = min_conf
        self.reacquire_radius = reacquire_radius
        self.refine = refine
        self.target_box_area = None
        self.result = HybridCandidate()

    def _fit(self, distance, expected, around, predicted, window, allow_scale):
        """在 around 周围 ±window 内匹配旋转轮廓，返回 (总代价, 轮廓代价, 角度, 位置)。"""
        o = self.outline
        h, w = distance.shape
        half = o.size / 2
        x0 = max(0, int(around[0] - half - window))
        y0 = max(0, int(around[1] - half - window))
        x1 = min(w, int(around[0] + half + window + 1))
        y1 = min(h, int(around[1] + half + window + 1))
        if y1 - y0 < o.size or x1 - x0 < o.size:
            return None
        roi = distance[y0:y1, x0:x1]
        best = None
        for delta in (-6, -4, -2, 0, 2, 4, 6):
            angle = int(round((expected + delta) / 2) * 2) % 360
            for scale in ((1., 1.08) if allow_scale else (1.,)):
                kernel = o._kernel(angle, scale)
                fit = cv2.matchTemplate(roi, kernel, cv2.TM_CCORR) / kernel.sum()
                if o.omega is not None:
                    fit = fit + .01 * delta ** 2
                yy, xx = np.mgrid[:fit.shape[0], :fit.shape[1]]
                total = fit + self.motion_weight * ((xx + x0 + half - predicted[0]) ** 2 +
                                                    (yy + y0 + half - predicted[1]) ** 2)
                iy, ix = np.unravel_index(total.argmin(), total.shape)
                pos = np.array([ix + x0 + half, iy + y0 + half])
                item = (float(total[iy, ix]), float(fit[iy, ix]), angle, pos)
                if best is None or item[0] < best[0]:
                    best = item
        return best

    def update(self, frame, index, visible, is_board, detections):
        o = self.outline
        if visible is not None or not is_board or o.center is None or o.stopped:
            r = o.update(frame, index, visible, is_board)
            if visible is not None and detections and r.x is not None:
                # 记下可见期目标对应 YOLO 框的面积，淡出后用来过滤大小差太多的诱饵框
                d = min(detections, key=lambda d: np.hypot(d.x + d.w / 2 - r.x, d.y + d.h / 2 - r.y))
                self.target_box_area = d.w * d.h
            self.result = HybridCandidate(r.state, r.x, r.y, r.angle, r.fit_cost,
                                          "visible" if r.state == "VISIBLE" else None)
            return self.result

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        o.gray_history.append(gray)
        predicted = o.center + o.velocity
        expected = o.angle + o.angular_velocity
        if o.omega is not None:
            anchor_frame, anchor_angle = o.angle_anchor
            expected = anchor_angle + o.omega * (index - anchor_frame)
        distance, _ = o._distance(gray, None)

        proposals = []
        fit = self._fit(distance, expected, predicted, predicted, 12, True)
        if fit is not None:
            proposals.append((*fit, "outline", None))
        for d in detections:
            if d.confidence < self.min_conf:
                continue
            c = np.array([d.x + d.w / 2, d.y + d.h / 2])
            if np.linalg.norm(c - predicted) > self.reacquire_radius:
                continue
            if self.target_box_area and not .5 < d.w * d.h / self.target_box_area < 2.:
                continue
            fit = self._fit(distance, expected, c, predicted, self.refine, True)
            if fit is not None:
                proposals.append((*fit, "yolo", d.confidence))
        if not proposals:
            self.result = HybridCandidate("OUT_OF_SEARCH_AREA")
            return self.result

        proposals.sort(key=lambda p: p[0])
        total, fit_cost, angle, pos, source, conf = proposals[0]
        rivals = [p[0] for p in proposals[1:] if np.linalg.norm(p[3] - pos) > 30]
        margin = rivals[0] - total if rivals else None
        outline_fit = next((p[1] for p in proposals if p[4] == "outline"), None)
        yolo = [p for p in proposals if p[4] == "yolo"]
        best_yolo = min(yolo, key=lambda p: p[1]) if yolo else None

        o.velocity = .5 * o.velocity + .5 * (pos - o.center)
        da = (angle - o.angle + 180) % 360 - 180
        o.angular_velocity = .5 * o.angular_velocity + .5 * da
        o.center, o.angle = pos, float(angle)
        self.result = HybridCandidate("UNVERIFIED", float(pos[0]), float(pos[1]), float(angle),
                                      fit_cost, source, conf, len(proposals), margin, outline_fit,
                                      best_yolo[1] if best_yolo else None,
                                      float(np.linalg.norm(best_yolo[3] - predicted)) if best_yolo else None)
        return self.result


def replay(video: Path, model: str, output: Path, motion_weight: float):
    OnnxDetector = load_onnx_detector()
    detector = OnnxDetector(model, conf=.05)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"无法读取视频: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output / "hybrid.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    hybrid = HybridTrial(motion_weight=motion_weight)
    baseline = OutlineTrial(compensated=True)
    rows, timings, index = [], [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            is_board = board_present(frame)
            visible = white_target(frame, hybrid.outline.center) if is_board else None
            detections = detector.detect(frame) if is_board else []
            result = hybrid.update(frame, index, visible, is_board, detections)
            timings.append(time.perf_counter() - t0)
            base = baseline.update(frame, index, visible, is_board)
            rows.append({"frame": index, "seconds": index / fps, **asdict(result),
                         "baseline_x": base.x, "baseline_y": base.y})

            for d in detections:
                if d.confidence >= .25:
                    cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), (255, 140, 0), 1)
            if base.x is not None and result.state != "VISIBLE":
                cv2.drawMarker(frame, (round(base.x), round(base.y)), (255, 0, 255),
                               cv2.MARKER_TILTED_CROSS, 20, 2)
            if result.x is not None:
                color = {"visible": (60, 230, 80), "yolo": (0, 0, 255)}.get(result.source, (0, 220, 255))
                cv2.circle(frame, (round(result.x), round(result.y)), 14, color, 2)
            text = f"f{index} {result.state} {result.source or ''}"
            if result.runner_up_margin is not None:
                text += f" margin={result.runner_up_margin:.2f}"
            cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            writer.write(frame)
            index += 1
            if index % round(fps * 5) == 0:
                print(f"processed {index} frames", flush=True)
    finally:
        writer.release()
        cap.release()

    with (output / "hybrid_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    after = [r for r in rows if r["state"] == "UNVERIFIED"]
    margins = np.array([r["runner_up_margin"] for r in after if r["runner_up_margin"] is not None])
    gaps = np.array([np.hypot(r["x"] - r["baseline_x"], r["y"] - r["baseline_y"])
                     for r in after if r["baseline_x"] is not None])
    summary = {
        "video": str(video.resolve()), "model": str(Path(model).resolve()),
        "motion_weight": motion_weight, "frames": index,
        "after_fade_frames": len(after),
        "winner_source": {s: sum(r["source"] == s for r in after) for s in ("yolo", "outline")},
        "runner_up_margin": {"median": float(np.median(margins)),
                             "frames_below_0.1": int((margins < .1).sum())} if len(margins) else None,
        "fit_cost_median": float(np.median([r["fit_cost"] for r in after])) if after else None,
        "distance_to_outline_only_px": {"median": float(np.median(gaps)),
                                        "first_100_median": float(np.median(gaps[:100])),
                                        "within_40px": float(np.mean(gaps < 40))} if len(gaps) else None,
        "processing_ms_median": float(np.median(timings) * 1000),
        "processing_ms_p95": float(np.percentile(timings, 95) * 1000),
        "ground_truth_after_fade_available": False, "game_pass_confirmed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", required=True, help="yolo26n12 的 best.onnx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-weight", type=float, default=.0009,
                        help="离预测点距离² 的惩罚系数；越小越容易跳到远处的 YOLO 候选")
    args = parser.parse_args()
    replay(args.video, args.model, args.output, args.motion_weight)


if __name__ == "__main__":
    main()
