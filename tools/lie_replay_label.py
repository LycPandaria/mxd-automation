"""回放跟踪标注：放慢播放测谎视频，人像玩测谎一样用鼠标跟住目标，记下每帧鼠标位置当答案。

买来的 zhaopian/cycle_xxxxx.avi 只录了画板、没有准星、也没人跟，目标淡出后在哪没有答案；
人在回放里跟（能放慢、能倒回去重跟）就是答案。输出和 LieRecorder 录像同一格式（frames.csv），
tools/lie_recording_labels.py 可以直接拿来生成训练数据。

用法：
    # 标买来的视频（一个文件、一个目录或通配都行），结果存到 replay_labels/<视频名>/
    python tools/lie_replay_label.py D:/videos/zhaopian/cycle_00001/cycle_00001.avi
    python tools/lie_replay_label.py D:/videos/zhaopian --limit 10

    # 先拿自己录过、有答案的测谎录像测准不准：画面里的准星会先藏掉，标完自动和真答案比
    python tools/lie_replay_label.py lie_recordings/20261024_090628

按键（在视频窗口里按）：
    空格  开始 / 暂停          [ ]  慢一档 / 快一档（0.25x ~ 1x）
    a     倒回 1 秒            z    倒回 3 秒          r  从头开始
    b     开 / 关「跟丢了」：开着时记的帧不当答案（目标确实找不到时按一下，找回来再按一下）
    回车  保存，下一个          Esc  连按两次：不保存，退出
倒回去重跟时，新跟的位置会覆盖原来记的。
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("lie_shadow", REPO / "src" / "perception" / "lie_shadow.py")
shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shadow)

KEYS_HELP = """按键（在视频窗口里按）：
  空格 开始/暂停   [ ] 慢一档/快一档   a 倒回 1 秒   z 倒回 3 秒   r 从头开始
  b 开/关「跟丢了」（开着时窗口有红框，这段不当答案）   回车 保存并下一个   Esc 连按两次=不保存退出"""

SPEEDS = (.25, .35, .5, .75, 1.)
DEFAULT_SPEED = .5
BAR_H = 36
WINDOW = "lie replay label"


class Track:
    """每帧的标注：鼠标位置（画板坐标）和是否「跟丢了」。重跟时后来的覆盖先前的。"""

    def __init__(self, frames: int):
        self.pos = [None] * frames
        self.skip = [False] * frames

    def record(self, i, xy, skip):
        self.pos[i] = (int(round(xy[0])), int(round(xy[1])))
        self.skip[i] = bool(skip)

    @property
    def labeled(self):
        return sum(p is not None and not s for p, s in zip(self.pos, self.skip))


class Source:
    """视频来源：买来的纯画板视频直接用；自己的测谎录像先裁出画板、把准星藏掉（不然等于看着答案标）。"""

    def __init__(self, path: Path):
        self.path = path
        self.cursor_reference_valid = True
        self.cursor_reference_note = ""
        if path.is_dir():
            self.recording = path
            self.video = path / "video.avi"
            meta_path = path / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            self.cursor_reference_valid = meta.get("cursor_reference_valid") is not False
            self.cursor_reference_note = meta.get("cursor_reference_note", "")
            with open(path / "frames.csv", encoding="utf-8") as f:
                self.cursor = {int(r["frame"]): r for r in csv.DictReader(f)}
        else:
            self.recording, self.video, self.cursor = None, path, {}
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise ValueError(f"打不开视频 {self.video}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.
        self.origin = (0, 0)
        if self.recording is not None:
            origins = []
            for i in range(0, self.frames, 5):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = self.cap.read()
                if ok and (o := shadow.board_box(frame)) is not None:
                    origins.append(o)
            if not origins:
                raise ValueError(f"{self.video} 里没找到画板")
            self.origin = tuple(int(v) for v in np.median(origins, axis=0))
        self.start = 0  # 从第一帧有画板的地方开始放（前面是说明框，跟不了）
        for i in range(self.frames):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i) if i == 0 else None
            ok, frame = self.cap.read()
            if not ok:
                break
            if shadow.board_box(frame) is not None:
                self.start = i
                break
        self.rng = random.Random(0)
        self._pos = None  # 上一次读到第几帧；None = 不知道（比如刚扫过一遍找画板），下次先跳到位

    @property
    def name(self):
        return self.path.name if self.recording is not None else self.path.stem

    def board(self, i):
        """第 i 帧的画板图（750×500）；读不到返回 None。"""
        if self._pos is None or i != self._pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        self._pos = i
        if not ok:
            return None
        bx, by = self.origin
        w, h = shadow.BOARD_SIZE
        board = np.ascontiguousarray(frame[by:by + h, bx:bx + w])
        if board.shape[:2] != (h, w):
            board = cv2.resize(board, (w, h))
        row = self.cursor.get(i)
        if row and row["video_x"]:
            aim = shadow.find_crosshair(board, (int(row["video_x"]) - bx, int(row["video_y"]) - by))
            if aim is not None:
                board = shadow.hide_crosshair(board, shadow.crosshair_mask(board, aim), self.rng)
        return board


def save(out_dir: Path, source: Source, track: Track, speed_log):
    """写 frames.csv（录像同一格式，video_x/y 是原视频坐标，多一列 skip）和 meta.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    bx, by = source.origin
    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "seconds", "client_x", "client_y", "video_x", "video_y", "left_down", "skip"])
        for i, (p, s) in enumerate(zip(track.pos, track.skip)):
            if p is None:
                wr.writerow([i, f"{i / source.fps:.4f}", "", "", "", "", "", ""])
            else:
                x, y = p[0] + bx, p[1] + by
                wr.writerow([i, f"{i / source.fps:.4f}", x, y, x, y, 0, int(s)])
    meta = {"source": "replay_label", "video": str(source.video.resolve()), "fps": source.fps,
            "frames": source.frames, "board_origin": list(source.origin),
            "cursor_is_aim": True,  # 记的就是人瞄的位置，不用再找准星偏移
            "crosshair_hidden_from_labeler": source.recording is not None,
            "labeled_frames": track.labeled, "skipped_frames": sum(track.skip),
            "speeds_used": sorted(set(speed_log)), "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fade_range(source: Source):
    """(淡出后第一帧, 画板最后一帧)：白色目标最后一次看得见之后、画板关掉之前，才是要靠跟的部分。"""
    sys.path.insert(0, str(REPO / "tools"))
    from lie_detector_video_trial import board_present
    start = fade_start(source)
    if start is None:
        return None
    end = start
    for i in range(start, source.frames):
        board = source.board(i)
        if board is None:
            break
        if board_present(board):
            end = i
    return start, end


def fade_start(source: Source):
    """白色目标最后一次看得见之后的第一帧（之后才是要靠跟的部分）。"""
    sys.path.insert(0, str(REPO / "tools"))
    from lie_detector_video_trial import board_present, white_target
    last, prev = None, None
    for i in range(source.frames):
        board = source.board(i)
        if board is None:
            break
        if not board_present(board):  # 画板关掉以后是地图和对话框，白色的对话框不是目标
            prev = None
            continue
        v = white_target(board, prev)
        if v is not None:
            last, prev = i, v[1].copy()
    return None if last is None else last + 1


def compare(source: Source, track: Track):
    """自己的录像：与原录像准星参考比较；已确认有误的参考不计算误差。"""
    if not source.cursor_reference_valid:
        return {"reference_valid": False, "reference_kind": "recorded_crosshair",
                "reason": source.cursor_reference_note or "原录像准星参考已标记为不可靠"}
    sys.path.insert(0, str(REPO / "tools"))
    from lie_recording_eval import crosshair_offset
    offset, _ = crosshair_offset(source.recording / "frames.csv", source.video)
    span = fade_range(source)
    if span is None:
        return None
    bx, by = source.origin
    # 真答案（录像时的鼠标）一帧跳出 100px 以上：那段玩家自己没在跟，不拿来比（和 lie_recording_labels 一样）
    jumped, prev = set(), None
    for i in sorted(source.cursor):
        row = source.cursor[i]
        if not row["video_x"]:
            prev = None
            continue
        cur = np.array([int(row["video_x"]), int(row["video_y"])])
        if prev is not None and np.linalg.norm(cur - prev) > 100:
            jumped.update(range(i, i + int(source.fps * .67) + 1))
        prev = cur
    pairs = []
    for i in range(span[0], span[1] + 1):
        if i in jumped:
            continue
        row = source.cursor.get(i)
        p = track.pos[i]
        if p is None or track.skip[i] or not row or not row["video_x"]:
            continue
        truth = np.array([int(row["video_x"]) - bx, int(row["video_y"]) - by]) + offset
        if 0 <= truth[0] < shadow.BOARD_SIZE[0] and 0 <= truth[1] < shadow.BOARD_SIZE[1]:
            pairs.append((i, np.array(p, float), truth))
    if not pairs:
        return None
    err = np.array([np.linalg.norm(p - t) for _, p, t in pairs])
    # 人跟目标总会慢半拍：试着把标注往前挪几帧，看最小误差在哪（只是参考，答案用的是没挪的）
    by_frame = {i: p for i, p, _ in pairs}
    lags = {}
    for lag in range(0, int(source.fps * .6) + 1, max(1, int(source.fps / 15))):
        e = [np.linalg.norm(by_frame[i + lag] - t) for i, _, t in pairs if i + lag in by_frame]
        if len(e) > len(pairs) / 2:
            lags[lag] = float(np.median(e))
    best = min(lags, key=lags.get) if lags else 0
    return {"after_fade_frames": len(pairs), "median_px": round(float(np.median(err)), 1),
            "p90_px": round(float(np.percentile(err, 90)), 1),
            "within_30px": round(float(np.mean(err < 30)), 3), "within_60px": round(float(np.mean(err < 60)), 3),
            "skipped_frames": sum(track.skip),
            "best_lag_seconds": round(best / source.fps, 2), "median_px_at_best_lag": round(lags.get(best, 0), 1)}


def label(source: Source, out_dir: Path, scale: float):
    """交互式标一个视频；返回 (True, 原因) 保存了，(None, 原因) 没保存、要退出。"""
    track = Track(source.frames)
    state = {"mouse": None}
    speed_i = SPEEDS.index(DEFAULT_SPEED)
    speeds_used = []

    def on_mouse(event, x, y, flags, param):
        if y >= BAR_H:
            state["mouse"] = (x / scale, (y - BAR_H) / scale)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    i, playing, skip = source.start, False, False
    board = source.board(i)
    next_t = time.perf_counter()
    esc_at = None   # 按过一次 Esc 的时间：3 秒内再按一次才真的不保存退出
    while True:
        if playing:
            now = time.perf_counter()
            if now >= next_t:
                if state["mouse"] is not None:
                    track.record(i, state["mouse"], skip)
                    speeds_used.append(SPEEDS[speed_i])
                if i + 1 >= source.frames:
                    playing = False
                else:
                    i += 1
                    nxt = source.board(i)
                    if nxt is None:
                        playing = False
                    else:
                        board = nxt
                next_t += 1. / (source.fps * SPEEDS[speed_i])
                if next_t < now:
                    next_t = now  # 电脑跟不上就别补，按实际节奏放
        view = board if scale == 1 else cv2.resize(board, None, fx=scale, fy=scale)
        canvas = np.zeros((view.shape[0] + BAR_H, view.shape[1], 3), np.uint8)
        canvas[BAR_H:] = view
        if skip:
            cv2.rectangle(canvas, (0, BAR_H), (canvas.shape[1] - 1, canvas.shape[0] - 1), (0, 0, 255), 6)
        status = "PLAY" if playing else ("END - Enter save" if i + 1 >= source.frames else "PAUSE - space")
        if esc_at is not None and time.perf_counter() - esc_at < 3:
            status = "Esc again = quit WITHOUT saving, Enter = save"
        text = (f"{source.name}  {i / source.fps:5.1f}/{source.frames / source.fps:.1f}s  "
                f"x{SPEEDS[speed_i]:g}  {status}  labeled {track.labeled}" + ("  [LOST]" if skip else ""))
        cv2.putText(canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1 if playing else 30) & 0xFF
        if key == 255:
            continue
        if key == 27:
            if esc_at is not None and time.perf_counter() - esc_at < 3:
                return None, "按了两次 Esc，没保存"
            esc_at, playing = time.perf_counter(), False
            continue
        esc_at = None
        if key in (13, 10):
            save(out_dir, source, track, speeds_used)
            return True, "按了回车"
        if key == ord(" "):
            playing = not playing and i + 1 < source.frames
            next_t = time.perf_counter()
        elif key == ord("["):
            speed_i = max(0, speed_i - 1)
        elif key == ord("]"):
            speed_i = min(len(SPEEDS) - 1, speed_i + 1)
        elif key in (ord("a"), ord("z"), ord("r")):
            back = {ord("a"): 1., ord("z"): 3.}.get(key)
            i = source.start if back is None else max(source.start, i - int(back * source.fps))
            board = source.board(i)
            next_t = time.perf_counter()
        elif key == ord("b"):
            skip = not skip


def inputs(args):
    paths = []
    for a in args.inputs:
        p = Path(a)
        if p.is_dir() and (p / "frames.csv").exists():
            paths.append(p)  # 自己的测谎录像
        elif p.is_dir():
            paths += sorted(Path(v) for v in glob.glob(str(p / "**" / "*.avi"), recursive=True))
        else:
            paths += sorted(Path(v) for v in glob.glob(a))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="视频文件 / 目录 / 自己的测谎录像目录")
    parser.add_argument("--output", type=Path, default=REPO / "replay_labels")
    parser.add_argument("--scale", type=float, default=1., help="窗口放大倍数（屏幕小就 0.8）")
    parser.add_argument("--limit", type=int, help="这次最多标几个")
    parser.add_argument("--redo", action="store_true", help="标过的也重标")
    args = parser.parse_args()

    print(KEYS_HELP)
    done = 0
    for path in inputs(args):
        source = Source(path)
        out_dir = args.output / source.name
        if (out_dir / "meta.json").exists() and not args.redo:
            continue
        print(f"[{done + 1}] {source.name}：{source.frames} 帧，{source.frames / source.fps:.1f} 秒，"
              f"先把鼠标放在白色目标上，按空格开始")
        ok, reason = label(source, out_dir, args.scale)
        if ok is None:
            print(f"退出：{reason}")
            break
        print(f"    已保存（{reason}）→ {out_dir}")
        if source.recording is not None:
            report = compare(source, load_track(out_dir, source))
            if report:
                (out_dir / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
                if report.get("reference_valid") is False:
                    print(f"    不计算标注误差：{report['reason']}；回放标注已保存")
                else:
                    print(f"    和原录像准星参考比（淡出后 {report['after_fade_frames']} 帧）：中位 {report['median_px']}px，"
                          f"30px 内 {report['within_30px']:.0%}，60px 内 {report['within_60px']:.0%}；"
                          f"相对参考的延迟约 {report['best_lag_seconds']} 秒")
        done += 1
        if args.limit and done >= args.limit:
            break
    cv2.destroyAllWindows()


def load_track(out_dir: Path, source: Source) -> Track:
    """从保存的 frames.csv 读回标注（画板坐标）。"""
    track = Track(source.frames)
    bx, by = source.origin
    with open(out_dir / "frames.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video_x"]:
                track.record(int(r["frame"]), (int(r["video_x"]) - bx, int(r["video_y"]) - by), r["skip"] == "1")
    return track


if __name__ == "__main__":
    main()
