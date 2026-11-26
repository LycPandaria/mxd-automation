"""检查测谎目标模型是不是在「找准星」而不是找目标；不发送键盘或鼠标输入。

例如：
    python tools/lie_crosshair_check.py lie_recordings/20261024_090628 \
        --dataset lie_detector_trials/target_dataset_v2 --model models/lie_target/best.onnx

游戏把光标画成绿色准星，玩家跟目标时准星总在目标上，训练答案又来自准星位置，模型很容易学成找准星。
程序自己接管鼠标后准星在它算的位置，会自己追自己。两个检查：
  1. 挪准星：淡出后每帧把真准星挪到 200px 外（原处抹掉、新处贴上），再按实际运行的办法藏准星
     （lie_shadow.erase_for_model），看框留在目标上还是跟着准星走；
  2. 光标瞬移（--teleport）：录像里玩家鼠标一帧跳出很远的那几帧，看框跟没跟着准星跳。
答案（目标位置）取数据集里的标签（lie_recording_labels.py 生成，sources.csv 标明哪些帧已经淡出）。
两帧对比模型（model_info.json 里 input_kind = two_frame）自动按实际运行的办法拼前两步：
挪准星时前两步的准星也挪同样的距离（像程序拿着鼠标在别处）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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

MOVE_PX = 200
NEAR_PX = 60


def move_crosshair(board, aim, rng, shift=None, two_frame=False):
    """把准星复制到 MOVE_PX 外的随机位置（或挪 shift），原处按实际运行的办法抹掉（不贴假准星）。
    单帧模型连同外面一圈底纹一起搬；两帧对比模型只搬准星画到的像素（和它的抹法一致）。
    返回 (新图, 新准星中心, 原处抹过的地方) 或 None（挪不开）。"""
    mask = shadow.ring_mask(board, aim) if two_frame else shadow.crosshair_mask(board, aim)
    if not mask.any():
        return None
    h, w = mask.shape
    halo = mask > 0 if two_frame else cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(halo)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch, patch_halo = board[y0:y1, x0:x1].copy(), halo[y0:y1, x0:x1]
    for _ in range(50 if shift is None else 1):
        angle = rng.uniform(0, 2 * np.pi)
        dx, dy = (int(MOVE_PX * np.cos(angle)), int(MOVE_PX * np.sin(angle))) if shift is None else shift
        nx, ny = x0 + dx, y0 + dy
        if 0 <= nx and nx + patch.shape[1] <= w and 0 <= ny and ny + patch.shape[0] <= h:
            break
    else:
        return None
    if two_frame:
        out, erased = cv2.inpaint(board, mask, 3, cv2.INPAINT_TELEA), mask > 0
    else:
        e = []
        out, erased = shadow.hide_crosshair(board, mask, rng, fakes=0, erased=e), e[0]
    region = out[ny:ny + patch.shape[0], nx:nx + patch.shape[1]]
    region[patch_halo] = patch[patch_halo]
    return out, np.asarray(aim, float) + [dx, dy], erased


def nearest(dets, point):
    if not dets:
        return None
    return min(float(np.hypot(d.x + d.w / 2 - point[0], d.y + d.h / 2 - point[1])) for d in dets)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--teleport", action="store_true", help="另外检查光标瞬移的那几帧")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    name = args.recording.name
    with open(args.dataset / "sources.csv", encoding="utf-8") as f:
        faded = sorted(int(r["stem"].split("_")[-1]) for r in csv.DictReader(f)
                       if r["stem"].startswith(name + "_") and r["source"] == "cursor")
    with open(args.recording / "frames.csv", encoding="utf-8") as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    detector = OnnxDetector(args.model, conf=args.conf)
    two_frame = shadow.model_input_kind(args.model) == "two_frame"
    seconds = [float(rows[i]["seconds"]) for i in sorted(rows)]
    step = max(1, round(1 / (shadow.RATE * float(np.median(np.diff(seconds))))))  # 每步隔几帧录像
    rng = random.Random(0)
    cap = cv2.VideoCapture(str(args.recording / "video.avi"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    def board_at(i):
        origin = shadow.board_box(frames[i])
        if origin is None or not rows[i]["video_x"]:
            return None
        bx, by = origin
        board = np.ascontiguousarray(frames[i][by:by + shadow.BOARD_SIZE[1], bx:bx + shadow.BOARD_SIZE[0]])
        aim = shadow.find_crosshair(board, (int(rows[i]["video_x"]) - bx, int(rows[i]["video_y"]) - by))
        return board, aim

    def hidden(board, aim):
        return shadow.erase_for_model(board, aim, rng, two_frame)

    def model_image(i, shift=None):
        """第 i 帧给模型看的图（准星可以先挪 shift）；两帧对比模型再拼上前两步。返回 (图, 准星中心) 或 None。"""
        frames_needed = [i] + ([i - k * step for k in range(1, shadow.MOTION_STEPS + 1)] if two_frame else [])
        boards, erased, cur_aim = [], [], None
        for n, j in enumerate(frames_needed):
            got = board_at(j) if j >= 0 else None
            if got is None or got[1] is None:
                if n == 0:
                    return None
                boards.append(None)
                erased.append(None)
                continue
            board, aim = got
            moved_from = None
            if shift is not None:
                m = move_crosshair(board, aim, rng, shift, two_frame)
                if m is None:
                    if n == 0:
                        return None
                    boards.append(None)
                    erased.append(None)
                    continue
                board, aim, moved_from = m
            b, e = hidden(board, aim)
            if moved_from is not None:
                e = moved_from if e is None else e | moved_from  # 原处的抹痕也和实际运行时的抹痕一样清零，不然差值图里它正好指着目标
            boards.append(b)
            erased.append(e)
            if n == 0:
                cur_aim = aim
        image = shadow.motion_input(boards[0], boards[1:], erased) if two_frame else boards[0]
        return image, cur_aim

    # 1. 挪准星
    moved = {"frames": 0, "box_on_target": 0, "box_on_moved_crosshair": 0, "no_box": 0}
    normal = {"frames": 0, "box_on_target": 0, "no_box": 0}
    for i in faded:
        got = board_at(i)
        if got is None or got[1] is None:
            continue
        board, aim = got
        stem = f"{name}_{i:05d}"
        _, cx, cy, _, _ = map(float, (args.dataset / "labels" / f"{stem}.txt").read_text().split())
        truth = np.array([cx * shadow.BOARD_SIZE[0], cy * shadow.BOARD_SIZE[1]])

        dets = detector.detect(model_image(i)[0])
        normal["frames"] += 1
        d = nearest(dets, truth)
        normal["no_box"] += d is None
        normal["box_on_target"] += d is not None and d < NEAR_PX

        for _ in range(20):
            angle = rng.uniform(0, 2 * np.pi)
            got = model_image(i, (int(MOVE_PX * np.cos(angle)), int(MOVE_PX * np.sin(angle))))
            if got is not None:
                break
        if got is None:
            continue
        image, new_aim = got
        dets = detector.detect(image)
        moved["frames"] += 1
        d_t, d_c = nearest(dets, truth), nearest(dets, new_aim)
        moved["no_box"] += d_t is None
        moved["box_on_target"] += d_t is not None and d_t < NEAR_PX
        moved["box_on_moved_crosshair"] += d_c is not None and d_c < NEAR_PX
    result = {"recording": name, "model": args.model, "conf": args.conf, "two_frame": two_frame,
              "after_fade_normal": normal, "after_fade_crosshair_moved_200px": moved}

    # 2. 光标瞬移
    if args.teleport:
        jumps, prev = [], None
        for i in sorted(rows):
            if not rows[i]["video_x"]:
                prev = None
                continue
            cur = np.array([int(rows[i]["video_x"]), int(rows[i]["video_y"])])
            if prev is not None and np.linalg.norm(cur - prev) > 100 and i < len(frames):
                jumps.append((i, prev.copy()))
            prev = cur
        teleports = []
        for j, before in jumps:
            for i in range(j, min(j + 18, len(frames)), 3):
                got = board_at(i)
                if got is None or got[1] is None:
                    continue
                board, aim = got
                origin = shadow.board_box(frames[i])
                old = before - origin  # 跳之前准星（≈目标）所在的位置
                dets = detector.detect(model_image(i)[0])
                teleports.append({"frame": i, "jump_from_frame": j,
                                  "box_near_crosshair": (nearest(dets, aim) or 999) < NEAR_PX,
                                  "box_near_pre_jump_spot": (nearest(dets, old) or 999) < 2 * NEAR_PX,
                                  "boxes": [[round(d.confidence, 2), d.x + d.w // 2, d.y + d.h // 2] for d in dets]})
        result["teleport_frames"] = teleports
        if teleports:
            result["teleport_box_followed_crosshair"] = round(
                float(np.mean([t["box_near_crosshair"] for t in teleports])), 3)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "crosshair_check.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
