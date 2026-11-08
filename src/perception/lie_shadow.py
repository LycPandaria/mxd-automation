"""测谎影子跟踪：测谎时在后台算目标位置写日志；开了接管（takeover）时，程序把鼠标移过去跟目标。

和 LieRecorder 一起用：录制把画面和光标存下来，影子跟踪把每帧算出的目标位置写到同一目录的 shadow.csv，
事后对比「程序算的位置」和「实际的准星」。接管规则见 MouseTakeover：你点一下鼠标左键才交还给你，这次测谎不再接管。

流程（每秒最多 RATE 帧，和离线评估时数据集「隔 2 帧取 1 张」的节奏一致）：
  1. 找金色画板（750×500），裁出来；
  2. 藏准星（erase_for_model，训练图也这样做，不然模型会去找准星）：单帧模型先贴几个假准星再统一抹掉；
     两帧对比模型只抹准星画到的那些像素，不贴假的；
  3. 只框真目标的模型（tools/lie_recording_labels.py 的数据训练）出框；两帧对比模型（model_info.json 里
     input_kind = two_frame）看的不是原图，而是 motion_input：当前帧灰度 + 和前两步对齐相减的差；
     形状先验模型（two_frame_prior）再多一个通道：倒计时记下的图形转到当前角度、画在上一步的位置（ShapePrior）；
  4. 运动预测跟踪（PredictTracker，参数和 tools/lie_target_model_eval.py 里一样）。
2~4 步在 TargetFinder 里，离线闭环评估（tools/lie_closed_loop_eval.py）用同一份。

接管实际用的是 ShapeFinder：看得见白色图形就用它的重心，之后拿记下的图形做形状匹配（不看模型）；
这一局没记到图形时才退回上面的 TargetFinder。
"""
from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
from collections import deque
from typing import Callable, Optional, Sequence, Tuple

import cv2
import numpy as np

BOARD_SIZE = (750, 500)
RATE = 15.

GATE, GATE_GROW, GATE_MAX = 50, 15, 140
MAX_SPEED = 25
LOW_CONF = .25
REACQUIRE_CONF = .6
START_CONF = .6
"""还没锁定时，第一个这个分以上的框当作起点（目标还白着时模型几乎只出一个框）"""


class PredictTracker:
    """运动预测跟踪：位置 + 速度（tools/lie_target_model_eval.py 离线评估也用这一份）。
    门限内有框就挑「分高、离预测近」的那个、更新速度；没有就按速度往前推，速度慢慢衰减。
    低分框（LOW_CONF 以下，多半是诱饵）只在离预测很近时才用；门限外连续两帧同一处出现高分框，说明跟丢了，跳过去。
    参数按每秒 RATE 步调的（离线评估：数据集隔 2 帧录像取 1 张）。"""

    def __init__(self, start, bounds):
        self.bounds = np.asarray(bounds, float)
        self.pos = np.asarray(start, float)
        self.vel = np.zeros(2)
        self.missed = 0
        self.far_strong = None

    def update(self, boxes):
        predicted = self.pos + self.vel
        gate = min(GATE_MAX, GATE + GATE_GROW * self.missed)
        scored = []
        for conf, c in boxes:
            d = np.linalg.norm(c - predicted)
            if d <= gate and (conf >= LOW_CONF or d <= GATE / 2):
                scored.append((conf * np.exp(-(d / gate) ** 2), c))
        strong = [c for conf, c in boxes if conf >= REACQUIRE_CONF and np.linalg.norm(c - predicted) > gate]
        jump = None
        if not scored and strong and self.far_strong is not None:
            jump = min(strong, key=lambda c: np.linalg.norm(c - self.far_strong))
            if np.linalg.norm(jump - self.far_strong) > 40:
                jump = None
        self.far_strong = strong[0] if strong and not scored else None
        if scored:
            best = max(scored, key=lambda t: t[0])[1]
            self.vel = .6 * self.vel + .4 * (best - self.pos) / (self.missed + 1)
            self.pos, self.missed = best, 0
        elif jump is not None:
            self.pos, self.vel, self.missed, self.far_strong = jump, np.zeros(2), 0, None
        else:
            self.pos, self.vel, self.missed = predicted, self.vel * .95, self.missed + 1
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel *= MAX_SPEED / speed
        clipped = np.clip(self.pos, 0, self.bounds - 1)
        # 撞到画板边：朝墙那个方向的速度清零，不再往墙里推（目标贴边时模型常认不出，原来会一直卡在边上）
        self.vel[clipped != self.pos] = 0
        self.pos = clipped
        return self.pos.copy()


def board_box(frame) -> Optional[Tuple[int, int]]:
    """金色画板左上角，没有返回 None（与 tools/lie_recording_eval.py 的 board_box 同一判据）。
    内部先在半分辨率上找（形状/面积判据都按比例缩，容差足够），坐标 ×2 回乘——单帧 7.5→2.2ms，
    每步都要跑一次，这是步频预算里的大头之一。"""
    scale = 2
    h, w = frame.shape[:2]
    small = frame if scale == 1 else cv2.resize(frame, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gold = np.uint8((hsv[..., 0] >= 10) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 65))
    gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((max(3, 9 // scale),) * 2, np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(gold)
    if count < 2:
        return None
    x, y, w, h, area = stats[1 + np.argmax(stats[1:, 4])]
    if (abs(w - BOARD_SIZE[0] // scale) > 20 // scale or abs(h - BOARD_SIZE[1] // scale) > 20 // scale
            or area < .7 * w * h):
        return None
    return int(x) * scale, int(y) * scale


def white_target(frame: np.ndarray, previous: Optional[np.ndarray] = None):
    """还没淡出的白色图形：只认面积足够大的白色连通域，排除说明窗口和小号倒计时文字。
    返回 (掩码, 重心, 面积) 或 None。训练标注（tools/lie_recording_labels.py）和形状先验（ShapePrior）共用。"""
    h, w = frame.shape[:2]
    mask = np.uint8(np.min(frame, axis=2) > 238) * 255
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    reference = previous if previous is not None else np.array([w / 2, h / 2])
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not .008 * w * h < area < .09 * w * h:
            continue
        if bw > .4 * w or bh > .45 * h or area / (bw * bh) < .35:
            continue
        if not .25 * h < centers[i, 1] < .85 * h:
            continue
        if previous is not None and np.linalg.norm(centers[i] - previous) > .2 * w:
            continue
        candidates.append((float(np.linalg.norm(centers[i] - reference)), i))
    if not candidates:
        return None
    _, i = min(candidates)
    return np.uint8(labels == i) * 255, centers[i], int(stats[i, cv2.CC_STAT_AREA])


FAKE_CROSSHAIRS = 3
"""藏准星：先把真准星（连同外面一圈）原样复制到画板上这么多个随机位置，再把所有准星用同一种办法抹掉。
只抹真准星时，抹完那块总留着肉眼看不出的痕迹（描边、阴影、压缩纹），模型学会了「找准星」而不是找目标——
玩家鼠标瞬移时框跟着准星跳走、原地还看得见的目标不管。贴了假准星后，痕迹到处都有，不再指向目标。
训练（tools/lie_recording_labels.py）和实际运行都用这一份。"""


def find_crosshair(board, cursor) -> Optional[np.ndarray]:
    """光标附近绿色准星的中心（准星中心和光标热点之间可能有偏移）；没看到返回 None。"""
    h, w = board.shape[:2]
    cx, cy = int(cursor[0]), int(cursor[1])
    x0, y0, x1, y1 = max(0, cx - 45), max(0, cy - 45), min(w, cx + 46), min(h, cy + 46)
    if x0 >= x1 or y0 >= y1:
        return None
    p = board[y0:y1, x0:x1].astype(int)
    ys, xs = np.nonzero((p[..., 1] > 150) & (p[..., 1] > p[..., 2] + 60) & (p[..., 1] > p[..., 0] + 60))
    if len(xs) < 15:
        return None
    return np.array([x0 + xs.mean(), y0 + ys.mean()])


OFFSET_MIN_PX = 20.
"""准星和光标差这么多才算偏移：平时差不到 1px，手猛甩时最近几步的中位数能到约 11px（准星晚一步），偏的局差约 42px"""
OFFSET_SAMPLES = 15
OFFSET_MIN_SAMPLES = 5


class CrosshairOffset:
    """准星相对真光标的偏移。大多数局准星就画在光标处，少数局（20261024_090628、20261101_220530）整局偏 (+30, +30)，
    原因不明。这两局一局光标在目标上、一局准星在目标上，都通过了，看不出游戏判定看哪个点，所以偏的局把目标放在两者中间；
    不偏的局走位不变。"""

    def __init__(self):
        self._seen = deque(maxlen=OFFSET_SAMPLES)
        self.measured: Optional[np.ndarray] = None  # 最近几步的中位数，样本不够时 None
        self.value: Optional[np.ndarray] = None  # 够大才有，走位按它修正

    def observe(self, aim, cursor):
        """aim：这一步看到的准星；cursor：同一帧的光标，同一个坐标系。任一个没有就跳过。"""
        if aim is None or cursor is None:
            return
        self._seen.append((aim[0] - cursor[0], aim[1] - cursor[1]))
        if len(self._seen) >= OFFSET_MIN_SAMPLES:
            self.measured = np.median(self._seen, axis=0)
            self.value = self.measured if np.linalg.norm(self.measured) >= OFFSET_MIN_PX else None

    def cursor_spot(self, target) -> np.ndarray:
        """要让目标落在光标和准星中间，光标该放哪。"""
        return np.asarray(target, float) - (0. if self.value is None else self.value / 2)


def crosshair_mask(board, center):
    """准星（绿色圆环 + 深色描边）在画板图上的掩码；center 是准星中心。"""
    h, w = board.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0, x1, y1 = max(0, cx - 32), max(0, cy - 32), min(w, cx + 33), min(h, cy + 33)
    if x0 >= x1 or y0 >= y1:
        return mask
    p = board[y0:y1, x0:x1].astype(int)
    green = (p[..., 1] > 150) & (p[..., 1] > p[..., 2] + 60) & (p[..., 1] > p[..., 0] + 60)
    if not green.any():
        return mask
    mask[y0:y1, x0:x1] = np.uint8(green) * 255
    cv2.circle(mask, (cx, cy), 21, 255, -1)  # 圆环里面的十字和中心点不是绿色，整块抹
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))  # 带上深色描边


RING_GROW = 3
"""准星绿色部分往外扩这么多 px，带上深色描边"""


def ring_mask(board, center):
    """准星本身（绿色圆环、四根刻度、中心点和深色描边）的掩码：只是准星画到的像素，不像 crosshair_mask 整块抹圆。"""
    h, w = board.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, y0, x1, y1 = max(0, cx - 32), max(0, cy - 32), min(w, cx + 33), min(h, cy + 33)
    if x0 >= x1 or y0 >= y1:
        return mask
    p = board[y0:y1, x0:x1].astype(int)
    green = (p[..., 1] > 150) & (p[..., 1] > p[..., 2] + 60) & (p[..., 1] > p[..., 0] + 60)
    mask[y0:y1, x0:x1] = np.uint8(green) * 255
    return cv2.dilate(mask, np.ones((2 * RING_GROW + 1,) * 2, np.uint8))


def erase_for_model(board, aim, rng, two_frame, avoid_box=None, on_target=False):
    """按模型类型藏准星，返回 (画板, 抹过的地方 bool 掩码或 None)。aim 为 None（没准星）原样返回。
    单帧模型：贴假准星再统一抹掉（hide_crosshair）。
    两帧对比模型：只把准星画到的像素补掉（inpaint），差值图里这些像素清零（motion_input），不贴假准星。
    假准星不用贴，是因为训练时补画的准星放在哪和答案无关（tools/lie_recording_labels.py），抹痕在哪不指向目标；
    这样差值图里只少一圈准星那么大的地方，不再满屏黑洞。"""
    if aim is None:
        return board, None
    if two_frame:
        mask = ring_mask(board, aim)
        if not mask.any():
            return board, None
        return cv2.inpaint(board, mask, 3, cv2.INPAINT_TELEA), mask > 0
    erased = []
    board = hide_crosshair(board, crosshair_mask(board, aim), rng, avoid_box, on_target, erased=erased)
    return board, erased[0]


def _box_mask(shape, box, margin):
    m = np.zeros(shape, bool)
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        m[max(0, y0 - margin):max(0, y1 + margin), max(0, x0 - margin):max(0, x1 + margin)] = True
    return m


def hide_crosshair(board, mask, rng, avoid_box=None, on_target=False, fakes=FAKE_CROSSHAIRS, erased=None):
    """贴 fakes 个假准星，再把真假准星统一抹掉（从 60~110px 外搬纹理盖上，边缘羽化）。
    avoid_box：目标框，搬纹理不从这搬（不然会在别处复制出一块白色「假目标」）；假准星可以贴在上面。
    on_target：真准星压在还看得见的白色目标上，用 inpaint 补成白色（搬纹理会在白纸上打出深色洞）。
    erased：给一个 list 就往里追加这次抹过的地方（真假准星，bool 掩码），两帧对比时差值图里把这些地方清零。"""
    if not mask.any():
        if erased is not None:
            erased.append(np.zeros(mask.shape, bool))
        return board
    h, w = mask.shape
    halo = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0  # 准星外面一圈也一起复制，残留痕迹跟着走
    ys, xs = np.nonzero(halo)
    hy0, hy1, hx0, hx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch, patch_halo = board[hy0:hy1, hx0:hx1].copy(), halo[hy0:hy1, hx0:hx1]
    patch_mask = mask[hy0:hy1, hx0:hx1]
    out = board.copy()
    # 假准星可以贴在目标上：只要真准星落在目标里、假的永远不在，模型就能学「目标中间有块抹痕」
    taken = cv2.dilate(np.uint8(halo), np.ones((21, 21), np.uint8)) > 0
    regions = [mask > 0]
    for _ in range(fakes):
        for _ in range(30):
            x = rng.randrange(0, w - patch.shape[1])
            y = rng.randrange(0, h - patch.shape[0])
            if not taken[y:y + patch.shape[0], x:x + patch.shape[1]].any():
                break
        else:
            continue
        region = out[y:y + patch.shape[0], x:x + patch.shape[1]]
        region[patch_halo] = patch[patch_halo]
        fake = np.zeros((h, w), bool)
        fake[y:y + patch.shape[0], x:x + patch.shape[1]] = patch_mask > 0
        regions.append(fake)
        taken[max(0, y - 10):y + patch.shape[0] + 10, max(0, x - 10):x + patch.shape[1] + 10] = True

    if erased is not None:
        erased.append(np.logical_or.reduce(regions))
    result = out.astype(np.float32)
    if on_target:
        result = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
        regions = regions[1:]
    keep_out = taken | _box_mask(halo.shape, avoid_box, 10)  # 搬纹理不能从任何准星、目标那里搬
    for inside in regions:
        shift = None
        for low, high in ((60, 110), (110, 250)):
            for _ in range(20):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(low, high)
                candidate = (int(dist * np.sin(angle)), int(dist * np.cos(angle)))
                if not (np.roll(keep_out, (-candidate[0], -candidate[1]), axis=(0, 1)) & inside).any():
                    shift = candidate
                    break
            if shift is not None:
                break
        if shift is None:
            # 不能使用最后一次失败的采样，它可能把另一个绿准星搬进来。
            result = cv2.inpaint(np.uint8(np.clip(result, 0, 255)),
                                 np.uint8(inside) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
            continue
        source = np.roll(result, (-shift[0], -shift[1]), axis=(0, 1))
        alpha = cv2.GaussianBlur(np.float32(inside), (7, 7), 0)[..., None]
        result = result * (1 - alpha) + source * alpha
    return np.uint8(np.clip(result, 0, 255))


MOTION_STEPS = 2
"""两帧对比模型除了当前帧，还看前 1、2 步（每步 1/RATE 秒）的画面"""
DIFF_GAIN = 4.
"""对齐相减后的差乘这么多倍再存成 0~255（花纹残差一般十几，目标轮廓二三十）"""


ERASED_MARGIN = 2
"""抹准星的地方在差值图里清零时往外多清这么多 px（inpaint 边缘、对齐有亚像素误差）"""


def aligned_diff(prev_gray, gray, prev_erased=None):
    """画板花纹整体在转圈平移（每 1/15 秒挪 5~10px），淡出的目标混在里面单看一帧认不出。
    把前一帧按整体移动量对齐到当前帧再相减，花纹抵消，跟花纹走得不一样的目标轮廓就亮出来。
    返回 (差, 前一帧抹准星的地方对齐后的掩码或 None)。"""
    (dx, dy), _ = cv2.phaseCorrelate(prev_gray, gray)
    warp = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(prev_gray, warp, gray.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    erased = None if prev_erased is None else cv2.warpAffine(np.uint8(prev_erased), warp, gray.shape[::-1]) > 0
    return np.abs(gray - moved), erased


def motion_input(board, history: Sequence[Optional[np.ndarray]], erased: Sequence[Optional[np.ndarray]] = ()):
    """两帧对比模型的输入（3 通道，训练和实际运行共用）：当前帧灰度、和前 1 步 / 前 2 步对齐相减的差。
    board 和 history 都是已经藏过准星的画板图，history 新的在前；缺的那步当作没变化（全 0）。
    erased：[当前帧, 前 1 步, 前 2 步] 抹准星的地方（erase_for_model 返回的），差值图里清零——
    补出来的那块两帧里不一样，准星又不跟着花纹走，相减后是很亮的一圈，比淡出的目标亮得多。"""
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
    erased = list(erased) + [None] * (MOTION_STEPS + 1 - len(erased))
    kernel = np.ones((2 * ERASED_MARGIN + 1,) * 2, np.uint8)
    channels = [gray]
    for k in range(MOTION_STEPS):
        prev = history[k] if k < len(history) else None
        if prev is None or prev.shape != board.shape:
            channels.append(np.zeros_like(gray))
            continue
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff, prev_erased = aligned_diff(prev_gray, gray, erased[k + 1])
        blank = np.zeros(gray.shape, bool)
        for m in (erased[0], prev_erased):
            if m is not None:
                blank |= m
        if blank.any():
            diff[cv2.dilate(np.uint8(blank), kernel) > 0] = 0
        channels.append(diff * DIFF_GAIN)
    return np.uint8(np.clip(np.dstack(channels), 0, 255))


def model_input_kind(model_path: str) -> str:
    """模型旁边 model_info.json 的 input_kind：single（看原图，默认）、two_frame（看 motion_input）
    或 two_frame_prior（motion_input + 形状先验，4 通道，见 with_prior）。"""
    try:
        with open(os.path.join(os.path.dirname(model_path), "model_info.json"), encoding="utf-8") as f:
            return str(json.load(f).get("input_kind", "single"))
    except (OSError, ValueError):
        return "single"


SPIN_DEG_PER_S = -114.7
"""图形转速（度/秒，cv2.getRotationMatrix2D 的方向，负数是顺时针）。倒计时时量的（tools/lie_recording_labels.measure_spin）：
买来视频每段 -114~-115.3，自己的 3 段录像 -114.7~-114.8；淡出后照样匀速转、大小不变"""
FULL_AREA = .9
"""白色面积到目前最大值的这么多倍才当完整图形记下（淡出途中、被字挡住一块时不记）"""


def _fill_holes(mask):
    """准星（抹过的地方）、字压出来的洞补上，只留外轮廓。"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


class ShapePrior:
    """形状先验：倒计时时白色图形看得清，记下它的样子（以重心为中心的掩码）和时刻；之后图形按 SPIN_DEG_PER_S
    匀速转、大小不变，任何时刻都能把它转到当时的角度，画在给定位置（上一步算出的目标位置）。
    模型多看这一张剪影，就知道这一局要找的是什么形状、大概在哪。训练（tools/lie_recording_labels.py）和实际运行共用。"""

    def __init__(self, spin: float = SPIN_DEG_PER_S):
        self.spin = spin
        self.template: Optional[np.ndarray] = None
        self.taken_at: Optional[float] = None
        self._max_area = 0

    def observe(self, board, t: float, near=None) -> bool:
        """画板上还看得见完整白色图形就记下（near：上一步的位置，只认它附近的）。返回这次有没有记。"""
        seen = white_target(board, None if near is None else np.asarray(near, float))
        if seen is None:
            return False
        mask, _, area = seen
        self._max_area = max(self._max_area, area)
        if area < FULL_AREA * self._max_area:
            return False
        mask = _fill_holes(mask)
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean(), ys.mean()
        size = int(2 * np.hypot(xs - cx, ys - cy).max()) + 8
        shift = np.float32([[1, 0, size / 2 - cx], [0, 1, size / 2 - cy]])
        self.template = cv2.warpAffine(mask, shift, (size, size), flags=cv2.INTER_NEAREST)
        self.taken_at = t
        return True

    def render(self, center, t: float, shape=(BOARD_SIZE[1], BOARD_SIZE[0]), turn: float = 0.) -> np.ndarray:
        """t 时刻的图形剪影（0/255）画在 center（重心）处；还没记下图形或 center 为 None 返回全 0。turn：多转几度（训练加扰动）。"""
        if self.template is None or center is None:
            return np.zeros(shape, np.uint8)
        size = self.template.shape[0]
        warp = cv2.getRotationMatrix2D((size / 2, size / 2), self.spin * (t - self.taken_at) + turn, 1.)
        warp[:, 2] += np.asarray(center, float) - size / 2
        return cv2.warpAffine(self.template, warp, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)


def with_prior(image, prior):
    """形状先验模型的输入（4 通道，按网络看到的顺序）：motion_input 的 3 个通道按 3 通道模型看到的顺序
    （ultralytics 把 3 通道图当 BGR 翻成 RGB 再送进网络，4 通道不翻）+ 先验剪影。
    这样第 4 通道权重为 0 时和原来的两帧对比模型完全一样，可以从它接着训。"""
    return np.dstack([image[..., ::-1], prior])


class TargetFinder:
    """一帧画板 → 目标位置：藏准星 → 模型输入（单帧 / 两帧对比 / 再加形状先验）→ 出框 → 运动预测跟踪。
    LieShadow 和离线闭环评估（tools/lie_closed_loop_eval.py）共用这一份，评估和实际运行不会对不上。"""

    def __init__(self, detector, kind: str = "single", rng: Optional[random.Random] = None):
        self.detector, self.kind = detector, kind
        self.two_frame = kind in ("two_frame", "two_frame_prior")
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        """画板没了（这次测谎结束）：跟踪、历史帧、记下的图形都清掉。"""
        self.tracker: Optional[PredictTracker] = None
        self.history = deque(maxlen=MOTION_STEPS)  # 前几步（藏过准星的画板, 抹过的地方），新的在前
        self.shape = ShapePrior() if self.kind == "two_frame_prior" else None
        self.boxes = []

    def step(self, board, aim, t: float):
        """board：裁好的画板图（可能带准星）；aim：准星中心（画板坐标）或 None；t：秒。返回目标位置，还没锁定返回 None。"""
        board, erased = erase_for_model(board, aim, self.rng, self.two_frame)
        image = (motion_input(board, [h[0] for h in self.history], [erased] + [h[1] for h in self.history])
                 if self.two_frame else board)
        self.history.appendleft((board, erased))
        if self.shape is not None:
            last = None if self.tracker is None else self.tracker.pos
            self.shape.observe(board, t, last)
            image = with_prior(image, self.shape.render(last, t, board.shape[:2]))
        self.boxes = [(d.confidence, np.array([d.x + d.w / 2, d.y + d.h / 2])) for d in self.detector.detect(image)]
        if self.tracker is not None:
            return self.tracker.update(self.boxes)
        top = max(self.boxes, key=lambda b: b[0], default=None)
        if top is not None and top[0] >= START_CONF:
            self.tracker = PredictTracker(top[1], BOARD_SIZE)
            return self.tracker.pos.copy()
        return None


MATCH_EDGE_PX = 5
"""形状匹配的轮廓线宽（px）"""
MATCH_WIDE_PX = 31
"""轮廓两边这么宽的一圈算背景，轮廓比背景亮才算匹配"""
MATCH_SPREAD_PX = 15
"""累积时上一步的分数往四周摊开这么多（目标每步最多挪这么多，分数能接上）"""
MATCH_DECAY = .92
"""累积时上一步分数留多少：越大越不容易被单帧的诱饵带走"""
MATCH_FREE_PX = 40
MATCH_JUMP_COST = .05
"""离上一步超过 MATCH_FREE_PX 后，每多 1px 从分数里扣这么多，位置不会随便跳到远处的诱饵上"""
WHITE_OVER_S = .5
"""白色图形消失这么久后不再认白色：它淡出后不会回来，再看到的白色是「成功！」字之类。
量过：8 段实机录像的白色阶段中间都没断过 0.15 秒以上。"""
MODEL_WAIT_S = .5
"""画板出现后这么久还没认出白色图形，才退回模型。之前白色图形还没淡入，模型的框不可信，鼠标不动。
量过：9 段自己的录像画板出现后 0~0.13 秒就认出，72 段买来的视频开头 0.2 秒内就认出"""
SHAPE_SCALE = 1.5
"""形状匹配（对齐差分、响应、累积、跳变代价、取峰）在 1/1.5 分辨率上算，峰值再局部放大细化。
实机每步 90~106ms 超出 15Hz 的 66ms 预算，步频掉到 8~9 步/秒，形状匹配按 15 步/秒调的参数接不上，
20261107 两局测谎因此失败（人工回放标注验收：同一段录像 15Hz 重放 within60 从 55%/56% 回到 98%/97%）。
降分辨率是砍单步耗时的大头：核的轮廓细节也随分辨率变糊，降太狠反而引入新的诱饵捕获——
20261107 两段标注录像上扫过 1/1、1/1.25、1/1.5、1/2（within60：98/97 → 98/94 → 98/95 → 97/89，
单步 45 → 28 → 22 → 13ms），1/1.5 是质量达标的下一档。白图重心、先验记录、抹准星仍是全分辨率。
验收：15Hz 重放两段标注录像 within60 ≥95%，裸跑单步 ≤25ms（实测 19~24ms）。"""


def match_kernel(template, angle: float, scale: int = 1):
    """图形转到 angle 度后的匹配核：轮廓线上为正、两边一圈为负，各自加起来是 ±1，平坦的地方响应是 0。
    scale：模板已是 1/scale 分辨率时，形态学参数按比例缩小。"""
    n = template.shape[0]
    turned = cv2.warpAffine(template, cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1.), (n, n), flags=cv2.INTER_LINEAR)
    mask = np.uint8(turned > 127)
    edge_px = max(1, round(MATCH_EDGE_PX / scale))
    wide_px = max(1, round(MATCH_WIDE_PX / scale))
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((edge_px,) * 2, np.uint8)).astype(np.float32)
    wide = np.ones((wide_px,) * 2, np.uint8)
    around = np.clip(cv2.dilate(mask, wide).astype(np.float32) - cv2.erode(mask, wide).astype(np.float32) - edge, 0, 1)
    return edge / max(edge.sum(), 1) - around / max(around.sum(), 1)


class ShapeFinder:
    """形状匹配找目标，接口和 TargetFinder 一样。图形大小不变、匀速转（ShapePrior），每一步都知道它现在长什么样，
    拿这个轮廓在两帧对比的差值图（motion_input）上找：花纹对齐相减后基本抵消，诱饵的轮廓也对不上形状。
    单帧证据很弱，分数按步累积（摊开一点再衰减），取最高的地方。不用模型，也就不会学到准星、不会认准自己放的位置。
    图形还白着就直接用重心；画板出现 MODEL_WAIT_S 后还没记到图形（比如白着的时候不在 white_target 认的范围里）才交给 fallback。"""

    def __init__(self, fallback: TargetFinder, rng: Optional[random.Random] = None):
        self.fallback = fallback
        self.rng = rng or random.Random()
        spread_small = max(3, round(MATCH_SPREAD_PX / SHAPE_SCALE)) | 1  # 摊开核按分辨率缩，保持奇数
        self.spread_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spread_small,) * 2)
        self._tpl_small: Optional[np.ndarray] = None  # 半分辨率的图形模板（记下图形后重建一次）
        self._ogrid: Optional[Tuple[np.ndarray, np.ndarray]] = None  # 半分辨率坐标网格，画板尺寸变了才重建
        self.reset()

    def reset(self):
        self.fallback.reset()
        self.history = deque(maxlen=MOTION_STEPS)
        self.shape = ShapePrior()
        self.score = None
        self.pos: Optional[np.ndarray] = None
        self.boxes = []
        self.started_at: Optional[float] = None
        self.white_at: Optional[float] = None  # 最后一次看到白色图形的时刻
        self._tpl_small = None  # 新的一局图形可能变了，重建
        self.source = ""  # 这一步的位置哪来的：wait（还在等白色图形，没有位置）/ white / shape / model

    def _small(self, board, erased):
        """半分辨率的画板和抹准星掩码（形状匹配专用的差分在这里算；白图/先验/抹准星都用全分辨率原图）。"""
        h, w = board.shape[:2]
        dsize = (int(round(w / SHAPE_SCALE)), int(round(h / SHAPE_SCALE)))
        small = cv2.resize(board, dsize, interpolation=cv2.INTER_AREA)
        small_erased = None if erased is None else cv2.resize(erased.astype(np.uint8), dsize) > 0
        return small, small_erased

    def step(self, board, aim, t: float):
        """board：裁好的画板图（可能带准星）；aim：准星中心（画板坐标）或 None；t：秒。返回目标位置，还没锁定返回 None。"""
        raw = board
        if self.started_at is None:
            self.started_at = t
        board, erased = erase_for_model(board, aim, self.rng, True)
        small, erased_small = self._small(board, erased)
        image = motion_input(small, [h[0] for h in self.history], [erased_small] + [h[1] for h in self.history])
        self.history.appendleft((small, erased_small))
        white_over = self.white_at is not None and t - self.white_at > WHITE_OVER_S
        seen = None if white_over else white_target(board, self.pos)
        if seen is not None:
            self.white_at = t
        if not white_over:
            self.shape.observe(board, t, self.pos)
        self.boxes = []
        if seen is not None:
            self.source, self.score = "white", None
            self.pos = np.array(seen[1], float)
            return self.pos.copy()
        if self.shape.template is None:
            if t - self.started_at < MODEL_WAIT_S:
                self.source = "wait"
                return None
            self.source = "model"
            found = self.fallback.step(raw, aim, t)
            self.boxes = self.fallback.boxes
            return found
        self.source = "shape"
        h, w = image.shape[:2]
        if self._ogrid is None or self._ogrid[0].shape[0] != h:
            self._ogrid = np.ogrid[:h, :w]
        ys, xs = self._ogrid
        diff = cv2.GaussianBlur(np.float32(image[..., 1]) + np.float32(image[..., 2]), (3, 3), 0)
        if self._tpl_small is None:
            self._tpl_small = cv2.resize(self.shape.template, (0, 0), fx=1. / SHAPE_SCALE, fy=1. / SHAPE_SCALE,
                                         interpolation=cv2.INTER_AREA)
        kernel = match_kernel(self._tpl_small, self.shape.spin * (t - self.shape.taken_at), scale=SHAPE_SCALE)
        response = cv2.filter2D(diff, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        response = (response - response.mean()) / (response.std() + 1e-6)
        self.score = response if self.score is None else cv2.dilate(self.score, self.spread_small) * MATCH_DECAY + response
        ranked = self.score
        if self.pos is not None:
            ranked = ranked - (MATCH_JUMP_COST / SHAPE_SCALE) * np.clip(
                np.hypot(xs - self.pos[0] / SHAPE_SCALE, ys - self.pos[1] / SHAPE_SCALE) - MATCH_FREE_PX / SHAPE_SCALE, 0, None)
        y, x = np.unravel_index(np.argmax(ranked), ranked.shape)
        # 半分辨率峰值局部 ×4 放大再取一次：定位精度损失从 2px 降到半像素以内
        x0, y0 = max(0, x - 2), max(0, y - 2)
        crop = ranked[y0:y0 + 5, x0:x0 + 5]
        big = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        by2, bx2 = np.unravel_index(np.argmax(big), big.shape)
        fx = (x0 * SHAPE_SCALE + bx2 * SHAPE_SCALE / 4.)
        fy = (y0 * SHAPE_SCALE + by2 * SHAPE_SCALE / 4.)
        self.pos = np.array([np.clip(fx, 0, w * SHAPE_SCALE - 1), np.clip(fy, 0, h * SHAPE_SCALE - 1)])
        return self.pos.copy()


CLICK_POLL_S = .01
"""每隔这么久看一次鼠标左键。一下点击只有 50ms 左右（录像里量的），算一步要 50ms 上下，只在每步看会漏掉"""
MOVE_SUBSTEPS = 3
"""每步（1/15 秒）分几小步挪过去，准星不会一跳一跳的"""


class MouseTakeover:
    """测谎默认由程序把鼠标移到影子算的位置；你点一下鼠标左键才交还给你，这次测谎不再接管。
    光有光标移动不算：远程连接会把光标瞬移走，手也可能碰到鼠标，这些都不该让程序撒手（下一步就把光标移回目标上）。
    read() 返回光标客户区坐标 (x, y, 左键按着没) 或 None；move(x, y) 把光标移到客户区 (x, y)，游戏窗口不在前台返回 False。
    state: waiting（还没锁定目标）→ program（程序拿着鼠标）/ user（你点了左键，只录制）。"""

    def __init__(self, read, move, on_log=None):
        self._read, self._move = read, move
        self._log = on_log or (lambda m: None)
        self.state = "waiting"
        self.moved = False  # 程序动过鼠标
        self._ref: Optional[np.ndarray] = None  # 程序上次放的位置
        self._released = False  # 看到过左键松开：一开始就按着的不算这次点的
        self._clicked = False
        self._away = False  # 游戏窗口不在前台，已经写过日志

    def poll(self):
        """看一眼左键：松开过之后又按下，就是你点了。"""
        cur = self._read()
        if cur is None:
            return
        if not cur[2]:
            self._released = True
        elif self._released:
            self._clicked = True

    def watch(self, stop: threading.Event):
        """后台一直看左键，直到 stop 或已经交还（LieShadow 开一个线程跑）。"""
        while self.state != "user" and not self._clicked and not stop.wait(CLICK_POLL_S):
            self.poll()

    def _give_back_if_clicked(self) -> bool:
        self.poll()
        if not self._clicked:
            return False
        self.state = "user"
        self._log("[测谎接管] 检测到你点了鼠标左键，鼠标交还给你，这次测谎只录制")
        return True

    def step(self, target, deadline: float):
        """target：这步要去的客户区坐标（None = 还没锁定，不动）；在 deadline（perf_counter）前分几小步挪完。"""
        if self.state == "user" or self._give_back_if_clicked() or target is None:
            return
        target = np.asarray(target, float)
        start = target if self.state == "waiting" else self._ref
        pause = max(0., deadline - time.perf_counter()) / MOVE_SUBSTEPS
        for k in range(1, MOVE_SUBSTEPS + 1):
            if k > 1:
                time.sleep(pause)
                if self._give_back_if_clicked():
                    return
            spot = np.round(start + (target - start) * k / MOVE_SUBSTEPS)
            if not self._move(int(spot[0]), int(spot[1])):
                if not self._away:
                    self._log("[测谎接管] 游戏窗口不在前台，先不动鼠标，回到前台接着跟")
                self._away = True
                return
            self._away = False
            self._ref, self.moved = spot, True
            if self.state == "waiting":
                self.state = "program"
                self._log("[测谎接管] 程序接管鼠标跟目标（点一下鼠标左键才交还给你）")


class LieShadow:
    """用法：shadow.start(out_dir) → 每帧 shadow.offer(frame, cursor)（很快，只存最新一帧）→ shadow.stop()。"""

    def __init__(self, model_path: str, conf: float = .1,
                 on_log: Optional[Callable[[str], None]] = None):
        self.model_path = model_path
        self.conf = conf
        self.takeover: Optional[Tuple[Callable, Callable]] = None  # (read, move)：设了就按 MouseTakeover 的规则接管鼠标
        self.took_over = False  # 这次测谎程序拿过鼠标（录像里的光标就不是你跟的了，不能当答案）
        self.crosshair_offset: Optional[Tuple[int, int]] = None  # 这次量到的准星相对光标的偏移（见 CrosshairOffset）
        self._log = on_log or (lambda m: None)
        self._detector = None
        self._two_frame = False
        self._kind: Optional[str] = None  # model_info.json 的 input_kind；没设时按 _two_frame
        self._rng = random.Random()
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, out_dir: str):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._latest = None
        self.took_over = False
        self.crosshair_offset = None
        self._thread = threading.Thread(target=self._loop, args=(out_dir,), daemon=True)
        self._thread.start()
        self._log("[测谎影子] 后台开始算目标位置" + ("（程序接管鼠标，点一下左键才交还给你）" if self.takeover else "（只记录，不动鼠标）"))

    def offer(self, index: int, t: float, frame: np.ndarray, cursor):
        with self._lock:
            self._latest = (index, t, frame, cursor)

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.)

    def _loop(self, out_dir: str):
        try:
            if self._detector is None:
                from .yolo_detector import OnnxDetector
                self._detector = OnnxDetector(self.model_path, conf=self.conf)
                self._kind = model_input_kind(self.model_path)
            kind = self._kind or ("two_frame" if self._two_frame else "single")
            self._log("[测谎影子] 模型看的是" + {
                "two_frame": "两帧对比（只抹准星本身）",
                "two_frame_prior": "两帧对比 + 形状先验（倒计时记下图形，按转速画在上一步的位置）"}.get(
                kind, "单帧原图（假准星统一抹掉）") + "；只在画板出现 0.5 秒后还没认出白色图形时才用，平时用形状匹配")
        except Exception as e:
            self._log(f"[测谎影子] 模型加载失败，不做影子跟踪：{e}")
            return
        mouse = None if self.takeover is None else MouseTakeover(*self.takeover, on_log=self._log)
        if mouse is not None:
            threading.Thread(target=mouse.watch, args=(self._stop,), daemon=True).start()
        finder = ShapeFinder(TargetFinder(self._detector, kind, self._rng), self._rng)
        offset, offset_told = CrosshairOffset(), False
        last_index, gaps, timings, held_from, held_to = -1, [], [], None, None
        with open(os.path.join(out_dir, "shadow.csv"), "w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(["frame", "seconds", "state", "board_left", "board_top", "pred_client_x",
                          "pred_client_y", "aim_client_x", "aim_client_y", "boxes", "top_conf", "infer_ms", "control",
                          "source"])
            while not self._stop.is_set():
                tick = time.perf_counter()
                with self._lock:
                    item = self._latest
                if item is None or item[0] == last_index:
                    time.sleep(.005)
                    continue
                index, t, frame, cursor = item
                last_index = index
                control = "off" if mouse is None else mouse.state
                go_to = None
                origin = board_box(frame)
                if origin is None:
                    finder.reset()
                    out.writerow([index, f"{t:.4f}", "no_board"] + [""] * 9 + [control, ""])
                else:
                    bx, by = origin
                    board = np.ascontiguousarray(frame[by:by + BOARD_SIZE[1], bx:bx + BOARD_SIZE[0]])
                    aim = None if cursor is None else find_crosshair(board, (cursor[0] - bx, cursor[1] - by))
                    offset.observe(aim, None if cursor is None else (cursor[0] - bx, cursor[1] - by))
                    if offset.measured is not None:
                        self.crosshair_offset = (int(round(offset.measured[0])), int(round(offset.measured[1])))
                    if offset.value is not None and not offset_told:
                        offset_told = True
                        self._log(f"[测谎接管] 这局准星画在光标旁边，偏 {self.crosshair_offset}，把目标放在光标和准星中间")
                    t0 = time.perf_counter()
                    pred = finder.step(board, aim, t)
                    timings.append((time.perf_counter() - t0) * 1000)
                    boxes = finder.boxes
                    top = max((b[0] for b in boxes), default=None)
                    if pred is not None and aim is not None and control != "program":  # 程序拿着鼠标时准星就是它自己放的
                        gaps.append(float(np.linalg.norm(pred - aim)))
                    out.writerow([index, f"{t:.4f}", "tracking" if pred is not None else "searching", bx, by,
                                  "" if pred is None else round(pred[0] + bx, 1),
                                  "" if pred is None else round(pred[1] + by, 1),
                                  "" if aim is None else round(aim[0] + bx, 1),
                                  "" if aim is None else round(aim[1] + by, 1),
                                  len(boxes), "" if top is None else round(float(top), 3),
                                  round(timings[-1], 1), control, finder.source])
                    if pred is not None:
                        spot = offset.cursor_spot(pred)
                        go_to = (spot[0] + bx, spot[1] + by)
                        if control == "program":
                            held_from = t if held_from is None else held_from
                            held_to = t
                if mouse is not None:
                    mouse.step(go_to, tick + 1. / RATE)
                    self.took_over = mouse.moved
                wait = 1. / RATE - (time.perf_counter() - tick)
                if wait > 0:
                    time.sleep(wait)
        if gaps:
            g = np.array(gaps)
            self._log(f"[测谎影子] 算了 {len(g)} 帧，程序位置离你的准星中位 {np.median(g):.0f}px，"
                      f"60px 内 {np.mean(g < 60):.0%}，推理中位 {np.median(timings):.0f}ms（详见 shadow.csv）")
        elif held_from is None:
            self._log("[测谎影子] 没算出目标位置（没找到画板或模型没出框），详见 shadow.csv")
        if held_from is not None:
            self._log(f"[测谎接管] 程序拿着鼠标跟了 {held_to - held_from:.1f} 秒（shadow.csv 的 control 列）")
