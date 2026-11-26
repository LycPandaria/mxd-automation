"""从通过了测谎的录像（LieRecorder 录的）生成「只框真目标」的 YOLO 训练数据；不发送键盘或鼠标输入。

例如：
    python tools/lie_recording_labels.py lie_recordings/20261024_090628 lie_recordings/<另一段> \
        --output train/lie_target_dataset

卖家的 yolo26n12 只有一类、画板上每个像目标的形状都框（每张 7~21 个框），分不出哪个是真的。
这里每张图只标一个框：
  - 白色目标还看得见时：框 = 白色区域的外接框；
  - 淡出途中半透明、还能看出一块发白：框 = 这块发白区域（玩家准星这时常常还落后一截）；
  - 完全淡出以后：框中心 = 玩家的准星中心（光标 + 准星偏移，玩家通过了测谎，说明一直套在目标上），
    框大小 = 看得见时白色区域外接框的中位数。
游戏把光标画成绿色准星，正好盖在目标上，模型会学成「找准星」；只抹掉也不行，抹完总有看不出的残留，
模型照样学会找它。所以先把真准星原样复制到别处贴几个假的，再统一抹掉（hide_crosshair）。
买来的视频没有准星，要补画一个；补画的大多不放在目标上（OFF_TARGET_CROSSHAIR），不然抹痕永远在答案正中，
模型又会学成「找抹痕」（在花纹画板上假准星带过去的一圈底纹看得出来，真抹痕就能和假的分开）。
放在目标上的那些也随机偏开 ON_TARGET_JITTER_PX：淡出后的答案就是人瞄的位置，准星正盖在答案中心的话，
模型会学成「目标上有抹痕就框抹痕中心」，程序自己拿鼠标时准星总落后目标一步，框就被拽回上一步的位置。
光标一帧跳出很远（鼠标拿起来放下）后的一小段，准星不在目标上，不拿来当答案。
two_frame=True 时存的不是原图，而是两帧对比模型的输入（lie_shadow.motion_input）：前 1/15、2/15 秒那两帧
也照样补画准星、藏准星（补画的准星跟着人瞄的位置走、和当前帧保持同样的偏移，像程序拿着鼠标在后面跟）。
两帧对比只抹准星画到的像素、不贴假准星（lie_shadow.erase_for_model）；为此补画的准星放在哪和答案无关
（two_frame_aim）：大部分随便放，一部分放在目标附近随机偏 0~NEAR_TARGET_PX，让模型见过准星压在目标上、
又学不到「准星在哪目标就在哪」。
自己的录像（own=True）准星正套在答案上，抹了也留痕迹，直接训练模型会学成「框抹痕」：程序拿着鼠标时抹痕就是上一步的位置，
框跟着原地不动。所以先用别的帧的花纹把真准星补掉（clean_crosshairs），之后和买来的视频一样补画准星（位置和答案无关）。
prior=True（--prior）时再加第 4 个通道「形状先验」（lie_shadow.ShapePrior / with_prior）：倒计时记下的白色图形，
按固定转速转到这一帧的角度，画在上一步答案附近（实际运行时是跟踪器上一步算的位置）；随机偏开、转歪，
一部分不给、一部分滑开 60~150px、一部分放错地方（training_prior），存成 4 通道 PNG，每张的先验种类记在 priors.csv。
买来视频里目标很少贴边，edge_every 让贴边的帧多取几张。
只用游戏判定「成功通过」的录像，失败的录像鼠标没跟上目标，不能当答案。
"""
from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import random
from pathlib import Path

import cv2
import numpy as np

# 藏准星、找画板、认白色图形、形状先验都和实际运行共用 src/perception/lie_shadow.py 里的同一份；按文件路径加载，
# 绕开 src.perception/__init__ 里只在 Windows 上有的模块（Windows 训练包里也只带这一个文件）
_spec = importlib.util.spec_from_file_location(
    "lie_shadow", Path(__file__).resolve().parent.parent / "src" / "perception" / "lie_shadow.py")
_shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shadow)
erase_for_model = _shadow.erase_for_model
motion_input, MOTION_STEPS, RATE = _shadow.motion_input, _shadow.MOTION_STEPS, _shadow.RATE
BOARD_SIZE, board_box, white_target = _shadow.BOARD_SIZE, _shadow.board_box, _shadow.white_target
ShapePrior, with_prior = _shadow.ShapePrior, _shadow.with_prior
JPEG_444 = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
"""两帧对比的 3 个通道不是颜色，不能按 4:2:0 压色度（会把两个差值通道糊掉、混在一起）"""

TELEPORT_PX = 100
"""光标一帧跳这么远（px）不是在跟目标（鼠标拿起来放下、或者玩家走神），之后 TELEPORT_SKIP 帧不用准星当答案"""
TELEPORT_SKIP = 20
OFF_TARGET_CROSSHAIR = .7
"""补画准星时有这么大比例放在离目标 OFF_TARGET_MIN_PX 以外的随机位置，抹痕在不在目标上不再能当线索"""
OFF_TARGET_MIN_PX = 80
CROSSHAIR_MARGIN = 35
NEAR_TARGET_CROSSHAIR = .4
"""两帧对比：补画准星有这么大比例放在目标附近（离答案中心 0~NEAR_TARGET_PX，方向随机），其余随便放"""
NEAR_TARGET_PX = 60
CROSSHAIR_GREEN, CROSSHAIR_DARK = (30, 238, 120), (20, 45, 20)
ON_TARGET_JITTER_PX = (15, 30)
"""补画在目标上的准星离答案中心这么远（px，方向随机），像程序跟在目标后面差一点"""


CLEAN_FRAME_GAPS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96)
"""补真准星时依次试前后隔这么多帧（按 30 帧/秒算）的画面：花纹在平移，隔一会儿准星底下的花纹就露出来了。
花纹是转圈平移，净位移有限，隔得近的帧准星常常还盖着同一块花纹，所以要试到隔 3 秒"""
CLEAN_TRIES = 3
"""试到这么多个能用的就挑最好的（每试一个要做一次整张的相位相关）"""
CLEAN_MAX_ERROR = 60.
"""准星周围一圈和对齐后的别的帧差多少（每通道平均）算对不上，改用 inpaint。不能卡太严：准星正套着目标，
周围有目标的轮廓、还在转，正常就差 30~50（不贴着目标的地方约 12）"""
CLEAN_PALE_SHARE = .3
"""准星周围一圈这么大比例是白色或半透明的发白（比淡出途中的目标判据 pale_target 一样）：压在图形上，别的帧只有花纹，
补过去是一个花纹洞，改用 inpaint 补成白色"""
CLEAN_NOISE = 3.3
"""补上去的花纹经过插值比真画面平滑，差值图里是一块偏暗的圆（在假位置量：只有原样的 0.4~0.5 倍），位置又正好在答案上，
所以加这么大的噪声（灰度标准差）把差值补回原样的水平"""
RESULT_CUT_S = 1.5
"""「成功！」字在画板关掉前约 1 秒变亮（淡入更早一点），之后测谎结束、玩家不再跟目标：自己的录像最后这么多秒不算答案"""


def _green(board):
    p = board.astype(int)
    return (p[..., 1] > 150) & (p[..., 1] > p[..., 2] + 60) & (p[..., 1] > p[..., 0] + 60)


def clean_crosshairs(boards, fps):
    """把录像里的真准星抹干净（原地改）：boards 是每帧的画板图，没画板的帧是 None。
    画板花纹整体平移，把别的帧对齐过来，准星底下的花纹就有了，比 inpaint 接近真实（Mac 上在不贴着目标的地方量：
    补的地方和真实差 12，相邻两步对齐相减的自然差 8，inpaint 25）。压在白色图形上时别的帧只有花纹，用 inpaint。
    返回 {temporal: 用别的帧补的帧数, inpaint: 用 inpaint 的帧数}。"""
    scale = max(1, round(fps / 30))
    gaps = [sign * gap * scale for gap in CLEAN_FRAME_GAPS for sign in (-1, 1)]
    h, w = BOARD_SIZE[1], BOARD_SIZE[0]
    gray = lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    report = {"temporal": 0, "inpaint": 0}
    noise = np.random.default_rng(0)
    for i, board in enumerate(boards):
        if board is None:
            continue
        ys, xs = np.nonzero(_green(board))
        if len(xs) < 15:
            continue
        mask = _shadow.ring_mask(board, (xs.mean(), ys.mean()))
        wide = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ring = (cv2.dilate(mask, np.ones((17, 17), np.uint8)) > 0) & (wide == 0)
        around = board[ring].astype(int)
        on_shape = np.mean((around.min(axis=1) > 185) & (around.max(axis=1) - around.min(axis=1) < 45)) > CLEAN_PALE_SHARE
        here, best, tried = gray(board), None, 0
        for gap in () if on_shape else gaps:
            j = i + gap
            if not 0 <= j < len(boards) or boards[j] is None:
                continue
            (dx, dy), _ = cv2.phaseCorrelate(gray(boards[j]), here)
            warp = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(boards[j], warp, (w, h), borderMode=cv2.BORDER_REPLICATE)
            valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), warp, (w, h))
            source_ring = cv2.warpAffine(np.uint8(_green(boards[j])) * 255, warp, (w, h))
            area = wide > 0
            if (valid[area] < 255).any() or (source_ring[area] > 0).any() or (moved[area].min(axis=1) > 238).any():
                continue  # 对过来的地方出了画板、盖着那一帧的准星或白色图形
            error = float(np.abs(moved[ring].astype(float) - board[ring].astype(float)).mean())
            tried += 1
            if best is None or error < best[0]:
                best = (error, moved)
            if tried == CLEAN_TRIES:
                break
        if best is None or best[0] > CLEAN_MAX_ERROR:
            board[:] = cv2.inpaint(board, mask, 3, cv2.INPAINT_TELEA)
            report["inpaint"] += 1
            continue
        alpha = np.maximum(cv2.GaussianBlur(np.float32(wide > 0), (5, 5), 0), np.float32(mask > 0))[..., None]
        patch = best[1] + noise.normal(0, CLEAN_NOISE, (h, w, 1))
        board[:] = np.uint8(np.clip(board * (1 - alpha) + patch * alpha, 0, 255))
        report["temporal"] += 1
    return report


def add_training_crosshair(board, aim):
    """照游戏里的绿准星画：绿圆环（半径 10~14）+ 四根刻度（里到 6、外到 16）+ 中心点，外面一圈深色描边。"""
    board = board.copy()
    x, y = (int(round(v)) for v in aim)
    for color, ring, tick, dot in ((CROSSHAIR_DARK, 8, 2, 2), (CROSSHAIR_GREEN, 4, 1, 1)):
        cv2.circle(board, (x, y), 12, color, ring)
        inner, outer = (3, 18) if color == CROSSHAIR_DARK else (6, 16)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ax, ay = x + dx * inner, y + dy * inner
            bx, by = x + dx * outer, y + dy * outer
            cv2.rectangle(board, (min(ax, bx) - tick * abs(dy), min(ay, by) - tick * abs(dx)),
                          (max(ax, bx) + tick * abs(dy), max(ay, by) + tick * abs(dx)), color, -1)
        cv2.rectangle(board, (x - dot, y - dot), (x + dot, y + dot), color, -1)
    return board


def off_target_aim(rng, aim, width, height):
    """画板里离 aim 至少 OFF_TARGET_MIN_PX 的随机位置（补画准星用）。"""
    while True:
        spot = np.array([rng.uniform(CROSSHAIR_MARGIN, width - CROSSHAIR_MARGIN),
                         rng.uniform(CROSSHAIR_MARGIN, height - CROSSHAIR_MARGIN)])
        if np.hypot(*(spot - aim)) >= OFF_TARGET_MIN_PX:
            return spot


def two_frame_aim(rng, aim, width, height):
    """两帧对比补画准星的位置，和答案 aim 无关：NEAR_TARGET_CROSSHAIR 的比例放在目标附近随机偏开，其余随便放。
    返回 (位置, 是否在目标附近)。"""
    if rng.random() < NEAR_TARGET_CROSSHAIR:
        angle, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, NEAR_TARGET_PX)
        spot = np.asarray(aim, float) + r * np.array([np.cos(angle), np.sin(angle)])
        return np.clip(spot, [CROSSHAIR_MARGIN] * 2, [width - CROSSHAIR_MARGIN, height - CROSSHAIR_MARGIN]), True
    return np.array([rng.uniform(CROSSHAIR_MARGIN, width - CROSSHAIR_MARGIN),
                     rng.uniform(CROSSHAIR_MARGIN, height - CROSSHAIR_MARGIN)]), False


def jittered_aim(rng, aim, width, height):
    """aim 往随机方向偏开 ON_TARGET_JITTER_PX（补画在目标上的准星用），不出画板。"""
    angle, r = rng.uniform(0, 2 * np.pi), rng.uniform(*ON_TARGET_JITTER_PX)
    spot = np.asarray(aim, float) + r * np.array([np.cos(angle), np.sin(angle)])
    return np.clip(spot, [CROSSHAIR_MARGIN, CROSSHAIR_MARGIN], [width - CROSSHAIR_MARGIN, height - CROSSHAIR_MARGIN])


PRIOR_NONE = .1
"""形状先验：这么大比例不给先验（全 0，像刚开始还没锁定），模型得自己找"""
PRIOR_WRONG = .1
"""这么大比例把剪影放在离答案 PRIOR_WRONG_MIN_PX 以外的随机位置（像跟踪器跟丢了），模型得学会不照抄先验"""
PRIOR_WRONG_MIN_PX = 150
PRIOR_DRIFT = .15
"""这么大比例把剪影放在离答案 PRIOR_DRIFT_PX 的地方（跟踪器跟着剪影慢慢滑开了一截）。
20261026_125240 里剪影滑开 100px 左右后模型就在剪影处出高分框、自己确认自己；之前训练里没有 60~150px 这一档"""
PRIOR_DRIFT_PX = (60, 150)
PRIOR_NEAR_PX, PRIOR_FAR_PX = 20, 60
"""其余放在上一步答案处再随机偏开：六成 0~PRIOR_NEAR_PX，四成 PRIOR_NEAR_PX~PRIOR_FAR_PX（实际运行时跟踪器也有误差）"""
EDGE_PX = 80
"""答案离画板边这么近（px）的帧多取几张（label_recording 的 edge_every）：买来视频里目标很少贴边，自己的录像里常见"""
PRIOR_TURN_DEG = 10
"""剪影角度再随机多转 ±这么多度（转速外推、记图形那一刻的误差）"""


def training_prior(rng, shape, last, truth, t, size):
    """训练图的形状先验通道：照实际运行，剪影（shape 在 t 时刻的样子）画在上一步的答案 last 处，再加扰动
    （见 PRIOR_*），不让模型学成「剪影在哪框就在哪」。返回 (先验图, 种类 none/wrong/drift/near, 剪影离答案多远或 None)。"""
    roll = rng.random()
    if shape.template is None or last is None or roll < PRIOR_NONE:
        return shape.render(None, t, size), "none", None
    if roll < PRIOR_NONE + PRIOR_WRONG:
        while True:
            spot = np.array([rng.uniform(0, size[1]), rng.uniform(0, size[0])])
            if np.hypot(*(spot - truth)) >= PRIOR_WRONG_MIN_PX:
                break
        kind = "wrong"
    elif roll < PRIOR_NONE + PRIOR_WRONG + PRIOR_DRIFT:
        for _ in range(20):  # 尽量落在画板里，剪影不被切掉
            angle = rng.uniform(0, 2 * np.pi)
            spot = np.asarray(truth, float) + rng.uniform(*PRIOR_DRIFT_PX) * np.array([np.cos(angle), np.sin(angle)])
            if 0 <= spot[0] < size[1] and 0 <= spot[1] < size[0]:
                break
        kind = "drift"
    else:
        r = rng.uniform(0, PRIOR_NEAR_PX) if rng.random() < .6 else rng.uniform(PRIOR_NEAR_PX, PRIOR_FAR_PX)
        angle = rng.uniform(0, 2 * np.pi)
        spot, kind = np.asarray(last, float) + r * np.array([np.cos(angle), np.sin(angle)]), "near"
    mask = shape.render(spot, t, size, rng.uniform(-PRIOR_TURN_DEG, PRIOR_TURN_DEG))
    return mask, kind, float(np.hypot(*(spot - truth)))


def measure_spin(snapshots):
    """倒计时时图形的转速（度/秒）：每张记下的模板和第一张比，找转多少度最重合，再拟合直线。
    用这段视频自己量的，视频帧率标错（cycle_00322 实际 30 帧、文件头写 60）时角度也对得上。
    点太少或拟合不直返回 None，用 lie_shadow.SPIN_DEG_PER_S。"""
    if len(snapshots) < 8:
        return None
    size = max(s[1].shape[0] for s in snapshots)

    def padded(m):
        o = (size - m.shape[0]) // 2
        return cv2.copyMakeBorder(m, o, size - m.shape[0] - o, o, size - m.shape[0] - o, cv2.BORDER_CONSTANT)

    ref = padded(snapshots[0][1])
    t0 = snapshots[0][2]
    angle, t_prev, rate, points = 0., t0, _shadow.SPIN_DEG_PER_S, []
    for _, template, t in snapshots:
        cur = padded(template) > 127

        def overlap(a):
            r = cv2.warpAffine(ref, cv2.getRotationMatrix2D((size / 2, size / 2), a, 1.), (size, size),
                               flags=cv2.INTER_NEAREST) > 127
            return (r & cur).sum() / max(1, (r | cur).sum())

        # 边量边更新转速：帧率标错时第一步只差一倍，±25° 找得到，之后猜得就准了
        guess = angle + rate * (t - t_prev)
        angle, t_prev = max(np.arange(guess - 25, guess + 25.5, 1.), key=overlap), t
        if t > t0:
            rate = angle / (t - t0)
        points.append((t, angle))
    t, a = np.array(points).T
    keep = np.ones(len(t), bool)
    for _ in range(2):  # 倒计时数字压在图形上那几帧量歪了，去掉再拟合
        slope, intercept = np.polyfit(t[keep], a[keep], 1)
        keep = np.abs(a - (slope * t + intercept)) < 6
    residual = a[keep] - (slope * t[keep] + intercept)
    return float(slope) if keep.sum() >= 8 and np.std(residual) < 3 else None


def pale_target(board, aim, size):
    """淡出途中半透明的目标：比纯白暗、但仍然发白、不带金色。在准星附近找，返回外接框或 None。"""
    p = board.astype(int)
    pale = np.uint8((p.min(axis=2) > 185) & (p.max(axis=2) - p.min(axis=2) < 45)) * 255
    pale = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(pale)
    area = size[0] * size[1]
    best = None
    for i in range(1, count):
        x, y, bw, bh, a = stats[i]
        if not .25 * area < a < 1.5 * area or max(bw, bh) > 1.5 * max(size):
            continue
        d = float(np.hypot(*(centers[i] - aim)))
        if d < 1.2 * max(size) and (best is None or d < best[0]):
            best = (d, (x, y, x + bw, y + bh))
    return best[1] if best else None


def crosshair_offset(frames_csv, video):
    """游戏画的绿色准星中心相对光标热点的偏移（取中位数）。"""
    with open(frames_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cap = cv2.VideoCapture(str(video))
    offsets, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = int(rows[i]["video_x"] or -1), int(rows[i]["video_y"] or -1)
        if 40 <= cx < frame.shape[1] - 40 and 40 <= cy < frame.shape[0] - 40:
            sub = frame[cy - 40:cy + 40, cx - 40:cx + 40]
            ys, xs = np.nonzero((sub[..., 1] > 180) & (sub[..., 0] < 120) & (sub[..., 2] < 120))
            if len(xs) > 15:
                offsets.append((xs.mean() - 40, ys.mean() - 40))
        i += 1
    cap.release()
    return np.median(offsets, axis=0) if offsets else np.zeros(2), len(offsets)


def inside(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def label_recording(recording: Path, out: Path, every: int, rng: random.Random, two_frame: bool = False,
                    prior: bool = False, video: Path = None, edge_every: int = 0, own: bool = False):
    """prior=True：存形状先验模型的输入（4 通道 PNG，见 training_prior），隐含 two_frame。
    video：源视频（Windows 上视频不在标注里记的 Mac 路径，由 lie_target_prepare.find_videos 找到）。
    edge_every：答案离画板边 EDGE_PX 以内的帧改成每这么多帧取一张（0 = 和别的帧一样每 every 帧）。
    own=True：画面里有真准星的录像拿来训练（见模块说明），只支持两帧对比；最后 RESULT_CUT_S 秒不要。"""
    two_frame = two_frame or prior
    if own and not two_frame:
        raise ValueError("自己的录像进训练只支持两帧对比模型（own 要配 two_frame 或 prior）")
    with open(recording / "frames.csv", encoding="utf-8") as f:
        rows = {int(r["frame"]): r for r in csv.DictReader(f)}
    meta = json.loads((recording / "meta.json").read_text(encoding="utf-8")) if (recording / "meta.json").exists() else {}
    if meta.get("cursor_reference_valid") is False:
        raise ValueError(f"{recording} 的准星参考已确认不可靠，请改用人工复核后的回放标注目录")
    if video is None:
        video = recording / "video.avi" if (recording / "video.avi").exists() else Path(meta["video"])
    if meta.get("cursor_is_aim"):
        offset = np.zeros(2)  # 回放标注（tools/lie_replay_label.py）记的就是人瞄的位置
    else:
        offset, _ = crosshair_offset(recording / "frames.csv", video)
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
        raise ValueError(f"{recording} 里没找到金色画板")
    bx, by = (int(v) for v in np.median(found[:, :2], axis=0))
    bw, bh = BOARD_SIZE
    seconds = [float(rows[i]["seconds"]) for i in sorted(rows)]
    fps = 1 / float(np.median(np.diff(seconds))) if len(seconds) > 1 else 30.
    step = max(1, round(fps / RATE))  # 两帧对比：前一步隔几帧录像

    # 第一遍：看得见的白色目标（外接框），顺便量目标大小和玩家准星离它多远
    visible, sizes, aim_errors, last = {}, [], [], None
    for i, frame in enumerate(frames):
        if boxes[i] is None:
            last = None
            continue
        board = frame[by:by + bh, bx:bx + bw]
        v = white_target(board, last)
        if v is None:
            continue
        ys, xs = np.nonzero(v[0])
        visible[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        sizes.append((xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()))
        last = v[1].copy()
    if not visible:
        raise ValueError(f"{recording} 里没看到白色目标")
    tw, th = np.median(sizes, axis=0)
    first_visible = min(visible)
    aim_window = sorted(visible)[max(0, len(visible) - 30)]

    # 光标瞬移：之后一段时间准星不在目标上
    teleport = set()
    prev = None
    for i in sorted(rows):
        if not rows[i]["video_x"]:
            prev = None
            continue
        cur = np.array([int(rows[i]["video_x"]), int(rows[i]["video_y"])])
        if prev is not None and np.linalg.norm(cur - prev) > TELEPORT_PX:
            teleport.update(range(i, i + TELEPORT_SKIP + 1))
        prev = cur

    # 第二遍：从目标第一次出现到画板关掉，每 every 帧出一张图
    name = recording.name
    written, from_cursor, from_pale, skipped_teleport, skipped_lost = 0, 0, 0, 0, 0
    synthetic_crosshairs, off_target_crosshairs, near_target_crosshairs = 0, 0, 0
    synthetic = meta.get("source") == "replay_label" and meta.get("crosshair_hidden_from_labeler") is False
    cleaned, last_frame = None, len(frames) - 1
    if own:
        cleaned = clean_crosshairs([None if b is None else f[by:by + bh, bx:bx + bw] for f, b in zip(frames, boxes)], fps)
        synthetic = True  # 真准星没了，接下来和买来的视频一样
        last_frame = max(i for i, b in enumerate(boxes) if b is not None) - round(RESULT_CUT_S * fps)

    def aim_at(j):
        """第 j 帧准星（光标 + 偏移）在画板上的位置；那帧没有光标返回 None。"""
        r = rows.get(j)
        if not r or not r["video_x"]:
            return None
        return np.array([int(r["video_x"]) - bx, int(r["video_y"]) - by]) + offset

    def answer(j):
        """第 j 帧的答案框 (x0, y0, x1, y1) 和来源（visible / pale / cursor）；当不了答案返回 (None, 原因)。"""
        if j < 0 or boxes[j] is None or j not in rows or not rows[j]["video_x"]:
            return None, "no_cursor"
        aim = aim_at(j)
        if j in visible:
            return visible[j], "visible"
        pale = pale_target(frames[j][by:by + bh, bx:bx + bw], aim, (tw, th))
        if pale is not None:
            return pale, "pale"  # 淡出途中还看得出来，比玩家准星（常常落后一截）准
        if not (0 <= aim[0] < bw and 0 <= aim[1] < bh):
            return None, "outside"
        if j in teleport:
            return None, "teleport"
        if rows[j].get("skip") == "1":  # 回放标注时标了「跟丢了」
            return None, "lost"
        return (aim[0] - tw / 2, aim[1] - th / 2, aim[0] + tw / 2, aim[1] + th / 2), "cursor"

    def prepared(j, aim, box, on_target):
        """第 j 帧画板，按实际运行的办法处理准星：（补画）+ 贴假准星 + 统一抹掉。返回 (画板, 抹过的地方或 None)。"""
        board = frames[j][by:by + bh, bx:bx + bw].copy()
        if aim is None:
            return board, None
        if synthetic:
            board = add_training_crosshair(board, aim)
        return erase_for_model(board, aim, rng, two_frame, box, on_target)

    def time_at(j):
        return float(rows[j]["seconds"]) if j in rows else j / fps

    # 形状先验：按实际运行的节奏（每步）看白色图形、记下完整的；每张图用它之前最近记下的那张
    snapshots, spin, prior_kinds = [], None, {}
    if prior:
        shape = ShapePrior()
        for j in range(0, len(frames), step):
            if boxes[j] is not None and shape.observe(frames[j][by:by + bh, bx:bx + bw], time_at(j)):
                snapshots.append((j, shape.template, shape.taken_at))
        spin = measure_spin(snapshots)
    shape_frames = [s[0] for s in snapshots]

    edge_extra = 0
    for i in range(first_visible, last_frame + 1):
        if (i - first_visible) % every:
            # 贴边的帧多取：每 edge_every 帧一张（只看答案框中心离画板边多近）
            if not edge_every or (i - first_visible) % edge_every:
                continue
            box, _ = answer(i)
            if box is None or min((box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                                  bw - (box[0] + box[2]) / 2, bh - (box[1] + box[3]) / 2) >= EDGE_PX:
                continue
            edge_extra += 1
        box, source = answer(i)
        if box is None:
            skipped_teleport += source == "teleport"
            skipped_lost += source == "lost"
            continue
        x0, y0, x1, y1 = box
        aim = cursor_aim = aim_at(i)
        on_target = source != "cursor"
        # 倒计时时玩家还没去套目标（有人隔着几十 px 等着），只看淡出前最后 1 秒玩家套得准不准
        if source == "visible" and i >= aim_window:
            aim_errors.append(float(np.hypot(aim[0] - (x0 + x1) / 2, aim[1] - (y0 + y1) / 2)))
        from_pale += source == "pale"
        from_cursor += source == "cursor"
        if synthetic and two_frame:
            aim, near = two_frame_aim(rng, aim, bw, bh)
            near_target_crosshairs += near
            off_target_crosshairs += not near
        elif synthetic and rng.random() < OFF_TARGET_CROSSHAIR:
            aim = off_target_aim(rng, aim, bw, bh)
            off_target_crosshairs += 1
        elif synthetic:
            aim = jittered_aim(rng, aim, bw, bh)
        on_target = on_target and inside((x0, y0, x1, y1), aim)  # 准星确实压在白色目标上
        synthetic_crosshairs += synthetic
        board, erased = prepared(i, aim, (x0, y0, x1, y1), on_target)
        if two_frame:
            history = []
            for k in range(1, MOTION_STEPS + 1):
                j = i - k * step
                if j < 0 or boxes[j] is None:
                    history.append((None, None))
                    continue
                prev_aim = aim_at(j)
                if synthetic:  # 补画的准星和当前帧保持同样的偏移，跟着人瞄的位置走
                    prev_aim = aim if prev_aim is None else prev_aim + (aim - cursor_aim)
                history.append(prepared(j, prev_aim, (x0, y0, x1, y1),
                                        prev_aim is not None and j in visible and inside(visible[j], prev_aim)))
            board = motion_input(board, [h[0] for h in history], [erased] + [h[1] for h in history])
        stem = f"{name}_{i:05d}"
        if prior:
            k = bisect.bisect_right(shape_frames, i) - 1
            shape = ShapePrior() if spin is None else ShapePrior(spin)
            if k >= 0:
                _, shape.template, shape.taken_at = snapshots[k]
            last, _ = answer(i - step)
            last = None if last is None else np.array([(last[0] + last[2]) / 2, (last[1] + last[3]) / 2])
            mask, kind, gap = training_prior(rng, shape, last, np.array([(x0 + x1) / 2, (y0 + y1) / 2]),
                                             time_at(i), (bh, bw))
            board = with_prior(board, mask)
            prior_kinds[kind] = prior_kinds.get(kind, 0) + 1
            new = not (out / "priors.csv").exists()
            with (out / "priors.csv").open("a", encoding="utf-8") as f:
                f.write(("stem,kind,gap_px\n" if new else "") + f"{stem},{kind},{'' if gap is None else round(gap, 1)}\n")
        x0, y0, x1, y1 = max(0., x0), max(0., y0), min(float(bw), x1), min(float(bh), y1)
        if prior:  # 4 通道不能存 JPEG；PNG 无损，和实际运行送进模型的一样
            cv2.imwrite(str(out / "images" / f"{stem}.png"), board)
        else:
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), board,
                        JPEG_444 if two_frame else [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / f"{stem}.txt").write_text(
            f"0 {(x0 + x1) / 2 / bw:.6f} {(y0 + y1) / 2 / bh:.6f} {(x1 - x0) / bw:.6f} {(y1 - y0) / bh:.6f}\n",
            encoding="utf-8")
        with (out / "sources.csv").open("a", encoding="utf-8") as f:
            f.write(f"{stem},{source}\n")
        written += 1
    kind = "two_frame_prior" if prior else "two_frame" if two_frame else "single"
    return {"recording": name, "images": written, "input_kind": kind,
            **({"prior": {"spin_deg_per_s": round(spin if spin is not None else _shadow.SPIN_DEG_PER_S, 1),
                          "spin_measured": spin is not None, "templates": len(snapshots),
                          "kinds": prior_kinds}} if prior else {}),
            "frames_per_step": step, "edge_extra_frames": edge_extra, "fading_from_pale": from_pale,
            "after_fade_from_cursor": from_cursor, "skipped_after_cursor_teleport": skipped_teleport,
            "skipped_marked_lost": skipped_lost,
            "synthetic_crosshair_frames": synthetic_crosshairs, "crosshair_cleaned": cleaned,
            "off_target_crosshair_frames": off_target_crosshairs,
            "near_target_crosshair_frames": near_target_crosshairs,
            "visible_frames": len(visible), "target_size_px": [round(float(tw)), round(float(th))],
            "crosshair_offset_px": [round(float(v), 1) for v in offset],
            "aim_error_while_visible_px": {
                "median": round(float(np.median(aim_errors)), 1),
                "p90": round(float(np.percentile(aim_errors, 90)), 1)} if aim_errors else None}


def preview(out: Path, path: Path, count=12, prefixes=None):
    """抽几张画上标签框，方便人眼检查（形状先验图显示灰度通道，剪影描成黄色）。prefixes：只看这几段录像。"""
    images = sorted(p for p in (out / "images").iterdir() if p.suffix in (".jpg", ".png")
                    and (prefixes is None or p.stem.rsplit("_", 1)[0] in prefixes))
    tiles = []
    for p in images[::max(1, len(images) // count)][:count]:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4:
            contours, _ = cv2.findContours(im[..., 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            im = cv2.cvtColor(im[..., 2], cv2.COLOR_GRAY2BGR)
            cv2.drawContours(im, contours, -1, (0, 255, 255), 2)
        h, w = im.shape[:2]
        _, cx, cy, bw, bh = map(float, (out / "labels" / f"{p.stem}.txt").read_text().split())
        cv2.rectangle(im, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                      (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 0, 255), 3)
        cv2.putText(im, p.stem, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        tiles.append(cv2.resize(im, (375, 250)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", type=Path, nargs="+", help="通过了测谎的 lie_recordings/<时间> 目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=2, help="每几帧取一张（相邻帧几乎一样）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--two-frame", action="store_true", help="存两帧对比模型的输入（lie_shadow.motion_input）")
    parser.add_argument("--prior", action="store_true", help="两帧对比 + 形状先验（4 通道 PNG，lie_shadow.with_prior）")
    args = parser.parse_args()

    for sub in ("images", "labels"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    (args.output / "sources.csv").write_text("stem,source\n", encoding="utf-8")  # 每张图的框从哪来
    reports = [label_recording(r, args.output, args.every, rng, args.two_frame, args.prior) for r in args.recordings]
    (args.output / "classes.txt").write_text("target\n", encoding="utf-8")
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images\nval: images\nnames:\n  0: target\n",
        encoding="utf-8")
    preview(args.output, args.output / "preview.jpg")
    summary = {"output": str(args.output), "recordings": reports,
               "images": sum(r["images"] for r in reports)}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
