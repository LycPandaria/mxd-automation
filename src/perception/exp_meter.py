"""经验记录：定时读状态栏的 EXP 数字，算出这次打了多少、每小时多少，并按分钟写到 exp_log/日期.csv。

状态栏写着 "EXP 436935[23.57%]"，位置随窗口大小变，所以先找 "EXP" 这三个字母（游戏里是固定的白色点阵字），
再用 OCR 只认它右边那一小块。主循环只把截图交过来，读数在后台线程做，不耽误打怪。

读错怎么办：经验和百分比能互相对账（经验 ÷ 百分比 = 这一级升级要的总经验，同一级里不变），
对不上、或者数字变小的读数先不信，连续两次都这样才认（升级了、死了掉经验）。
"""
import csv
import os
import re
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

LABEL = [
    "######.#....#..#####.",
    "#####..##..##..######",
    "##......####...##..##",
    "##......####...##..##",
    "#####....##....######",
    "#####...####...#####.",
    "##......####...##....",
    "#####..##..##..##....",
    "######.#....#..##....",
]
"""状态栏 "EXP" 三个字母的点阵（1920×1080 录像里取的）"""
LABEL_WHITE = 170
"""三个通道都高于这个算白字"""
LABEL_MIN_SCORE = 0.8
"""和点阵的相似度低于这个算没找到"""
SEARCH_BOTTOM = 0.15
"""只在画面最下面这么大一块找（状态栏在最下面）"""
TEXT_BOX = (26, -3, 130, 14)
"""数字相对 "EXP" 左上角的范围 (左, 上, 右, 下)（px）"""
TEXT_ZOOM = 3
"""字很小，放大这么多倍再认"""
TEXT_RE = re.compile(r"(\d+)\s*[\[(（]\s*(\d+(?:\.\d+)?)\s*%")
"""从 OCR 结果里取 经验 和 百分比"""
MIN_SCORE = 0.6
"""OCR 置信度低于这个不用"""
PCT_TOLERANCE = 0.012
"""百分比只显示两位小数，对账时放宽这么多（%）"""
LEVEL_UP_DROP = 15.0
"""百分比一下掉这么多才算升级（死亡最多掉 10%）"""
LOST_MIN_DROP = 0.1
"""百分比掉这么多才算真的掉经验；更少的是之前把数字读大了一点，悄悄改回来"""
LOST_RETRY_READS = 6
"""连续这么多次没读到数字，就重新找一次 "EXP" 的位置"""
RATE_WINDOW_SECONDS = 600
"""「每小时」按最近这么久算（秒）"""
LOG_DIR = "exp_log"
CSV_HEADER = ["时间", "经验", "百分比", "本次累计获得", "比上一行多", "备注"]


def _label_mask() -> np.ndarray:
    return np.array([[255 if c == "#" else 0 for c in row] for row in LABEL], np.uint8)


def find_label(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """状态栏 "EXP" 字样左上角在画面里的位置，没找到返回 None。"""
    top = int(frame.shape[0] * (1 - SEARCH_BOTTOM))
    white = (frame[top:].min(axis=2) > LABEL_WHITE).astype(np.uint8) * 255
    mask = _label_mask()
    if white.shape[0] < mask.shape[0] or white.shape[1] < mask.shape[1]:
        return None
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(white, mask, cv2.TM_CCOEFF_NORMED))
    return (x, y + top) if score >= LABEL_MIN_SCORE else None


def parse_exp(text: str) -> Optional[Tuple[int, float]]:
    """ "436935[23.57%]" → (436935, 23.57)；不是这个样子返回 None。"""
    m = TEXT_RE.search(text.replace(" ", ""))
    if not m:
        return None
    exp, pct = int(m.group(1)), float(m.group(2))
    return (exp, pct) if 0 <= pct <= 100 else None


def _need_range(exp: int, pct: float) -> Tuple[float, float]:
    """这个读数说明升级要的总经验在什么范围。"""
    low = exp * 100 / (pct + PCT_TOLERANCE)
    high = exp * 100 / (pct - PCT_TOLERANCE) if pct > PCT_TOLERANCE else float("inf")
    return low, high


class ExpTracker:
    """把一连串读数变成"这次获得多少"。时间都由调用方传进来。"""

    def __init__(self):
        self.reset(0.0)

    def reset(self, now: float):
        self.start_at = now
        self.exp: Optional[int] = None
        self.pct = 0.0
        self.gained = 0
        self.levels = 0
        self.note = ""  # 下一行记录的备注（升级、掉经验）
        self._need: Optional[Tuple[float, float]] = None
        self._pending: Optional[Tuple[int, float]] = None
        self._samples = deque()  # (时间, 本次累计获得)，算每小时用

    def feed(self, now: float, exp: int, pct: float) -> str:
        """喂一个读数，返回 ""（信了）、"pending"（先不信）、"level_up"、"lost"（掉经验）。"""
        need = _need_range(exp, pct)
        if self.exp is None:  # 刚开始：连续两次对得上才信，第一次可能就是读错的
            confirmed = self._confirmed(exp, pct, need)
            self._pending = (exp, pct)
            if confirmed:
                self._accept(now, exp, pct, need)
                return ""
            return "pending"
        if exp >= self.exp and self._overlap(self._need, need):
            self._gain(now, exp, pct, exp - self.exp, self._merge(self._need, need))
            return ""
        confirmed = self._confirmed(exp, pct, need)
        self._pending = (exp, pct)
        if not confirmed:
            return "pending"
        # 连续两次都这样，是真的：不是升级就是掉经验
        if self.pct - pct >= LEVEL_UP_DROP:
            left = max(0, int(self._need[0]) - self.exp)
            self.levels += 1
            self.note = "升级"
            self._gain(now, exp, pct, left + exp, need)
            return "level_up"
        if self.pct - pct >= LOST_MIN_DROP:
            self.note = f"掉经验 {self.exp - exp}"
            self._gain(now, exp, pct, exp - self.exp, need)
            return "lost"
        self._gain(now, exp, pct, exp - self.exp, need)
        return ""

    def rate_per_hour(self, now: float) -> Optional[float]:
        """最近一段时间折算的每小时经验；刚开始（不到 1 分钟）返回 None。"""
        if not self._samples or now - self._samples[0][0] < 60:
            return None
        then, gained = self._samples[0]
        return (self.gained - gained) * 3600 / (now - then)

    def _confirmed(self, exp: int, pct: float, need: Tuple[float, float]) -> bool:
        if self._pending is None:
            return False
        p_exp, p_pct = self._pending
        return exp >= p_exp and self._overlap(_need_range(p_exp, p_pct), need)

    def _accept(self, now: float, exp: int, pct: float, need: Tuple[float, float]):
        self.exp, self.pct, self._need, self._pending = exp, pct, need, None
        self._samples.append((now, self.gained))
        while now - self._samples[0][0] > RATE_WINDOW_SECONDS:
            self._samples.popleft()

    def _gain(self, now: float, exp: int, pct: float, delta: int, need: Tuple[float, float]):
        self.gained += delta
        self._accept(now, exp, pct, need)

    @staticmethod
    def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return a[0] <= b[1] and b[0] <= a[1]

    @staticmethod
    def _merge(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return max(a[0], b[0]), min(a[1], b[1])


class ExpLog:
    """按天写 csv。文件被别的程序占着（比如 Excel 开着）时先攒着，下次再写。"""

    def __init__(self, folder: str = LOG_DIR, on_log: Optional[Callable[[str], None]] = None):
        self.folder = folder
        self._on_log = on_log or (lambda m: None)
        self._rows: List[Tuple[str, list]] = []
        self._warned = False

    def add(self, now: float, exp: int, pct: float, gained: int, delta: int, note: str):
        row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), exp, f"{pct:.2f}", gained, delta, note]
        self._rows.append((self.path_for(now), row))
        self.flush()

    def path_for(self, now: float) -> str:
        return os.path.join(self.folder, time.strftime("%Y%m%d", time.localtime(now)) + ".csv")

    def flush(self):
        while self._rows:
            path, row = self._rows[0]
            try:
                os.makedirs(self.folder, exist_ok=True)
                new = not os.path.isfile(path)
                # utf-8-sig：Excel 打开中文不乱码
                with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(CSV_HEADER)
                    w.writerow(row)
            except OSError as e:
                if not self._warned:
                    self._warned = True
                    self._on_log(f"[经验] 写不进 {path}（是不是用 Excel 开着？）先攒着，关掉后会补上: {e}")
                return
            self._rows.pop(0)
            self._warned = False


def read_history(path: str) -> List[List[str]]:
    """读一个 csv 的全部记录（不含表头），读不了返回空。"""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.reader(f))[1:]
    except OSError:
        return []


class ExpMeter:
    """主循环每帧调 feed(frame)；到时间了就在后台读一次，界面用 summary() 显示。"""

    def __init__(self, on_log: Optional[Callable[[str], None]] = None, read_text: Optional[Callable] = None):
        self._on_log = on_log or (lambda m: None)
        self._read_text = read_text or self._ocr_text
        self.tracker = ExpTracker()
        self.log = ExpLog(on_log=self._on_log)
        self.read_seconds = 5.0
        self.row_seconds = 60.0
        self.log_minutes = 10.0
        self._lock = threading.Lock()
        self._busy = False
        self._label: Optional[Tuple[int, int]] = None
        self._misses = 0
        self._warned_missing = self._warned_error = False
        self._last_read = self._last_row = self._last_note = 0.0
        self._row_gained = 0
        self._engine = None

    def start(self, now: float):
        with self._lock:
            self.tracker.reset(now)
            self._last_read = self._last_row = float("-inf")  # 第一次读到就记一行
            self._last_note = now
            self._row_gained = 0
            self._label, self._misses, self._warned_missing, self._warned_error = None, 0, False, False

    def stop(self, now: float):
        """停止打怪：把最后的数写进记录。"""
        with self._lock:
            if self.tracker.exp is not None and self.tracker.gained != self._row_gained:
                self._add_row(now)

    def feed(self, frame: np.ndarray, now: float):
        if self._busy or now - self._last_read < self.read_seconds:
            return
        self._last_read = now
        self._busy = True
        threading.Thread(target=self._read_in_background, args=(frame, now), daemon=True).start()

    def summary(self, now: float) -> str:
        with self._lock:
            t = self.tracker
            if t.exp is None:
                return "经验: 还没读到（开始打怪后几秒）"
            rate = t.rate_per_hour(now)
            text = f"经验: {t.exp} ({t.pct:.2f}%)  本次 {t.gained:+d}"
            return text + (f"  每小时 {rate:+.0f}" if rate is not None else "")

    # ---- 后台读数 ----

    def _read_in_background(self, frame: np.ndarray, now: float):
        try:
            self.read_once(frame, now)
        except Exception as e:
            if not self._warned_error:  # 每 5 秒读一次，别刷屏
                self._warned_error = True
                self._on_log(f"[经验] 读取出错: {e}")
        finally:
            self._busy = False

    def read_once(self, frame: np.ndarray, now: float):
        reading = self._read(frame)
        if reading is None:
            self._misses += 1
            if self._misses >= LOST_RETRY_READS:
                self._label, self._misses = None, 0
                if self.tracker.exp is None and not self._warned_missing:
                    self._warned_missing = True
                    self._on_log("[经验] 状态栏上读不到 EXP 数字（只在 1920×1080 的游戏画面验证过），先继续试")
            return
        self._misses = 0
        with self._lock:
            result = self.tracker.feed(now, *reading)
            if result in ("level_up", "lost") or (result == "" and now - self._last_row >= self.row_seconds):
                self._add_row(now)
            if result == "level_up":
                self._on_log(f"[经验] 升级了！现在 {reading[0]} ({reading[1]:.2f}%)，本次共 {self.tracker.gained:+d}")
            elif result == "lost":
                self._on_log(f"[经验] 经验变少了（死亡？）: {self.tracker.note}")
            if result == "" and self.log_minutes > 0 and now - self._last_note >= self.log_minutes * 60:
                self._last_note = now
                self._on_log("[经验] " + self._summary_no_lock(now))

    def _summary_no_lock(self, now: float) -> str:
        t = self.tracker
        rate = t.rate_per_hour(now)
        text = f"{t.exp} ({t.pct:.2f}%)，本次 {t.gained:+d}"
        return text + (f"，每小时约 {rate:+.0f}" if rate is not None else "")

    def _add_row(self, now: float):
        t = self.tracker
        self.log.add(now, t.exp, t.pct, t.gained, t.gained - self._row_gained, t.note)
        self._row_gained, self._last_row, t.note = t.gained, now, ""

    def _read(self, frame: np.ndarray) -> Optional[Tuple[int, float]]:
        if self._label is None:
            self._label = find_label(frame)
            if self._label is None:
                return None
        x, y = self._label
        left, top, right, bottom = TEXT_BOX
        crop = frame[max(0, y + top):y + bottom, x + left:x + right]
        if crop.size == 0:
            return None
        big = cv2.resize(crop, None, fx=TEXT_ZOOM, fy=TEXT_ZOOM, interpolation=cv2.INTER_CUBIC)
        text = self._read_text(big)
        return parse_exp(text) if text else None

    def _ocr_text(self, image: np.ndarray) -> str:
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
        try:
            result, _ = self._engine(np.ascontiguousarray(image), use_det=False, use_cls=False, use_rec=True)
        except TypeError:  # 老版本 rapidocr 没有这几个参数
            result, _ = self._engine(np.ascontiguousarray(image))
        texts = []
        for item in result or []:
            # 只识别时是 [文字, 分数]，带检测时是 [框, 文字, 分数]
            text, score = item[-2], item[-1]
            if float(score) >= MIN_SCORE:
                texts.append(str(text))
        return "".join(texts)
