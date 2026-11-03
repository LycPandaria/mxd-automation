"""自动补充飞镖（充值）。

流程（全部在游戏窗口里点，期间主循环暂停，不做别的操作）：
  1. 点左侧快捷图标里的枫叶（冒险岛会员，排第几个都行）→ 弹出「冒险岛会员」窗口
  2. 点「便捷杂货店」那一行的「打开」→ 打开商店
  3. 点右边背包的「消耗」页
  4. 这一屏有几个「充值」按钮就点几个（满的那行没有按钮），每次点弹窗里的「确定」
  5. 按 Esc 关掉商店（会员窗口还开着就再按一次）

按钮位置不写死：每一步截一张图用 OCR 找字，窗口弹在哪都能点到。
商店里的按钮（消耗页签、充值、确定）字太小，OCR 常认不出，就用旁边认得出的字定位：
  - 充值按钮：同一行「…金币 / 充值：720」那行字的右边（ANCHOR_RECHARGE）
  - 消耗页签：右边背包上方的金币数「2,140,106」的左下（ANCHOR_TAB）
  - 确定：弹窗里「真充值吗？」的右下（ANCHOR_OK）
偏移按 1920×1080 量的，运行时按画面高度等比换算。
只有枫叶图标没有字：在配置的高度（restock_icon_y，按 1920×1080 填写）那一排图标里找最大的橙色块
（restock_icon_x 只在两块一样大时用来挑近的）；
找不到说明左侧图标栏收起来了，先点灯泡下面的喇叭（restock_horn_x/y）展开，看到枫叶再点。

任何一步找不到要点的东西就按 Esc 收尾并返回 False，不会乱点。

补飞镖占着主循环，每张截图都要交给测谎监视（on_frame），不然这十几秒里弹出测谎也认不出来，
白色图形阶段就错过了（录像 20261101_104735）。认出测谎（abort）就马上停：不再点，也不按 Esc，
窗口等测谎结束后 close_after_abort 再关。
"""
import os
import re
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

REF_W, REF_H = 1920, 1080
"""restock_icon_x/y 按这个分辨率填写，运行时按实际画面等比换算"""

STEP_TIMEOUT = 5.0
"""每一步等界面出现的最长时间（秒），期间每 0.5 秒截图 OCR 一次（整帧 OCR 一次约 1 秒）"""

DEBUG_DIR = "restock_debug"
"""每次补飞镖的截图存这里：点之前（红圈标出点击位置）、点之后、失败时的画面"""

ICON_SEARCH = 30
"""在配置的 y 上下多大范围（px，1080 高画面下）找枫叶图标的橙色"""

ICON_ROW_RIGHT = 480
"""左侧快捷图标栏的右端（px，1080 高画面下）：整排图标都在 0~这里之间。
枫叶在栏里的位置会变（第三个、第四个……），所以整排找，不只看配置的 x 附近"""

ICON_HORN_GUARD = 35
"""喇叭（restock_horn_x）左右这么宽（px，1080 高画面下）里的橙色不算枫叶：
图标栏收起来时喇叭那里也有一块橙色，以前会被当成枫叶点下去，结果只是把图标栏展开/收起"""

ICON_MAX_SIZE = 60
"""一个图标的橙色块最大多宽/多高（px，1080 高画面下），再大就是怪物、伤害数字之类，不是图标"""

ICON_SAMPLES = 8
ICON_SAMPLE_GAP = 0.2
"""图标栏的图标会一闪一闪：枫叶大约每 3 秒有将近 1 秒褪成白色（几乎没有橙色），这时第二个图标（橙色蘑菇）
的橙色反而最多，只看一张图就会点到蘑菇。所以连截 ICON_SAMPLES 张、每张隔 ICON_SAMPLE_GAP 秒（盖过褪色那一段），
每个图标取它最亮那一刻的橙色量再比"""

ANCHOR_RECHARGE = (71, -17)
"""充值按钮 = 「金币/充值：N」那行字中心（几行取中位数 x）+ 这个偏移"""
ANCHOR_TAB = (-120, 35)
"""消耗页签 = 背包金币数中心 + 这个偏移"""
ANCHOR_OK = (53, 55)
"""确定按钮 = 「真充值吗？」中心 + 这个偏移"""

ROW_COLUMN_PX = 40
"""价格行靠左对齐，字数不同时中心差不了多少：离这一列超过这么多（px，1080 高画面下）的不是商店的行，
多半是聊天栏里带"金币""充值"的广告，不能当充值按钮点"""
ROW_MAX_CHARS = 22
"""价格行最长的样子是 "12,345金币/充值：1,000"，再长的是聊天"""
ROW_SAME_PX = 15
"""认出的按钮和按价格行推算的位置相差这么多（px，1080 高画面下）以内，算同一行"""

PRICE_RE = re.compile(r"^\d[\d,，]*金币")
"""商店里的价格行（"50金币"、"3,220金币/充值：59"），聊天栏、拾取提示（"获取金币(+321)"）不会这样开头"""

MESO_RE = re.compile(r"^\d{1,3}([,，]\d{3})+$")


class _Aborted(Exception):
    """测谎弹出来了，停止补飞镖"""


class StarRestock:
    def __init__(self, capture, executor, config, on_log: Optional[Callable[[str], None]] = None):
        self.capture = capture
        self.executor = executor
        self.config = config
        self._log = on_log or (lambda m: None)
        self._engine = None
        self._shot_dir = DEBUG_DIR
        self._shot_n = 0
        self.on_frame: Callable[[np.ndarray], None] = lambda f: None  # 每张截图都给它（测谎监视）
        self.abort: Callable[[], bool] = lambda: False  # 返回 True 就马上停（认出测谎了）
        self.aborted = False  # 这次是中途停的，窗口可能还开着

    # ---- 公开接口 ----

    def run(self) -> bool:
        """执行一次补飞镖，成功返回 True。调用方负责暂停其他操作。"""
        self._shot_dir = os.path.join(DEBUG_DIR, time.strftime("%Y%m%d_%H%M%S"))
        self._shot_n = 0
        self.aborted = False
        self._log(f"[补飞镖] 开始：打开冒险岛会员 → 便捷杂货店 → 消耗 → 有几个充值几个"
                  f"（每一步的截图存到 {os.path.abspath(self._shot_dir)}）")
        try:
            ok = self._run()
        except _Aborted:
            self._stop_for_lie_test()
            return False
        except Exception as e:
            self._log(f"[补飞镖] 出错: {e}")
            ok = False
        if not ok:
            self._close_all()
        return ok

    def close_after_abort(self):
        """测谎结束后，把中途停下时可能还开着的窗口关掉；没有中途停过什么也不做。"""
        if self.aborted:
            self.aborted = False
            self._close_all()

    # ---- 流程 ----

    def _run(self) -> bool:
        frame = self._grab()
        self._check_abort()
        h, w = frame.shape[:2]
        s = h / REF_H

        # 1. 枫叶图标：没看到就先点灯泡下面的喇叭把图标栏展开，看到枫叶了再点。
        #    点完没出现「便捷杂货店」（点错了、或那一下只是把图标栏展开了）就重新找一次再点
        items = None
        for attempt in range(2):
            pos = self._find_icon(self._icon_frames())
            if pos is None:
                hx = int(float(getattr(self.config, "restock_horn_x", 32)) * w / REF_W)
                hy = int(float(getattr(self.config, "restock_horn_y", 571)) * h / REF_H)
                self._log(f"[补飞镖] 没看到枫叶图标，先点灯泡下面的喇叭展开图标栏 ({hx},{hy})")
                self._click(hx, hy)
                time.sleep(0.5)
                pos = self._find_icon(self._icon_frames())
                if pos is None:
                    self._save(self._grab(), "点喇叭后还是没有枫叶")
                    self._log("[补飞镖] 点了喇叭还是没看到枫叶图标，放弃本次补充"
                              "（图标位置变了的话改「枫叶图标 x / y」）")
                    return False
            ix, iy = pos
            self._log(f"[补飞镖] 点枫叶图标 ({ix},{iy})")
            self._click(ix, iy)

            # 2. 便捷杂货店 → 打开
            items = self._wait_for(lambda it: self._find(it, "杂货"), "便捷杂货店", give_up=attempt == 1)
            if items is not None:
                break
        if items is None:
            return False
        shop = self._shop_entry(items)
        opens = [t for t in items if "打开" in t[0] and t[1] > shop[1]
                 and abs(t[2] - shop[2]) <= 30 * s]
        if opens:
            ox, oy = min(opens, key=lambda t: t[1])[1:3]
        else:
            ox, oy = shop[1] + int(120 * s), shop[2]  # 「打开」按钮在同一行右边
        self._log(f"[补飞镖] 点「便捷杂货店」的打开 ({ox},{oy})")
        self._click(ox, oy)

        # 3. 消耗页（背包在右边）：认得出「消耗」就点它，否则按金币数的位置推算
        items = self._wait_for(lambda it: self._find(it, "消耗") or self._meso(it), "商店（消耗页/金币数）")
        if items is None:
            return False
        tabs = self._find(items, "消耗")
        if tabs:
            tx, ty = max(tabs, key=lambda t: t[1])[1:3]
        else:
            m = self._meso(items)
            tx, ty = m[1] + int(ANCHOR_TAB[0] * s), m[2] + int(ANCHOR_TAB[1] * s)
        self._log(f"[补飞镖] 点消耗页 ({tx},{ty})")
        self._click(tx, ty)

        # 4. 这一屏所有「充值」按钮
        items = self._wait_for(lambda it: self._recharge_buttons(it, s), "充值按钮（「金币/充值」那几行）")
        if items is None:
            return False
        buttons = self._recharge_buttons(items, s)
        self._log(f"[补飞镖] 看到 {len(buttons)} 个可以充值")
        done = 0
        for i, (bx, by) in enumerate(buttons, 1):
            self._log(f"[补飞镖] 点第 {i} 个充值 ({bx},{by})")
            self._click(bx, by)
            got = self._wait_for(lambda it: self._find(it, "确定") or self._find(it, "充值吗"),
                                 "确定（「真充值吗？」）", timeout=2.0)
            if got is None:
                continue
            oks = self._find(got, "确定")
            if oks:
                cx, cy = oks[0][1:3]
            else:
                q = self._find(got, "充值吗")[0]
                cx, cy = q[1] + int(ANCHOR_OK[0] * s), q[2] + int(ANCHOR_OK[1] * s)
            self._log(f"[补飞镖] 点确定 ({cx},{cy})")
            self._click(cx, cy)
            done += 1
            time.sleep(0.4)
        self._log(f"[补飞镖] 充值完成 {done}/{len(buttons)} 个")

        # 5. 关窗口
        self._close_all()
        return done > 0

    def _close_all(self):
        """关掉补飞镖打开的窗口：每次按 Esc 前都截图确认还有窗口开着，按完等 1 秒再看。

        什么都没开时按 Esc 会弹出游戏菜单（切换频道/系统选项/退出游戏），之后按键都被菜单吃掉；
        万一弹出来了，最后再按一次 Esc 把它关掉。"""
        for _ in range(4):
            open_ = self._open_windows(self._ocr(self._grab()))
            if not open_:
                break
            if self.abort():
                self._stop_for_lie_test()
                return
            self._log(f"[补飞镖] 按 Esc 关闭（还开着：{'、'.join(open_)}）")
            self.executor.press_key("esc")
            time.sleep(1.0)
        if "游戏菜单" in self._open_windows(self._ocr(self._grab())):
            if self.abort():
                self._stop_for_lie_test()
                return
            self._log("[补飞镖] 游戏菜单弹出来了，再按一次 Esc 关掉")
            self.executor.press_key("esc")
            time.sleep(1.0)

    def _stop_for_lie_test(self):
        self.aborted = True
        self._log("[补飞镖] 测谎弹出来了，马上停止补充（窗口等测谎结束再关）")

    def _open_windows(self, items) -> List[str]:
        """画面里还开着哪些窗口。只认各窗口自己特有的字——聊天栏里常有"充值""确定"之类的广告，
        不能当成窗口还开着。"""
        out = []
        if self._find(items, "真充值"):
            out.append("确认框")
        price = [t for t in items if PRICE_RE.match(t[0].replace(" ", "")) and len(t[0]) <= 18]
        if len(price) >= 2 or self._meso(items):
            out.append("商店")
        if [t for t in items if ("便捷杂货店" in t[0] and len(t[0]) <= 7) or "冒险岛会员" in t[0]]:
            out.append("会员窗口")
        if self._find(items, "切换频道") or self._find(items, "系统选项") or self._find(items, "退出游戏"):
            out.append("游戏菜单")
        return out

    # ---- 工具 ----

    def _grab(self) -> np.ndarray:
        frame = self.capture.grab()
        self.on_frame(frame)
        return frame

    def _check_abort(self):
        if self.abort():
            raise _Aborted()

    def _click(self, x: int, y: int):
        self._save(self._grab(), "点之前", (int(x), int(y)))
        self._check_abort()
        self.executor.real_click(int(x), int(y))
        time.sleep(0.5)
        self._save(self._grab(), "点之后")

    def _save(self, frame, what: str, mark: Optional[Tuple[int, int]] = None):
        """调试截图：mark 给出时在点击位置画红圈。存不了就算了，不影响流程。"""
        try:
            import cv2
            os.makedirs(self._shot_dir, exist_ok=True)
            img = frame.copy()
            if mark is not None:
                cv2.circle(img, mark, 18, (0, 0, 255), 3)
                cv2.drawMarker(img, mark, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            self._shot_n += 1
            name = f"{self._shot_n:02d}_{what}.jpg"
            # 路径里有中文时 cv2.imwrite 会失败，用 imencode + tofile
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                buf.tofile(os.path.join(self._shot_dir, name))
        except Exception:
            pass

    def _wait_for(self, cond, what: str, timeout: float = STEP_TIMEOUT, give_up: bool = True):
        """每 0.5 秒截图 OCR，直到 cond(items) 有结果；超时写日志返回 None。
        give_up=False 时日志写「再试一次」而不是「放弃」（调用方还会重试）。"""
        end = time.time() + timeout
        while True:
            items = self._ocr(self._grab())
            self._check_abort()
            if cond(items):
                return items
            if time.time() >= end:
                self._save(self._grab(), f"没找到_{what[:8]}")
                self._log(f"[补飞镖] {timeout:.0f} 秒内没找到「{what}」，"
                          f"{'放弃本次补充' if give_up else '再试一次'}"
                          f"（截图见 {os.path.abspath(self._shot_dir)}）")
                return None
            time.sleep(0.5)

    @staticmethod
    def _find(items, key: str):
        return [t for t in items if key in t[0]]

    @staticmethod
    def _buttons(items):
        return [t for t in items if t[0].replace(" ", "") in ("充值", "充值>")]

    @staticmethod
    def _shop_entry(items):
        """会员窗口里「便捷杂货店」那一项（不要说明文字里的"杂货店"）。"""
        exact = [t for t in items if "便捷杂货店" in t[0] and len(t[0]) <= 7]
        if exact:
            return exact[0]
        return min((t for t in items if "杂货" in t[0]), key=lambda t: len(t[0]))

    @staticmethod
    def _meso(items):
        """背包里的金币数（2,140,106 这种带逗号的数字），取最靠上的那个。"""
        ms = [t for t in items if MESO_RE.match(t[0].replace(" ", ""))]
        return min(ms, key=lambda t: t[2]) if ms else None

    def _recharge_buttons(self, items, s: float) -> List[Tuple[int, int]]:
        """充值按钮位置，从上到下。认得出按钮字就用它；按钮字小，常常只认出一部分，
        所以每个「金币/充值：N」行也推算一个位置，没被认出的按钮靠它补上。"""
        out = []
        btns = self._buttons(items)
        if btns:
            bx = np.median([t[1] for t in btns])
            out = [(t[1], t[2]) for t in btns if abs(t[1] - bx) <= ROW_COLUMN_PX * s]
        lines = [t for t in items if "充值" in t[0] and "金" in t[0]
                 and len(t[0].replace(" ", "")) <= ROW_MAX_CHARS]
        if lines:
            # 列的位置优先按真正的价格行算，聊天广告不会把它带偏
            priced = [t for t in lines if PRICE_RE.match(t[0].replace(" ", ""))] or lines
            cx = np.median([t[1] for t in priced])  # 取中位数免得字数不同带偏
            for t in lines:
                if abs(t[1] - cx) > ROW_COLUMN_PX * s:
                    continue
                y = t[2] + int(ANCHOR_RECHARGE[1] * s)
                if not any(abs(y - oy) <= ROW_SAME_PX * s for _, oy in out):
                    out.append((int(cx) + int(ANCHOR_RECHARGE[0] * s), y))
        return sorted(out, key=lambda p: p[1])

    def _ocr(self, frame: np.ndarray) -> List[Tuple[str, int, int]]:
        """OCR 整帧，返回 [(文字, 中心x, 中心y), ...]。"""
        engine = self._get_engine()
        if engine is None:
            return []
        result, _ = engine(np.ascontiguousarray(frame))
        out = []
        for box, text, _score in result or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((text.strip(), int(sum(xs) / 4), int(sum(ys) / 4)))
        return out

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                # 商店的字很小，不缩小画面（det_limit_side_len 给到 2000）
                self._engine = RapidOCR(det_limit_type="max", det_limit_side_len=2000, use_cls=False)
            except Exception as e:
                self._log(f"[补飞镖] OCR 初始化失败: {e}")
                self._engine = False
        return self._engine or None

    def _icon_frames(self) -> List[np.ndarray]:
        """连截几张图给 _find_icon：图标会闪，一张图可能正赶上枫叶褪色（见 ICON_SAMPLES）。"""
        frames = []
        for i in range(ICON_SAMPLES):
            if i:
                time.sleep(ICON_SAMPLE_GAP)
            frames.append(self._grab())
            self._check_abort()
        return frames

    def _find_icon(self, frames) -> Optional[Tuple[int, int]]:
        """在左侧图标栏那一排里找枫叶图标，找不到返回 None（图标栏多半收起来了）。
        frames 可以是一张图，也可以是连截的几张（每个图标取最亮那一张的橙色量）。

        枫叶在栏里的位置会变，所以整排（x 从 0 到 ICON_ROW_RIGHT）找橙色块，取最大的一块：
        蘑菇图标的橙色大约只有枫叶的三分之一，红包图标偏红，只有边缘零星几个像素算橙色。
        太大的橙色块（怪物、伤害数字）和喇叭位置上的橙色不算；一样大时取离配置 x 近的。"""
        if isinstance(frames, np.ndarray):
            frames = [frames]
        h, w = frames[0].shape[:2]
        s = h / REF_H
        x0 = int(float(getattr(self.config, "restock_icon_x", 200)) * w / REF_W)
        near = ICON_MAX_SIZE * s / 2
        slots = []  # [橙色像素数（各张里最多的）, cx, cy]，同一个图标在几张图里合成一项
        for frame in frames:
            for n, cx, cy in self._orange_blobs(frame):
                slot = next((t for t in slots if abs(t[1] - cx) < near and abs(t[2] - cy) < near), None)
                if slot is None:
                    slots.append([n, cx, cy])
                elif n > slot[0]:
                    slot[:] = [n, cx, cy]
        if not slots:
            return None
        _, cx, cy = max(slots, key=lambda t: (t[0], -abs(t[1] - x0)))
        return cx, cy

    def _orange_blobs(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """图标栏那一排里图标大小的橙色块：[(橙色像素数, cx, cy), ...]，喇叭上的不算。"""
        import cv2
        h, w = frame.shape[:2]
        s = h / REF_H
        y0 = int(float(getattr(self.config, "restock_icon_y", 571)) * h / REF_H)
        hx = int(float(getattr(self.config, "restock_horn_x", 32)) * w / REF_W)
        r = int(ICON_SEARCH * s)
        xa, xb, ya, yb = 0, min(w, int(ICON_ROW_RIGHT * s)), max(0, y0 - r), min(h, y0 + r)
        roi = frame[ya:yb, xa:xb].astype(int)
        b, g, rr = roi[..., 0], roi[..., 1], roi[..., 2]
        mask = np.uint8((rr > 200) & (g > 80) & (g < 180) & (b < 90))   # 枫叶的橙色
        # 先膨胀一下把同一个图标里断开的橙色连起来，再按连通块分开各个图标
        count, labels = cv2.connectedComponents(cv2.dilate(mask, np.ones((5, 5), np.uint8)))
        out = []
        for i in range(1, count):
            ys, xs = np.nonzero((labels == i) & (mask > 0))
            if len(xs) < 30 * s * s:
                continue
            if xs.max() - xs.min() > ICON_MAX_SIZE * s or ys.max() - ys.min() > ICON_MAX_SIZE * s:
                continue
            cx, cy = xa + int(np.median(xs)), ya + int(np.median(ys))
            if abs(cx - hx) < ICON_HORN_GUARD * s:
                continue  # 喇叭，不是枫叶
            out.append((len(xs), cx, cy))
        return out
