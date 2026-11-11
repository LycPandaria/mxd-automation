"""切换频道：这一层没怪了就换一个频道，换完按住方向键走一段。

流程（期间主循环暂停，和录像 视频/20261031_212446.mp4 里手动操作一样）：
  1. 按 Esc 弹出游戏菜单，点最上面一行「切换频道」
  2. 弹出频道窗口（5 行 6 列，当前频道是青色），双击别的频道
  3. 窗口关掉后约 1.3 秒黑屏 = 真的在换；约 3.5 秒后画面回来
  4. 按住右方向键 5 秒

有时候第 2 步窗口关了却一直不黑屏（录像里第一次就是这样，游戏没有任何提示），
所以看到黑屏才算换成，否则换一个频道从第 1 步重来。

菜单和频道窗口的颜色、版面是固定的，不用 OCR：按颜色找，比 OCR 快，也不怕小字认不出。
位置和大小按 1920×1080 量的，运行时按画面高度等比换算。
"""
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


def _app_dir() -> str:
    """程序数据目录：打包后是 exe 所在目录，开发环境是项目根目录。

    与 logger / config_loader 的算法一致。这里单独算一份是为了让本模块
    能被测试按文件路径单独加载（tests/test_quit.py），不依赖包结构。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REF_H = 1080
"""下面的像素值按这个画面高度量的"""

MENU_ROW = (91, 28)
"""游戏菜单一行的宽、高（px）"""
MENU_ROW_GAP = 29
"""游戏菜单相邻两行的间距（px）"""
MENU_MIN_ROWS = 4
"""至少这么多行上下对齐才算菜单：切换频道/系统选项/隐私政策/…/退出游戏，一共 6 行"""
MENU_QUIT_ROW = 5
"""「退出游戏」是第几行（最上面的「切换频道」是第 0 行）"""

DIALOG_BAR = (432, 22)
"""频道窗口顶部蓝条（写着服务器名）的宽、高（px）：整个窗口按它定位"""
CELL_OFFSET = (33, 39)
"""第 1 行第 1 列频道格子中心相对蓝条左上角的偏移（px）"""
CELL_GAP = (70, 22)
"""频道格子的横向、纵向间距（px）"""
GRID = (5, 6)
"""频道窗口一屏 5 行 6 列"""
CHANGE_BUTTON_OFFSET = (325, 157)
"""「切换频道」按钮相对蓝条左上角的偏移（px）"""
CANCEL_BUTTON_OFFSET = (404, 157)
"""「取消」按钮相对蓝条左上角的偏移（px）"""
GRID_MIN_CELLS = 24
"""30 个格子里至少这么多个看着像格子才算频道窗口（没选中的是浅灰，选中的是青色）"""

ATTEMPTS = 4
"""最多试几次：点了频道没反应就换一个再试"""
STEP_TIMEOUT = 2.0
"""菜单、频道窗口等多久没出现就重试（秒）"""
BLACK_TIMEOUT = 6.0
"""点了频道后等多久没黑屏算没反应（秒）。录像里成功时约 1.3 秒黑屏，没反应的那次 4.6 秒后还没有"""
LOAD_TIMEOUT = 30.0
"""黑屏后最多等多久画面回来（秒）。录像里约 3.5 秒"""
BLACK_LEVEL = 8.0
"""画面平均亮度低于这个算黑屏（实际约 0.2，只有一个鼠标光标；正常画面 130 以上）"""
RECOVER_LEVEL = 30.0
"""画面平均亮度高于这个算回来了。录像里正常约 145；定得低一点，暗色地图也能认出来"""
BLACK_FRAMES = 2
"""连续这么多次看到黑屏才算，闪一下不算"""
POLL_SECONDS = 0.1
"""等界面变化时多久看一次（秒）"""
DOUBLE_CLICK_WAIT = 0.5
"""双击后等这么久还开着窗口，就改点「切换频道」按钮（秒）"""

RETRY_SECONDS = 60
"""没换成的话这么久后再试（秒）"""
MAX_IDLE_RUNS = 5
"""连着换这么多次都没打到怪就不再自动换：多半不是"这层没怪"，而是角色不在打怪的地方或认不出怪"""

DEBUG_DIR = os.path.join(_app_dir(), "logs", "channel_debug")
"""每次换频道的截图存这里：点之前（红圈标出点击位置）和没反应时的画面"""

QUIT_DEBUG_DIR = os.path.join(_app_dir(), "logs", "quit_debug")
"""每次定时结束游戏的截图存这里，没退出去时看这些图"""
QUIT_LEAVE_TIMEOUT = 6.0
"""点「退出游戏」或按回车后，等这么久还在游戏里就进行下一步（秒）"""
QUIT_LEFT_FRAMES = 5
"""连续这么多次看不到游戏画面（底部按钮没了、截不到图）才算离开了游戏"""
HUD_BUTTON = (67, 24)
"""游戏画面底部状态栏上青蓝色小按钮的宽、高（px），一直都在；登录界面没有"""
HUD_AREA = 0.92
"""这些按钮在画面高度这个比例以下"""


def _cyan_mask(frame: np.ndarray) -> np.ndarray:
    """菜单行、频道窗口蓝条、选中的频道格子都是这种青蓝色。"""
    b, g, r = [frame[..., i].astype(int) for i in range(3)]
    return ((b > 170) & (g > 130) & (g < 210) & (r < 110) & (b - r > 90)).astype(np.uint8)


def _boxes(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    """连通块的 (x, y, 宽, 高)。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= min_area]


def _scales(frame: np.ndarray) -> List[float]:
    """UI 的两种可能缩放：随画面等比缩放，或保持固定像素大小。

    常量是按 1920×1080 量的。老客户端（含不少怀旧服）的 UI 是固定像素，
    换分辨率只改变可视范围、按钮/菜单尺寸不变；也有的客户端会整体缩放。
    两种都试一遍，谁认出来用谁——不然 1366×768 下会按 0.711 去量固定的 UI，
    怎么都对不上（实测 2026-11-04 换频道一直卡在"按 Esc 后认不出菜单"）。
    """
    s = frame.shape[0] / REF_H
    return [s] if abs(s - 1.0) <= 0.02 else [s, 1.0]


def find_menu(frame: np.ndarray) -> Optional[Tuple[int, int]]:
    """游戏菜单开着 → 返回最上面一行「切换频道」的中心；没开返回 None。"""
    hit = find_menu_scaled(frame)
    return hit[0] if hit is not None else None


def find_menu_scaled(frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
    """菜单位置 + 命中的缩放倍数。

    退出游戏要按"第 6 行"算点击位置，行距必须用**同一个**缩放倍数：
    固定像素 UI 和等比缩放 UI 的行距差一截，混用会点到别的行去。
    """
    for s in _scales(frame):
        pos = _find_menu_at(frame, s)
        if pos is not None:
            return pos, s
    return None


def _find_menu_at(frame: np.ndarray, s: float) -> Optional[Tuple[int, int]]:
    """按缩放倍数 s 找菜单（实现见 find_menu）。"""
    w, h = MENU_ROW[0] * s, MENU_ROW[1] * s
    rows = sorted((b for b in _boxes(_cyan_mask(frame), int(w * h * 0.4))
                   if abs(b[2] - w) <= 14 * s and abs(b[3] - h) <= 6 * s), key=lambda b: b[1])
    for top in rows:
        run, y = 1, top[1]
        for b in rows:
            if abs(b[0] - top[0]) <= 6 * s and abs(b[1] - (y + MENU_ROW_GAP * s)) <= 6 * s:
                run, y = run + 1, b[1]
        if run >= MENU_MIN_ROWS:
            return top[0] + top[2] // 2, top[1] + top[3] // 2
    return None


class ChannelDialog:
    """认出来的频道窗口：格子中心、当前频道、两个按钮的位置。"""

    def __init__(self, bar: Tuple[int, int], s: float, current: Optional[int]):
        self.bar, self.s, self.current = bar, s, current

    def cell(self, index: int) -> Tuple[int, int]:
        """第 index 格（从 0 开始，从左到右、从上到下）的中心。"""
        row, col = divmod(index, GRID[1])
        return (self.bar[0] + int((CELL_OFFSET[0] + col * CELL_GAP[0]) * self.s),
                self.bar[1] + int((CELL_OFFSET[1] + row * CELL_GAP[1]) * self.s))

    def button(self, offset: Tuple[int, int]) -> Tuple[int, int]:
        return self.bar[0] + int(offset[0] * self.s), self.bar[1] + int(offset[1] * self.s)


def find_dialog(frame: np.ndarray) -> Optional[ChannelDialog]:
    """频道窗口开着 → 返回 ChannelDialog；没开返回 None。"""
    for s in _scales(frame):
        dialog = _find_dialog_at(frame, s)
        if dialog is not None:
            return dialog
    return None


def _find_dialog_at(frame: np.ndarray, s: float) -> Optional[ChannelDialog]:
    """按缩放倍数 s 找频道窗口（实现见 find_dialog）。"""
    w, h = DIALOG_BAR[0] * s, DIALOG_BAR[1] * s
    cyan = _cyan_mask(frame)
    for x, y, bw, bh in _boxes(cyan, int(w * h * 0.5)):
        if abs(bw - w) > 50 * s or abs(bh - h) > 8 * s:
            continue
        d = ChannelDialog((x, y), s, None)
        cells, current = 0, None
        for i in range(GRID[0] * GRID[1]):
            cx, cy = d.cell(i)
            # 看格子左端（中间有字）
            px = frame[max(0, cy - 2):cy + 3, max(0, cx - int(26 * s)):max(1, cx - int(22 * s))]
            if px.size == 0:
                continue
            b, g, r = px.reshape(-1, 3).mean(axis=0)
            if b - r > 60:  # 青色（格子左端有个白色小三角，所以看平均色而不是单个点）
                cells, current = cells + 1, i
            elif min(b, g, r) > 170 and max(b, g, r) - min(b, g, r) < 40:
                cells += 1
        if cells >= GRID_MIN_CELLS:
            d.current = current
            return d
    return None


def is_black(frame: np.ndarray) -> bool:
    return float(frame[::4, ::4].mean()) < BLACK_LEVEL


def in_game(frame: np.ndarray) -> bool:
    """画面是不是在游戏里（底部状态栏的按钮还在）。登录界面、黑屏都不是。"""
    top = int(frame.shape[0] * HUD_AREA)
    roi = frame[top:]
    for s in _scales(frame):
        w, h = HUD_BUTTON[0] * s, HUD_BUTTON[1] * s
        if any(abs(b[2] - w) <= 14 * s and abs(b[3] - h) <= 8 * s
               for b in _boxes(_cyan_mask(roi), int(w * h * 0.4))):
            return True
    return False


class ChannelTimer:
    """多久没攻击、该不该换频道。时间都由调用方传进来。"""

    def __init__(self):
        self.idle_from = 0.0  # 从这时起算没攻击（开始、补飞镖完、测谎完、换频道完）
        self.runs = 0  # 上次放技能之后换了几次

    def start(self, now: float):
        self.idle_from, self.runs = now, 0

    def restart(self, now: float):
        """暂停打怪的事做完了：停着的这段时间不算没攻击。"""
        self.idle_from = now

    def due(self, now: float, last_cast: float, idle_seconds: float) -> bool:
        if last_cast > self.idle_from:
            self.runs = 0
        return (idle_seconds > 0 and now - max(last_cast, self.idle_from) >= idle_seconds
                and self.runs < MAX_IDLE_RUNS)

    def finished(self, now: float, ok: bool, idle_seconds: float):
        """换完了：换成了重新计时，没换成 RETRY_SECONDS 后再试。"""
        self.runs += 1
        self.idle_from = now if ok else now - max(0.0, idle_seconds - RETRY_SECONDS)


class ChannelSwitcher:
    def __init__(self, capture, executor, config, on_log: Optional[Callable[[str], None]] = None):
        self.capture = capture
        self.executor = executor
        self.config = config
        self._log = on_log or (lambda m: None)
        self.abort: Callable[[], bool] = lambda: False  # 要立刻停下（点了停止、测谎弹窗）
        self.on_frame: Callable[[np.ndarray], None] = lambda f: None  # 每张截图都给它（测谎监视不能断）
        self.aborted = False  # 上次是中途停下的，窗口可能还开着，等 close_after_abort 收尾
        self._last_index = -1
        self._tried = False
        self._frame: Optional[np.ndarray] = None
        self._shot_dir = DEBUG_DIR
        self._shot_n = 0

    # ---- 公开接口 ----

    def run(self) -> bool:
        """换一次频道并按住方向键走一段，换成返回 True。调用方负责暂停其他操作。"""
        self._shot_dir = os.path.join(DEBUG_DIR, time.strftime("%Y%m%d_%H%M%S"))
        self._shot_n = 0
        self._tried = False
        self.aborted = False
        self._log(f"[换频道] 开始（截图存到 {os.path.abspath(self._shot_dir)}）")
        try:
            for attempt in range(1, ATTEMPTS + 1):
                if self.abort():
                    return False
                if self._try_once(attempt):
                    self._hold_after()
                    return True
            if self.abort():
                return False
            self._log(f"[换频道] 试了 {ATTEMPTS} 次都没换成，放弃本次")
            self._close_windows()
            return False
        except Exception as e:
            self._log(f"[换频道] 出错: {e}")
            self._close_windows()
            return False
        finally:
            if self.abort():
                self.aborted = True
                self._log("[换频道] 中途停下，现在不碰窗口（测谎的话，通过后再关）")

    def close_after_abort(self):
        """测谎结束后，把中途停下时可能还开着的菜单和频道窗口关掉；没有中途停下就什么都不做。"""
        if self.aborted:
            self.aborted = False
            try:
                self._close_windows()
            except Exception as e:
                self._log(f"[换频道] 关窗口出错: {e}")

    # ---- 流程 ----

    def _try_once(self, attempt: int) -> bool:
        frame = self._grab()
        if frame is not None and is_black(frame):  # 上一次点的其实生效了，只是黑屏来得晚
            return self._wait_loaded()
        dialog = self._open_dialog()
        if self.abort():
            return False
        if dialog is None:
            self._log(f"[换频道] 第 {attempt} 次：频道窗口没打开")
            self._save(self._grab(), "频道窗口没打开")
            self._close_windows()
            return False
        index = self._next_index(dialog)
        row, col = divmod(index, GRID[1])
        x, y = dialog.cell(index)
        self._log(f"[换频道] 第 {attempt} 次：双击第 {row + 1} 行第 {col + 1} 列的频道 ({x},{y})"
                  f"，当前频道在{self._cell_name(dialog.current)}")
        self._save(self._grab(), "点频道之前", (x, y))
        if self.abort():
            return False
        self.executor.real_double_click(x, y)

        self._sleep(DOUBLE_CLICK_WAIT)
        frame = self._grab()
        if frame is not None and not self.abort() and find_dialog(frame) is not None:
            bx, by = dialog.button(CHANGE_BUTTON_OFFSET)
            self._log(f"[换频道] 双击后窗口还开着，点「切换频道」按钮 ({bx},{by})")
            self.executor.real_click(bx, by)

        if not self._wait(is_black, BLACK_TIMEOUT, BLACK_FRAMES):
            if not self.abort():
                self._log(f"[换频道] 第 {attempt} 次没反应：{BLACK_TIMEOUT:.0f} 秒内没有黑屏，换一个频道再试")
                self._save(self._grab(), "没反应")
                self._close_windows()
            return False
        return self._wait_loaded()

    def _wait_loaded(self) -> bool:
        """黑屏了：等画面回来。"""
        self._log("[换频道] 黑屏了，正在换频道，等画面回来")
        if not self._wait(lambda f: float(f[::4, ::4].mean()) > RECOVER_LEVEL, LOAD_TIMEOUT):
            if not self.abort():
                self._log(f"[换频道] {LOAD_TIMEOUT:.0f} 秒画面还没回来（掉线了？）")
            return False
        self._log("[换频道] 画面回来了，换频道成功")
        self._sleep(1.0)
        return True

    def _open_dialog(self) -> Optional[ChannelDialog]:
        """按 Esc 弹菜单 → 点「切换频道」→ 等频道窗口；每一步没出来就再来一遍。"""
        for _ in range(2):
            frame = self._grab()
            if frame is None or self.abort():
                return None
            dialog = find_dialog(frame)
            if dialog is not None:
                return dialog
            pos = find_menu(frame)
            if pos is None:
                self._log("[换频道] 按 Esc 打开游戏菜单")
                self.executor.press_key("esc")
                # 先等一下让菜单画出来，并把这一帧存下来：
                # 认不出菜单时，看这张图就能分清"Esc 没打开菜单"还是"菜单开了但没认出来"
                self._sleep(0.4)
                probe = self._grab()
                if not self._wait(lambda f: find_menu(f) is not None, STEP_TIMEOUT):
                    if probe is not None:
                        self._save(probe, "按Esc后没认出菜单")
                        hgt = probe.shape[0]
                        self._log(
                            f"[换频道] 按 Esc 后没认出菜单（画面 {probe.shape[1]}x{hgt}，"
                            f"期望菜单行 {MENU_ROW[0]}x{MENU_ROW[1]}（固定像素）或 "
                            f"{int(MENU_ROW[0] * hgt / REF_H)}x"
                            f"{int(MENU_ROW[1] * hgt / REF_H)}（按高度缩放））"
                        )
                    continue
                frame = self._frame
                pos = find_menu(frame)
            self._log(f"[换频道] 点「切换频道」({pos[0]},{pos[1]})")
            self._save(frame, "点菜单之前", pos)
            if self.abort():
                return None
            self.executor.real_click(pos[0], pos[1])
            if self._wait(lambda f: find_dialog(f) is not None, STEP_TIMEOUT):
                return find_dialog(self._frame)
        return None

    def _next_index(self, dialog: ChannelDialog) -> int:
        """下一个要去的格子：第一次去当前频道的下一格，没反应就继续往后，不重复试同一个，也不点当前频道。"""
        total = GRID[0] * GRID[1]
        start = dialog.current if dialog.current is not None and not self._tried else self._last_index
        index = (start + 1) % total
        if index == dialog.current:
            index = (index + 1) % total
        self._last_index, self._tried = index, True
        return index

    @staticmethod
    def _cell_name(index: Optional[int]) -> str:
        if index is None:
            return "这一屏外（没看到青色格子）"
        row, col = divmod(index, GRID[1])
        return f"第 {row + 1} 行第 {col + 1} 列"

    def _hold_after(self):
        """换完按住方向键走一段（默认右键 5 秒）。"""
        key = str(getattr(self.config, "channel_hold_key", "right"))
        seconds = float(getattr(self.config, "channel_hold_seconds", 5.0))
        if seconds <= 0 or self.abort():
            return
        self._log(f"[换频道] 按住 {key} 键 {seconds:.0f} 秒")
        self.executor.key_down(key)
        try:
            end = time.time() + seconds
            while time.time() < end and not self.abort():
                self._grab()
                time.sleep(POLL_SECONDS)
        finally:
            self.executor.key_up(key)

    def _close_windows(self):
        """关掉还开着的频道窗口和菜单。什么都没开时不按 Esc：按了反而会弹出菜单。"""
        for _ in range(3):
            frame = self._grab()
            if frame is None or self.abort():
                return
            dialog = find_dialog(frame)
            if dialog is not None:
                self.executor.real_click(*dialog.button(CANCEL_BUTTON_OFFSET))
            elif find_menu(frame) is not None:
                self.executor.press_key("esc")
            else:
                return
            self._sleep(0.5)

    # ---- 工具 ----

    def _grab(self) -> Optional[np.ndarray]:
        """截图并交给测谎监视；黑屏切换时截图可能失败，当成这次没看到。"""
        try:
            frame = self.capture.grab()
        except Exception:
            return None
        self.on_frame(frame)
        return frame

    def _sleep(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end and not self.abort():
            self._grab()
            time.sleep(POLL_SECONDS)

    def _wait(self, cond, timeout: float, frames: int = 1) -> bool:
        """等 cond(frame) 连续成立 frames 次（满足时的画面留在 self._frame）；超时或要停下返回 False。"""
        end, hits = time.time() + timeout, 0
        while time.time() < end and not self.abort():
            frame = self._grab()
            hits = hits + 1 if frame is not None and cond(frame) else 0
            if hits >= frames:
                self._frame = frame
                return True
            time.sleep(POLL_SECONDS)
        return False

    def _save(self, frame: Optional[np.ndarray], what: str, mark: Optional[Tuple[int, int]] = None):
        """调试截图：mark 给出时在点击位置画红圈。存不了就算了，不影响流程。"""
        if frame is None:
            return
        try:
            os.makedirs(self._shot_dir, exist_ok=True)
            img = frame.copy()
            if mark is not None:
                cv2.circle(img, (int(mark[0]), int(mark[1])), 18, (0, 0, 255), 3)
            self._shot_n += 1
            # 路径里有中文时 cv2.imwrite 会失败，用 imencode + tofile
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                buf.tofile(os.path.join(self._shot_dir, f"{self._shot_n:02d}_{what}.jpg"))
        except Exception:
            pass


def next_quit_at(hhmm, now: float) -> Optional[float]:
    """"06:30" 这个钟点下一次到来的时间戳（今天已经过了就是明天）；写得不对返回 None。"""
    try:
        hour, minute = str(hhmm).split(":")
        t = datetime.fromtimestamp(now).replace(hour=int(hour), minute=int(minute),
                                                second=0, microsecond=0)
    except ValueError:
        return None
    if t.timestamp() <= now:
        t += timedelta(days=1)
    return t.timestamp()


class GameQuitter(ChannelSwitcher):
    """定时结束游戏：按 Esc 弹菜单 → 点「退出游戏」→ 弹了确认框就按回车 → 等离开游戏。

    只要看到已经离开游戏（登录界面、窗口关了）就到此为止，不再按任何键：
    登录界面按回车会点「连接」，又登录回去。
    """

    def run(self) -> bool:
        """离开游戏了返回 True。调用方负责暂停其他操作。"""
        self._shot_dir = os.path.join(QUIT_DEBUG_DIR, time.strftime("%Y%m%d_%H%M%S"))
        self._shot_n = 0
        self.aborted = False
        self._log(f"[定时] 开始退出游戏（截图存到 {os.path.abspath(self._shot_dir)}）")
        try:
            for attempt in range(1, ATTEMPTS + 1):
                if self.abort():
                    return False
                if self._try_quit(attempt):
                    return True
            if not self.abort():
                self._log(f"[定时] 试了 {ATTEMPTS} 次还在游戏里，没退出去，看一下截图")
            return False
        except Exception as e:
            self._log(f"[定时] 退出游戏出错: {e}")
            return False
        finally:
            self.aborted = self.abort()
            if self.aborted:
                # 对齐换频道的收尾日志：02:00 的退出被测谎/手动停止打断时，
                # 日志里要能看出"退出流程发生过、被挂起、之后会接着退"
                self._log("[定时] 中途停下，退出流程挂起（测谎的话，结束后接着退）")

    def _try_quit(self, attempt: int) -> bool:
        frame = self._grab()
        if frame is None or not in_game(frame):
            self._log("[定时] 画面里看不到游戏（底部状态栏不在），不再按键")
            self._save(frame, "看不到游戏")
            return self._wait_left()  # 上一次点的生效了，或者本来就不在游戏里：不碰
        hit = find_menu_scaled(frame)
        if hit is None:
            self._log("[定时] 按 Esc 打开游戏菜单")
            self.executor.press_key("esc")
            if not self._wait(lambda f: find_menu(f) is not None, STEP_TIMEOUT):
                self._log(f"[定时] 第 {attempt} 次：游戏菜单没打开")
                self._save(self._grab(), "菜单没打开")
                return False
            frame = self._frame
            hit = find_menu_scaled(frame)
            if hit is None:
                return False
        pos, scale = hit
        # 行距必须用认出菜单时那个缩放倍数（固定像素 UI 与等比缩放 UI 差一截）
        x = pos[0]
        y = pos[1] + int(MENU_ROW_GAP * MENU_QUIT_ROW * scale)
        self._log(f"[定时] 点「退出游戏」({x},{y})")
        self._save(frame, "点退出之前", (x, y))
        if self.abort():
            return False
        self.executor.real_click(x, y)
        if self._wait_left():
            return True
        # 还在游戏里。菜单还开着 = 没点上，下一轮再点；菜单关了多半是弹了确认框，才按回车。
        # 已经离开游戏就不能再按：登录界面会点「连接」，窗口没了回车会落到别的程序上
        frame = self._grab()
        if frame is None or not in_game(frame):
            return self._wait_left()
        if self.abort():
            return False
        if find_menu(frame) is not None:
            self._log("[定时] 点了之后菜单还开着，再点一次")
            self._save(frame, "点了没反应")
            return False
        self._log("[定时] 点了之后还在游戏里，按回车确认")
        self._save(frame, "点退出之后")
        self.executor.press_key("enter")
        if self._wait_left():
            return True
        self._save(self._grab(), "按回车之后")
        return False

    def _wait_left(self) -> bool:
        """等离开游戏（连续看不到游戏画面）；超时或要停下返回 False。"""
        end, hits = time.time() + QUIT_LEAVE_TIMEOUT, 0
        while time.time() < end and not self.abort():
            frame = self._grab()
            hits = hits + 1 if frame is None or not in_game(frame) else 0
            if hits >= QUIT_LEFT_FRAMES:
                return True
            time.sleep(POLL_SECONDS)
        return False
