"""测谎报警浮层：置顶 + 鼠标穿透的红框（贴在游戏窗口上）。

================================================================================
为什么需要它
================================================================================

  测谎弹窗出现时玩家多半正盯着游戏画面，而不是盯着本工具的窗口。
  只弹工具窗口里的横幅 / 只响一声，很容易错过（而错过 = 测谎失败受罚）。
  所以在游戏窗口上叠一层会闪的红框，视觉上无法忽略。

================================================================================
鼠标穿透
================================================================================

  小游戏要求玩家把真实鼠标光标压在移动图形上，浮层绝对不能吃掉鼠标事件：

  Qt 层: WA_TransparentForMouseEvents + WA_ShowWithoutActivating
  Win32 层: WS_EX_TRANSPARENT（点穿）+ WS_EX_NOACTIVATE（不抢焦点）
            + WS_EX_TOOLWINDOW（不出现在任务栏/Alt+Tab）

  只这两层都设上才真正"点穿且不抢焦点"，因此这里直接改 exstyle。

================================================================================
不做的事
================================================================================

  不把工具主窗口拉到前台（不 SetForegroundWindow / 不 activateWindow）：
  游戏窗口需要保持前台，抢焦点可能让小游戏收不到输入或直接暂停。
  提示只靠"声音 + 任务栏闪烁 + 本浮层"。
"""
import ctypes

from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

# ---- Win32 扩展窗口样式 ----
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020   # 鼠标事件穿透到下层窗口
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000    # 点击/显示均不激活本窗口
WS_EX_TOOLWINDOW = 0x00000080    # 不显示在任务栏

BORDER_WIDTH = 12                # 红框粗细（px）
BANNER_HEIGHT = 64               # 顶部文字带高度（px）
TASKBAR_FLASH_MS = 15000         # 任务栏闪烁时长；用固定时长而非"直到激活窗口"，
                                 # 因为玩家可能按热键（不激活窗口）来确认报警


class AlarmOverlay(QWidget):
    """贴在游戏窗口上的置顶红框浮层（闪烁 + 顶部提示文字）。

    用法:
        ov = AlarmOverlay()
        ov.show_alarm((left, top, w, h), "测谎弹窗！请接管鼠标")
        ov.stop_alarm()
    """

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool,
        )
        # 背景透明（只画描边）+ 不吃鼠标事件 + 显示时不激活
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._text = ""
        self._blink = False
        self._timer = QTimer(self)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._tick)
        self.hide()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def show_alarm(self, rect, text: str):
        """在指定屏幕矩形处显示报警浮层。

        Args:
            rect: (left, top, width, height)，屏幕坐标（游戏窗口 GetWindowRect）
            text: 顶部提示文字
        """
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            raise ValueError(f"无效矩形: {rect}")
        self._text = text
        self.setGeometry(QRect(int(left), int(top), int(width), int(height)))
        self._apply_click_through()
        self._blink = False
        self.show()
        self._timer.start()
        self.update()

    def stop_alarm(self):
        """停止闪烁并隐藏浮层。"""
        self._timer.stop()
        self.hide()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _apply_click_through(self):
        """给原生窗口加上 WS_EX_TRANSPARENT / NOACTIVATE（点穿 + 不抢焦点）。"""
        try:
            hwnd = int(self.winId())   # 触发原生窗口创建
            user32 = ctypes.windll.user32
            ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ex |= (WS_EX_TRANSPARENT | WS_EX_LAYERED
                   | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
        except Exception:
            # 失败也不影响报警：声音 + 任务栏闪烁 + 主窗口横幅仍在
            pass

    def _tick(self):
        self._blink = not self._blink
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()

        # ---- 四周红框（闪烁）----
        color = QColor(255, 70, 70) if self._blink else QColor(165, 10, 10)
        pen = QPen(color)
        pen.setWidth(BORDER_WIDTH)
        p.setPen(pen)
        half = BORDER_WIDTH // 2
        p.drawRect(QRect(half, half, max(1, w - BORDER_WIDTH),
                         max(1, h - BORDER_WIDTH)))

        # ---- 顶部文字带（深底白字，不遮挡下方游戏区域）----
        band = QRect(0, 0, w, BANNER_HEIGHT)
        p.fillRect(band, QColor(165, 10, 10, 235) if self._blink
                   else QColor(80, 0, 0, 235))
        p.setPen(QPen(QColor(255, 255, 255)))
        font = QFont("Microsoft YaHei", 20)
        font.setBold(True)
        p.setFont(font)
        p.drawText(band, Qt.AlignCenter,
                   self._text or "测谎弹窗！请立即接管鼠标作答")
        p.end()


def flash_taskbar(widget):
    """任务栏闪烁提醒（不抢焦点，固定时长）。"""
    try:
        QApplication.alert(widget, TASKBAR_FLASH_MS)
    except Exception:
        pass
