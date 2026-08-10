"""预览标签控件：显示游戏画面，支持拖拽框选血/蓝条区域。

``region_selected`` 信号以原始帧坐标发出（已做缩放换算）。

从原 ``app/main_window.py`` 中的 ``PreviewLabel`` 类原样迁移，未做行为改动。
"""
from PyQt5.QtWidgets import QLabel, QRubberBand
from PyQt5.QtCore import Qt, pyqtSignal, QRect


class PreviewLabel(QLabel):
    """预览标签：显示游戏画面，支持拖拽框选血条区域。"""

    region_selected = pyqtSignal(str, int, int, int, int)  # target, x, y, w, h

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.setText("尚未锁定窗口或未启动")
        self._selecting = False
        self._select_target = "hp"     # 当前框选目标: "hp" 或 "mp"
        self._origin = None
        self._rubber = None
        self._frame_size = (0, 0)      # 当前帧的原始尺寸 (w, h)
        self._pixmap_rect = QRect()    # 当前显示 pixmap 在 label 内的实际矩形

    def set_select_mode(self, on: bool, target: str = ""):
        self._selecting = on
        if target:
            self._select_target = target
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def update_frame(self, pixmap):
        from PyQt5.QtGui import QPixmap
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        # 计算 letterbox 偏移
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._pixmap_rect = QRect(x, y, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def set_frame_size(self, w, h):
        self._frame_size = (w, h)

    def mousePressEvent(self, event):
        if not self._selecting or event.button() != Qt.LeftButton:
            return
        self._origin = event.pos()
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber.setGeometry(QRect(self._origin, self._origin))
        self._rubber.show()

    def mouseMoveEvent(self, event):
        if self._rubber is not None and self._origin is not None:
            self._rubber.setGeometry(
                QRect(self._origin, event.pos()).normalized()
            )

    def mouseReleaseEvent(self, event):
        if not self._selecting or self._rubber is None:
            return
        rect = QRect(self._origin, event.pos()).normalized()
        self._rubber.hide()
        self._rubber = None
        self._origin = None
        self.set_select_mode(False)

        # 限定到 pixmap 区域内
        inter = rect.intersected(self._pixmap_rect)
        if inter.width() < 3 or inter.height() < 3:
            return

        # 换算到原始帧坐标
        fw, fh = self._frame_size
        if fw == 0 or fh == 0:
            return
        sx = fw / self._pixmap_rect.width()
        sy = fh / self._pixmap_rect.height()
        fx = int((inter.x() - self._pixmap_rect.x()) * sx)
        fy = int((inter.y() - self._pixmap_rect.y()) * sy)
        fw_sel = int(inter.width() * sx)
        fh_sel = int(inter.height() * sy)
        self.region_selected.emit(self._select_target, fx, fy, fw_sel, fh_sel)
