"""MXD 游戏辅助控制台主窗口。

布局:
  顶部: 窗口锁定 (下拉选择 + 刷新 + 锁定)
  左侧: 实时预览 (检测框叠加) + 开始/停止 + 日志
  右侧: 配置面板 (检测 / 血量 / 战斗 / 热键)
"""
import time

import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QSlider, QGroupBox, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QPlainTextEdit, QCheckBox, QFileDialog,
    QMessageBox, QSplitter, QAbstractItemView, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainterPath, QCursor

from .config import load_config, save_config, config_path
from .detector import create_detector
from .automation import Automation


class PreviewLabel(QLabel):
    """预览标签：显示游戏画面，支持拖拽框选血条区域。

    region_selected 信号以原始帧坐标发出（已做缩放换算）。
    """

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

    def update_frame(self, pixmap: QPixmap):
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
        from PyQt5.QtWidgets import QRubberBand
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


class ColorPickerOverlay(QWidget):
    """全屏屏幕拾色器：点击任意位置获取该像素颜色。"""

    color_picked = pyqtSignal(int, int, int)  # r, g, b
    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setWindowState(Qt.WindowFullScreen)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._mag_size = 150
        self._last_pixel = None  # 缓存上次像素颜色

        # 用 QTimer 节流放大镜更新（30fps），避免 paintEvent 卡顿
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, event):
        super().showEvent(event)
        self.grabMouse()
        self.setFocus()
        self.setFocusPolicy(Qt.StrongFocus)
        self._timer.start()

    def closeEvent(self, event):
        self._timer.stop()
        self.releaseMouse()
        super().closeEvent(event)

    def _tick(self):
        """定时器回调：刷新放大镜区域。"""
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            gp = event.globalPos()
            try:
                r, g, b = self._get_pixel_color(gp.x(), gp.y())
                self.color_picked.emit(r, g, b)
            except Exception:
                self.cancelled.emit()
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        # 半透明黑色背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

        cursor_pos = self.mapFromGlobal(QCursor.pos())

        # 放大镜
        self._draw_magnifier(painter, cursor_pos)

        # 十字准星
        painter.setPen(QPen(QColor(255, 255, 0), 2))
        cx, cy = cursor_pos.x(), cursor_pos.y()
        painter.drawLine(cx - 25, cy, cx - 6, cy)
        painter.drawLine(cx + 6, cy, cx + 25, cy)
        painter.drawLine(cx, cy - 25, cx, cy - 6)
        painter.drawLine(cx, cy + 6, cx, cy + 25)

        # 准星中心点
        painter.setBrush(QColor(255, 0, 0))
        painter.drawEllipse(cx - 2, cy - 2, 4, 4)

        # 顶部提示
        painter.setPen(QPen(QColor(255, 255, 255)))
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(20, 30, "点击鼠标取色  |  ESC 取消")

        # 底部 RGB 信息（使用缓存颜色，避免每次 paintEvent 都调用 pixel()）
        if self._last_pixel:
            r, g, b = self._last_pixel
            font.setBold(False)
            font.setPointSize(10)
            painter.setFont(font)
            gp = QCursor.pos()
            color_text = f"RGB: ({r}, {g}, {b})    HEX: #{r:02X}{g:02X}{b:02X}    坐标: ({gp.x()}, {gp.y()})"
            painter.drawText(20, self.height() - 20, color_text)

    def _draw_magnifier(self, painter, local_pos):
        """在鼠标旁边绘制放大镜。"""
        gp = QCursor.pos()
        size = 20

        try:
            img = ImageGrab.grab(bbox=(
                gp.x() - size // 2,
                gp.y() - size // 2,
                gp.x() + size // 2,
                gp.y() + size // 2,
            ))
            # 同时更新缓存颜色
            arr = np.array(img)
            center = arr[arr.shape[0] // 2, arr.shape[1] // 2]
            self._last_pixel = (int(center[0]), int(center[1]), int(center[2]))
        except Exception:
            return

        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._mag_size, self._mag_size,
            Qt.KeepAspectRatio, Qt.FastTransformation
        )

        # 放大镜显示位置（鼠标右下侧）
        margin = 30
        mag_x = local_pos.x() + margin
        mag_y = local_pos.y() + margin

        # 如果超出右/下边界，放到左上
        if mag_x + self._mag_size > self.width():
            mag_x = local_pos.x() - self._mag_size - margin
        if mag_y + self._mag_size > self.height():
            mag_y = local_pos.y() - self._mag_size - margin

        # 白色外边框
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(mag_x, mag_y, self._mag_size, self._mag_size)

        # 圆形裁剪后绘制放大图像
        path = QPainterPath()
        path.addEllipse(mag_x, mag_y, self._mag_size, self._mag_size)
        painter.setClipPath(path)
        painter.drawPixmap(mag_x, mag_y, pixmap)
        painter.setClipping(False)

        # 中心十字标记
        center_x = mag_x + self._mag_size // 2
        center_y = mag_y + self._mag_size // 2
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(center_x - 8, center_y, center_x + 8, center_y)
        painter.drawLine(center_x, center_y - 8, center_x, center_y + 8)

    @staticmethod
    def _get_pixel_color(x, y):
        """获取指定屏幕坐标的像素颜色。"""
        import pyautogui
        pixel = pyautogui.pixel(x, y)
        return pixel.red, pixel.green, pixel.blue


class MainWindow(QMainWindow):
    # 跨线程信号：自动化线程 → GUI 线程
    log_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(object, object, object, object)  # frame, detections, hp, mp
    hotkey_signal = pyqtSignal()  # F12 触发

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXD 游戏辅助控制台")
        self.resize(1180, 760)

        self.config = load_config()
        self.detector = create_detector(
            self.config.model_path, self.config.confidence, self._log
        )
        self.automation = Automation(
            self.config, self.detector,
            on_log=self.log_signal.emit,
            on_frame=self.frame_signal.emit,
        )

        self._fps_counter = [0, time.time()]
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_fps)

        self._init_ui()
        self._load_config_to_ui()

        self.log_signal.connect(self._on_log)
        self.frame_signal.connect(self._on_frame)
        self.hotkey_signal.connect(self._toggle_run)

        self._register_hotkey()

    # ---------------- UI 构建 ----------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 顶部: 窗口锁定
        root.addWidget(self._build_window_bar())

        # 主体: 左侧预览+日志 / 右侧配置
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.statusBar().showMessage("就绪。请锁定游戏窗口后点击「开始」。")

    def _build_window_bar(self):
        box = QGroupBox("游戏窗口")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("窗口标题:"))
        self.window_combo = QComboBox()
        self.window_combo.setEditable(True)
        self.window_combo.setMinimumWidth(360)
        h.addWidget(self.window_combo, 1)
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self._refresh_windows)
        h.addWidget(self.refresh_btn)
        self.lock_btn = QPushButton("锁定窗口")
        self.lock_btn.clicked.connect(self._lock_window)
        h.addWidget(self.lock_btn)
        self.win_status = QLabel("未锁定")
        self.win_status.setStyleSheet("color: #c0392b; font-weight:bold;")
        h.addWidget(self.win_status)
        return self._wrap(box)

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)

        # 预览
        self.preview = PreviewLabel()
        self.preview.region_selected.connect(self._on_region_selected)
        v.addWidget(self.preview, 1)

        # FPS
        self.fps_label = QLabel("FPS: -")
        self.fps_label.setStyleSheet("color: #27ae60; font-weight:bold;")
        v.addWidget(self.fps_label)

        # 控制按钮
        ctl = QHBoxLayout()
        self.run_btn = QPushButton("▶ 开始自动打怪")
        self.run_btn.setStyleSheet(
            "padding:10px; font-size:14px; font-weight:bold; "
            "background-color:#27ae60; color:white;"
        )
        self.run_btn.clicked.connect(self._toggle_run)
        ctl.addWidget(self.run_btn)
        self.hp_pick_btn = QPushButton("框选血条区域")
        self.hp_pick_btn.clicked.connect(
            lambda: self.preview.set_select_mode(True, "hp")
        )
        ctl.addWidget(self.hp_pick_btn)
        self.mp_pick_btn = QPushButton("框选蓝条区域")
        self.mp_pick_btn.clicked.connect(
            lambda: self.preview.set_select_mode(True, "mp")
        )
        ctl.addWidget(self.mp_pick_btn)
        v.addLayout(ctl)

        # 日志
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(180)
        self.log_box.setStyleSheet("background-color:#111; color:#ddd;")
        v.addWidget(self.log_box)
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.addWidget(self._build_detect_group())
        v.addWidget(self._build_hp_group())
        v.addWidget(self._build_mp_group())
        v.addWidget(self._build_combat_group())
        v.addStretch()
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self._save_config)
        v.addWidget(save_btn)
        return panel

    def _build_detect_group(self):
        box = QGroupBox("检测设置 (YOLO)")
        g = QGridLayout(box)
        g.addWidget(QLabel("模型路径:"), 0, 0)
        self.model_edit = QLineEdit()
        g.addWidget(self.model_edit, 0, 1)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._browse_model)
        g.addWidget(browse, 0, 2)

        g.addWidget(QLabel("置信度:"), 1, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100:.2f}")
        )
        self.conf_label = QLabel("0.50")
        self.conf_label.setMinimumWidth(40)
        g.addWidget(self.conf_slider, 1, 1)
        g.addWidget(self.conf_label, 1, 2)

        g.addWidget(QLabel("怪物类别:"), 2, 0)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("逗号分隔, 如 monster,boss")
        g.addWidget(self.classes_edit, 2, 1, 1, 2)

        g.addWidget(QLabel("检测FPS:"), 3, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(8)
        g.addWidget(self.fps_spin, 3, 1, 1, 2)
        return box

    def _build_hp_group(self):
        box = QGroupBox("血量设置")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("按键:"))
        self.hp_key_edit = QLineEdit()
        self.hp_key_edit.setFixedWidth(50)
        self.hp_key_edit.setPlaceholderText("f")
        h.addWidget(self.hp_key_edit)

        h.addWidget(QLabel("阈值%:"))
        self.hp_thr_spin = QSpinBox()
        self.hp_thr_spin.setRange(0, 100)
        self.hp_thr_spin.setValue(50)
        self.hp_thr_spin.setFixedWidth(55)
        h.addWidget(self.hp_thr_spin)

        h.addWidget(QLabel("颜色:"))
        self.hp_swatch = QLabel()
        self.hp_swatch.setFixedSize(24, 24)
        self.hp_swatch.setStyleSheet("background-color: rgb(255,0,0); border:1px solid #333;")
        h.addWidget(self.hp_swatch)

        h.addWidget(QLabel("区域:"))
        self.hp_region_label = QLabel("未设置")
        self.hp_region_label.setMinimumWidth(60)
        h.addWidget(self.hp_region_label)

        h.addStretch()
        return box

    def _build_mp_group(self):
        box = QGroupBox("蓝量设置")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("按键:"))
        self.mp_key_edit = QLineEdit()
        self.mp_key_edit.setFixedWidth(50)
        self.mp_key_edit.setPlaceholderText("g")
        h.addWidget(self.mp_key_edit)

        h.addWidget(QLabel("阈值%:"))
        self.mp_thr_spin = QSpinBox()
        self.mp_thr_spin.setRange(0, 100)
        self.mp_thr_spin.setValue(30)
        self.mp_thr_spin.setFixedWidth(55)
        h.addWidget(self.mp_thr_spin)

        h.addWidget(QLabel("颜色:"))
        self.mp_swatch = QLabel()
        self.mp_swatch.setFixedSize(24, 24)
        self.mp_swatch.setStyleSheet("background-color: rgb(0,120,255); border:1px solid #333;")
        h.addWidget(self.mp_swatch)

        h.addWidget(QLabel("区域:"))
        self.mp_region_label = QLabel("未设置")
        self.mp_region_label.setMinimumWidth(60)
        h.addWidget(self.mp_region_label)

        h.addStretch()
        return box

    def _build_combat_group(self):
        box = QGroupBox("战斗设置")
        v = QVBoxLayout(box)

        g = QGridLayout()
        g.addWidget(QLabel("选中目标键:"), 0, 0)
        self.target_key_edit = QLineEdit("tab")
        g.addWidget(self.target_key_edit, 0, 1)
        v.addLayout(g)

        self.move_cb = QCheckBox("检测到怪时点击移动到怪位置 (否则原地放技能)")
        v.addWidget(self.move_cb)

        # 技能表
        v.addWidget(QLabel("技能列表 (轮转释放):"))
        self.skill_table = QTableWidget(0, 3)
        self.skill_table.setHorizontalHeaderLabels(["名称", "按键", "冷却(秒)"])
        self.skill_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.skill_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        v.addWidget(self.skill_table)

        skill_btns = QHBoxLayout()
        add_btn = QPushButton("+ 添加")
        add_btn.clicked.connect(self._add_skill_row)
        del_btn = QPushButton("- 删除选中")
        del_btn.clicked.connect(self._del_skill_row)
        skill_btns.addWidget(add_btn)
        skill_btns.addWidget(del_btn)
        skill_btns.addStretch()
        v.addLayout(skill_btns)
        return box

    @staticmethod
    def _wrap(widget):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(widget)
        return w

    # ---------------- 配置 ↔ UI ----------------
    def _load_config_to_ui(self):
        c = self.config
        self.model_edit.setText(c.model_path)
        self.conf_slider.setValue(int(c.confidence * 100))
        self.classes_edit.setText(c.monster_classes)
        self.fps_spin.setValue(c.fps)
        self.hp_key_edit.setText(c.hp_key)
        self.hp_thr_spin.setValue(int(c.hp_threshold * 100))
        if c.hp_color:
            self._set_swatch(self.hp_swatch, c.hp_color)
        if c.hp_region:
            self.hp_region_label.setText(
                f"{c.hp_region[0]},{c.hp_region[1]} {c.hp_region[2]}x{c.hp_region[3]}"
            )
        # MP
        self.mp_key_edit.setText(c.mp_key)
        self.mp_thr_spin.setValue(int(c.mp_threshold * 100))
        if c.mp_color:
            self._set_swatch(self.mp_swatch, c.mp_color)
        if c.mp_region:
            self.mp_region_label.setText(
                f"{c.mp_region[0]},{c.mp_region[1]} {c.mp_region[2]}x{c.mp_region[3]}"
            )
        self.target_key_edit.setText(c.target_key)
        self.move_cb.setChecked(c.move_to_monster)
        # 技能表
        self.skill_table.setRowCount(0)
        for s in c.skills:
            self._add_skill_row(s.get("name", ""), s.get("key", ""), s.get("cooldown", 1.0))

    def _read_ui_to_config(self):
        c = self.config
        c.model_path = self.model_edit.text().strip()
        c.confidence = self.conf_slider.value() / 100
        c.monster_classes = self.classes_edit.text().strip() or "monster"
        c.fps = self.fps_spin.value()
        c.hp_key = self.hp_key_edit.text().strip()
        c.hp_threshold = self.hp_thr_spin.value() / 100
        c.hp_color = self.config.hp_color
        # mp
        c.mp_key = self.mp_key_edit.text().strip()
        c.mp_threshold = self.mp_thr_spin.value() / 100
        c.mp_color = self.config.mp_color
        c.target_key = self.target_key_edit.text().strip()
        c.move_to_monster = self.move_cb.isChecked()
        # 技能
        skills = []
        for r in range(self.skill_table.rowCount()):
            name = self.skill_table.item(r, 0).text() if self.skill_table.item(r, 0) else ""
            key = self.skill_table.item(r, 1).text() if self.skill_table.item(r, 1) else ""
            cd_text = self.skill_table.item(r, 2).text() if self.skill_table.item(r, 2) else "1.0"
            try:
                cd = float(cd_text)
            except ValueError:
                cd = 1.0
            if name or key:
                skills.append({"name": name, "key": key, "cooldown": cd})
        c.skills = skills

    def _add_skill_row(self, name="", key="", cd=1.0):
        r = self.skill_table.rowCount()
        self.skill_table.insertRow(r)
        self.skill_table.setItem(r, 0, QTableWidgetItem(str(name)))
        self.skill_table.setItem(r, 1, QTableWidgetItem(str(key)))
        self.skill_table.setItem(r, 2, QTableWidgetItem(str(cd)))

    def _del_skill_row(self):
        rows = {i.row() for i in self.skill_table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.skill_table.removeRow(r)

    def _save_config(self):
        self._read_ui_to_config()
        save_config(self.config)
        self._log(f"[配置] 已保存到 {config_path()}")

    # ---------------- 事件处理 ----------------
    def _refresh_windows(self):
        self.window_combo.clear()
        try:
            windows = self.automation.list_windows()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"枚举窗口失败: {e}")
            return
        for hwnd, title in windows:
            self.window_combo.addItem(title)
        if self.config.window_title:
            self.window_combo.setCurrentText(self.config.window_title)

    def _lock_window(self):
        title = self.window_combo.currentText().strip()
        if not title:
            QMessageBox.warning(self, "提示", "请先选择或输入窗口标题")
            return
        try:
            locked = self.automation.lock_window(title)
        except Exception as e:
            self.win_status.setText("锁定失败")
            self.win_status.setStyleSheet("color:#c0392b;font-weight:bold;")
            QMessageBox.warning(self, "锁定失败", str(e))
            return
        self.config.window_title = title
        self.win_status.setText(f"已锁定: {locked}")
        self.win_status.setStyleSheet("color:#27ae60;font-weight:bold;")
        self._log(f"[窗口] 已锁定: {locked}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型", "", "模型文件 (*.pt *.onnx)"
        )
        if path:
            self.model_edit.setText(path)

    def _start_color_picker(self, target):
        """启动屏幕拾色器，点击后获取颜色。"""
        self._color_target = target
        # 不传 parent，作为独立窗口显示，避免被主窗口遮挡
        self._color_picker = ColorPickerOverlay()
        self._color_picker.color_picked.connect(self._on_color_picked)
        self._color_picker.cancelled.connect(self._on_color_pick_cancelled)
        self._color_picker.showFullScreen()

    def _on_color_pick_cancelled(self):
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _on_color_picked(self, r, g, b):
        color = [r, g, b]
        if self._color_target == "mp":
            self.config.mp_color = color
            self._set_swatch(self.mp_swatch, color)
        else:
            self.config.hp_color = color
            self._set_swatch(self.hp_swatch, color)
        self._log(f"[取色] {self._color_target.upper()} 颜色已设置为 RGB({r},{g},{b})")
        self.showNormal()
        self.activateWindow()
        self.raise_()

    @staticmethod
    def _set_swatch(swatch, rgb):
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #333;"
        )

    def _on_region_selected(self, target, x, y, w, h):
        if target == "mp":
            self.config.mp_region = [x, y, w, h]
            self.mp_region_label.setText(f"{x},{y} {w}x{h}")
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.mp_color = color
                self._set_swatch(self.mp_swatch, color)
                self._log(f"[蓝量] 区域已设置: ({x},{y}) {w}x{h} | 自动识别颜色: RGB{tuple(color)}")
            else:
                self._log(f"[蓝量] 区域已设置: ({x},{y}) {w}x{h} | 颜色识别失败")
        else:
            self.config.hp_region = [x, y, w, h]
            self.hp_region_label.setText(f"{x},{y} {w}x{h}")
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.hp_color = color
                self._set_swatch(self.hp_swatch, color)
                self._log(f"[血量] 区域已设置: ({x},{y}) {w}x{h} | 自动识别颜色: RGB{tuple(color)}")
            else:
                self._log(f"[血量] 区域已设置: ({x},{y}) {w}x{h} | 颜色识别失败")

    def _detect_region_color(self, x, y, w, h):
        """从当前窗口截图中识别指定区域的主颜色。"""
        try:
            frame = self.automation.capture.grab()
            from .detector import detect_region_color
            return detect_region_color(frame, [x, y, w, h])
        except Exception:
            return None

    def _toggle_run(self):
        if self.automation.running:
            self.automation.stop()
            self.run_btn.setText("▶ 开始自动打怪")
            self.run_btn.setStyleSheet(
                "padding:10px;font-size:14px;font-weight:bold;"
                "background-color:#27ae60;color:white;"
            )
            return

        # 启动前: 读 UI → 存配置 → (必要时)重建检测器 → 锁窗口 → 启动
        self._read_ui_to_config()
        save_config(self.config)

        if not self.automation.window_locked:
            if self.config.window_title:
                try:
                    self.automation.lock_window(self.config.window_title)
                    self.win_status.setText(f"已锁定: {self.config.window_title}")
                    self.win_status.setStyleSheet("color:#27ae60;font-weight:bold;")
                except Exception as e:
                    QMessageBox.warning(self, "错误", f"锁定窗口失败: {e}")
                    return
            else:
                QMessageBox.warning(self, "提示", "请先锁定游戏窗口")
                return

        # 模型路径变化时重建检测器
        if self.config.model_path and self.config.model_path != getattr(self.detector, "_path", ""):
            self.detector = create_detector(
                self.config.model_path, self.config.confidence, self._log
            )
            self.automation.set_detector(self.detector)

        self.automation.config = self.config
        try:
            self.automation.start()
        except Exception as e:
            QMessageBox.warning(self, "启动失败", str(e))
            return

        self.run_btn.setText("⏹ 停止")
        self.run_btn.setStyleSheet(
            "padding:10px;font-size:14px;font-weight:bold;"
            "background-color:#c0392b;color:white;"
        )
        self._preview_timer.start(1000)

    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_frame(self, frame, detections, hp_ratio, mp_ratio):
        if frame is None:
            return
        h, w = frame.shape[:2]
        self.preview.set_frame_size(w, h)

        disp = frame.copy()
        # 画检测框
        for d in detections:
            color = (0, 255, 0) if d.cls_name in [
                c.strip() for c in self.config.monster_classes.split(",")
            ] else (0, 165, 255)
            cv2.rectangle(disp, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            cv2.putText(disp, f"{d.cls_name} {d.confidence:.2f}",
                        (d.x, max(0, d.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # 画血条区域
        if self.config.hp_region:
            rx, ry, rw, rh = self.config.hp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
        # 画蓝条区域
        if self.config.mp_region:
            rx, ry, rw, rh = self.config.mp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 128, 0), 1)
        # HP / MP 文本
        hp_text = f"HP: {hp_ratio:.0%}" if hp_ratio is not None else "HP: -"
        cv2.putText(disp, hp_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        mp_text = f"MP: {mp_ratio:.0%}" if mp_ratio is not None else "MP: -"
        cv2.putText(disp, mp_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.preview.update_frame(QPixmap.fromImage(qimg.copy()))

        self._fps_counter[0] += 1

    def _update_fps(self):
        now = time.time()
        fps = self._fps_counter[0] / max(0.001, now - self._fps_counter[1])
        self._fps_counter = [0, now]
        self.fps_label.setText(f"FPS: {fps:.1f}")
        if not self.automation.running:
            self._preview_timer.stop()

    def _log(self, msg):
        self.log_signal.emit(msg)

    def _register_hotkey(self):
        try:
            import keyboard
            keyboard.add_hotkey(self.config.start_stop_hotkey,
                                self.hotkey_signal.emit)
        except Exception as e:
            self._log(f"[热键] 注册 {self.config.start_stop_hotkey} 失败: {e}")

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        try:
            self.automation.stop()
        except Exception:
            pass
        try:
            import keyboard
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        super().closeEvent(event)
