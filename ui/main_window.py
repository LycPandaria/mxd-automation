"""MXD 游戏辅助控制台主窗口。

布局：
  顶部: 窗口锁定 (下拉选择 + 刷新 + 锁定)
  左侧: 实时预览 (检测框叠加) + 开始/停止 + 日志
  右侧: 配置面板 (检测 / 血量 / 蓝量 / 战斗 / 热键)

依赖：
  - ``src.main.Automation``：主循环
  - ``src.utils.config_loader``：配置加载
  - ``src.perception``：检测器与区域颜色识别
  - ``ui.preview_label.PreviewLabel``：预览与框选
"""
import time

import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QComboBox, QSlider, QGroupBox, QGridLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QPlainTextEdit, QCheckBox, QFileDialog,
    QMessageBox, QSplitter, QAbstractItemView, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from src.utils.config_loader import (
    load_config, save_config, save_user_config, config_path,
)
from src.perception.yolo_detector import create_detector
from src.perception.hp_mp_detector import detect_region_color
from src.main import Automation

from .preview_label import PreviewLabel


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

        g.addWidget(QLabel("自身名字:"), 4, 0)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入角色名, 如 我是立立")
        g.addWidget(self.name_edit, 4, 1, 1, 2)
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
        self.name_edit.setText(c.self_name)
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
        c.self_name = self.name_edit.text().strip()
        c.hp_key = self.hp_key_edit.text().strip()
        c.hp_threshold = self.hp_thr_spin.value() / 100
        # hp_color / mp_color 由框选区域时自动写入，此处不覆盖
        # mp
        c.mp_key = self.mp_key_edit.text().strip()
        c.mp_threshold = self.mp_thr_spin.value() / 100
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
        save_user_config(self.config)
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

    @staticmethod
    def _set_swatch(swatch, rgb):
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #333;"
        )

    def _on_region_selected(self, target, x, y, w, h):
        """框选区域后自动识别颜色并写入配置。"""
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
        save_user_config(self.config)

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
        self.automation._render_name_template()  # 重新渲染名字模板
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
        # 按类别使用不同颜色绘制检测框
        monster_classes = [c.strip() for c in self.config.monster_classes.split(",")]
        floor_classes = [c.strip() for c in self.config.floor_classes.split(",")]
        rope_classes = [c.strip() for c in self.config.rope_classes.split(",")]
        player_classes = [c.strip() for c in self.config.player_classes.split(",")]
        for d in detections:
            if d.cls_name in player_classes:
                color = (255, 0, 255)        # 品红: 其他玩家
            elif d.cls_name in monster_classes:
                color = (0, 255, 0)          # 绿色: 怪物
            elif d.cls_name in rope_classes:
                color = (0, 165, 255)        # 橙色: 绳索
            elif d.cls_name in floor_classes:
                color = (128, 128, 128)      # 灰色: 地板
            else:
                color = (0, 165, 255)        # 默认橙色
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
        # 自身位置：HP条偏移优先，名字模板匹配兜底
        self_pos = self.automation._locate_self(frame)
        if self_pos:
            sx, sy = self_pos
            cv2.circle(disp, (sx, sy), 6, (255, 255, 0), -1)
            cv2.putText(disp, "self", (sx + 8, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
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