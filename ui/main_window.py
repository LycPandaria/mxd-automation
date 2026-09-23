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
import ctypes
import os
import sys
import threading
import time
from datetime import datetime

# 确保项目根目录在 sys.path 中，使得 `src` / `ui` 可作为顶层包导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSlider, QGroupBox,
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QCheckBox, QFileDialog, QMessageBox, QSplitter,
    QAbstractItemView, QSpinBox, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QMetaObject
from PyQt5.QtGui import QImage, QPixmap

from src.utils.config_loader import (
    load_config, save_config, save_user_config, config_path, resolve_model_path,
    APP_DIR, BUNDLE_DIR,
)
from src.perception.yolo_detector import create_detector
from src.perception.hp_mp_detector import detect_region_color
from src.main import Automation
from src.utils.logger import get_logger

from ui.preview_label import PreviewLabel
from ui.alarm_overlay import AlarmOverlay, flash_taskbar


class MainWindow(QMainWindow):
    # 跨线程信号：自动化线程 → GUI 线程
    log_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(object, object, object, object)  # frame, detections, hp, mp
    hotkey_signal = pyqtSignal()  # F12 触发
    lie_signal = pyqtSignal(float)  # 测谎弹窗命中（匹配得分）

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LYC WORKSPACE")
        self.resize(1180, 760)

        self.config = load_config()
        # 先解析为绝对路径再创建检测器，避免 CWD 不对导致 os.path.exists 失败
        model_path = resolve_model_path(self.config.model_path) if self.config.model_path else ""
        self.detector = create_detector(
            model_path, self.config.confidence, self._log
        )
        self.automation = Automation(
            self.config, self.detector,
            on_log=self.log_signal.emit,
            on_frame=self.frame_signal.emit,
            on_lie_detected=self.lie_signal.emit,
        )

        # ---- 测谎报警状态 ----
        self._alarm_on = False          # 报警音循环开关（确认后置 False）
        self._alarm_blink_on = False    # 横幅闪烁相位
        self._alarm_thread = None
        self._overlay = None            # 游戏窗口上的红框浮层（惰性创建）

        # ---- 启停热键轮询备份 ----
        # 见 _register_hotkey 的注释：光靠键盘钩子会出现"注册成功但按了没反应"。
        self._poll_vk = None            # 热键的虚拟键码（None=不走轮询）
        self._poll_key_down = False     # 上一轮检测到的按下状态（做按下沿判定）
        self._last_toggle_time = 0.0    # 上次 toggle 时间（双通道去重用）
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_stop_key)

        self._fps_counter = [0, time.time()]
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_fps)

        # 自动截图定时器：按配置间隔截取游戏画面保存到 train/data/raw/
        self._screenshot_timer = QTimer(self)
        self._screenshot_timer.setInterval(5000)
        self._screenshot_timer.timeout.connect(self._on_auto_screenshot_tick)

        self._init_ui()
        self._load_config_to_ui()

        # 攻击距离/近战距离（统一）修改时实时同步到 YAML
        self.distance_spin.valueChanged.connect(self._on_distance_changed)
        self.attack_range_y_spin.valueChanged.connect(self._on_attack_range_y_changed)
        # 近身击退参数：改动即写入配置并落盘（运行中也即时生效）
        self.knockback_key_edit.textChanged.connect(self._on_knockback_changed)
        self.knockback_name_edit.textChanged.connect(self._on_knockback_changed)
        self.knockback_cd_spin.valueChanged.connect(self._on_knockback_changed)
        self.knockback_range_spin.valueChanged.connect(self._on_knockback_changed)
        self.attack_type_combo.currentIndexChanged.connect(self._on_attack_type_changed)
        self.stand_mode_checkbox.toggled.connect(self._on_stand_mode_changed)
        self.stand_facing_combo.currentIndexChanged.connect(self._on_stand_facing_changed)
        self.stand_default_attack_checkbox.toggled.connect(self._on_stand_default_attack_changed)
        self.stand_skill2_checkbox.toggled.connect(self._on_stand_skill2_changed)

        self.log_signal.connect(self._on_log)
        self.frame_signal.connect(self._on_frame)
        self.hotkey_signal.connect(self._toggle_run)
        self.lie_signal.connect(self._on_lie_detected)

        # 报警横幅闪烁定时器（400ms 一次相位翻转）
        self._alarm_blink_timer = QTimer(self)
        self._alarm_blink_timer.setInterval(400)
        self._alarm_blink_timer.timeout.connect(self._blink_alarm)

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

        # 测谎报警横幅（默认隐藏，命中弹窗时闪烁显示）
        self.alarm_bar = QWidget()
        ab = QHBoxLayout(self.alarm_bar)
        ab.setContentsMargins(10, 8, 10, 8)
        self.alarm_label = QLabel("")
        self.alarm_label.setStyleSheet(
            "color:white; font-weight:bold; font-size:15px;"
        )
        ab.addWidget(self.alarm_label, 1)
        self.alarm_ack_btn = QPushButton("我已接管（静音）")
        self.alarm_ack_btn.clicked.connect(self._ack_lie_alarm)
        ab.addWidget(self.alarm_ack_btn)
        self.alarm_bar.setStyleSheet("background-color:#c0392b; border-radius:4px;")
        self.alarm_bar.setVisible(False)
        v.addWidget(self.alarm_bar)

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
        self.preview_once_btn = QPushButton("当前帧预览")
        self.preview_once_btn.clicked.connect(self._on_preview_frame)
        ctl.addWidget(self.preview_once_btn)
        self.auto_shot_check = QCheckBox("自动截图")
        self.auto_shot_check.setToolTip("开启后按右侧间隔截取游戏画面，保存到 train/data/raw/")
        self.auto_shot_check.toggled.connect(self._on_toggle_auto_shot)
        ctl.addWidget(self.auto_shot_check)
        self.screenshot_interval_spin = QSpinBox()
        self.screenshot_interval_spin.setRange(1, 3600)
        self.screenshot_interval_spin.setValue(5)
        self.screenshot_interval_spin.setSuffix(" s")
        self.screenshot_interval_spin.setToolTip("自动截图间隔（秒）")
        self.screenshot_interval_spin.setFixedWidth(80)
        ctl.addWidget(self.screenshot_interval_spin)
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
        g.addWidget(QLabel("模型/EXE路径:"), 0, 0)
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
        self.hp_key_edit.setFixedWidth(80)
        self.hp_key_edit.setPlaceholderText("f / hm")
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
        self.mp_key_edit.setFixedWidth(80)
        self.mp_key_edit.setPlaceholderText("g / pu")
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

        # 攻击类型 + 攻击距离（放到上面）
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("攻击类型:"))
        self.attack_type_combo = QComboBox()
        self.attack_type_combo.addItem("长手 (远程攻击)", "long")
        self.attack_type_combo.addItem("短手 (近战攻击)", "short")
        self.attack_type_combo.setToolTip(
            "长手: 远程职业(弓/弩/法), 在较远距离攻击\n"
            "短手: 近战职业(战/盗), 贴近怪物才能攻击"
        )
        self.attack_type_combo.setFixedWidth(140)
        type_row.addWidget(self.attack_type_combo)

        self.distance_label = QLabel("攻击距离px:")
        type_row.addWidget(self.distance_label)
        self.distance_spin = QSpinBox()
        self.distance_spin.setRange(10, 800)
        self.distance_spin.setValue(200)
        self.distance_spin.setToolTip("人物与怪物水平差小于此值才触发攻击")
        self.distance_spin.setFixedWidth(70)
        type_row.addWidget(self.distance_spin)

        type_row.addStretch()
        v.addLayout(type_row)

        # 第二行：跳跃键 + 垂直容差
        row = QHBoxLayout()
        row.addWidget(QLabel("跳跃键:"))
        self.jump_key_edit = QLineEdit("alt")
        self.jump_key_edit.setPlaceholderText("alt / space")
        self.jump_key_edit.setFixedWidth(60)
        row.addWidget(self.jump_key_edit)

        row.addWidget(QLabel("垂直容差px:"))
        self.attack_range_y_spin = QSpinBox()
        self.attack_range_y_spin.setRange(10, 300)
        self.attack_range_y_spin.setValue(60)
        self.attack_range_y_spin.setToolTip("垂直差小于此值时即使不同层也直接攻击（不绕路）")
        self.attack_range_y_spin.setFixedWidth(70)
        row.addWidget(self.attack_range_y_spin)

        row.addStretch()
        v.addLayout(row)

        # 近身击退：怪贴到触发距离内时改放击退技能把它推开（原地，不移动）
        kb_row = QHBoxLayout()
        kb_row.addWidget(QLabel("近身击退:"))
        self.knockback_name_edit = QLineEdit("")
        self.knockback_name_edit.setPlaceholderText("名称(退魔箭)")
        self.knockback_name_edit.setFixedWidth(90)
        kb_row.addWidget(self.knockback_name_edit)
        self.knockback_key_edit = QLineEdit("")
        self.knockback_key_edit.setPlaceholderText("按键")
        self.knockback_key_edit.setFixedWidth(50)
        kb_row.addWidget(self.knockback_key_edit)
        kb_row.addWidget(QLabel("冷却s:"))
        self.knockback_cd_spin = QDoubleSpinBox()
        self.knockback_cd_spin.setRange(0.05, 60.0)
        self.knockback_cd_spin.setSingleStep(0.05)
        self.knockback_cd_spin.setDecimals(2)
        self.knockback_cd_spin.setValue(0.3)
        self.knockback_cd_spin.setFixedWidth(70)
        kb_row.addWidget(self.knockback_cd_spin)
        kb_row.addWidget(QLabel("触发距离px:"))
        self.knockback_range_spin = QSpinBox()
        self.knockback_range_spin.setRange(10, 800)
        self.knockback_range_spin.setValue(240)
        self.knockback_range_spin.setFixedWidth(70)
        kb_row.addWidget(self.knockback_range_spin)
        kb_row.addStretch()
        v.addLayout(kb_row)
        _kb_tip = (
            "怪贴到触发距离（双方中心x差）以内时改放击退技能，把它推开后继续站定输出，\n"
            "替代“贴脸就往后走”的移动型后撤（不移动=不落崖、不打断输出，还带伤害）。\n"
            "按键留空 = 不启用。冷却期内会照常放普通技能，所以击退没生效也不会卡住输出。\n"
            "只对长手(远程)生效；按键不要与技能/buff/加血/加蓝/拾取键重复。\n"
            "触发距离参考：实测参考帧为 224/232px（精灵框间隙约 115~120px），默认 240。"
        )
        for _w in (self.knockback_name_edit, self.knockback_key_edit,
                   self.knockback_cd_spin, self.knockback_range_spin):
            _w.setToolTip(_kb_tip)

        # 站桩模式：角色完全不动，只打朝向正前方射程内的怪
        stand_row = QHBoxLayout()
        self.stand_mode_checkbox = QCheckBox("站桩模式")
        self.stand_mode_checkbox.setToolTip(
            "勾选后角色完全不动（不移动/不转向/不后撤/不探索），\n"
            "只攻击朝向正前方、射程内的怪；攻击逻辑与普通模式一致"
        )
        stand_row.addWidget(self.stand_mode_checkbox)
        stand_row.addWidget(QLabel("站桩朝向:"))
        self.stand_facing_combo = QComboBox()
        self.stand_facing_combo.addItem("朝右 (right)", "right")
        self.stand_facing_combo.addItem("朝左 (left)", "left")
        self.stand_facing_combo.setToolTip(
            "站桩时角色的固定朝向（需与游戏内实际朝向一致）：\n"
            "只会攻击这一侧、射程内的怪"
        )
        self.stand_facing_combo.setFixedWidth(120)
        stand_row.addWidget(self.stand_facing_combo)
        self.stand_default_attack_checkbox = QCheckBox("无怪也攻击")
        self.stand_default_attack_checkbox.setToolTip(
            "站桩模式下没有识别到怪物时也按技能键盲打（两个技能交替），\n"
            "用于应对模型漏检；不移动、不转向"
        )
        stand_row.addWidget(self.stand_default_attack_checkbox)
        self.stand_skill2_checkbox = QCheckBox("只用技能2")
        self.stand_skill2_checkbox.setToolTip(
            "站桩模式下攻击只放技能2（爆炸箭这类群攻）：\n"
            "技能2 冷却中就等下一帧，不退回技能1\n"
            "（普通模式的 AOE 连发会用技能1兜底，站桩按需求去掉）"
        )
        stand_row.addWidget(self.stand_skill2_checkbox)
        stand_row.addStretch()
        v.addLayout(stand_row)

        # 站桩定时微动：每过 N 秒"反方向点一下 + 朝向点一下"（反"定点一动不动"特征）
        micro_row = QHBoxLayout()
        self.micro_move_checkbox = QCheckBox("定时微动")
        self.micro_move_checkbox.setToolTip(
            "站桩模式下每过一段时间，朝背离朝向的方向极短点一下，\n"
            "再朝朝向极短点一下（两次等长、方向相反 ≈ 原地晃一下）。\n"
            "不关心挪了几像素；最后一下朝朝向，所以朝向保持不变。"
        )
        micro_row.addWidget(self.micro_move_checkbox)
        micro_row.addWidget(QLabel("间隔:"))
        self.micro_interval_spin = QSpinBox()
        self.micro_interval_spin.setRange(10, 3600)
        self.micro_interval_spin.setValue(300)
        self.micro_interval_spin.setSuffix(" s")
        self.micro_interval_spin.setToolTip("微动间隔（秒），默认 300s=5分钟")
        self.micro_interval_spin.setFixedWidth(80)
        micro_row.addWidget(self.micro_interval_spin)
        micro_row.addWidget(QLabel("点按:"))
        self.micro_tap_spin = QSpinBox()
        self.micro_tap_spin.setRange(20, 300)
        self.micro_tap_spin.setValue(40)
        self.micro_tap_spin.setSuffix(" ms")
        self.micro_tap_spin.setToolTip(
            "每次方向键点按的时长（毫秒）。越小动作越轻，默认 40ms"
        )
        self.micro_tap_spin.setFixedWidth(80)
        micro_row.addWidget(self.micro_tap_spin)
        micro_row.addStretch()
        v.addLayout(micro_row)

        # 拾取设置
        pickup_row = QHBoxLayout()
        self.pickup_checkbox = QCheckBox("自动拾取")
        self.pickup_checkbox.setToolTip("勾选后自动按拾取键捡东西")
        pickup_row.addWidget(self.pickup_checkbox)
        pickup_row.addWidget(QLabel("拾取键:"))
        self.pickup_key_edit = QLineEdit("z")
        self.pickup_key_edit.setPlaceholderText("z")
        self.pickup_key_edit.setFixedWidth(50)
        pickup_row.addWidget(self.pickup_key_edit)
        pickup_row.addWidget(QLabel("间隔(ms):"))
        self.pickup_interval_spin = QSpinBox()
        self.pickup_interval_spin.setRange(100, 2000)
        self.pickup_interval_spin.setValue(333)
        self.pickup_interval_spin.setSuffix("ms")
        self.pickup_interval_spin.setToolTip("拾取间隔（毫秒），333ms=每秒3次")
        self.pickup_interval_spin.setFixedWidth(80)
        pickup_row.addWidget(self.pickup_interval_spin)
        pickup_row.addStretch()
        v.addLayout(pickup_row)

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

        # Buff 表（定期释放，不看战斗状态）
        v.addWidget(QLabel("自动加Buff (定期释放):"))
        self.buff_table = QTableWidget(0, 3)
        self.buff_table.setHorizontalHeaderLabels(["名称", "按键", "间隔(秒)"])
        self.buff_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.buff_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.buff_table.setToolTip(
            "开启自动打怪后定期按这些键（不移动、不看战斗状态）。\n"
            "间隔(秒) = 重新释放间隔，建议填略小于 buff 游戏内持续时间\n"
            "（如 180s 的 buff 填 170）。\n"
            "注意：按键不要与技能/加血/加蓝/拾取键重复。"
        )
        v.addWidget(self.buff_table)

        buff_btns = QHBoxLayout()
        b_add_btn = QPushButton("+ 添加")
        b_add_btn.clicked.connect(lambda: self._add_buff_row())
        b_del_btn = QPushButton("- 删除选中")
        b_del_btn.clicked.connect(self._del_buff_row)
        buff_btns.addWidget(b_add_btn)
        buff_btns.addWidget(b_del_btn)
        buff_btns.addStretch()
        v.addLayout(buff_btns)
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
        # 显示解析后的完整路径，让用户能直观看到模型实际位置
        # （打包后相对路径 best.onnx 会解析为 _internal\best.onnx）
        self.model_edit.setText(resolve_model_path(c.model_path) if c.model_path else "")
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
                f"x={c.hp_region[0]:.1%} y={c.hp_region[1]:.1%} "
                f"w={c.hp_region[2]:.1%} h={c.hp_region[3]:.1%}"
            )
        # MP
        self.mp_key_edit.setText(c.mp_key)
        self.mp_thr_spin.setValue(int(c.mp_threshold * 100))
        if c.mp_color:
            self._set_swatch(self.mp_swatch, c.mp_color)
        if c.mp_region:
            self.mp_region_label.setText(
                f"x={c.mp_region[0]:.1%} y={c.mp_region[1]:.1%} "
                f"w={c.mp_region[2]:.1%} h={c.mp_region[3]:.1%}"
            )
        self.jump_key_edit.setText(c.jump_key)
        # 攻击类型
        atk_type = getattr(c, "attack_type", "long")
        idx = self.attack_type_combo.findData(atk_type)
        self.attack_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        # 攻击距离：长手读 config.attack_range，短手固定 50（_sync_distance_ui 内处理）
        self.distance_spin.setValue(int(getattr(c, "attack_range", 200)))
        self.attack_range_y_spin.setValue(int(getattr(c, "attack_range_y", 60)))
        # 近身击退
        _kb = getattr(c, "knockback_skill", None) or {}
        self.knockback_name_edit.setText(str(_kb.get("name", "") or ""))
        self.knockback_key_edit.setText(str(_kb.get("key", "") or ""))
        try:
            self.knockback_cd_spin.setValue(float(_kb.get("cooldown", 0.3) or 0.3))
        except (TypeError, ValueError):
            self.knockback_cd_spin.setValue(0.3)
        try:
            self.knockback_range_spin.setValue(int(_kb.get("range", 240) or 240))
        except (TypeError, ValueError):
            self.knockback_range_spin.setValue(240)
        # 站桩模式
        self.stand_mode_checkbox.setChecked(bool(getattr(c, "stand_mode", False)))
        _sf = getattr(c, "stand_facing", "right")
        _sidx = self.stand_facing_combo.findData(_sf)
        self.stand_facing_combo.setCurrentIndex(_sidx if _sidx >= 0 else 0)
        self.stand_default_attack_checkbox.setChecked(
            bool(getattr(c, "stand_default_attack", True))
        )
        self.stand_skill2_checkbox.setChecked(
            bool(getattr(c, "stand_skill2_only", True))
        )
        # 站桩定时微动
        self.micro_move_checkbox.setChecked(
            bool(getattr(c, "stand_micro_move_enabled", True))
        )
        self.micro_interval_spin.setValue(
            int(float(getattr(c, "stand_micro_move_interval", 300.0) or 300.0))
        )
        self.micro_tap_spin.setValue(
            int(getattr(c, "stand_micro_move_tap_ms", 40) or 40)
        )
        self._sync_distance_ui()
        # 拾取
        self.pickup_checkbox.setChecked(getattr(c, "pickup_enabled", True))
        self.pickup_key_edit.setText(getattr(c, "pickup_key", "z"))
        self.pickup_interval_spin.setValue(int(getattr(c, "pickup_interval", 0.333) * 1000))
        # 自动截图间隔（秒）
        self.screenshot_interval_spin.setValue(int(getattr(c, "screenshot_interval", 5)))
        # 技能表
        self.skill_table.setRowCount(0)
        for s in c.skills:
            self._add_skill_row(s.get("name", ""), s.get("key", ""), s.get("cooldown", 1.0))
        # Buff 表
        self.buff_table.setRowCount(0)
        for b in (getattr(c, "buff_skills", None) or []):
            self._add_buff_row(b.get("name", ""), b.get("key", ""), b.get("cooldown", 60.0))

    def _read_ui_to_config(self):
        c = self.config
        c.window_title = self.window_combo.currentText().strip()
        # 若文本框里是 exe 旁边(APP_DIR)或打包内(BUNDLE_DIR)模型的绝对路径，
        # 保存时转回相对路径(best.onnx)，避免文件夹移动后路径失效
        _text = self.model_edit.text().strip()
        if getattr(sys, "frozen", False):
            _norm = os.path.normpath(_text)
            for _base in (APP_DIR, BUNDLE_DIR):
                _base_n = os.path.normpath(_base)
                if _norm == _base_n or _norm.startswith(_base_n + os.sep):
                    _text = os.path.relpath(_norm, _base_n)
                    break
        c.model_path = _text
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
        c.jump_key = self.jump_key_edit.text().strip() or "alt"
        c.attack_type = self.attack_type_combo.currentData()
        if c.attack_type == "long":
            c.attack_range = self.distance_spin.value()  # 长手距离可在界面修改
        c.attack_range_y = self.attack_range_y_spin.value()
        # 近身击退（按键留空 = 不启用）
        c.knockback_skill = {
            "name": self.knockback_name_edit.text().strip() or "击退技能",
            "key": self.knockback_key_edit.text().strip(),
            "cooldown": float(self.knockback_cd_spin.value()),
            "range": int(self.knockback_range_spin.value()),
        }
        # 站桩模式
        c.stand_mode = self.stand_mode_checkbox.isChecked()
        c.stand_facing = self.stand_facing_combo.currentData()
        c.stand_default_attack = self.stand_default_attack_checkbox.isChecked()
        c.stand_skill2_only = self.stand_skill2_checkbox.isChecked()
        # 站桩定时微动
        c.stand_micro_move_enabled = self.micro_move_checkbox.isChecked()
        c.stand_micro_move_interval = float(self.micro_interval_spin.value())
        c.stand_micro_move_tap_ms = self.micro_tap_spin.value()
        # 拾取
        c.pickup_enabled = self.pickup_checkbox.isChecked()
        c.pickup_key = self.pickup_key_edit.text().strip() or "z"
        c.pickup_interval = self.pickup_interval_spin.value() / 1000.0
        c.screenshot_interval = self.screenshot_interval_spin.value()
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

        # Buff
        buffs = []
        for r in range(self.buff_table.rowCount()):
            name = self.buff_table.item(r, 0).text() if self.buff_table.item(r, 0) else ""
            key = self.buff_table.item(r, 1).text() if self.buff_table.item(r, 1) else ""
            cd_text = self.buff_table.item(r, 2).text() if self.buff_table.item(r, 2) else "60"
            try:
                cd = float(cd_text)
            except ValueError:
                cd = 60.0
            if name or key:
                buffs.append({"name": name, "key": key, "cooldown": cd})
        c.buff_skills = buffs

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

    def _add_buff_row(self, name="", key="", cd=60.0):
        r = self.buff_table.rowCount()
        self.buff_table.insertRow(r)
        self.buff_table.setItem(r, 0, QTableWidgetItem(str(name)))
        self.buff_table.setItem(r, 1, QTableWidgetItem(str(key)))
        self.buff_table.setItem(r, 2, QTableWidgetItem(str(cd)))

    def _del_buff_row(self):
        rows = {i.row() for i in self.buff_table.selectedIndexes()}
        for r in sorted(rows, reverse=True):
            self.buff_table.removeRow(r)

    def _save_config(self):
        self._read_ui_to_config()
        save_user_config(self.config)
        self._log(f"[配置] 已保存到 {config_path()}")


    def _on_attack_range_y_changed(self, value):
        """垂直容差微调框变化时，实时同步到 config 并保存到 YAML。"""
        self.config.attack_range_y = value
        save_user_config(self.config)

    def _on_knockback_changed(self, *_):
        """近身击退参数变化时，实时同步到 config 并保存到 YAML。

        在打怪运行中改也能立刻生效（决策层每帧从 config 读）。
        """
        self.config.knockback_skill = {
            "name": self.knockback_name_edit.text().strip() or "击退技能",
            "key": self.knockback_key_edit.text().strip(),
            "cooldown": float(self.knockback_cd_spin.value()),
            "range": int(self.knockback_range_spin.value()),
        }
        save_user_config(self.config)

    def _on_stand_mode_changed(self, checked):
        """站桩模式勾选变化时，实时同步到 config 并保存到 YAML。"""
        self.config.stand_mode = bool(checked)
        save_user_config(self.config)

    def _on_stand_facing_changed(self, index):
        """站桩朝向变化时，实时同步到 config 并保存到 YAML。"""
        self.config.stand_facing = self.stand_facing_combo.currentData()
        save_user_config(self.config)

    def _on_stand_default_attack_changed(self, checked):
        """站桩默认攻击(无怪也攻击)勾选变化时，实时同步到 config 并保存到 YAML。"""
        self.config.stand_default_attack = bool(checked)
        save_user_config(self.config)

    def _on_stand_skill2_changed(self, checked):
        """站桩"只用技能2"勾选变化时，实时同步到 config 并保存到 YAML。"""
        self.config.stand_skill2_only = bool(checked)
        save_user_config(self.config)

    def _on_attack_type_changed(self, index):
        """攻击类型切换时，更新距离 UI 并同步到 config。"""
        self.config.attack_type = self.attack_type_combo.currentData()
        self._sync_distance_ui()
        save_user_config(self.config)

    def _sync_distance_ui(self):
        """根据攻击类型同步距离输入框状态。

        长手: 距离在界面可调（写入 config.attack_range，exe 可修改）。
        短手: 贴脸距离固定 50px，禁用输入框，不读 config.attack_range。
        """
        if self.config.attack_type == "short":
            self.distance_label.setText("近战距离px: (固定50)")
            self.distance_spin.setEnabled(False)
            self.distance_spin.setValue(50)
            self.distance_spin.setToolTip(
                "短手(近战): 贴脸距离固定 50px，不允许修改；"
                "与长手攻击距离完全独立"
            )
        else:
            self.distance_label.setText("攻击距离px:")
            self.distance_spin.setEnabled(True)
            self.distance_spin.setValue(int(getattr(self.config, "attack_range", 200)))
            self.distance_spin.setToolTip(
                "长手(远程): 与怪物水平差≤此值才触发攻击，攻击中轻微波动不中断"
            )

    def _on_distance_changed(self, value):
        """攻击距离微调框变化时，实时同步到 config 并保存到 YAML。

        长手/短手共用同一个 attack_range。
        """
        self.config.attack_range = value
        save_user_config(self.config)

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
            self, "选择 YOLO 模型或 EXE", "", "模型/EXE (*.pt *.onnx *.exe);;所有文件 (*.*)"
        )
        if path:
            self.model_edit.setText(path)

    @staticmethod
    def _set_swatch(swatch, rgb):
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #333;"
        )

    def _on_region_selected(self, target, x, y, w, h):
        """框选区域后自动识别颜色并写入配置（存储为百分比）。"""
        frame = self.automation.capture.grab()
        fh, fw = frame.shape[:2]
        # 转换为百分比存储
        region_pct = [x / fw, y / fh, w / fw, h / fh]
        if target == "mp":
            self.config.mp_region = region_pct
            self.mp_region_label.setText(
                f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                f"w={region_pct[2]:.1%} h={region_pct[3]:.1%}"
            )
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.mp_color = color
                self._set_swatch(self.mp_swatch, color)
                self._log(
                    f"[蓝量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色: RGB{tuple(color)}"
                )
            else:
                self._log(
                    f"[蓝量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色识别失败"
                )
        else:
            self.config.hp_region = region_pct
            self.hp_region_label.setText(
                f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                f"w={region_pct[2]:.1%} h={region_pct[3]:.1%}"
            )
            color = self._detect_region_color(x, y, w, h)
            if color:
                self.config.hp_color = color
                self._set_swatch(self.hp_swatch, color)
                self._log(
                    f"[血量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色: RGB{tuple(color)}"
                )
            else:
                self._log(
                    f"[血量] 区域已设置: "
                    f"x={region_pct[0]:.1%} y={region_pct[1]:.1%} "
                    f"w={region_pct[2]:.1%} h={region_pct[3]:.1%} "
                    f"| 颜色识别失败"
                )

    def _detect_region_color(self, x, y, w, h):
        """从当前窗口截图中识别指定区域的主颜色。"""
        try:
            frame = self.automation.capture.grab()
            return detect_region_color(frame, [x, y, w, h])
        except Exception:
            return None

    def _on_preview_frame(self):
        """单帧预览：截图 + 分析 + 预览，并在日志输出总耗时（供调试）。"""
        if self.automation.running:
            QMessageBox.information(self, "提示", "自动打怪运行中已持续预览，请先停止后再单帧预览。")
            return
        if not self.automation.window_locked:
            QMessageBox.warning(self, "提示", "请先锁定游戏窗口")
            return
        self._on_log("[预览] 触发单帧分析预览 ...")
        self.automation.preview_frame_once()

    def _on_toggle_auto_shot(self, checked):
        """自动截图开关：开启后按配置间隔保存游戏画面到 train/data/raw/。"""
        if checked:
            if not self.automation.window_locked:
                QMessageBox.warning(self, "提示", "请先锁定游戏窗口")
                self.auto_shot_check.blockSignals(True)
                self.auto_shot_check.setChecked(False)
                self.auto_shot_check.blockSignals(False)
                return
            secs = self.screenshot_interval_spin.value()
            self._screenshot_timer.start(secs * 1000)
            self._log(f"[截图] 自动截图已开启（每{secs}秒），保存到 train/data/raw/")
        else:
            self._screenshot_timer.stop()
            self._log("[截图] 自动截图已关闭")

    def _on_auto_screenshot_tick(self):
        """截取当前锁定窗口画面，保存为 PNG 到 train/data/raw/。"""
        try:
            frame = self.automation.capture.grab()
        except Exception as e:
            self._log(f"[截图] 截取失败: {e}")
            return

        raw_dir = os.path.join(_PROJECT_ROOT, "train", "data", "raw")
        try:
            os.makedirs(raw_dir, exist_ok=True)
        except Exception as e:
            self._log(f"[截图] 创建目录失败: {e}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(raw_dir, f"mxd_{timestamp}.png")
        try:
            cv2.imwrite(path, frame)
        except Exception as e:
            self._log(f"[截图] 保存失败: {e}")
            return
        self._log(f"[截图] 已保存: {path}")

    @pyqtSlot()
    def _stop_run(self):
        """停止自动打怪并复位按钮状态（供手动停止 / 测谎报警共用）。"""
        self.automation.stop()
        self.run_btn.setText("▶ 开始自动打怪")
        self.run_btn.setStyleSheet(
            "padding:10px;font-size:14px;font-weight:bold;"
            "background-color:#27ae60;color:white;"
        )
        self.preview_once_btn.setEnabled(True)

    @pyqtSlot()
    def _toggle_run(self):
        # ---- 去重：同一次按键可能被钩子和轮询各触发一次 ----
        # 不去重的话，一次 F6 会 toggle 两次（停止+立刻开始）＝看起来"按 F6 没反应"。
        now = time.time()
        if now - self._last_toggle_time < self._TOGGLE_DEBOUNCE:
            return
        self._last_toggle_time = now

        if self.automation.running:
            self._stop_run()
            return

        # 报警正在响时，第一次按只静音，不启动：
        # 否则玩家"想让机器人停下来"的这一按，反而会把机器人重新拉起来。
        if self._alarm_on:
            self._ack_lie_alarm()
            self._log("[热键] 已静音测谎报警（再按一次才开始自动打怪）")
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

        # 模型路径变化时重建检测器（用 resolved 路径比较，避免 UI 截断/相对绝对路径差异导致误重建）
        resolved_path = resolve_model_path(self.config.model_path) if self.config.model_path else ""
        detector_path = getattr(self.detector, "_path", "")
        # detector._path 可能是相对路径，也做一次 resolve 再比较
        if detector_path:
            detector_path = resolve_model_path(detector_path)
        if resolved_path and resolved_path != detector_path:
            self.detector = create_detector(
                resolved_path, self.config.confidence, self._log
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
        self.preview_once_btn.setEnabled(False)

    # ---------------- 测谎报警 ----------------
    #
    # 测谎弹窗命中后由工作线程发 lie_signal 到这里（GUI 线程），做三件事：
    #   1. 停止自动打怪（工作线程那边已 stop()，这里只复位按钮状态）
    #   2. 循环报警音，直到玩家点「我已接管（静音）」或按启停热键
    #   3. 视觉提醒：主窗口红横幅闪烁 + 任务栏闪烁 + 游戏窗口上的红框浮层
    #
    # 刻意不把主窗口拉到前台：游戏窗口需要保持前台，抢焦点可能让小游戏
    # 收不到输入甚至暂停。提示靠"声音 + 闪烁 + 浮层"。

    @pyqtSlot(float)
    def _on_lie_detected(self, score: float):
        self._log(f"[测谎] 收到报警信号（score={score:.3f}）→ 已停机，请接管鼠标")
        self._stop_run()

        # 1) 主窗口横幅（闪烁 + 确认按钮）
        self.alarm_label.setText(
            f"⚠ 测谎弹窗！已停机 —— 请立即切到游戏，用鼠标作答（score={score:.2f}）"
        )
        self.alarm_bar.setVisible(True)
        self._alarm_blink_on = False
        self._blink_alarm()
        self._alarm_blink_timer.start()

        # 2) 游戏窗口上的置顶红框浮层（看得见、点得穿、不抢焦点）
        self._show_alarm_overlay()

        # 3) 声音 + 任务栏闪烁
        self._start_alarm_sound()
        flash_taskbar(self)

    def _ack_lie_alarm(self):
        """确认已接管：静音、停闪烁、隐藏浮层与横幅。"""
        self._alarm_on = False
        self._alarm_blink_timer.stop()
        self.alarm_bar.setVisible(False)
        self._hide_alarm_overlay()

    def _blink_alarm(self):
        """横幅闪烁（红 / 亮红交替）。"""
        self._alarm_blink_on = not self._alarm_blink_on
        color = "#ff6b6b" if self._alarm_blink_on else "#c0392b"
        self.alarm_bar.setStyleSheet(f"background-color:{color}; border-radius:4px;")

    def _start_alarm_sound(self):
        """启动循环报警音线程（已在响则不重复启动）。"""
        if self._alarm_on:
            return
        self._alarm_on = True
        self._alarm_thread = threading.Thread(
            target=self._alarm_sound_loop, daemon=True
        )
        self._alarm_thread.start()

    def _alarm_sound_loop(self):
        """循环播放报警音，直到 _alarm_on 被置 False。

        winsound.Beep 是阻塞调用，所以放在独立线程里；
        某些环境 Beep 不可用（抛异常）→ 降级为系统提示音。
        """
        import winsound

        fallback = False
        while self._alarm_on:
            try:
                if fallback:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                    time.sleep(0.5)
                else:
                    winsound.Beep(1568, 180)     # G6
                    if not self._alarm_on:
                        break
                    winsound.Beep(1046, 260)     # C6
            except Exception:
                if not fallback:
                    fallback = True          # 换系统提示音再试
                    continue
                time.sleep(0.5)              # 连提示音都不行 → 静默重试

    def _show_alarm_overlay(self):
        """在游戏窗口上显示红框浮层（失败不影响声音/横幅提醒）。"""
        try:
            rect = self.automation.get_window_rect()
            if not rect:
                return
            if self._overlay is None:
                self._overlay = AlarmOverlay()
            self._overlay.show_alarm(
                rect, "⚠ 测谎弹窗！请立即用鼠标作答"
            )
        except Exception as e:
            self._log(f"[测谎] 报警浮层显示失败（声音与横幅不受影响）: {e}")

    def _hide_alarm_overlay(self):
        if self._overlay is not None:
            try:
                self._overlay.stop_alarm()
            except Exception:
                pass

    def _on_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{ts}] {msg}")
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
        # 同时持久化到 logs/mxd.log
        get_logger().info(msg)

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
        for d in detections:
            if d.cls_name in monster_classes:
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
        hp_region = self.config.scale_region(
            self.config.hp_region, frame.shape[1], frame.shape[0]
        )
        if hp_region:
            rx, ry, rw, rh = hp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
        # 画蓝条区域
        mp_region = self.config.scale_region(
            self.config.mp_region, frame.shape[1], frame.shape[0]
        )
        if mp_region:
            rx, ry, rw, rh = mp_region
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 128, 0), 1)
        # 自身位置：直接复用主循环已算好的缓存中心点（避免 UI 线程执行 OCR 卡死）
        center = self.automation._get_last_center()
        if center:
            sx, sy = center
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

    # ---------------- 全局启停热键 ----------------
    #
    # 两条通道并行，互为备份：
    #   1. keyboard 库的低级键盘钩子：支持任意组合键，但依赖 WH_KEYBOARD_LL。
    #      该钩子的回调是 Python 代码，若被 GIL 拖住超过系统 LowLevelHooksTimeout
    #      （默认 300ms），Windows 会【静默摘掉钩子】，此后按键完全没反应；
    #      部分游戏的反外挂也会屏蔽外来钩子。
    #   2. GetAsyncKeyState 轮询：不装钩子、不抢键、不受上述限制，代价是只支持单键。
    #
    # 两条路各自打日志（"触发（钩子）" / "触发（轮询）"），所以下次按 F6 没反应时，
    # 看日志就能判断是哪条通道失效。

    # 常见单键热键 → 虚拟键码（仅轮询用；组合键不在表内则只靠钩子）
    # 键统一小写：_register_hotkey 查表前会把配置值 .lower()
    _VK_MAP = {f"f{i}": 0x6F + i for i in range(1, 13)}              # f1~f12
    _VK_MAP.update({chr(c): c for c in range(0x30, 0x3A)})           # 0~9
    _VK_MAP.update({chr(c).lower(): c for c in range(0x41, 0x5B)})   # a~z
    _VK_MAP.update({
        "space": 0x20, "tab": 0x09, "esc": 0x1B, "enter": 0x0D,
        "home": 0x24, "end": 0x23, "insert": 0x2D, "delete": 0x2E,
        "pause": 0x13, "numlock": 0x90, "scrolllock": 0x91,
    })

    # 启停热键去重窗口（秒）：一次实体按键可能被钩子与轮询各触发一次，
    # 不去重会 toggle 两次（停止后立刻又启动）＝看起来"按热键没反应"。
    _TOGGLE_DEBOUNCE = 0.25

    def _register_hotkey(self):
        key_name = self.config.start_stop_hotkey

        # ---- 通道1：键盘钩子 ----
        try:
            import keyboard
            keyboard.add_hotkey(key_name, self._on_hotkey)
            self._log(f"[热键] 钩子已注册 {key_name}（启动/停止）")
        except Exception as e:
            self._log(f"[热键] 钩子注册 {key_name} 失败: {e}")

        # ---- 通道2：GetAsyncKeyState 轮询备份 ----
        self._poll_vk = self._VK_MAP.get(str(key_name).strip().lower())
        if self._poll_vk:
            self._poll_key_down = False
            self._poll_timer.start()
            self._log(f"[热键] 已启用 {key_name} 轮询备份（不依赖键盘钩子）")
        else:
            self._log(
                f"[热键] {key_name} 不在轮询表内（组合键？）→ 仅靠钩子触发"
            )

    def _on_hotkey(self):
        """keyboard 钩子线程回调：显式排队到 GUI 线程执行 _toggle_run。

        keyboard 库的 add_hotkey 回调运行在它自建的后台线程里，直接
        emit Qt 信号跨线程可能静默丢失；改用 QMetaObject.invokeMethod
        显式 QueuedConnection 排队到 GUI 线程，确保热键可靠触发。
        """
        self.log_signal.emit(
            f"[热键] {self.config.start_stop_hotkey} 触发（钩子）"
        )
        QMetaObject.invokeMethod(self, "_toggle_run", Qt.QueuedConnection)

    def _poll_stop_key(self):
        """轮询启停热键（GetAsyncKeyState，不依赖键盘钩子）。

        只在"按下沿"触发一次：最高位 0x8000 表示当前是否处于按下状态。
        轮询跑在 GUI 线程（QTimer），因此直接调用 _toggle_run 即可，
        比钩子线程 invokeMethod 更可靠。
        """
        if not self._poll_vk:
            return
        try:
            down = bool(
                ctypes.windll.user32.GetAsyncKeyState(self._poll_vk) & 0x8000
            )
        except Exception:
            self._poll_timer.stop()   # 轮询不可用 → 停掉，仍靠钩子
            return
        if down and not self._poll_key_down:
            self._log(f"[热键] {self.config.start_stop_hotkey} 触发（轮询）")
            self._toggle_run()
        self._poll_key_down = down

    # ---------------- 关闭 ----------------
    def closeEvent(self, event):
        # 先静音：报警音跑在独立线程里，不关会一直响到进程退出
        try:
            self._ack_lie_alarm()
        except Exception:
            pass
        try:
            self._poll_timer.stop()
        except Exception:
            pass
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())