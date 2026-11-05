"""主程序入口与自动打怪主循环。

================================================================================
架构概览（三层架构）
================================================================================

  感知层 (perception/)         决策层 (decision/)        执行层 (execution/)
  ┌──────────────────┐       ┌──────────────────┐     ┌──────────────────┐
  │ ScreenCapture    │──帧──▶│ Context          │     │ ActionExecutor   │
  │  (截图)          │       │  (数据载体)       │     │  (动作聚合)      │
  │                  │       │                  │     │                  │
  │ YoloDetector     │──框──▶│ DecisionEngine   │──▶│ KeyboardController│
  │  (YOLO检测)      │       │  (反应式决策)     │     │  (按键注入)      │
  │                  │       │                  │     │                  │
  │ detect_bar_ratio │──比─▶│ 同平台追击/爬绳  │     │ MouseController  │
  │  (HP/MP检测)     │       │ 下落/探索/技能   │     │  (鼠标注入)      │
  │                  │       │                  │     │                  │
  │ YoloDetector   │──坐─▶│ self_position    │     │                  │
  │  (YOLO检测)       │       │  (自身坐标)      │     │                  │
  └──────────────────┘       └──────────────────┘     └──────────────────┘

================================================================================
数据流（每帧）
================================================================================

  1. ScreenCapture.grab()          → frame (numpy BGR 数组)
  2. YoloDetector.detect(frame)    → [Detection, ...]  (玩家/怪物/地板/绳索)
  3. detect_bar_ratio(hp_region)   → hp_ratio (0.0~1.0)
  4. detect_bar_ratio(mp_region)   → mp_ratio (0.0~1.0)
  5. 过滤 player 框 → self_position (脚底) / self_center (中心)
  6. Context(monsters, floors, ..., hp_ratio, mp_ratio, self_position)
  7. DecisionEngine.decide(ctx)    → 反应式决策（同平台追击/爬绳/下落/探索/技能）
  8. on_frame(frame, ...)          → 预览回调（UI 渲染检测框）

================================================================================
自身定位策略
================================================================================

  YOLO player 框（唯一方案，不再用 OCR）
    原理: YOLO 检测出画面中的 player（玩家）框；
          脚底 = bbox 底部中点 (y+h)，角色中心 = bbox 中心点 (y+h//2)。
          比 OCR 的"名字中心"更贴近真实站位。
    多人同屏时区分"自己":
          - 有历史位置 → 取离上一帧脚底最近的 player（位置连续性）
          - 无历史位置（启动/换图）→ 取面积最大的 player（兜底启发式）

================================================================================
运行方式
================================================================================

  GUI 模式:  python main.py
  CLI 模式:  python -m src.main
"""
import os
import time
import threading
from datetime import datetime
from typing import Callable, Optional, Any, Tuple

from .perception.screen_capture import ScreenCapture
from .perception.yolo_detector import Detector, create_detector
from .perception.hp_mp_detector import detect_bar_ratio
from .perception.lie_detector import LieDetector
from .perception.lie_shadow import LieShadow, board_box
from .perception.lie_detector_watch import LieDetectorWatcher
from .perception.lie_recorder import LieRecorder, LieLog, client_cursor
from .perception.minimap_locator import MinimapLocator
from .perception.exp_meter import ExpMeter
from .execution.channel_switch import (
    ChannelSwitcher, ChannelTimer, GameQuitter, MAX_IDLE_RUNS, RETRY_SECONDS,
    next_quit_at,
)
from .execution.action_executor import ActionExecutor
from .decision.context import Context, DecisionEngine
from .utils.config_loader import (
    Config, resolve_model_path, resolve_asset_path, APP_DIR,
)


# =============================================================================
# 自动加 buff 的"施法窗口"参数
# =============================================================================
# 为什么需要窗口：按 buff 键时若角色正在攻击动画中，游戏会忽略该按键，
# 而 press_key 仍记为"已按下" → 整个 cooldown 内不再重试 → buff 加不上。
# 因此 buff 到期时先让决策层暂停攻击，等攻击动画收尾后再按键。
#
# 窗口长度 = BUFF_PRESS_DELAY + (buff 个数 × BUFF_MULTI_GAP) + BUFF_CAST_TAIL

BUFF_PRESS_DELAY_SECONDS = 0.4   # 开窗后等这么久再按（让上一次攻击动画收尾）
BUFF_MULTI_GAP_SECONDS = 0.25    # 多个 buff 依次释放之间的间隔
BUFF_CAST_TAIL_SECONDS = 0.6     # 按完最后一个 buff 后，留给它施法动画的余量

LIE_RESULT_WAIT_SECONDS = 5.0
"""画板消失后再录这么久，等 OCR 认出「成功通过」对话框：结果要写进这一局的 meta.json / results.csv。
通过的话通常 1~2 秒内就认出来，会提前结束；没通过就录满这段时间。"""


class Automation:
    """自动打怪主循环控制器。

    【职责】
    整合三层架构，在独立线程中循环执行"截图 → 检测 → 决策 → 执行"。

    【线程模型】
    - 主线程: PyQt5 GUI 事件循环
    - 工作线程: Automation._loop() 运行控制循环
    - 通过回调 (on_log, on_frame) 把结果推回主线程

    【生命周期】
    1. 构造 Automation(config, detector, on_log, on_frame)
    2. lock_window(title)    锁定游戏窗口
    3. start()               启动工作线程
    4. stop()                停止工作线程
    5. unlock_window()       释放窗口

    Args:
        config:   全局配置对象（窗口/检测/按键/技能等）
        detector: YOLO 检测器实例，None 时自动创建 MockDetector
        on_log:   日志回调，参数 (message: str)
        on_frame: 预览回调，参数
                  (frame, detections, hp_ratio, mp_ratio, locked_target)
                  locked_target 为当前锁定的怪物 Detection（无目标时为 None）
    """

    def __init__(self, config: Config, detector: Optional[Detector] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_frame: Optional[Callable[..., None]] = None,
                 on_lie_detected: Optional[Callable[[float], None]] = None,
                 on_lie_resume: Optional[Callable[[], None]] = None):
        # 日志转发：正常输出，同时留最近几行，测谎录制开始时写进这一局的 log.txt
        self._lie_log = LieLog(on_log or (lambda m: None))
        on_log = self._lie_log.write
        # ---- 感知层 ----
        self.config = config
        self.capture = ScreenCapture()  # 窗口截图（客户区 BitBlt）

        # 检测器：如果传了就用，否则根据配置自动创建（模型不存在时回退 Mock）
        self.model_path = resolve_model_path(config.model_path)
        self.detector = detector if detector is not None else create_detector(
            self.model_path, config.confidence, on_log or (lambda m: None)
        )

        # ---- 执行层 ----
        # 聚合键盘 + 鼠标控制器；on_log 用于上报按键注入结果（是否已锁定窗口等）
        # 使用 SendInput 驱动层模拟真实全局按键，游戏窗口必须在前台
        self.executor = ActionExecutor(
            on_log=on_log or (lambda m: None),
        )

        # ---- 决策层 ----
        # 反应式决策引擎：基于 YOLO 画面实时决策，不需要地图
        self.engine = DecisionEngine(
            config, self.executor,
            on_log=on_log or (lambda m: None)
        )

        # ---- 回调 ----
        self.on_log = on_log or (lambda m: None)
        self.on_frame = on_frame or (lambda f, d, h, m, t: None)
        # 测谎弹窗命中回调（参数为匹配得分）。UI 侧据此弹报警。
        self.on_lie_detected = on_lie_detected or (lambda score: None)
        self.on_lie_resume = on_lie_resume or (lambda: None)

        # ---- 测谎弹窗检测（反外挂）----
        # 弹窗出现 → 3 秒后开始"鼠标跟随图形"小游戏 → 必须马上停手让玩家接管。
        # 检测放在每帧最前面（YOLO 之前）：命中当帧就 return，反而更省。
        # 实测约 2.6ms/帧，相对 YOLO 的 ~30ms 可忽略。
        self._lie_detector: Optional[LieDetector] = None
        self._lie_latched = False      # 一次弹窗只报警一次（重启时复位）
        if getattr(config, "lie_detect_enabled", False):
            try:
                tpl = resolve_asset_path(
                    getattr(config, "lie_detect_template", "") or ""
                )
                self._lie_detector = LieDetector(
                    tpl,
                    threshold=float(
                        getattr(config, "lie_detect_threshold", 0.78) or 0.78
                    ),
                    on_log=self.on_log,
                )
            except Exception as e:
                self.on_log(f"[测谎] 初始化失败，测谎检测已禁用: {e}")

        # ---- 测谎小游戏自动作答（mxd-lie-logic-code 方案）----
        # 弹窗命中停机后，LieRecorder 录画面+光标（60fps，影子结果写同目录），
        # LieShadow 跟踪目标并接管鼠标：白图形可见时直接用重心，淡出后用记下的
        # 轮廓在两帧对齐差值图上做形状匹配（不看模型），0.5s 没认出白图形才退回
        # 模型。点一下鼠标左键才交还。认出「成功通过」对话框后按 Esc 关掉并自动
        # 恢复打怪（lie_auto_resume）；没通过就一直停着。
        self._lie_shadow: Optional[LieShadow] = None
        self._lie_watcher: Optional[LieDetectorWatcher] = None
        self._lie_game_recorder: Optional[LieRecorder] = None
        self._lie_game_thread: Optional[threading.Thread] = None
        self._lie_game_stop = threading.Event()
        if getattr(config, "lie_shadow_enabled", False):
            try:
                self._lie_shadow = LieShadow(
                    resolve_model_path(
                        getattr(config, "lie_shadow_model", "") or ""
                    ),
                    conf=float(getattr(config, "lie_shadow_conf", 0.1) or 0.1),
                    on_log=self.on_log,
                )
                if getattr(config, "lie_shadow_takeover", True):
                    self._lie_shadow.takeover = (self._read_cursor_client,
                                                 self._move_cursor_client)
                self._lie_watcher = LieDetectorWatcher(on_log=self.on_log)
                self._lie_game_recorder = LieRecorder(
                    self.capture.grab,
                    lambda: client_cursor(self.capture.hwnd),
                    on_log=self.on_log,
                    out_root=self._lie_evidence_dir(),
                    shadow=self._lie_shadow,
                    run_log=self._lie_log,
                    result=lambda: ("通过" if self._lie_watcher is not None and self._lie_watcher.passed
                                    else "没看到通过对话框"),
                )
                self.on_log("[测谎] 自动作答已启用（形状匹配跟踪 + OCR 通过判定）")
            except Exception as e:
                self._lie_shadow = None
                self.on_log(f"[测谎] 自动作答初始化失败，已禁用: {e}")

        # ---- 线程控制 ----
        self._running = False  # 控制循环是否继续
        self._thread = None    # 工作线程对象
        self._frame_count = 0  # 帧计数器（用于限频日志）

        # ---- 自身定位（YOLO player 框，不再用 OCR）----
        # _last_self_foot: 上一帧自身脚底坐标，用于多 player 时的位置连续性选择
        # _cached_center:  最近一次定位到的角色中心点（供状态日志显示）
        self._last_self_foot: Optional[Tuple[int, int]] = None
        self._cached_center: Optional[Tuple[int, int]] = None

        # ---- HP/MP 变化追踪 ----
        self._last_hp_ratio: Optional[float] = None
        self._last_mp_ratio: Optional[float] = None

        # ---- 自动拾取 ----
        self._last_pickup_time = 0.0  # 上次拾取按键的时间戳

        # ---- 自动加 buff（施法窗口状态机）----
        self._buff_next_time = {}     # key -> 下次到期时间戳
        self._buff_queue = []         # 本轮待释放的 buff [{name,key,cd}]
        self._buff_open_time = 0.0    # 本轮窗口打开的时间戳
        self._buff_last_press = 0.0   # 本轮上次按键的时间戳

        # ---- 活动范围：小地图黄点（镜头跟着角色走，只有小地图能看出自己在地图哪里）----
        self._minimap = MinimapLocator()
        self._minimap_dot: Optional[Tuple[int, int]] = None  # 黄点在画面里的位置（预览用）

        # 主循环外（换频道）这一张截图认出了测谎
        self._lie_probe_hit = False

        # ---- 自动换频道：channel_idle_min 分钟没放过技能（这层没怪了）就换一个频道 ----
        self._channel = ChannelSwitcher(self.capture, self.executor, config, on_log=self.on_log)
        self._channel.on_frame = self._lie_probe_frame
        self._channel.abort = lambda: self._lie_probe_hit
        self._channel_requested = False  # 「测试换频道」按钮：打怪中按下 → 下一帧执行
        self._channel_busy = threading.Lock()
        self._channel_timer = ChannelTimer()

        # ---- 定时结束游戏：到 quit_time 就不打了，按 Esc 点「退出游戏」----
        self._quitter = GameQuitter(self.capture, self.executor, config, on_log=self.on_log)
        self._quitter.on_frame = self._lie_probe_frame
        self._quitter.abort = lambda: self._lie_probe_hit
        self._quit_at: Optional[float] = None

        # ---- 经验记录：定时读状态栏的 EXP，写到 exp_log/日期.csv ----
        self._exp = ExpMeter(on_log=self.on_log)

    # =========================================================================
    # 窗口管理
    # =========================================================================

    def list_windows(self):
        """枚举所有可见窗口，返回 [(hwnd, title), ...]。

        用于 UI 下拉框选择要锁定的窗口。
        """
        return self.capture.list_windows()

    def lock_window(self, title: str) -> str:
        """按标题锁定游戏窗口。

        锁定后：
        1. ScreenCapture 可以截取该窗口画面
        2. ActionExecutor 的 SendInput 会注入到该窗口

        Returns:
            锁定后的窗口标题（用于确认）
        """
        locked = self.capture.lock(title=title)
        # 把窗口句柄传给执行层，这样按键/鼠标消息会发到游戏中
        self.executor.set_target_window(self.capture.hwnd)
        return locked

    def unlock_window(self):
        """释放窗口锁定。"""
        self.capture.unlock()

    @property
    def window_locked(self):
        return self.capture.locked

    def get_window_rect(self):
        """获取窗口在屏幕中的矩形 (left, top, width, height)。"""
        return self.capture.get_rect() if self.capture.locked else None

    # =========================================================================
    # 检测器管理
    # =========================================================================

    def set_detector(self, detector: Detector):
        """运行时替换检测器（切换模型时用）。"""
        self.detector = detector

    # =========================================================================
    # 主循环控制
    # =========================================================================

    @property
    def running(self):
        """是否正在运行中。"""
        return self._running

    def start(self):
        """启动自动打怪主循环。

        在独立线程中运行 _loop()，不阻塞 UI 线程。

        Raises:
            RuntimeError: 未锁定窗口时调用
        """
        if self._running:
            return
        if not self.capture.locked:
            raise RuntimeError("请先锁定游戏窗口")

        # UI 可能修改了配置，同步到决策引擎
        self.engine.update_config(self.config)
        self.engine.reset()  # 清空技能轮转索引、冷却记录

        # 清空 buff 状态：启动时所有 buff 立即到期（先加满一轮）
        self._buff_next_time.clear()
        self._buff_queue = []
        self._buff_open_time = 0.0
        self._buff_last_press = 0.0

        # 测谎锁存复位：若弹窗还挂着就重启，允许再报一次警（然后立刻被再次拦下）
        self._lie_latched = False

        # 上次测谎的影子跟踪还在跑（玩家没等结束就重启）→ 先收手并把鼠标交还。
        # 这是影子在"程序停止"之外唯一的强制退出路径：重启=玩家接管。
        self._stop_lie_shadow("玩家手动重启")

        # 测谎检测状态：模板加载日志发生在 __init__，那时 UI 还没接上日志信号会被丢掉，
        # 所以这里补一条 —— 阈值调错（比如测试时改成 0.2 忘了改回）必须一眼能看见。
        if self._lie_detector is not None:
            self.on_log(f"[测谎] 检测状态：{self._lie_detector.describe()}")

        # ---- 外围自动化状态复位 ----
        self._channel_timer.start(time.time())
        self._schedule_quit()
        self._exp.read_seconds = float(getattr(self.config, "exp_read_seconds", 5) or 5)
        self._exp.log_minutes = float(getattr(self.config, "exp_log_minutes", 10) or 0)
        self._exp.start(time.time())

        self._running = True
        # daemon=True: 主线程退出时自动结束，不会卡住进程
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.on_log("[启动] 自动打怪已开始")

    def stop(self):
        """停止自动打怪。

        设置 _running = False，_loop() 会在下一次迭代时退出。
        同时释放所有按住的移动键，防止停止后方向键卡住。

        注意：这里【不】停止测谎影子跟踪 —— 弹窗命中本来就是先停机、
        再由影子接管鼠标（20261031 复盘：UI 报警回调 _stop_run() 也会调
        本方法，曾把刚启动的影子当场杀掉，shadow.csv 只有表头）。
        影子自己的退出路径：画板消失 / 超时 / 用户手动重启（start()）。
        """
        if not self._running:
            return
        self._running = False
        self._buff_queue = []       # 丢弃未处理完的 buff 队列
        self.engine.release_keys()  # 释放按住的方向键/上键
        # 物理兜底：把键盘/鼠标控制器认为"还按着"的键全部释放。
        # 只靠 engine.release_keys() 时，若引擎的 _held_key 与控制器的 _held_keys
        # 不一致（换键/异常路径），停机会漏掉那个键 → 角色停止后仍一直走、一直打。
        self.executor.reset()
        try:
            self._exp.stop(time.time())  # 把最后的经验数写进记录
        except Exception:
            pass
        self.on_log("[停止] 自动打怪已停止")

    # =========================================================================
    # 外围自动化（换频道 / 经验）
    # =========================================================================

    def _lie_probe_frame(self, frame):
        """换频道期间每张截图都喂给测谎检测。

        换频道会占住主循环十几秒，期间没有别的地方能发现测谎弹窗。
        这里只"认出来 + 让操作自己停"（abort），真正的停机/报警交给主循环
        下一帧的 _handle_lie_detected —— 在那个线程里调 stop() 会等自己结束。
        """
        if self._lie_probe_hit:
            return
        if self._lie_detector is not None and self._lie_detector.available:
            hit, _score = self._lie_detector.detect(frame)
            if hit:
                self._lie_probe_hit = True

    def _locate_on_minimap(self, frame):
        """开了活动范围时在框选的小地图区域里找自己，返回 (黄点 x, 区域宽)；没开/没框选/没找到时 x 为 None。"""
        self._minimap_dot = None
        if getattr(self.config, "range_limit", "none") not in ("left", "right"):
            return None, None
        region = self.config.scale_region(
            getattr(self.config, "minimap_region", None), frame.shape[1], frame.shape[0]
        )
        if region is None:
            return None, None
        dot = self._minimap.locate(frame, region)
        if dot is None:
            return None, region[2]
        self._minimap_dot = (region[0] + int(dot[0]), region[1] + int(dot[1]))
        return dot[0], region[2]

    def exp_summary(self) -> str:
        """界面显示的一行经验信息。"""
        return self._exp.summary(time.time())

    def channel_switch_now(self):
        """「测试换频道」按钮：打怪中 → 下一帧暂停打怪去换；没在打怪 → 直接在后台线程换一次。"""
        if not self.capture.locked:
            self.on_log("[换频道] 请先锁定游戏窗口")
            return
        if self._running:
            self._channel_requested = True
            self.on_log("[换频道] 已安排，下一帧暂停打怪去换频道")
        else:
            threading.Thread(target=self._do_channel_switch, daemon=True).start()

    def _channel_due(self) -> bool:
        """该自动换频道了吗（channel_idle_min 分钟没放过技能）。"""
        if not getattr(self.config, "channel_enabled", False):
            return False
        idle = float(getattr(self.config, "channel_idle_min", 3) or 0) * 60
        return self._channel_timer.due(time.time(), self.engine.last_cast_at, idle)

    def _do_channel_switch(self):
        """暂停一切操作，换一次频道（在主循环线程里调用时，期间不会有任何别的按键）。"""
        if not self._channel_busy.acquire(blocking=False):
            return
        in_loop = self._running
        ok = False
        try:
            self._lie_probe_hit = False
            self.engine.release_keys()
            time.sleep(0.8)  # 等落地、技能动作做完
            self._channel.config = self.config  # 界面「开始」时会换成新的配置对象
            self._channel.abort = lambda: (in_loop and not self._running) or self._lie_probe_hit
            ok = self._channel.run()
        finally:
            idle = float(getattr(self.config, "channel_idle_min", 3) or 0) * 60
            self._channel_timer.finished(time.time(), ok, idle)
            if in_loop and not ok and self._running and not self._lie_probe_hit:
                self.on_log(f"[换频道] 本次没换成，{RETRY_SECONDS} 秒后再试")
            if in_loop and self._channel_timer.runs == MAX_IDLE_RUNS:
                self.on_log(f"[换频道] 连着换了 {MAX_IDLE_RUNS} 次都没打到怪，先不自动换了"
                            "（检查角色是不是在打怪的地方、怪认不认得出）；打到怪后恢复")
            if self._lie_probe_hit:
                self.on_log("[换频道] 检测到测谎弹窗，已中止；菜单等测谎结束后再关")
            self.engine.reset()
            self._channel_busy.release()

    def _schedule_quit(self):
        """点「开始」时算好这次几点结束游戏：钟点今天已经过了就是明天。"""
        self._quit_at = None
        if not getattr(self.config, "quit_enabled", False):
            return
        text = getattr(self.config, "quit_time", "")
        now = time.time()
        self._quit_at = next_quit_at(text, now)
        if self._quit_at is None:
            self.on_log(f"[定时] 结束时间 {text} 写得不对，要写成 06:30 这样，这次不定时")
            return
        minutes = int((self._quit_at - now) // 60)
        self.on_log(f"[定时] {time.strftime('%m-%d %H:%M', time.localtime(self._quit_at))} 结束游戏"
                    f"（还有 {minutes // 60} 小时 {minutes % 60} 分）")

    def _do_quit(self):
        """到时间了：松开所有键，退出游戏，然后停止打怪。没退出去也不再打。"""
        self.on_log("[定时] 到时间了，不打了")
        self.engine.release_keys()
        time.sleep(0.8)  # 等落地、技能动作做完
        self._lie_probe_hit = False
        self._quitter.abort = lambda: not self._running or self._lie_probe_hit
        ok = self._quitter.run()
        if self._quitter.aborted:
            return  # 点了停止就到此为止；测谎弹出来了就等测谎结束，再回到这里接着退
        self._quit_at = None
        self.on_log("[定时] 已离开游戏，不再操作" if ok
                    else "[定时] 没退出去，打怪已停止，游戏还开着")
        self.stop()
        if ok:
            self.capture.unlock()  # 窗口可能没了，下次点「开始」重新锁定

    # =========================================================================
    # 测谎弹窗（反外挂）
    # =========================================================================

    def _handle_lie_detected(self, frame, score: float):
        """测谎弹窗命中：立即停机 + 录屏取证 + 报警（同一次弹窗只报警一次）。

        为什么必须先停机：
          1. 弹窗盖住画面正中，此刻 YOLO 的输出全是垃圾，决策层会乱走乱放技能；
          2. 小游戏要求玩家用真实鼠标把光标压在移动图形上，机器人必须让出控制权。

        为什么停机后还要录屏：
          停机只是不再操作，弹窗还在游戏里，玩家接手的这段处置过程（有没有
          及时作答、小游戏怎么失败）是事后复核的唯一依据。录屏线程独立于
          主循环，stop() 之后仍继续录 lie_record_seconds 秒。

        停机后不自动恢复：误恢复（在小游戏中途继续打怪）的代价远大于少打一分钟怪。
        玩家按启停热键即可重新开始（会自动解除报警）。
        """
        if self._lie_latched:
            # 弹窗仍在（例如玩家重启后又立刻被拦下）→ 只保持停机
            self.stop()
            return

        self._lie_latched = True
        self.on_log(
            f"[测谎] ⚠ 检测到测谎弹窗！score={score:.3f} → 立即停机，请接管鼠标作答"
        )
        self.stop()

        # 截图与录屏共用同一个时间戳，方便事后按文件名对照
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 录制与跟踪在 _start_lie_game 里一起启动（LieRecorder 内含影子同步）
        try:
            path = self._save_lie_evidence(frame, ts)
            if path:
                self.on_log(f"[测谎] 现场截图已保存: {path}")
        except Exception as e:
            self.on_log(f"[测谎] 现场截图保存失败: {e}")

        try:
            self.on_lie_detected(score)
        except Exception as e:
            self.on_log(f"[测谎] 报警回调异常: {e}")

        # 推一帧预览，让 UI 显示命中画面（无检测框/无血蓝值）
        try:
            self.on_frame(frame, [], None, None, None)
        except Exception:
            pass

        # ---- 自动作答（录制 + 跟踪 + 接管 + 通过后自动恢复）----
        # 放在停机之后；抓帧各自独立（BitBlt 每次自建资源），并发安全。
        self._start_lie_game()

    # =========================================================================
    # 测谎影子跟踪 / 接管（mxd-lie-target 方案）
    # =========================================================================

    def _read_cursor_client(self):
        """光标在客户区里的坐标 + 左键是否按下；窗口没锁或取不到返回 None。

        接管规则是"点一下左键才交还"，所以光标读取必须带上按键状态
        （lie_shadow.MouseTakeover 的约定）。
        """
        return client_cursor(self.capture.hwnd)

    def _move_cursor_client(self, x: int, y: int) -> bool:
        """把光标移到客户区 (x, y)。游戏窗口不在前台就不动，返回 False。"""
        try:
            import win32api
            import win32gui
            hwnd = self.capture.hwnd
            if not hwnd:
                return False
            fg = win32gui.GetForegroundWindow()
            if fg != hwnd and not win32gui.IsChild(hwnd, fg):
                return False
            sx, sy = win32gui.ClientToScreen(hwnd, (int(x), int(y)))
            win32api.SetCursorPos((sx, sy))
            return True
        except Exception:
            return False

    def _start_lie_game(self):
        """命中弹窗停机后启动自动作答：录制+跟踪（LieRecorder 内含影子）+
        OCR 监视（通过判定），另起监督线程负责结束判定 / 按 Esc / 自动恢复。"""
        if self._lie_game_recorder is None:
            return
        if self._lie_game_thread is not None and self._lie_game_thread.is_alive():
            return
        if not self.capture.locked:
            self.on_log("[测谎] 窗口未锁定，跳过自动作答")
            return
        try:
            self._lie_game_stop.clear()
            self._lie_watcher.start()
            out_dir = self._lie_game_recorder.start("检测到测谎弹窗")
            if out_dir is None:
                self.on_log("[测谎] 已在录制中，跳过重复启动")
                return
        except Exception as e:
            self.on_log(f"[测谎] 自动作答启动失败: {e}")
            return
        self._lie_game_thread = threading.Thread(
            target=self._supervise_lie_game, daemon=True,
        )
        self._lie_game_thread.start()

    def _supervise_lie_game(self):
        """自动作答监督线程（主循环此时已停机）：

        阶段一（小游戏进行中）：按 15 次/秒喂帧给 OCR 监视器、盯画板——
        连续 N 帧不见就停录制（内含影子）；
        阶段二（结束后）：继续喂帧 —— 「成功通过」对话框出现在画板消失之后，
        OCR 要靠这些帧认它；按 watcher.next_step 处理：还开着就按 Esc
        （每次按之前都要有比上次更晚的画面确认它还在，避免没开任何窗口时
        Esc 弹出游戏菜单），关掉了就自动恢复打怪；没通过就一直停着。
        """
        interval = 1.0 / 15.0
        end_frames = int(getattr(self.config, "lie_shadow_end_frames", 15) or 15)
        max_seconds = float(
            getattr(self.config, "lie_shadow_max_seconds", 60) or 60
        )
        t0 = time.perf_counter()
        missing = 0
        seen_board = False
        gone_at = None  # 画板连续消失够帧数的时间点

        # ---- 阶段一：小游戏进行中 ----
        while not self._lie_game_stop.is_set():
            elapsed = time.perf_counter() - t0
            if elapsed > max_seconds:
                self.on_log(f"[测谎] 超过 {max_seconds:.0f}s，停止录制与跟踪")
                break
            try:
                frame = self.capture.grab()
            except Exception as e:
                self.on_log(f"[测谎] 截图失败，停止自动作答: {e}")
                break
            self._lie_watcher.feed(frame)
            if board_box(frame) is not None:
                seen_board = True
                missing = 0
                gone_at = None
            elif seen_board:
                missing += 1
                if missing >= end_frames:
                    if gone_at is None:
                        gone_at = time.perf_counter()
                        self.on_log("[测谎] 画板已消失，小游戏结束")
                    # 再喂几秒帧，等 OCR 认出「成功通过」对话框（结果要写进这一局的录像）
                    if (self._lie_watcher.passed
                            or time.perf_counter() - gone_at >= LIE_RESULT_WAIT_SECONDS):
                        break
            wait = interval - (time.perf_counter() - t0 - elapsed)
            if wait > 0:
                time.sleep(wait)

        self._lie_game_recorder.stop("画板消失" if seen_board else "停止")

        # ---- 阶段二：通过判定 + Esc + 自动恢复 ----
        # OCR 监视器继续跑（对话框在画板之后才出现），直到恢复或超时。
        if not getattr(self.config, "lie_auto_resume", True):
            self.on_log("[测谎] lie_auto_resume 已关闭，是否通过请看日志，"
                        "恢复打怪请手动操作")
            self._lie_watcher.stop()
            return
        esc_at, esc_count = 0.0, 0
        resume_deadline = time.time() + 60.0
        while time.time() < resume_deadline and not self._lie_game_stop.is_set():
            try:
                frame = self.capture.grab()
            except Exception as e:
                self.on_log(f"[测谎] 截图失败: {e}")
                break
            self._lie_watcher.feed(frame)
            step = self._lie_watcher.next_step(esc_at, esc_count)
            if step == "esc":
                self.on_log(f"[测谎] 测谎通过，按 Esc 关闭对话框"
                            f"（第 {esc_count + 1} 次）")
                self.executor.press_key("esc")
                esc_at = time.time()
                esc_count += 1
                time.sleep(1.0)
            elif step == "resume":
                self._lie_watcher.stop()
                self.on_log("[测谎] 通过对话框已关闭，自动恢复打怪")
                self._resume_after_lie()
                return
            else:
                time.sleep(0.5)
        self._lie_watcher.stop()
        self.on_log("[测谎] 未见「成功通过」对话框或未能关闭（可能没通过），"
                    "不自动恢复打怪；玩家按启停热键可重新开始")

    def _resume_after_lie(self):
        """测谎通过后恢复打怪（与 start() 相同的状态复位，但不重复锁窗）。"""
        self._lie_latched = False
        self._stop_lie_game("恢复打怪")
        # 测谎打断了换频道：菜单、频道窗口可能还开着，收尾关掉
        self._channel.close_after_abort()
        self._lie_probe_hit = False
        self._channel_timer.restart(time.time())
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        try:
            self.on_lie_resume()
        except Exception:
            pass
        self.on_log("[启动] 测谎通过，自动打怪已恢复")

    def _stop_lie_game(self, reason: str = ""):
        """停止自动作答（录制/影子/OCR 监视/监督线程，幂等）。"""
        running = (self._lie_game_thread is not None
                   and self._lie_game_thread.is_alive())
        self._lie_game_stop.set()
        try:
            self._lie_watcher.stop()
        except Exception:
            pass
        try:
            self._lie_game_recorder.stop(reason or "停止")
        except Exception as e:
            self.on_log(f"[测谎] 录制停止异常: {e}")
        self._lie_game_thread = None
        if running and reason:
            self.on_log(f"[测谎] 自动作答已停止（{reason}）")

    def _stop_lie_shadow(self, reason: str = ""):
        """兼容旧调用：现在自动作答由 _stop_lie_game 统一管理。"""
        self._stop_lie_game(reason)

    def _lie_evidence_dir(self) -> str:
        """测谎取证目录（截图与录屏共用；相对路径基于程序目录）。"""
        out_dir = getattr(self.config, "lie_record_dir", "") or "logs/lie_detected"
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(APP_DIR, out_dir)
        return out_dir

    def _save_lie_evidence(self, frame, ts: Optional[str] = None) -> str:
        """保存命中现场截图（留证 + 便于日后复核/重裁模板）。返回路径或空串。"""
        import cv2

        out_dir = self._lie_evidence_dir()
        os.makedirs(out_dir, exist_ok=True)
        if not ts:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(out_dir, f"lie_{ts}.png")
        if not cv2.imwrite(path, frame):
            return ""
        return path

    def shutdown(self):
        """释放后台资源（关闭程序时调用）：录制正常收尾、各线程退出。"""
        self._stop_lie_game("程序退出")
        self._lie_log.close()  # 程序要退了，不等 LOG_AFTER_S 的下一次写日志

    def _loop(self):
        """主循环（在独立线程中运行）。

        每帧执行:
          1. 截图（grab）
          2. YOLO 检测（detect）→ 按类别过滤
          3. HP/MP 检测（颜色数像素）→ 比例
          4. 自身定位（YOLO player 框）→ 坐标
          5. 组装 Context → DecisionEngine.decide()
          6. 预览回调 → UI 渲染

        FPS 控制: 通过 time.sleep() 补齐到 1/fps 秒
        """
        # 每帧的目标间隔时间（秒）
        interval = 1.0 / max(1, self.config.fps)

        while self._running:
            self._frame_count += 1

            # ---- 全局异常兜底 ----
            # 任何一步抛异常（截图/检测/定位/决策）都不允许静默崩溃线程，
            # 必须记录 traceback 到日志，便于定位问题。
            try:
                self._loop_frame(interval)
            except Exception:
                import traceback
                self.on_log("[错误] 主循环异常(已捕获, 继续运行):")
                for line in traceback.format_exc().splitlines():
                    self.on_log(f"  {line}")
                time.sleep(interval)

    def _loop_frame(self, interval: float):
        """单帧执行：截图→测谎弹窗检测→检测→HP/MP→定位→决策→预览。"""
        t0 = time.time()  # 帧开始时间
        try:
            frame = self.capture.grab()
            # 首次截图记录帧尺寸 + DPI 诊断
            if self._frame_count == 1:
                h, w = frame.shape[:2]
                self.on_log(f"[帧尺寸] {w}x{h}")
                self.on_log(
                    f"[配置] 参考分辨率: "
                    f"{self.config.reference_width}x{self.config.reference_height}"
                )
                # DPI 诊断：对比 GetClientRect 与实际帧尺寸
                try:
                    cw, ch = self.capture.get_client_rect()
                    if cw != w or ch != h:
                        self.on_log(
                            f"[DPI警告] GetClientRect={cw}x{ch} "
                            f"与实际帧 {w}x{h} 不一致，"
                            f"可能存在 DPI 缩放偏移！"
                        )
                    else:
                        self.on_log(
                            f"[DPI] 客户区尺寸匹配 ({cw}x{ch})，坐标应无偏移"
                        )
                except Exception:
                    pass
        except Exception as e:
            self.on_log(f"[错误] 截图失败: {e}")
            time.sleep(interval)
            return

        # ---- 1.5 测谎弹窗检测（反外挂）----
        # 放在 YOLO/决策之前：命中当帧直接 return，不浪费后面的推理，
        # 也保证不会在弹窗上继续按键（弹窗盖屏时 YOLO 输出全是垃圾）。
        if self._lie_detector is not None and self._lie_detector.available:
            lie_hit, lie_score = self._lie_detector.detect(frame)
            if lie_hit:
                self._handle_lie_detected(frame, lie_score)
                return

        # ---- 1.6 经验记录：定时读状态栏的 EXP（后台线程，不占本帧）----
        if getattr(self.config, "exp_enabled", True):
            self._exp.feed(frame, time.time())

        # ---- 1.65 定时结束游戏：测谎中走不到这里，测谎结束后再退 ----
        if self._quit_at is not None and time.time() >= self._quit_at:
            self._do_quit()
            return

        # ---- 1.7 自动换频道：这层好几分钟没打到怪（或按了测试按钮）就换一个频道 ----
        if self._channel_requested or self._channel_due():
            if not self._channel_requested:
                self.on_log(f"[换频道] 已经 "
                            f"{float(getattr(self.config, 'channel_idle_min', 3)):g} "
                            f"分钟没攻击，这层应该没怪了")
            self._channel_requested = False
            self._do_channel_switch()
            return

        # ---- 2~5. 感知流水线：YOLO 检测 → HP/MP → 自身定位 ----
        detections, hp_ratio, mp_ratio, self_pos = self._run_perception(frame)

        # 按类别名过滤（配置中可能用逗号分隔多个类别名）
        monster_classes = self._monster_classes()
        monsters = [d for d in detections if d.cls_name in monster_classes]
        floors = [d for d in detections if d.cls_name in self._floor_classes()]
        ropes = [d for d in detections if d.cls_name in self._rope_classes()]

        # 每 30 帧输出一次地图元素概览（YOLO 基于当前截图分析的结果）
        if self._frame_count % 30 == 0:
            parts = []
            if monsters:
                coords = ", ".join(
                    f"({d.center[0]},{d.center[1]})" for d in monsters
                )
                parts.append(f"怪{len(monsters)}只:{coords}")
            if floors:
                parts.append(f"平台{len(floors)}个")
            if ropes:
                parts.append(f"绳索{len(ropes)}条")
            if parts:
                self.on_log("[地图] " + " | ".join(parts))

        # ---- HP/MP 变化检测 ----
        if hp_ratio is not None:
            if self._last_hp_ratio is None or abs(hp_ratio - self._last_hp_ratio) >= 0.05:
                self._last_hp_ratio = hp_ratio
                self.on_log(f"[HP] {hp_ratio:.0%}")
        if mp_ratio is not None:
            if self._last_mp_ratio is None or abs(mp_ratio - self._last_mp_ratio) >= 0.05:
                self._last_mp_ratio = mp_ratio
                self.on_log(f"[MP] {mp_ratio:.0%}")

        # 每 30 帧输出一次状态
        if self._frame_count % 30 == 0:
            # HP/MP 比例
            hp_str = f"{hp_ratio:.0%}" if hp_ratio is not None else "N/A"
            mp_str = f"{mp_ratio:.0%}" if mp_ratio is not None else "N/A"

            # 测谎匹配分：贴在这条固定日志上，方便一眼看出阈值是否调错
            # （阈值 0.78 时正常画面应长期在 0.2~0.4；长期贴到 0.7+ 说明模板不匹配）
            lie_str = ""
            if self._lie_detector is not None and self._lie_detector.available:
                lie_str = f" 测谎分={self._lie_detector.last_score:.2f}"

            # 自身坐标
            center = self._get_last_center()
            if self_pos:
                if center:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"中心:({center[0]},{center[1]}) "
                        f"脚底:({self_pos[0]},{self_pos[1]}){lie_str}"
                    )
                else:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"脚底:({self_pos[0]},{self_pos[1]}){lie_str}"
                    )
            else:
                self.on_log(f"[状态] HP={hp_str} MP={mp_str} 自身未定位{lie_str}")

        # ---- 6. 决策与执行 ----
        # 活动范围：小地图上找自己的黄点（只有开了 range_limit 才跑，开销很小）
        minimap_x, minimap_width = self._locate_on_minimap(frame)
        # Context 是感知层 → 决策层的数据载体
        ctx = Context(
            monsters=monsters,
            floors=floors,
            ropes=ropes,
            self_position=self_pos,     # 自身脚底坐标 (cx, cy) 或 None
            self_center=self._get_last_center(),  # 角色中心点（距离推算用）
            hp_ratio=hp_ratio,          # 0.0~1.0
            mp_ratio=mp_ratio,          # 0.0~1.0
            detections=detections,      # 全部检测结果（供调试/日志用）
            frame_height=frame.shape[0],  # 像素参数缩放用
            frame_width=frame.shape[1],
            minimap_x=minimap_x,        # 小地图黄点 x（相对框选区域左边）
            minimap_width=minimap_width,
        )
        self.engine.decide(ctx)  # 决策引擎根据上下文执行动作

        # ---- 6.5. 自动拾取（定时按拾取键，每秒N次）----
        self._auto_pickup()

        # ---- 6.6. 自动加 buff（定期释放，冷却到就放）----
        self._auto_buff()

        # ---- 7. 预览回调 ----
        # 把 frame 和检测结果推给 UI 线程渲染；锁定目标随帧一起传，
        # 保证红框画的就是这一帧决策用的目标（不在 UI 线程读决策层状态）。
        self.on_frame(frame, detections, hp_ratio, mp_ratio,
                      self.engine.locked_target)

        # ---- 8. FPS 控制 ----
        elapsed = time.time() - t0
        if elapsed < interval:
            # 帧太快，sleep 补齐
            time.sleep(interval - elapsed)

    def _run_perception(self, frame):
        """感知流水线：YOLO 检测 → HP/MP 检测 → 自身定位（YOLO player 框）。

        供主循环 _loop_frame 与单帧预览 preview_frame_once 共用，
        返回 (detections, hp_ratio, mp_ratio, self_pos)。

        Args:
            frame: BGR 截图
        """
        # ---- YOLO 检测 ----
        # detect() 返回 [Detection, ...]，每个 Detection 包含:
        #   cls_name, confidence, x, y, w, h, center
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            self.on_log(f"[错误] 检测失败: {e}")
            detections = []

        # ---- HP 检测 ----
        # scale_region(): 把参考分辨率下的坐标缩放到当前帧的实际像素
        hp_region = self.config.scale_region(
            self.config.hp_region, frame.shape[1], frame.shape[0]
        )
        # detect_bar_ratio(): 多方法融合检测（边缘→亮度→颜色）
        hp_ratio = detect_bar_ratio(
            frame, hp_region,
            tuple(self.config.hp_color) if self.config.hp_color else None,
            self.config.hp_tolerance,
        )

        # ---- MP 检测 ----
        mp_region = self.config.scale_region(
            self.config.mp_region, frame.shape[1], frame.shape[0]
        )
        mp_ratio = detect_bar_ratio(
            frame, mp_region,
            tuple(self.config.mp_color) if self.config.mp_color else None,
            self.config.mp_tolerance,
            expand=3,  # MP 条下方是同色蓝色面板，减小扩展避免背景混入
        )

        # ---- 自身定位（YOLO player 框）----
        players = [d for d in detections if d.cls_name in self._player_classes()]
        self_pos, _ = self._locate_self_from_players(players)

        return detections, hp_ratio, mp_ratio, self_pos

    def preview_frame_once(self) -> float:
        """单次『截图 + 分析 + 预览』，供 UI“当前帧预览”按钮调试使用。

        只走感知 + 预览，不触发决策与按键注入，避免调试时误动角色。
        返回本次总耗时（秒）。
        """
        t0 = time.perf_counter()
        try:
            frame = self.capture.grab()
        except Exception as e:
            self.on_log(f"[预览] 截图失败: {e}")
            return time.perf_counter() - t0

        detections, hp_ratio, mp_ratio, self_pos = self._run_perception(frame)
        # 单帧预览也认一次小地图黄点：框选/调分界线时能立刻在预览里看到绿圈
        self._locate_on_minimap(frame)
        self.on_frame(frame, detections, hp_ratio, mp_ratio, None)

        # 输出自身坐标（脚底 + 角色中心）到日志，供调试
        center = self._get_last_center()
        if self_pos:
            if center:
                self.on_log(
                    f"[预览] 自身坐标 脚底=({self_pos[0]},{self_pos[1]}) "
                    f"中心=({center[0]},{center[1]})"
                )
            else:
                self.on_log(f"[预览] 自身坐标 脚底=({self_pos[0]},{self_pos[1]})")
        else:
            self.on_log("[预览] 自身未定位（未检测到 player）")

        elapsed = time.perf_counter() - t0
        self.on_log(f"[预览] 单帧分析+预览完成，总耗时 {elapsed * 1000:.1f}ms")
        return elapsed

    # =========================================================================
    # 类别名解析（从配置的逗号分隔字符串 → 列表）
    # 例如: "monster" → ["monster"]
    #       "monster,boss" → ["monster", "boss"]
    # =========================================================================

    def _auto_pickup(self):
        """自动拾取：按配置的间隔定时触发拾取键。

        仅在 pickup_enabled=True 且已锁定窗口时生效。
        间隔由 config.pickup_interval 控制（默认 0.333s = 每秒3次）。
        """
        enabled = getattr(self.config, "pickup_enabled", False)
        if not enabled:
            return
        if not self.capture.locked:
            return
        now = time.time()
        interval = getattr(self.config, "pickup_interval", 0.333)
        if now - self._last_pickup_time >= interval:
            pickup_key = getattr(self.config, "pickup_key", "z")
            self.executor.press_key(pickup_key, cooldown=0.0)
            self._last_pickup_time = now

    def _auto_buff(self):
        """自动加 buff：到期时先让决策层暂停攻击（施法窗口），再按 buff 键。

        流程：到期 → 请求施法窗口(暂停攻击) → 等攻击动画收尾 → 逐个按键 → 恢复。
        这样按下去时角色不在攻击动画中，游戏才能接收 buff 按键。
        多个 buff 同时到期时共用一个窗口，之间留 BUFF_MULTI_GAP 秒间隔。

        cooldown = 重新释放间隔(秒)，建议略小于 buff 游戏内持续时间。
        仅已锁定窗口时生效；按键为空或 cooldown<=0 的条目直接跳过。
        """
        if not self.capture.locked:
            return
        buffs = getattr(self.config, "buff_skills", None) or []
        if not buffs:
            return
        now = time.time()
        fps = max(1, int(getattr(self.config, "fps", 10) or 10))

        # ---- 1) 队列为空：检查到期 buff，有就开窗口（本帧先不按）----
        if not self._buff_queue:
            due = []
            for b in buffs:
                key = str(b.get("key", "") or "").strip()
                try:
                    cd = float(b.get("cooldown", 0) or 0)
                except (TypeError, ValueError):
                    cd = 0.0
                if not key or cd <= 0:
                    continue
                if now >= self._buff_next_time.get(key, 0.0):
                    due.append({"name": b.get("name", key), "key": key, "cd": cd})
            if not due:
                return
            self._buff_queue = due
            self._buff_open_time = now
            self._buff_last_press = 0.0
            window_s = (BUFF_PRESS_DELAY_SECONDS
                        + len(due) * BUFF_MULTI_GAP_SECONDS
                        + BUFF_CAST_TAIL_SECONDS)
            self.engine.request_buff_window(int(round(window_s * fps)))
            names = "、".join(d["name"] for d in due)
            self.on_log(f"[BUFF] {names} 到期 → 暂停攻击 {window_s:.1f}s 腾出施法窗口")
            return

        # ---- 2) 队列非空：等攻击动画收尾后逐个按键（此时攻击已被暂停）----
        if now - self._buff_open_time < BUFF_PRESS_DELAY_SECONDS:
            return
        if self._buff_last_press and now - self._buff_last_press < BUFF_MULTI_GAP_SECONDS:
            return
        b = self._buff_queue.pop(0)
        self.executor.press_key(b["key"], cooldown=0.0)
        self._buff_next_time[b["key"]] = now + b["cd"]
        self._buff_last_press = now
        self.on_log(f"[BUFF] 释放 {b['name']} ({b['key']})")

    def _monster_classes(self):
        return [c.strip() for c in self.config.monster_classes.split(",") if c.strip()]

    def _floor_classes(self):
        return [c.strip() for c in self.config.floor_classes.split(",") if c.strip()]

    def _rope_classes(self):
        return [c.strip() for c in self.config.rope_classes.split(",") if c.strip()]

    def _player_classes(self):
        return [c.strip() for c in self.config.player_classes.split(",") if c.strip()]

    # =========================================================================
    # 自身定位（YOLO player 框）
    # =========================================================================

    def _locate_self_from_players(self, players):
        """从 YOLO player 检测框定位自身，返回 (脚底, 角色中心)。

        脚底 = player bbox 底部中点 (y+h)；角色中心 = bbox 中心点 (y+h//2)。
        比 OCR 的"名字中心"更贴近真实站位。

        多人同屏时的选择策略（区分"自己"和"其他玩家"）：
          - 无历史位置（启动/换图）：取面积最大的 player（兜底启发式）；
          - 有历史位置：取离上一帧脚底最近的 player（位置连续性）。

        Returns:
            (foot, center) 两个 (x, y) 元组；未检测到 player 时返回 (None, None)
        """
        if not players:
            return None, None

        if self._last_self_foot is None or len(players) == 1:
            # 启动兜底 / 单玩家：取面积最大的（单玩家时即唯一那个）
            p = max(players, key=lambda d: d.w * d.h)
        else:
            # 位置连续性：取离上一帧脚底最近的 player
            lx, ly = self._last_self_foot
            p = min(players, key=lambda d:
                    (d.center[0] - lx) ** 2 + (d.center[1] - ly) ** 2)

        foot = (p.center[0], p.y + p.h)
        center = (p.center[0], p.y + p.h // 2)

        self._last_self_foot = foot
        self._cached_center = center
        return foot, center

    def _get_last_center(self) -> Optional[Tuple[int, int]]:
        """获取最近一次定位到的角色中心点（供日志/预览用）。"""
        return self._cached_center


def main():
    """CLI 入口（无 GUI）：加载配置并启动主循环，按 Ctrl+C 退出。

    用法: python -m src.main
    """
    from .utils.logger import get_logger
    log = get_logger()

    cfg = load_config()
    if not cfg.window_title:
        log.error("未配置 window_title，请在 config/user.yaml 中设置")
        return

    auto = Automation(cfg, on_log=lambda m: log.info(m))
    try:
        locked = auto.lock_window(cfg.window_title)
        log.info(f"已锁定窗口: {locked}")
    except Exception as e:
        log.error(f"锁定窗口失败: {e}")
        return

    auto.start()
    try:
        while auto.running:
            time.sleep(1)
    except KeyboardInterrupt:
        auto.stop()
    finally:
        # 让测谎录屏线程正常收尾（daemon 线程被直接杀掉会留下残缺视频）
        auto.shutdown()


if __name__ == "__main__":
    from .utils.config_loader import load_config
    main()