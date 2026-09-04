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
import time
import threading
from typing import Callable, Optional, Any, Tuple

from .perception.screen_capture import ScreenCapture
from .perception.yolo_detector import Detector, create_detector
from .perception.hp_mp_detector import detect_bar_ratio
from .execution.action_executor import ActionExecutor
from .decision.context import Context, DecisionEngine
from .utils.config_loader import Config, resolve_model_path


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
        on_frame: 预览回调，参数 (frame, detections, hp_ratio, mp_ratio)
    """

    def __init__(self, config: Config, detector: Optional[Detector] = None,
                 on_log: Optional[Callable[[str], None]] = None,
                 on_frame: Optional[Callable[..., None]] = None):
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
        self.on_frame = on_frame or (lambda f, d, h, m: None)

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

        self._running = True
        # daemon=True: 主线程退出时自动结束，不会卡住进程
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.on_log("[启动] 自动打怪已开始")

    def stop(self):
        """停止自动打怪。

        设置 _running = False，_loop() 会在下一次迭代时退出。
        同时释放所有按住的移动键，防止停止后方向键卡住。
        """
        if not self._running:
            return
        self._running = False
        self.engine.release_keys()  # 释放按住的方向键/上键
        self.on_log("[停止] 自动打怪已停止")

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
        """单帧执行：截图→检测→HP/MP→定位→决策→预览。"""
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

            # 自身坐标
            center = self._get_last_center()
            if self_pos:
                if center:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"中心:({center[0]},{center[1]}) "
                        f"脚底:({self_pos[0]},{self_pos[1]})"
                    )
                else:
                    self.on_log(
                        f"[状态] HP={hp_str} MP={mp_str} "
                        f"脚底:({self_pos[0]},{self_pos[1]})"
                    )
            else:
                self.on_log(f"[状态] HP={hp_str} MP={mp_str} 自身未定位")

        # ---- 6. 决策与执行 ----
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
        )
        self.engine.decide(ctx)  # 决策引擎根据上下文执行动作

        # ---- 6.5. 自动拾取（定时按拾取键，每秒N次）----
        self._auto_pickup()

        # ---- 7. 预览回调 ----
        # 把 frame 和检测结果推给 UI 线程渲染
        self.on_frame(frame, detections, hp_ratio, mp_ratio)

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
        self.on_frame(frame, detections, hp_ratio, mp_ratio)

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


if __name__ == "__main__":
    from .utils.config_loader import load_config
    main()