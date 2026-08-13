"""地图探索器 — 一边走路一边建图，类似扫地机器人的 SLAM。

================================================================================
设计理念
================================================================================

  角色在游戏地图里走动时，每帧 YOLO 检测到的 floors/ropes 被逐帧拼接
  到 GlobalMap 中。通过追踪角色在窗口内的位移，推算当前帧在全局地图中
  的偏移量，实现多帧拼接。

================================================================================
工作流程
================================================================================

  1. start_explore(map_id) → 创建或加载 GlobalMap
  2. 每帧 update(floors, ropes, self_pos) → 拼接到全局地图
  3. save() → 保存到磁盘

  探索完成后，战斗时直接:
    GlobalMap.load_by_id(map_id) → AStarPathfinder(global_map) → 规划路径

================================================================================
位置追踪
================================================================================

  通过比较前后两帧 self_position 的变化来推算角色在全局地图中的位移:
    delta = self_pos_current - self_pos_prev
    global_pos += delta

  局限:
    - 如果角色被传送（换地图/进传送门），位移追踪会失效，需要重新定位
    - 如果 self_position 来自 HP 条偏移（固定位置），不能反映角色移动
      → 此时需要 OCR 识别脚底名字来获取真实位移

  解决方案（TODO）:
    使用 OCR 识别的脚底名字坐标作为 self_position 的来源，
    因为名字会跟着角色移动，可以反映真实位移。
"""
from typing import Optional, Tuple, List, Callable

from .global_map import GlobalMap
from ..perception.yolo_detector import Detection


class MapExplorer:
    """地图探索器 — 管理探索建图的全生命周期。

    用法:
        explorer = MapExplorer(on_log=print)
        explorer.start_explore("henesys_field_01")

        # 每帧调用
        explorer.update(floors, ropes, self_pos=(300, 400), frame_w=1366, frame_h=768)

        # 探索完成
        explorer.save("assets/maps")
    """

    def __init__(self, on_log: Optional[Callable[[str], None]] = None):
        self._log = on_log or (lambda m: None)
        self._current_map: Optional[GlobalMap] = None

        # 角色在全局地图中的位置（像素坐标）
        self._global_x: int = 0
        self._global_y: int = 0

        # 上一帧角色在窗口内的坐标（用于计算位移）
        self._last_self_pos: Optional[Tuple[int, int]] = None

        # 累计探索的帧数
        self._frame_count: int = 0

    # =========================================================================
    # 探索生命周期
    # =========================================================================

    def start_explore(self, map_id: str, maps_dir: str = "assets/maps") -> GlobalMap:
        """开始探索一个地图。

        优先从磁盘加载已有地图，不存在则创建新的。

        Args:
            map_id:   地图标识（如 "henesys_field_01"）
            maps_dir: 地图文件存储目录

        Returns:
            GlobalMap 实例
        """
        existing = GlobalMap.load_by_id(map_id, maps_dir)
        if existing:
            self._log(f"[探索] 加载已有地图 {map_id}，已探索 {existing.explored_count} 格")
            self._current_map = existing
        else:
            self._log(f"[探索] 开始新地图 {map_id}")
            self._current_map = GlobalMap(map_id=map_id)

        self._global_x = 0
        self._global_y = 0
        self._last_self_pos = None
        self._frame_count = 0
        return self._current_map

    def update(self, floors: List[Detection], ropes: List[Detection],
               self_pos: Optional[Tuple[int, int]],
               frame_width: int, frame_height: int):
        """每帧更新地图。

        将当前帧的 YOLO 检测结果拼接到全局地图中。
        同时追踪角色在全局地图中的位置。

        Args:
            floors:       YOLO 检测到的地板列表
            ropes:        YOLO 检测到的绳索列表
            self_pos:     角色在窗口内的脚底坐标 (x, y)，None 表示无法定位
            frame_width:  帧宽度
            frame_height: 帧高度
        """
        if self._current_map is None:
            return

        self._frame_count += 1

        # 更新全局位置: 根据 self_pos 的变化推算位移
        if self_pos is not None:
            if self._last_self_pos is not None:
                dx = self_pos[0] - self._last_self_pos[0]
                dy = self_pos[1] - self._last_self_pos[1]
                self._global_x += dx
                self._global_y += dy

            self._last_self_pos = self_pos

        # 计算当前帧在全局地图中的偏移
        # 帧偏移 = 角色全局坐标 - 角色窗口坐标
        if self_pos is not None:
            frame_offset_x = self._global_x - self_pos[0]
            frame_offset_y = self._global_y - self_pos[1]
        else:
            # 无法定位自身时，用上一帧的偏移（近似）
            frame_offset_x = self._global_x
            frame_offset_y = self._global_y

        # 将当前帧拼接到全局地图
        self._current_map.merge_frame(
            floors, ropes,
            frame_offset_x, frame_offset_y,
            frame_width, frame_height
        )

        # 每 100 帧输出一次进度
        if self._frame_count % 100 == 0:
            self._log(f"[探索] 第 {self._frame_count} 帧，"
                      f"全局位置 ({self._global_x}, {self._global_y})，"
                      f"已探索 {self._current_map.explored_count} 格")

    def save(self, directory: str = "assets/maps") -> Optional[str]:
        """保存当前地图到磁盘。

        Args:
            directory: 保存目录

        Returns:
            保存的文件路径，没有地图则返回 None
        """
        if self._current_map is None:
            return None
        path = self._current_map.save(directory)
        self._log(f"[探索] 地图已保存: {path} ({self._current_map.summary()})")
        return path

    # =========================================================================
    # 查询
    # =========================================================================

    @property
    def current_map(self) -> Optional[GlobalMap]:
        """当前正在探索的地图。"""
        return self._current_map

    @property
    def global_position(self) -> Tuple[int, int]:
        """角色在全局地图中的位置（像素坐标）。"""
        return (self._global_x, self._global_y)

    def window_to_global(self, wx: int, wy: int) -> Tuple[int, int]:
        """窗口坐标 → 全局坐标。

        Args:
            wx: 窗口内 x 坐标
            wy: 窗口内 y 坐标

        Returns:
            全局坐标 (gx, gy)
        """
        if self._last_self_pos is None:
            return (wx, wy)
        return (
            wx + self._global_x - self._last_self_pos[0],
            wy + self._global_y - self._last_self_pos[1],
        )

    def global_to_window(self, gx: int, gy: int) -> Tuple[int, int]:
        """全局坐标 → 窗口坐标（用于判断目标是否在屏幕内）。"""
        if self._last_self_pos is None:
            return (gx, gy)
        return (
            gx - self._global_x + self._last_self_pos[0],
            gy - self._global_y + self._last_self_pos[1],
        )