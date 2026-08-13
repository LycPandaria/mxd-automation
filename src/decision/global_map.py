"""全局地图 — 持久化的网格地图，支持动态扩展。

================================================================================
设计理念
================================================================================

  类似扫地机器人的 SLAM：角色在游戏里走动时，每帧 YOLO 检测到的地板/绳索
  被逐帧"拼接"到一张全局地图中。探索完成后，地图保存到磁盘，下次进入同一张
  地图时直接加载，不需要重新探索。

================================================================================
数据结构
================================================================================

  用稀疏字典存储: dict[(gx, gy)] = cell_type
  因为地图可能很大，大部分区域是 UNEXPLORED，字典比二维数组更省内存。

  格子类型:
    UNEXPLORED = -1  未探索区域（灰色地带）
    AIR = 0          不可行走（空中/障碍物）
    FLOOR = 1        可站立/行走（地板）
    LADDER = 2       可攀爬（绳索/梯子）

================================================================================
坐标系统
================================================================================

  窗口坐标:  每一帧 YOLO 检测结果的坐标，原点在帧左上角
  全局坐标:  整张地图的坐标，原点在第一帧的左上角

  转换公式:
    全局坐标 = 窗口坐标 + 帧偏移
    帧偏移 = 角色全局坐标 - 角色窗口坐标

  例子:
    角色在窗口内坐标 (300, 400)，角色在全局地图中坐标 (1500, 800)
    则帧偏移 = (1500 - 300, 800 - 400) = (1200, 400)
    怪物在窗口内坐标 (500, 400) → 全局坐标 (500+1200, 400+400) = (1700, 800)

================================================================================
保存格式
================================================================================

  JSON 文件，结构:
    {
      "map_id": "henesys_field_01",
      "cell_size": 10,
      "bounds": {"min_x": -100, "max_x": 2000, "min_y": -50, "max_y": 1500},
      "cells": [[gx, gy, type], ...]
    }

  保存路径: assets/maps/{map_id}.json
"""
import json
import os
from typing import Dict, List, Optional, Tuple

# ---- 格子类型 ----
UNEXPLORED = -1
AIR = 0
FLOOR = 1
LADDER = 2

# 默认格子大小（像素）
CELL_SIZE = 10

# 地板站立区域: 地板检测框上方额外标记为可站立的格数
FLOOR_STAND_HEIGHT = 2


def pixel_to_grid(px: int) -> int:
    """像素坐标 → 网格坐标（向下取整）。"""
    return px // CELL_SIZE


def grid_to_pixel(g: int) -> int:
    """网格坐标 → 像素坐标（格子中心）。"""
    return g * CELL_SIZE + CELL_SIZE // 2


class GlobalMap:
    """全局地图 — 一张游戏地图的完整网格表示。

    支持:
      - 逐帧拼接: merge_frame() 将 YOLO 检测结果写入全局地图
      - 序列化: save() / load() 持久化到磁盘
      - 查询: get_cell() / is_walkable() / is_ladder() 供 A* 使用

    Attributes:
        map_id:    地图标识（如 "henesys_field_01"）
        cell_size: 每格像素大小
        grid:      稀疏网格字典 {(gx, gy): cell_type}
        bounds:    已探索区域的边界
    """

    def __init__(self, map_id: str = "", cell_size: int = CELL_SIZE):
        self.map_id = map_id
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], int] = {}
        self._min_gx = 0
        self._max_gx = 0
        self._min_gy = 0
        self._max_gy = 0

    # =========================================================================
    # 格子读写
    # =========================================================================

    def set_cell(self, gx: int, gy: int, cell_type: int):
        """设置格子类型，同时更新边界。"""
        self.grid[(gx, gy)] = cell_type
        if len(self.grid) == 1:
            self._min_gx = self._max_gx = gx
            self._min_gy = self._max_gy = gy
        else:
            self._min_gx = min(self._min_gx, gx)
            self._max_gx = max(self._max_gx, gx)
            self._min_gy = min(self._min_gy, gy)
            self._max_gy = max(self._max_gy, gy)

    def get_cell(self, gx: int, gy: int) -> int:
        """获取格子类型。未探索区域返回 UNEXPLORED(-1)。"""
        return self.grid.get((gx, gy), UNEXPLORED)

    def is_walkable(self, gx: int, gy: int) -> bool:
        """判断格子是否可行走（FLOOR 或 LADDER）。"""
        cell = self.get_cell(gx, gy)
        return cell in (FLOOR, LADDER)

    def is_ladder(self, gx: int, gy: int) -> bool:
        """判断格子是否是绳索/梯子。"""
        return self.get_cell(gx, gy) == LADDER

    def is_floor(self, gx: int, gy: int) -> bool:
        """判断格子是否是地板。"""
        return self.get_cell(gx, gy) == FLOOR

    @property
    def bounds(self) -> Dict[str, int]:
        """已探索区域的边界。"""
        return {
            "min_x": self._min_gx,
            "max_x": self._max_gx,
            "min_y": self._min_gy,
            "max_y": self._max_gy,
        }

    @property
    def width(self) -> int:
        return self._max_gx - self._min_gx + 1 if self.grid else 0

    @property
    def height(self) -> int:
        return self._max_gy - self._min_gy + 1 if self.grid else 0

    @property
    def explored_count(self) -> int:
        """已探索的格子数量。"""
        return len(self.grid)

    # =========================================================================
    # 逐帧拼接 — 探索建图的核心方法
    # =========================================================================

    def merge_frame(self, floors: List, ropes: List,
                    frame_offset_x: int, frame_offset_y: int,
                    frame_width: int, frame_height: int):
        """将一帧 YOLO 检测结果拼接到全局地图。

        Args:
            floors:         YOLO 检测到的地板列表 [Detection, ...]
            ropes:          YOLO 检测到的绳索列表 [Detection, ...]
            frame_offset_x: 当前帧左上角在全局地图中的 x 像素偏移
            frame_offset_y: 当前帧左上角在全局地图中的 y 像素偏移
            frame_width:    帧宽度（像素）
            frame_height:   帧高度（像素）
        """
        grid_offset_x = pixel_to_grid(frame_offset_x)
        grid_offset_y = pixel_to_grid(frame_offset_y)
        frame_gw = pixel_to_grid(frame_width) + 1
        frame_gh = pixel_to_grid(frame_height) + 1

        # 帧范围内未探索的格子标记为 AIR
        for gy in range(frame_gh):
            for gx in range(frame_gw):
                global_gx = grid_offset_x + gx
                global_gy = grid_offset_y + gy
                if (global_gx, global_gy) not in self.grid:
                    self.grid[(global_gx, global_gy)] = AIR

        # 标记地板: 地板框内 + 上方站立区域
        for d in floors:
            fx1, fy1 = d.x, d.y
            fx2, fy2 = d.x + d.w, d.y + d.h
            gx1 = grid_offset_x + pixel_to_grid(fx1)
            gy1 = grid_offset_y + pixel_to_grid(fy1)
            gx2 = grid_offset_x + pixel_to_grid(fx2)
            gy2 = grid_offset_y + pixel_to_grid(fy2)

            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    self.set_cell(gx, gy, FLOOR)

            stand_gy1 = max(gy1 - FLOOR_STAND_HEIGHT, 0)
            for gy in range(stand_gy1, gy1):
                for gx in range(gx1, gx2 + 1):
                    if self.get_cell(gx, gy) == AIR:
                        self.set_cell(gx, gy, FLOOR)

        # 标记绳索
        for d in ropes:
            fx1, fy1 = d.x, d.y
            fx2, fy2 = d.x + d.w, d.y + d.h
            gx1 = grid_offset_x + pixel_to_grid(fx1)
            gy1 = grid_offset_y + pixel_to_grid(fy1)
            gx2 = grid_offset_x + pixel_to_grid(fx2)
            gy2 = grid_offset_y + pixel_to_grid(fy2)

            for gy in range(gy1, gy2 + 1):
                for gx in range(gx1, gx2 + 1):
                    self.set_cell(gx, gy, LADDER)

    # =========================================================================
    # 序列化
    # =========================================================================

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "map_id": self.map_id,
            "cell_size": self.cell_size,
            "bounds": self.bounds,
            "cells": [[gx, gy, t] for (gx, gy), t in self.grid.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalMap":
        """从字典反序列化。"""
        m = cls(map_id=data["map_id"], cell_size=data["cell_size"])
        for gx, gy, t in data["cells"]:
            m.grid[(gx, gy)] = t
        if m.grid:
            m._min_gx = data["bounds"]["min_x"]
            m._max_gx = data["bounds"]["max_x"]
            m._min_gy = data["bounds"]["min_y"]
            m._max_gy = data["bounds"]["max_y"]
        return m

    def save(self, directory: str) -> str:
        """保存到磁盘。"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.map_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> Optional["GlobalMap"]:
        """从磁盘加载。"""
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_by_id(cls, map_id: str, directory: str) -> Optional["GlobalMap"]:
        """根据 map_id 从目录加载。"""
        return cls.load(os.path.join(directory, f"{map_id}.json"))

    def summary(self) -> str:
        """返回地图摘要信息。"""
        floor_count = sum(1 for t in self.grid.values() if t == FLOOR)
        ladder_count = sum(1 for t in self.grid.values() if t == LADDER)
        return (
            f"GlobalMap({self.map_id}) "
            f"bounds=({self._min_gx},{self._min_gy})~"
            f"({self._max_gx},{self._max_gy}) "
            f"cells={len(self.grid)} floors={floor_count} ladders={ladder_count}"
        )