"""A* 拓扑寻路（占位）。

读取 ``config/maps/*.json`` 中从 WZ 拆包提取的绝对坐标拓扑图，
计算跨平台跳跃路径。节点是平台锚点，边是可达跳跃关系。

TODO:
    - 加载地图 JSON（nodes/edges）
    - A* 启发式（欧氏距离）
    - 输出跳跃动作序列（坐标 + 方向 + 跳跃键）
"""
from typing import List, Optional


class AStarPathfinder:
    """A* 寻路器（占位）。"""

    def __init__(self, map_path: Optional[str] = None):
        self.map_path = map_path
        self._nodes = []
        self._edges = []

    def load_map(self, map_path: str):
        """加载地图 JSON。"""
        # TODO: 实现
        pass

    def find_path(self, start, goal) -> List:
        """计算从 start 到 goal 的路径。"""
        # TODO: 实现 A* 算法
        return []
