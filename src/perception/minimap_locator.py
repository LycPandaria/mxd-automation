"""在小地图上找自己（黄色星形小点）。

冒险岛的镜头跟着角色走，角色在画面里的 x 基本不变，看不出自己在地图的哪一半。
小地图是整张地图的缩略图，自己的位置就是上面那个黄点：核心 RGB(255,255,136)，
外圈 (255,255,0)，约 6×6 px。别的玩家是红点、队友/家族是橙点，都不会被认成自己。
"""
from typing import Optional, Tuple

import cv2
import numpy as np

DOT_MIN_RGB = (200, 195)
"""黄点的 R、G 下限（星形尖角抗锯齿后约 (207,204,0)）"""
DOT_MAX_B = 175
"""黄点的 B 上限：挡掉小地图上发白的云、文字"""
DOT_MIN_AREA = 10
"""比这小的黄色块不是自己（比如任务/传送口旁边 2~3 px 的小星星）"""
DOT_MAX_SIDE = 14
"""黄点边长上限（px）：约 6 px，放大显示时也到不了这么大"""
DOT_MAX_ASPECT = 1.5
"""黄点差不多是正方形：长条状的黄色块是地图上画的东西"""


def find_self_dots(frame: np.ndarray, region: Tuple[int, int, int, int]) -> list:
    """找出区域内所有像"自己"的黄点，返回 [(x, y, 面积), ...]，坐标相对区域左上角。"""
    rx, ry, rw, rh = region
    crop = frame[ry:ry + rh, rx:rx + rw]
    if crop.size == 0:
        return []
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    mask = ((r >= DOT_MIN_RGB[0]) & (g >= DOT_MIN_RGB[1]) & (b <= DOT_MAX_B)).astype(np.uint8)
    n, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots = []
    for i in range(1, n):
        w, h, area = stats[i][2:5]
        if area < DOT_MIN_AREA or max(w, h) > DOT_MAX_SIDE or max(w, h) > min(w, h) * DOT_MAX_ASPECT:
            continue
        dots.append((float(centers[i][0]), float(centers[i][1]), int(area)))
    return dots


class MinimapLocator:
    """每帧在框选的小地图区域里找自己；有好几个候选时选离上次最近的（第一次选面积最大的）。"""

    def __init__(self):
        self._last: Optional[Tuple[float, float]] = None

    def reset(self):
        self._last = None

    def locate(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """返回黄点中心 (x, y)，相对区域左上角；找不到返回 None。"""
        dots = find_self_dots(frame, region)
        if not dots:
            return None
        if self._last is not None:
            lx, ly = self._last
            best = min(dots, key=lambda d: abs(d[0] - lx) + abs(d[1] - ly))
        else:
            best = max(dots, key=lambda d: d[2])
        self._last = (best[0], best[1])
        return self._last
