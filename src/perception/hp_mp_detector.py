"""血量/蓝量检测（基于 OpenCV 颜色占比）。

原理：固定血条/蓝条位置，统计匹配目标颜色的像素占比。

血条区域固定后，每帧：
  1. 裁剪 (x, y, w, h) 区域
  2. 逐像素判断是否匹配目标颜色（RGB 差 < tolerance）
  3. 匹配像素数 / 总像素数 = 当前血量比例
"""
from typing import Optional, List

import numpy as np
import cv2


def detect_region_color(frame: np.ndarray, region) -> Optional[List[int]]:
    """从窗口截图的指定区域识别主颜色。

    Args:
        frame: 窗口截图 BGR
        region: (x, y, w, h) 窗口内坐标

    Returns:
        [r, g, b] RGB 颜色；识别失败返回 None
    """
    if not region or frame is None:
        return None
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # 取中间行像素
    mid_row = roi[roi.shape[0] // 2, :, :]

    # 排除接近黑色的背景像素（亮度 < 30）
    pixels = mid_row.reshape(-1, 3)
    brightness = np.mean(pixels, axis=1)
    mask = brightness > 30
    if mask.sum() > 0:
        valid_pixels = pixels[mask]
    else:
        valid_pixels = pixels

    # 计算平均颜色（BGR → RGB）
    avg_bgr = np.mean(valid_pixels, axis=0).astype(int)
    return [int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])]


def detect_bar_ratio(frame: np.ndarray, region, color,
                     tolerance: int = 20) -> Optional[float]:
    """检测血/蓝条剩余比例（颜色占比法）。

    固定区域，统计匹配颜色的像素占比。
    满状态: 比例 >= 0.95
    空状态: 比例 <= 0.05

    Args:
        frame: 窗口截图 BGR
        region: (x, y, w, h) 条的区域（窗口内坐标）
        color: (r, g, b) 目标颜色（RGB）
        tolerance: 颜色容差（各通道允许的最大差值）

    Returns:
        0.0-1.0 颜色占比；None 表示未配置/无法检测
    """
    if not region or frame is None:
        return None
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # 转 BGR → RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 目标颜色
    target = np.array(color[:3], dtype=np.int16)

    # 逐像素计算颜色差
    diff = np.abs(roi_rgb.astype(np.int16) - target)

    # 每个像素的 RGB 各通道都在容差内才算匹配
    match_mask = np.all(diff <= tolerance, axis=2)

    # 计算匹配比例
    total_pixels = match_mask.size
    if total_pixels == 0:
        return None

    matched = np.sum(match_mask)
    ratio = float(matched) / total_pixels

    return ratio