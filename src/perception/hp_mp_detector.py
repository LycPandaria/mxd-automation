"""血量/蓝量百分比检测（基于 OpenCV）。

多方法融合检测血/蓝条剩余比例，依次尝试：
  1. 边缘检测（最精确，找颜色跳变点）
  2. 亮度检测（最稳健）
  3. 颜色检测（兜底）

同时提供 ``detect_region_color`` 用于框选区域后自动识别主颜色：
取中间行像素均值，过滤掉接近黑色的背景（亮度 < 30）。
"""
from typing import Optional, List

import numpy as np
import cv2


# ---------------- 区域颜色识别 ----------------

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
    mid_row = roi[roi.shape[0] // 2, :, :]  # shape: (w, 3) BGR

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


# ---------------- 血/蓝条比例检测 ----------------

def detect_bar_ratio(frame: np.ndarray, region, color, tolerance=20) -> Optional[float]:
    """多方法融合检测血/蓝条剩余比例。

    依次尝试：边缘检测 → 亮度检测 → 颜色检测，返回第一个有效结果。

    Args:
        frame: 窗口截图 BGR
        region: (x, y, w, h) 条在窗口内的区域
        color: (r, g, b) 条颜色（RGB）
        tolerance: 颜色容差

    Returns:
        0.0-1.0 剩余比例；None 表示未配置/无法检测
    """
    if not region or frame is None:
        return None
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # 方法1：边缘检测（最精确）
    ratio = _detect_by_edge(roi, color, tolerance)
    if ratio is not None:
        return ratio

    # 方法2：亮度检测（最稳健）
    ratio = _detect_by_brightness(roi, color, tolerance)
    if ratio is not None:
        return ratio

    # 方法3：颜色检测（兜底）
    return _detect_by_color(roi, color, tolerance)


def _detect_by_edge(roi: np.ndarray, color, tolerance) -> Optional[float]:
    """通过边缘检测找血条填充末端的颜色跳变点。"""
    h, w = roi.shape[:2]

    # 转灰度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 统计每列的边缘像素数
    col_edges = np.sum(edges > 0, axis=0).astype(float)

    # 边缘密集的列可能是填充末端（颜色跳变处）
    edge_cols = np.where(col_edges > h * 0.3)[0]  # 超过30%行有边缘

    if len(edge_cols) == 0:
        return None

    # 边缘密集区域的右端点即为填充末端
    max_edge_col = edge_cols.max() + 1

    # 验证：该位置左侧应该大部分是亮的（用颜色验证）
    if color and len(color) >= 3:
        target_bgr = np.array([color[2], color[1], color[0]], dtype=np.int16)
        left_roi = roi[:, :max_edge_col]
        diff = np.abs(left_roi.astype(np.int16) - target_bgr)
        match_ratio = np.mean(np.all(diff <= tolerance, axis=2))
        if match_ratio < 0.1:  # 匹配率太低说明不可靠
            return None

    return float(max_edge_col) / w


def _detect_by_brightness(roi: np.ndarray, color, tolerance) -> Optional[float]:
    """通过亮度检测血条填充范围。"""
    h, w = roi.shape[:2]

    # 转灰度
    gray = np.mean(roi, axis=2)  # shape: (h, w)

    # 统计每列的平均亮度
    col_brightness = np.mean(gray, axis=0)  # shape: (w,)

    # 自适应阈值：取前10%最亮列的平均亮度
    sorted_bright = np.sort(col_brightness)[::-1]
    ref_count = max(1, len(sorted_bright) // 10)
    ref_bright = np.mean(sorted_bright[:ref_count]) if len(sorted_bright) > 0 else 128

    # 阈值：参考亮度的 35%
    threshold = max(25, ref_bright * 0.35)

    # 找亮列
    bright_cols = col_brightness >= threshold

    # 如果有颜色配置，用颜色辅助验证
    if color and len(color) >= 3:
        target_bgr = np.array([color[2], color[1], color[0]], dtype=np.int16)
        diff = np.abs(roi.astype(np.int16) - target_bgr)
        color_match = np.all(diff <= tolerance * 1.5, axis=2)
        col_match = np.any(color_match, axis=0)
        bright_cols = bright_cols & col_match  # 两者都要满足

    matched_cols = np.where(bright_cols)[0]
    if len(matched_cols) == 0:
        return None

    max_col = matched_cols.max() + 1
    return float(max_col) / w


def _detect_by_color(roi: np.ndarray, color, tolerance) -> Optional[float]:
    """纯颜色匹配（兜底方案）。"""
    if not color or len(color) < 3:
        return None

    h, w = roi.shape[:2]
    target_bgr = np.array([color[2], color[1], color[0]], dtype=np.int16)
    diff = np.abs(roi.astype(np.int16) - target_bgr)
    mask = np.all(diff <= tolerance, axis=2)

    col_match = np.any(mask, axis=0)
    matched_cols = np.where(col_match)[0]
    if len(matched_cols) == 0:
        return 0.0

    max_col = matched_cols.max() + 1
    return float(max_col) / w
