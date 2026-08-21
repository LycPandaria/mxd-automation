"""血量/蓝量检测（基于 OpenCV 颜色占比）。

原理：固定血条/蓝条位置，统计匹配目标颜色的像素占比。

血条区域固定后，每帧：
  1. 裁剪 (x, y, w, h) 区域（垂直方向扩展数像素，容纳标定偏移）
  2. 逐像素判断是否匹配目标颜色（RGB 容差 + HSV 色相/饱和度 双保险）
  3. 取匹配像素最多的行（条所在行），
     该行最长连续匹配段长度 / 条有效宽度 = 当前血量比例
"""
from typing import Optional, List

import numpy as np
import cv2


def detect_region_color(frame: np.ndarray, region) -> Optional[List[int]]:
    """从窗口截图的指定区域识别主颜色（用于 UI 框选标定，保持不变）。

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


# =========================================================================
# 内部工具
# =========================================================================

def _auto_target_color(roi_rgb: np.ndarray) -> np.ndarray:
    """自动在区域内找"最鲜艳"的像素作为目标色（颜色未配置时兜底）。

    以 饱和度 × 亮度 打分，取分数最高的像素颜色。
    """
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    score = hsv[:, :, 1].astype(np.float32) * hsv[:, :, 2].astype(np.float32)
    idx = int(score.argmax())
    row, col = np.unravel_index(idx, score.shape)
    return roi_rgb[row, col].astype(np.int16)


def _target_hue(target_rgb) -> int:
    """目标 RGB 颜色对应的 HSV 色相 (0~179)。"""
    img = np.array(target_rgb[:3], dtype=np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return int(hsv[0, 0, 0])


def _color_match_mask(roi_rgb: np.ndarray, target_rgb, tolerance: int) -> np.ndarray:
    """颜色匹配掩码：RGB 容差 与 HSV 色相/饱和度 双通道，任一命中即算匹配。

    - RGB 通道：像素与目标色各通道差 <= tolerance（覆盖标定色附近的精确色）
    - HSV 通道：色相接近目标色相 且 饱和度/亮度足够（覆盖血条的渐变/高光）
    """
    rgb = roi_rgb.astype(np.int16)
    diff = np.abs(rgb - target_rgb[:3])
    rgb_ok = np.all(diff <= tolerance, axis=2)

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    target_h = _target_hue(target_rgb)
    dh = np.abs(h - target_h)
    dh = np.minimum(dh, 180 - dh)          # 色相是环形的
    hsv_ok = (dh <= 25) & (s >= 40) & (v >= 30)

    return rgb_ok | hsv_ok


def _longest_run(row: np.ndarray) -> int:
    """一维 bool 数组中最长连续 True 的长度。"""
    row = row.astype(np.uint8)
    d = np.diff(np.concatenate(([0], row, [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        return 0
    return int((ends - starts).max())


def detect_bar_ratio(frame: np.ndarray, region, color=None,
                     tolerance: int = 20, expand: int = 8) -> Optional[float]:
    """检测血/蓝条剩余比例（自适应色带检测）。

    相比旧的"固定区域颜色占比"方案，这里做了三点改进，解决框选偏移导致的误检：

    1. 区域上下扩展 `expand` 像素，容纳窗口缩放 / 标定带来的 1~2px 垂直偏移；
    2. 匹配条件升级为 RGB 容差 + HSV 色相/饱和度双保险，能覆盖血条的渐变与高光，
       即使配置里标定的颜色不够准（如采样到暗部）也能匹配上；
    3. 比例改为"条所在行的最长连续色带长度 ÷ 条有效宽度"，
       不再被黑色背景、边框、文字等稀释。

    Args:
        frame: 窗口截图 BGR
        region: (x, y, w, h) 条的区域（窗口内像素坐标）
        color: (r, g, b) 目标颜色（RGB）；None 时自动采样区域内最鲜艳颜色
        tolerance: RGB 容差（各通道允许的最大差值）
        expand: 垂直方向额外扩展的像素数

    Returns:
        0.0-1.0 剩余比例；检测失败返回 None
    """
    if not region or frame is None:
        return None
    x, y, w, h = region
    if w <= 0 or h <= 0:
        return None

    # ---- 1. 扩展区域（垂直方向容纳偏移） ----
    y0 = max(0, y - expand)
    y1 = min(frame.shape[0], y + h + expand)
    x0 = max(0, x)
    x1 = min(frame.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return None

    roi = frame[y0:y1, x0:x1]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # ---- 2. 确定目标颜色（未配置时自动采样） ----
    target_rgb = _auto_target_color(roi_rgb) if color is None \
        else np.array(color[:3], dtype=np.int16)

    # ---- 3. 匹配掩码 ----
    mask = _color_match_mask(roi_rgb, target_rgb, tolerance)

    row_counts = mask.sum(axis=1)
    if int(row_counts.max()) <= 0:
        # 没有任何匹配：区分"空血"和"区域无效"
        # 若区域内几乎没有亮色内容（全黑/全空白），视为框选偏移导致检测失败
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_ratio = float(np.mean(gray > 25))
        if bright_ratio < 0.1:
            return None
        return 0.0  # 区域有内容但无目标色 → 空血

    # ---- 4. 取匹配像素最多的行（条中心行），算最长连续色带 ----
    # 比例 = 色带长度 / 区域宽。区域宽即条宽（框选时宽度对齐条两端），
    # 垂直偏移已由 expand 扩展吸收，横向按比例缩放误差很小，无需额外校正。
    best_row = int(row_counts.argmax())
    fill_len = _longest_run(mask[best_row])
    if fill_len <= 0:
        return None

    ratio = fill_len / roi_rgb.shape[1]
    return float(max(0.0, min(1.0, ratio)))
