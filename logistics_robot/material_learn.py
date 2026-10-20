"""从画面上点中的物料采样，学习该颜色的 HSV 范围。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def crop_patch(frame, x: float, y: float, radius: int = 42):
    """复制点击点周围的小块。点必须落在画面内。"""
    height, width = frame.shape[:2]
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("采样点在画面外")
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    return frame[y0:y1, x0:x1].copy()


def hsv_bounds_from_bgr(bgr) -> list[list[int]]:
    """根据一块物料表面估计 HSV 范围。红色跨过色相 0 时分成两段。"""
    import cv2
    if bgr.size == 0:
        raise ValueError("采样区域为空")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    keep = pixels[(pixels[:, 1] >= 25) & (pixels[:, 2] >= 25)]
    if len(keep) < 20:
        raise ValueError("采样点颜色太淡，请点在物料表面上")
    hues = keep[:, 0].astype(int)
    saturation = int(max(15, np.percentile(keep[:, 1], 5) - 90))
    value = int(max(15, np.percentile(keep[:, 2], 5) - 90))
    hue_pad = 18
    raw_low = int(np.percentile(hues, 5))
    raw_high = int(np.percentile(hues, 95))
    low = hues[hues < 90]
    high = hues[hues >= 90]
    crosses = (len(low) > 0.2 * len(hues) and len(high) > 0.2 * len(hues)
               and int(low.min()) < 20 and int(high.max()) > 160)
    if crosses or raw_low <= 12 or raw_high >= 168:
        high_end = _clamp_hue(float(np.percentile(low, 95)) + hue_pad) if len(low) else 18
        low_start = _clamp_hue(float(np.percentile(high, 5)) - hue_pad) if len(high) else 164
        return [
            [0, saturation, value, max(high_end, 18), 255, 255],
            [min(low_start, 164), saturation, value, 179, 255, 255],
        ]
    return [[
        _clamp_hue(float(raw_low) - hue_pad), saturation, value,
        _clamp_hue(float(raw_high) + hue_pad), 255, 255,
    ]]


def hsv_bounds_from_rgb(r: int, g: int, b: int, tolerance: int = 48,
                        value_floor: int = 0) -> list[list[int]]:
    """把一个 RGB 颜色扩成识别用的 HSV 范围。亮度下限不会低于 value_floor。"""
    import cv2
    red = int(np.clip(int(r), 0, 255))
    green = int(np.clip(int(g), 0, 255))
    blue = int(np.clip(int(b), 0, 255))
    spread = int(np.clip(int(tolerance), 8, 120))
    pixel = np.uint8([[[blue, green, red]]])
    hue, sat, val = (int(channel) for channel in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])
    hue_pad = int(np.clip(round(spread * 0.22), 8, 28))
    sat_low = int(np.clip(max(40, sat - spread), 0, 255))
    val_low = int(np.clip(max(int(value_floor), val - min(spread, 40)), 0, 255))
    low = hue - hue_pad
    high = hue + hue_pad
    if low < 0:
        return [
            [0, sat_low, val_low, min(179, high), 255, 255],
            [max(0, 180 + low), sat_low, val_low, 179, 255, 255],
        ]
    if high > 179:
        return [
            [0, sat_low, val_low, min(179, high - 180), 255, 255],
            [max(0, low), sat_low, val_low, 179, 255, 255],
        ]
    return [[low, sat_low, val_low, min(179, high), 255, 255]]


def rgb_sample_from_bounds(bounds) -> tuple[int, int, int]:
    """用色域里偏亮的一点还原 RGB，供调色图显示。"""
    import cv2
    if not bounds:
        return 220, 40, 40
    h1, s1, v1, h2, _s2, v2 = (int(value) for value in bounds[0])
    hue = 0 if len(bounds) > 1 else int(round((h1 + h2) / 2))
    sat = int(np.clip(max(s1, 100), 0, 255))
    val = int(np.clip(max(v1, v2, 200), 0, 255))
    bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """把学习后的配置写回 JSON。"""
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _clamp_hue(value: float) -> int:
    return max(0, min(179, int(round(value))))
