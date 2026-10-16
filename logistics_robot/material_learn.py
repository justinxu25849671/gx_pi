"""从画面上点中的物料采样，学习该颜色的 HSV 范围。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def crop_patch(frame, x: float, y: float, radius: int = 28):
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
    keep = pixels[(pixels[:, 1] >= 40) & (pixels[:, 2] >= 40)]
    if len(keep) < 20:
        raise ValueError("采样点颜色太淡，请点在物料表面上")
    hues = keep[:, 0].astype(int)
    saturation = int(max(30, np.percentile(keep[:, 1], 10) - 25))
    value = int(max(30, np.percentile(keep[:, 2], 10) - 30))
    low = hues[hues < 90]
    high = hues[hues >= 90]
    if (len(low) > 0.2 * len(hues) and len(high) > 0.2 * len(hues)
            and int(low.min()) < 20 and int(high.max()) > 160):
        return [
            [0, saturation, value, _clamp_hue(float(np.percentile(low, 95)) + 8), 255, 255],
            [_clamp_hue(float(np.percentile(high, 5)) - 8), saturation, value, 179, 255, 255],
        ]
    return [[
        _clamp_hue(float(np.percentile(hues, 5)) - 8), saturation, value,
        _clamp_hue(float(np.percentile(hues, 95)) + 8), 255, 255,
    ]]


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    """把学习后的配置写回 JSON。"""
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _clamp_hue(value: float) -> int:
    return max(0, min(179, int(round(value))))
