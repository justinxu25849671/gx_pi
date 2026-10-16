"""用缩小后的帧差区分静止和运动，避免每帧都重算不动的区域。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSample:
    """reuse 为真时调用方直接沿用上一帧结果。regions 为 None 表示整帧都允许新目标。"""

    reuse: bool
    regions: list[tuple[int, int, int, int]] | None


def point_in_regions(x: float, y: float,
                     regions: list[tuple[int, int, int, int]], pad: float = 28.0) -> bool:
    """判断一点是否落在某个运动区域里。"""
    for left, top, width, height in regions:
        if left - pad <= x <= left + width + pad and top - pad <= y <= top + height + pad:
            return True
    return False


def search_mask(shape, scale: float, regions, points, cv2):
    """运动区域加上已有追踪点，供检测时挡住静止背景。regions 为 None 时不限制。"""
    if regions is None:
        return None
    import numpy as np
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for left, top, width, height in regions:
        cv2.rectangle(
            mask,
            (int(left * scale), int(top * scale)),
            (int((left + width) * scale), int((top + height) * scale)),
            255, -1,
        )
    for x_px, y_px, radius_px in points:
        cv2.circle(mask, (int(float(x_px) * scale), int(float(y_px) * scale)),
                   max(8, int((float(radius_px) + 16.0) * scale)), 255, -1)
    return mask


class MotionGate:
    """画面不动就少算；隔一段时间整帧复核一次，避免新放上的静止物料一直被漏掉。"""

    def __init__(self, width: int = 160, diff_threshold: int = 24,
                 min_ratio: float = 0.012, refresh_every: int = 15) -> None:
        self.width = max(32, int(width))
        self.diff_threshold = int(diff_threshold)
        self.min_ratio = float(min_ratio)
        self.refresh_every = max(1, int(refresh_every))
        self._previous = None
        self._still = 0

    def update(self, frame) -> MotionSample:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_height, full_width = gray.shape[:2]
        small_width = min(self.width, full_width)
        small_height = max(1, int(round(full_height * small_width / full_width)))
        small = cv2.GaussianBlur(
            cv2.resize(gray, (small_width, small_height), interpolation=cv2.INTER_AREA),
            (5, 5), 0,
        )
        if self._previous is None or self._previous.shape != small.shape:
            self._previous = small
            self._still = 0
            return MotionSample(False, None)
        diff = cv2.absdiff(small, self._previous)
        self._previous = small
        changed = (diff > self.diff_threshold).astype(np.uint8) * 255
        ratio = float(np.count_nonzero(changed)) / changed.size
        if ratio < self.min_ratio:
            self._still += 1
            if self._still >= self.refresh_every:
                self._still = 0
                return MotionSample(False, None)
            return MotionSample(True, None)
        self._still = 0
        changed = cv2.dilate(changed, np.ones((7, 7), dtype=np.uint8))
        contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = full_width / float(small_width)
        scale_y = full_height / float(small_height)
        regions = []
        for contour in contours:
            x_px, y_px, width, height = cv2.boundingRect(contour)
            if width * height < 8:
                continue
            regions.append((
                int(x_px * scale_x), int(y_px * scale_y),
                max(1, int(width * scale_x)), max(1, int(height * scale_y)),
            ))
        return MotionSample(False, regions or None)
