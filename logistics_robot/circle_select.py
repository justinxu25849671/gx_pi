"""原画色块识别：色域轮廓、多圆拟合，增强后只留一个圆并跟踪。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .model import DetectedObject


@dataclass(frozen=True)
class CircleMark:
    """全图像素坐标里的一个圆。"""

    x: float
    y: float
    radius: float
    score: float


class CircleTracker:
    """按颜色实时跟踪唯一圆。窗口内跟上不换号，短暂丢失先沿速度外推。"""

    def __init__(self, stable_frames: int = 4, tolerance_px: float = 8.0,
                 smooth: float = 0.82, motion_px: float = 160.0, hold_frames: int = 4) -> None:
        self.required = max(1, int(stable_frames))
        self.tolerance = float(tolerance_px)
        self.smooth = min(1.0, max(0.05, float(smooth)))
        self.motion_px = float(motion_px)
        self.hold_frames = max(0, int(hold_frames))
        self._tracks: dict[str, dict[str, float]] = {}
        self._next_id = 1

    def clear(self) -> None:
        self._tracks.clear()

    def hint(self, color: str) -> tuple[float, float, float] | None:
        """预测下一帧圆心。锁定后只在它周围搜索。"""
        track = self._tracks.get(color)
        if track is None:
            return None
        return (float(track["x"]) + float(track.get("vx", 0.0)),
                float(track["y"]) + float(track.get("vy", 0.0)),
                float(track["r"]))

    def rejects_jump(self, color: str, mark: CircleMark) -> bool:
        """丢掉位置或面积跳得太多的一帧。下一帧仍然如此就立刻跟上。"""
        previous = self._tracks.get(color)
        if previous is None:
            return False
        prev_r = max(float(previous["r"]), 1.0)
        dist = math.hypot(mark.x - float(previous["x"]), mark.y - float(previous["y"]))
        speed = math.hypot(float(previous.get("vx", 0.0)), float(previous.get("vy", 0.0)))
        limit = max(90.0, prev_r * 1.2, speed * 2.5 + 24.0)
        area_ratio = (float(mark.radius) / prev_r) ** 2
        position_jump = dist > limit
        area_jump = area_ratio > 2.0 or area_ratio < 0.5
        if not position_jump and not area_jump:
            previous.pop("jump_x", None)
            previous.pop("jump_y", None)
            previous.pop("jump_r", None)
            return False
        pending_x = previous.get("jump_x")
        if pending_x is not None:
            pending_r = max(float(previous.get("jump_r", mark.radius)), 1.0)
            near_place = math.hypot(
                mark.x - float(pending_x), mark.y - float(previous.get("jump_y", mark.y))
            ) <= max(28.0, prev_r * 0.45)
            pending_ratio = (float(mark.radius) / pending_r) ** 2
            near_area = 0.5 <= pending_ratio <= 2.0
            if near_place and near_area:
                previous.pop("jump_x", None)
                previous.pop("jump_y", None)
                previous.pop("jump_r", None)
                return False
        previous["jump_x"] = float(mark.x)
        previous["jump_y"] = float(mark.y)
        previous["jump_r"] = float(mark.radius)
        return True

    def hold(self, color: str) -> bool:
        """这一帧没跟上。还在保持期内就留着原轨迹，不重新全图搜索。"""
        track = self._tracks.get(color)
        if track is None:
            return False
        track["misses"] = float(track.get("misses", 0.0)) + 1.0
        track["vx"] = float(track.get("vx", 0.0)) * 0.5
        track["vy"] = float(track.get("vy", 0.0)) * 0.5
        if track["misses"] <= self.hold_frames:
            return True
        self._tracks.pop(color, None)
        return False

    def coast(self, color: str) -> DetectedObject | None:
        """短暂丢失时继续画上一帧的色域圆，编号和稳定帧都不换。"""
        track = self._tracks.get(color)
        if track is None:
            return None
        center = (float(track["x"]), float(track["y"]))
        radius = float(track["r"])
        count = int(track["count"])
        return DetectedObject(
            color=color, center_x_px=center[0], center_y_px=center[1], area_px=0.0,
            circle_x_px=center[0], circle_y_px=center[1],
            reference_x_px=center[0], reference_y_px=center[1],
            stable=count >= self.required, stable_frames=count,
            confidence=min(1.0, count / self.required),
            track_id=int(track["id"]), top_valid=True, top_reason="短暂丢失，沿用上一帧",
            top_ellipse=(center, (radius * 2.0, radius * 2.0), 0.0),
            top_metrics=(("radius_px", radius), ("score", 0.0)),
        )

    def _gate(self) -> float:
        return max(self.tolerance, self.motion_px)

    def observe(self, color: str, mark: CircleMark | None,
                material_center: tuple[float, float] | None,
                area_px: float, reason: str = "", associated: bool = False) -> DetectedObject | None:
        if mark is None:
            self._tracks.pop(color, None)
            if material_center is None:
                return None
            return DetectedObject(
                color=color, center_x_px=material_center[0], center_y_px=material_center[1],
                area_px=area_px, stable=False, top_valid=False, top_reason=reason or "增强后没有留下圆",
            )
        previous = self._tracks.get(color)
        jumped = previous is None or (
            not associated and math.hypot(
                mark.x - float(previous["x"]), mark.y - float(previous["y"])) > self._gate())
        if jumped:
            track = {"id": self._next_id, "x": mark.x, "y": mark.y,
                     "r": mark.radius, "count": 1, "vx": 0.0, "vy": 0.0, "misses": 0.0}
            self._next_id += 1
        else:
            track = dict(previous)
            # 当前帧测到的浅色小圆直接用，不做时间平滑，避免跟踪滞后。
            track["x"] = mark.x
            track["y"] = mark.y
            track["r"] = mark.radius
            track["vx"] = float(track["x"]) - float(previous["x"])
            track["vy"] = float(track["y"]) - float(previous["y"])
            track["misses"] = 0.0
            track["count"] = float(previous["count"]) + 1
        self._tracks[color] = track
        count = int(track["count"])
        stable = count >= self.required
        center = (float(track["x"]), float(track["y"]))
        radius = float(track["r"])
        return DetectedObject(
            color=color,
            center_x_px=material_center[0] if material_center else center[0],
            center_y_px=material_center[1] if material_center else center[1],
            area_px=area_px,
            circle_x_px=center[0], circle_y_px=center[1],
            reference_x_px=center[0], reference_y_px=center[1],
            stable=stable, stable_frames=count,
            confidence=min(1.0, count / self.required),
            track_id=int(track["id"]),
            top_valid=True, top_reason="",
            top_ellipse=(center, (radius * 2.0, radius * 2.0), 0.0),
            top_metrics=(("radius_px", radius), ("score", float(mark.score))),
        )


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_gray(gray):
    """灰度图加大对比度后再锐化。"""
    contrasted = _CLAHE.apply(gray)
    blurred = cv2.GaussianBlur(contrasted, (0, 0), 1.2)
    return cv2.addWeighted(contrasted, 1.7, blurred, -0.7, 0)


def color_mask(frame, bounds_list, kernel_px: int = 5):
    """按色域取出掩膜。红色可以有多段色相。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for bounds in bounds_list:
        h1, s1, v1, h2, s2, v2 = (int(value) for value in bounds)
        partial = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
        mask = partial if mask is None else cv2.bitwise_or(mask, partial)
    if mask is None:
        return np.zeros(frame.shape[:2], np.uint8)
    size = max(1, int(kernel_px)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _dedupe(circles: list[CircleMark]) -> list[CircleMark]:
    """中心和半径都接近的圆合成一个，保留分数更高的。"""
    kept: list[CircleMark] = []
    for circle in sorted(circles, key=lambda item: item.score, reverse=True):
        if any(math.hypot(circle.x - other.x, circle.y - other.y)
               < 0.45 * max(circle.radius, other.radius, 1.0)
               and abs(circle.radius - other.radius)
               <= 0.22 * max(circle.radius, other.radius, 1.0)
               for other in kept):
            continue
        kept.append(circle)
    return kept


def _circles_on_gray(gray, offset_x: float, offset_y: float) -> list[CircleMark]:
    """在已经增强的灰度图上找互相分开的圆。坐标加回原图。"""
    height, width = gray.shape[:2]
    if height < 12 or width < 12:
        return []
    found: list[CircleMark] = []
    max_radius = min(height, width) // 2
    if max_radius >= 6:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(8.0, max_radius / 3.0),
            param1=100, param2=18, minRadius=6, maxRadius=max_radius,
        )
        if circles is not None:
            for x_px, y_px, radius in circles[0]:
                found.append(CircleMark(float(x_px) + offset_x, float(y_px) + offset_y,
                                        float(radius), 0.7))
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 12:
        contours = sorted(contours, key=len, reverse=True)[:12]
    for contour in contours:
        if len(contour) < 16:
            continue
        (x_px, y_px), radius = cv2.minEnclosingCircle(contour)
        if radius < 6:
            continue
        coverage = cv2.arcLength(contour, False) / max(2.0 * math.pi * float(radius), 1.0)
        if coverage < 0.45:
            continue
        found.append(CircleMark(float(x_px) + offset_x, float(y_px) + offset_y,
                                float(radius), 0.5 + 0.4 * min(coverage, 1.0)))
    return found


def propose_circles(frame, contour) -> list[CircleMark]:
    """最大色块上先做圆形拟合，分开成若干个不同的圆。"""
    (blob_x, blob_y), blob_radius = cv2.minEnclosingCircle(contour)
    proposals = [CircleMark(float(blob_x), float(blob_y), float(blob_radius), 0.4)]
    x_px, y_px, width, height = cv2.boundingRect(contour)
    pad = 8
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(0, x_px - pad), max(0, y_px - pad)
    x1, y1 = min(frame_w, x_px + width + pad), min(frame_h, y_px + height + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return proposals
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local = contour - np.array([[[x0, y0]]], dtype=np.int32)
    region = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(region, [local], -1, 255, -1)
    region = cv2.dilate(region, np.ones((3, 3), np.uint8))
    gray = cv2.bitwise_and(gray, region)
    proposals.extend(_circles_on_gray(gray, x0, y0))
    return _dedupe(proposals)


def _refine_one(frame, mark: CircleMark) -> CircleMark | None:
    """把候选圆裁出来，灰度、加大对比度、锐化，再拟合一次。"""
    frame_h, frame_w = frame.shape[:2]
    pad = max(4, int(round(mark.radius * 0.35)))
    x0 = max(0, int(round(mark.x - mark.radius)) - pad)
    y0 = max(0, int(round(mark.y - mark.radius)) - pad)
    x1 = min(frame_w, int(round(mark.x + mark.radius)) + pad)
    y1 = min(frame_h, int(round(mark.y + mark.radius)) + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = enhance_gray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    matches = _circles_on_gray(gray, x0, y0)
    if not matches:
        return None
    nearest = min(matches, key=lambda item: math.hypot(item.x - mark.x, item.y - mark.y)
                  + abs(item.radius - mark.radius))
    if math.hypot(nearest.x - mark.x, nearest.y - mark.y) > max(12.0, mark.radius * 0.45):
        return None
    if abs(nearest.radius - mark.radius) > max(8.0, mark.radius * 0.45):
        return None
    agreement = 1.0 - abs(nearest.radius - mark.radius) / max(mark.radius, 1.0)
    return CircleMark(nearest.x, nearest.y, nearest.radius, nearest.score + agreement)


def fit_blob_circles(frame, contour) -> tuple[list[CircleMark], CircleMark | None]:
    """拟合出多个圆，增强后按与色块外圆的一致程度只留一个。"""
    proposals = propose_circles(frame, contour)
    if not proposals:
        return [], None
    blob = max(proposals, key=lambda item: item.radius)
    ranked = sorted(proposals, key=lambda item: item.score, reverse=True)[:4]
    refined: list[CircleMark] = []
    for mark in ranked:
        hit = _refine_one(frame, mark)
        if hit is not None:
            refined.append(hit)
    if not refined:
        return proposals, None

    def consistency(mark: CircleMark) -> float:
        center_error = math.hypot(mark.x - blob.x, mark.y - blob.y) / max(blob.radius, 1.0)
        radius_error = abs(mark.radius - blob.radius) / max(blob.radius, 1.0)
        return mark.score - center_error - radius_error

    return proposals, max(refined, key=consistency)


def gamut_circle(contour, offset_x: float = 0.0, offset_y: float = 0.0) -> CircleMark | None:
    """杯口顶层：从上往下第一处最宽的位置。不取瓶身里面积最大的内切圆。"""
    x_px, y_px, width, height = cv2.boundingRect(contour)
    if width < 6 or height < 6:
        return None
    local = contour - np.array([[[x_px, y_px]]], dtype=np.int32)
    region = np.zeros((height, width), np.uint8)
    cv2.drawContours(region, [local], -1, 255, -1)
    widths = np.count_nonzero(region, axis=1).astype(np.float32)
    if float(widths.max()) < 6:
        return None
    smooth = widths
    if height >= 3:
        smooth = np.empty_like(widths)
        smooth[0] = widths[0]
        smooth[-1] = widths[-1]
        smooth[1:-1] = (widths[:-2] + 2.0 * widths[1:-1] + widths[2:]) / 4.0
    present = np.flatnonzero(smooth > 2)
    if len(present) == 0:
        return None
    peak_row = int(present[0])
    peak = float(smooth[peak_row])
    plateau_end = peak_row
    for row in range(peak_row + 1, height):
        value = float(smooth[row])
        if value > peak + 0.5:
            peak = value
            peak_row = row
            plateau_end = row
            continue
        if value + 1.0 >= peak * 0.97:
            plateau_end = row
            continue
        break
    peak_row = (peak_row + plateau_end) // 2
    cols = np.flatnonzero(region[peak_row])
    if len(cols) < 4:
        return None
    radius = (int(cols[-1]) - int(cols[0]) + 1) / 2.0
    if radius < 4:
        return None
    return CircleMark((int(cols[0]) + int(cols[-1])) / 2.0 + x_px + offset_x,
                      float(peak_row + y_px) + offset_y, radius, 0.9)


def fit_lighter_circle(crop, contour, offset_x: float = 0.0, offset_y: float = 0.0,
                       color: str = "") -> CircleMark | None:
    """色块里拟合圆，留下更小、颜色更浅的那个。四种颜色都只取最亮的一段。"""
    outer = gamut_circle(contour, offset_x, offset_y)
    x_px, y_px, width, height = cv2.boundingRect(contour)
    if crop is None or width < 8 or height < 8:
        return outer
    y1 = min(int(crop.shape[0]), y_px + height)
    x1 = min(int(crop.shape[1]), x_px + width)
    if y1 <= y_px or x1 <= x_px:
        return outer
    patch = crop[y_px:y1, x_px:x1]
    local = contour - np.array([[[x_px, y_px]]], dtype=np.int32)
    mask = np.zeros(patch.shape[:2], np.uint8)
    cv2.drawContours(mask, [local], -1, 255, -1)
    if cv2.countNonZero(mask) < 30:
        return outer
    value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    inside = value[mask > 0]
    dark = float(np.percentile(inside, 40))
    bright = float(np.percentile(inside, 96))
    if bright < dark + 8.0:
        return outer
    level = bright
    light = np.where((mask > 0) & (value >= level), np.uint8(255), np.uint8(0))
    light_area = int(cv2.countNonZero(light))
    blob_area = int(cv2.countNonZero(mask))
    if light_area < blob_area * 0.05 or light_area > blob_area * 0.8:
        return outer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    light = cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: CircleMark | None = None
    for item in contours:
        area = float(cv2.contourArea(item))
        perim = float(cv2.arcLength(item, True))
        if area < 40 or perim < 1:
            continue
        if 4.0 * math.pi * area / (perim * perim) < 0.45:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(item)
        if radius < 4:
            continue
        if outer is not None and float(radius) > outer.radius * 0.9:
            continue
        mark = CircleMark(float(cx) + x_px + offset_x, float(cy) + y_px + offset_y,
                          float(radius), 0.95)
        if best is None or mark.radius < best.radius:
            best = mark
    return best if best is not None else outer


def _centroid(contour) -> tuple[float, float] | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _detection_view(frame, limit: int):
    """全图搜索先缩小。返回检测图和相对原图的缩放。"""
    _height, width = frame.shape[:2]
    if limit <= 0 or width <= limit:
        return frame, 1.0
    scale = limit / float(width)
    size = (limit, max(1, int(round(frame.shape[0] * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR), scale


def _scale_mark(mark: CircleMark, scale: float) -> CircleMark:
    if scale == 1.0:
        return mark
    factor = 1.0 / scale
    return CircleMark(mark.x * factor, mark.y * factor, mark.radius * factor, mark.score)


def _follow_window(frame, bounds, hint: tuple[float, float, float], kernel_px: int,
                   minimum: float, pad: float, color: str = ""):
    """在预测点周围取色域，圆用色块内部能放下的最大圆。"""
    x_px, y_px, radius = hint
    frame_h, frame_w = frame.shape[:2]
    x0 = max(0, int(math.floor(x_px - radius - pad)))
    y0 = max(0, int(math.floor(y_px - radius - pad)))
    x1 = min(frame_w, int(math.ceil(x_px + radius + pad)))
    y1 = min(frame_h, int(math.ceil(y_px + radius + pad)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mask = color_mask(crop, bounds, kernel_px)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [item for item in contours if cv2.contourArea(item) >= max(16.0, minimum * 0.25)]
    if not usable:
        return None
    contour = max(usable, key=cv2.contourArea)
    center = _centroid(contour)
    mark = fit_lighter_circle(crop, contour, x0, y0, color)
    if center is None or mark is None:
        return None
    return mark, (center[0] + x0, center[1] + y0), float(cv2.contourArea(contour))


def follow_circle(frame, bounds, hint: tuple[float, float, float], kernel_px: int,
                  minimum: float, motion_px: float, color: str = ""):
    """先在速度预测点附近找，找不到再把窗口放大一圈。"""
    _x_px, _y_px, radius = hint
    pad = max(float(motion_px), float(radius), 48.0)
    found = _follow_window(frame, bounds, hint, kernel_px, minimum, pad, color)
    if found is not None:
        return found
    return _follow_window(frame, bounds, hint, kernel_px, minimum, pad * 2.2, color)


def _acquire_circle(frame, bounds, minimum: float, maximum: float,
                    kernel_px: int, limit: int, color: str = ""):
    """全图缩小后找最大色块，再把圆心换回原图坐标。"""
    view, scale = _detection_view(frame, limit)
    view_minimum = minimum * scale * scale
    view_kernel = max(1, int(round(kernel_px * scale))) | 1
    mask = color_mask(view, bounds, view_kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    usable = [item for item in contours if cv2.contourArea(item) >= view_minimum]
    if not usable:
        return None
    contour = max(usable, key=cv2.contourArea)
    area = float(cv2.contourArea(contour)) / max(scale * scale, 1e-9)
    center = _centroid(contour)
    if center is None:
        return None
    full_center = (center[0] / scale, center[1] / scale)
    if area > maximum:
        return None, full_center, area, "面积不符"
    winner = fit_lighter_circle(view, contour, color=color)
    if winner is not None:
        winner = _scale_mark(winner, scale)
    return winner, full_center, area, ""


def recognize_frame(frame, colors, hsv_colors, detection, tracker: CircleTracker) -> list[DetectedObject]:
    """锁定后只跟踪圆附近；丢失后再做一次缩小后的全图搜索。"""
    tracker.required = max(1, int(detection.get("stable_frames", tracker.required)))
    tracker.tolerance = float(detection.get("stable_center_tolerance_px", tracker.tolerance))
    tracker.motion_px = float(detection.get("track_motion_px", max(tracker.tolerance, 160.0)))
    tracker.hold_frames = max(0, int(detection.get("track_hold_frames", tracker.hold_frames)))
    tracker.smooth = min(1.0, max(0.05, float(detection.get("track_smooth", tracker.smooth))))
    minimum = float(detection.get("min_area_px", 900))
    maximum = float(detection.get("max_area_px", 250000))
    kernel_px = int(detection.get("morphology_kernel_px", 5))
    limit = int(detection.get("detect_max_width_px", 640))
    objects: list[DetectedObject] = []
    seen: set[str] = set()
    for color in colors:
        bounds = hsv_colors.get(color) or []
        if not bounds:
            continue
        hint = tracker.hint(color)
        if hint is not None:
            followed = follow_circle(frame, bounds, hint, kernel_px, minimum, tracker.motion_px, color)
            if followed is not None:
                mark, center, area = followed
                if tracker.rejects_jump(color, mark):
                    seen.add(color)
                    item = tracker.coast(color)
                    if item is not None:
                        objects.append(item)
                    continue
                seen.add(color)
                item = tracker.observe(color, mark, center, area, associated=True)
                if item is not None:
                    objects.append(item)
                continue
            if tracker.hold(color):
                seen.add(color)
                item = tracker.coast(color)
                if item is not None:
                    objects.append(item)
                continue
        found = _acquire_circle(frame, bounds, minimum, maximum, kernel_px, limit, color)
        if found is None:
            tracker.observe(color, None, None, 0.0)
            continue
        winner, center, area, reason = found
        seen.add(color)
        item = tracker.observe(color, winner, center, area, reason)
        if item is not None:
            objects.append(item)
    for color in list(tracker._tracks):
        if color not in seen:
            tracker.observe(color, None, None, 0.0)
    return objects
