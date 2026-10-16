"""放置区黑白同心环检测；与物料 HSV 分类完全独立。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from .vision import image_origin_px


@dataclass(frozen=True)
class RingCandidate:
    center_x_px: float
    center_y_px: float
    radius_px: float
    visible_arc_ratio: float
    fit_error_px: float
    concentric_layers: int
    position: int | None = None
    valid: bool = False
    stable: bool = False
    stable_frames: int = 0
    quality: float = 0.0
    reason: str = ""
    geometry_error_ratio: float = 0.0


def _size_ratio(new: float, old: float) -> float:
    small = min(new, old)
    if small <= 0:
        return float("inf")
    return max(new, old) / small


def _rect_px(raw, width: int, height: int) -> tuple[int, int, int, int]:
    values = [float(value) for value in raw]
    if len(values) != 4:
        raise ValueError("ROI/遮挡矩形必须包含四个数")
    if max(abs(value) for value in values) <= 1.0:
        values = [values[0] * width, values[1] * height,
                  values[2] * width, values[3] * height]
    x1, y1, x2, y2 = (round(value) for value in values)
    return (max(0, min(x1, width)), max(0, min(y1, height)),
            max(0, min(x2, width)), max(0, min(y2, height)))


class RingDetector:
    """用边缘上的圆作为放置环。不要求同心层、圆弧完整度或 1..3 号环位。"""

    def __init__(self, config: Mapping[str, Any], cv2_module=None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2_module
        self._config = dict(config)
        self._tracks: dict[int, tuple[float, float, int]] = {}
        self._observations: list[dict[str, float | int | None]] = []
        self._pixel_scale = 1.0

    def _masks(self, shape) -> tuple[Any, Any]:
        cv2 = self.cv2
        height, width = shape[:2]
        import numpy as np
        valid = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = _rect_px(self._config.get("roi", [0, 0, 1, 1]), width, height)
        cv2.rectangle(valid, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, -1)
        excluded = np.zeros_like(valid)
        for rect in self._config.get("occlusion_rects", []):
            ax, ay, bx, by = _rect_px(rect, width, height)
            cv2.rectangle(excluded, (ax, ay), (max(ax, bx - 1), max(ay, by - 1)), 255, -1)
        valid[excluded > 0] = 0
        return valid, excluded

    def _arc_metrics(self, edges, excluded, x: float, y: float,
                     radius: float) -> tuple[float, float]:
        height, width = edges.shape[:2]
        samples = max(72, int(2 * math.pi * radius / 3.0))
        hits = 0
        visible = 0
        distances: list[float] = []
        search = max(1, int(round(float(self._config.get("edge_search_px", 3)) * self._pixel_scale)))
        for index in range(samples):
            angle = 2.0 * math.pi * index / samples
            px = int(round(x + radius * math.cos(angle)))
            py = int(round(y + radius * math.sin(angle)))
            if not (0 <= px < width and 0 <= py < height) or excluded[py, px] != 0:
                continue
            visible += 1
            best = search + 1.0
            for offset in range(-search, search + 1):
                qx = int(round(x + (radius + offset) * math.cos(angle)))
                qy = int(round(y + (radius + offset) * math.sin(angle)))
                if 0 <= qx < width and 0 <= qy < height and edges[qy, qx] != 0:
                    best = min(best, abs(float(offset)))
            if best <= search:
                hits += 1
                distances.append(best)
        if visible == 0:
            return 0.0, float("inf")
        return hits / visible, (sum(distances) / len(distances) if distances else float(search + 1))

    def _group_concentric(self, circles, edges, excluded) -> list[RingCandidate]:
        center_tol = self._px("concentric_center_tolerance_px", 12.0)
        remaining = [tuple(float(value) for value in circle) for circle in circles]
        groups: list[list[tuple[float, float, float]]] = []
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            keep = []
            for circle in remaining:
                if math.hypot(circle[0] - seed[0], circle[1] - seed[1]) <= center_tol:
                    group.append(circle)
                else:
                    keep.append(circle)
            remaining = keep
            groups.append(group)

        del edges, excluded
        candidates: list[RingCandidate] = []
        for group in groups:
            outer = max(group, key=lambda item: item[2])
            candidates.append(RingCandidate(
                outer[0], outer[1], outer[2], 1.0, 0.0, 1,
                valid=True, quality=1.0,
            ))
        return candidates

    def _circle_proposals(self, gray, edges):
        """从边缘圆弧拟合候选，并以 Hough 结果补充完整圆。"""
        cv2 = self.cv2
        minimum = self._px("min_radius_px", 18)
        maximum = self._px("max_radius_px", 120)
        min_axis_ratio = float(self._config.get("min_ellipse_axis_ratio", 0.80))
        min_arc = float(self._config.get("proposal_min_arc_ratio", 0.45))
        proposals = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 8:
                continue
            _x, _y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 6 or box_height < 6:
                continue
            if min(box_width, box_height) / max(box_width, box_height) < min_axis_ratio:
                continue
            try:
                (x, y), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
            radius = (major + minor) / 4.0
            if not minimum <= radius <= maximum or major <= 0 or minor / major < min_axis_ratio:
                continue
            coverage = cv2.arcLength(contour, False) / max(2.0 * math.pi * radius, 1.0)
            if coverage >= min_arc:
                proposals.append((x, y, radius))
        min_radius = max(1, int(round(minimum)))
        max_radius = max(min_radius + 1, int(round(maximum)))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=float(self._config.get("hough_dp", 1.2)),
            minDist=max(1.0, self._px("hough_min_distance_px", 80)),
            param1=float(self._config.get("canny_high", 160)),
            param2=float(self._config.get("hough_accumulator", 48)),
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None:
            proposals.extend(tuple(float(value) for value in circle) for circle in circles[0])
        return proposals

    def _bind_positions(self, candidates: list[RingCandidate]) -> list[RingCandidate]:
        expected = self._config.get("position_centers_px", {})
        bound: list[RingCandidate] = []
        if expected:
            maximum = float(self._config.get("position_match_tolerance_px", 120.0))
            unused = list(candidates)
            for raw_position, raw_point in sorted(expected.items(), key=lambda item: int(item[0])):
                if not unused or raw_point is None:
                    continue
                position = int(raw_position)
                point = (float(raw_point[0]), float(raw_point[1]))
                ranked = sorted(((math.hypot(item.center_x_px - point[0],
                                              item.center_y_px - point[1]), item)
                                 for item in unused), key=lambda pair: pair[0])
                distance, candidate = ranked[0]
                ambiguous = (len(ranked) > 1 and
                             ranked[1][0] - distance <
                             float(self._config.get("candidate_ambiguity_px", 12.0)))
                unused.remove(candidate)
                valid = candidate.valid and distance <= maximum and not ambiguous
                bound.append(replace(candidate, position=position, valid=valid,
                                     reason=candidate.reason if valid else (
                                         "多个环候选无法区分" if ambiguous else
                                         f"与 {position} 号环预计位置相差 {distance:.1f}px")))
        else:
            return list(candidates)
        return bound

    def _stabilize(self, candidates: list[RingCandidate]) -> list[RingCandidate]:
        required = int(self._config.get("stable_frames", 4))
        tolerance = float(self._config.get("stable_center_tolerance_px", 5.0))
        output: list[RingCandidate] = []
        for item in candidates:
            if item.position is None or not item.valid:
                output.append(item)
                continue
            previous = self._tracks.get(item.position)
            count = 1
            if previous and math.hypot(item.center_x_px - previous[0],
                                       item.center_y_px - previous[1]) <= tolerance:
                count = previous[2] + 1
            self._tracks[item.position] = (item.center_x_px, item.center_y_px, count)
            output.append(replace(item, stable=count >= required, stable_frames=count))
        visible_positions = {item.position for item in candidates if item.valid}
        self._tracks = {position: track for position, track in self._tracks.items()
                        if position in visible_positions}
        return output

    def _px(self, name: str, default: float) -> float:
        """把全分辨率像素阈值换到当前检测图。"""
        return float(self._config.get(name, default)) * self._pixel_scale

    def _detection_view(self, frame, valid_mask, excluded):
        """缩小宽画面后再找圆。返回灰度图、遮挡图和相对原图的缩放。"""
        cv2 = self.cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        gray[valid_mask == 0] = 0
        _height, width = gray.shape[:2]
        limit = int(self._config.get("detect_max_width_px", 640))
        if limit <= 0 or width <= limit:
            return gray, excluded, 1.0
        scale = limit / float(width)
        size = (limit, max(1, int(round(gray.shape[0] * scale))))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        excluded = cv2.resize(excluded, size, interpolation=cv2.INTER_NEAREST)
        return gray, excluded, scale

    def _suppress_duplicates(self, candidates: list[RingCandidate]) -> list[RingCandidate]:
        """同一位置只保留边缘最完整的一个圆。同心圆已经合成一个候选。"""
        ranked = sorted(candidates, key=lambda item: (
            item.visible_arc_ratio, item.quality, item.radius_px), reverse=True)
        kept: list[RingCandidate] = []
        for item in ranked:
            if any(math.hypot(item.center_x_px - other.center_x_px,
                              item.center_y_px - other.center_y_px)
                   < 0.55 * max(item.radius_px, other.radius_px, 1.0)
                   for other in kept):
                continue
            kept.append(item)
        return kept

    def _to_full_resolution(self, candidates: list[RingCandidate], scale: float) -> list[RingCandidate]:
        if scale == 1.0:
            return candidates
        factor = 1.0 / scale
        return [replace(item,
                        center_x_px=item.center_x_px * factor,
                        center_y_px=item.center_y_px * factor,
                        radius_px=item.radius_px * factor,
                        fit_error_px=item.fit_error_px * factor)
                for item in candidates]

    def detect(self, frame) -> list[RingCandidate]:
        """检测画面中的圆。不按直径、圆弧完整度或环位编号剔除。"""
        cv2 = self.cv2
        valid_mask, excluded = self._masks(frame.shape)
        gray, excluded, scale = self._detection_view(frame, valid_mask, excluded)
        self._pixel_scale = scale
        edges = cv2.Canny(gray, int(self._config.get("canny_low", 60)),
                          int(self._config.get("canny_high", 160)))
        edges[cv2.resize(valid_mask, (gray.shape[1], gray.shape[0]),
                         interpolation=cv2.INTER_NEAREST) == 0] = 0
        raw = self._circle_proposals(gray, edges)
        grouped = self._suppress_duplicates(self._group_concentric(raw, edges, excluded))
        result = self._bind_positions(self._to_full_resolution(grouped, scale))
        self._pixel_scale = 1.0
        if not bool(self._config.get("calibrated", False)):
            result = [replace(item, valid=False, stable=False,
                              reason="环检测参数尚未用实物标定") for item in result]
        return self._persist(self._stabilize(result))

    def _persist(self, candidates: list[RingCandidate]) -> list[RingCandidate]:
        """连续出现才输出。半径突然变大或变小的那一帧不输出。"""
        required = int(self._config.get("stable_frames", 4))
        tolerance = float(self._config.get("stable_center_tolerance_px", 5.0))
        max_ratio = float(self._config.get("max_radius_change_ratio", 1.45))
        pairs = []
        for detection_index, item in enumerate(candidates):
            for track_index, track in enumerate(self._observations):
                distance = math.hypot(item.center_x_px - float(track["x"]),
                                       item.center_y_px - float(track["y"]))
                if distance <= tolerance:
                    pairs.append((distance, detection_index, track_index))
        pairs.sort()
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        matches: dict[int, int] = {}
        for _distance, detection_index, track_index in pairs:
            if detection_index in used_detections or track_index in used_tracks:
                continue
            used_detections.add(detection_index)
            used_tracks.add(track_index)
            matches[detection_index] = track_index
        next_tracks: list[dict[str, float | int | None]] = []
        emitted: list[RingCandidate] = []
        for detection_index, item in enumerate(candidates):
            track = None if detection_index not in matches else dict(self._observations[matches[detection_index]])
            if track is None:
                track = {
                    "x": item.center_x_px, "y": item.center_y_px, "radius": item.radius_px,
                    "count": 1, "pending_radius": None, "pending_count": 0,
                }
                next_tracks.append(track)
                if required <= 1:
                    emitted.append(replace(item, stable=True, stable_frames=1))
                continue
            track["x"], track["y"] = item.center_x_px, item.center_y_px
            if _size_ratio(item.radius_px, float(track["radius"])) > max_ratio:
                pending = track["pending_radius"]
                if pending is not None and _size_ratio(item.radius_px, float(pending)) <= max_ratio:
                    track["pending_count"] = int(track["pending_count"]) + 1
                    track["pending_radius"] = item.radius_px
                else:
                    track["pending_radius"] = item.radius_px
                    track["pending_count"] = 1
                if int(track["pending_count"]) < required:
                    next_tracks.append(track)
                    continue
                track["radius"] = item.radius_px
                track["count"] = int(track["pending_count"])
                track["pending_radius"] = None
                track["pending_count"] = 0
            else:
                track["radius"] = item.radius_px
                track["count"] = int(track["count"]) + 1
                track["pending_radius"] = None
                track["pending_count"] = 0
            next_tracks.append(track)
            if int(track["count"]) < required:
                continue
            emitted.append(replace(item, stable=True, stable_frames=int(track["count"])))
        self._observations = next_tracks
        return emitted

    @staticmethod
    def target(candidates: list[RingCandidate], position: int) -> RingCandidate | None:
        return next((item for item in candidates if item.position == position and item.valid), None)


def draw_ring_debug(frame, rings: list[RingCandidate]):
    """在复制的画面上显示环心、可见弧比例、拟合误差和稳定状态。"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
    output = frame.copy()
    height, width = output.shape[:2]
    for item in rings:
        color = (0, 220, 0) if item.valid and item.stable else ((0, 180, 255) if item.valid else (0, 0, 255))
        center = (round(item.center_x_px), round(item.center_y_px))
        origin = image_origin_px(item.center_x_px, item.center_y_px, width, height)
        cv2.circle(output, center, max(1, round(item.radius_px)), color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 18, 2)
        label = (f"ring={item.position or '?'} xy=({origin[0]},{origin[1]}) "
                 f"arc={item.visible_arc_ratio:.2f} err={item.fit_error_px:.1f} "
                 f"layers={item.concentric_layers} stable={item.stable_frames}")
        cv2.putText(output, label, (max(0, center[0] - round(item.radius_px)),
                                    max(18, center[1] - round(item.radius_px) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output
