"""放置区黑白同心环检测；与物料 HSV 分类完全独立。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping


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
    """通过可见圆弧、同心层数和位置关联筛选放置环。"""

    def __init__(self, config: Mapping[str, Any], cv2_module=None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2_module
        self._config = dict(config)
        self._tracks: dict[int, tuple[float, float, int]] = {}

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
        search = int(self._config.get("edge_search_px", 3))
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
        center_tol = float(self._config.get("concentric_center_tolerance_px", 12.0))
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

        candidates: list[RingCandidate] = []
        min_layers = int(self._config.get("min_concentric_layers", 2))
        min_arc = float(self._config.get("min_visible_arc_ratio", 0.30))
        max_error = float(self._config.get("max_fit_error_px", 2.5))
        for group in groups:
            # Hough 常会在同一条印刷线的内外边缘给出近似重复半径，先去重，
            # 再用已知直径关系寻找能匹配最多层的像素/mm 比例。
            radius_separation = float(self._config.get("min_radius_separation_px", 4.0))
            distinct: list[tuple[float, float, float]] = []
            for circle in sorted(group, key=lambda item: item[2]):
                if not distinct or abs(circle[2] - distinct[-1][2]) >= radius_separation:
                    distinct.append(circle)
            diameters = [float(self._config.get("material_base_diameter_mm", 50.0))]
            diameters.extend(float(value) for value in
                             self._config.get("ring_outer_diameters_mm", []))
            tolerance = float(self._config.get("diameter_tolerance_ratio", 0.10))
            best_group: list[tuple[float, float, float]] = []
            best_error = float("inf")
            for circle in distinct:
                for diameter in diameters:
                    scale = (2.0 * circle[2]) / diameter
                    chosen: dict[float, tuple[tuple[float, float, float], float]] = {}
                    for observed in distinct:
                        observed_mm = 2.0 * observed[2] / scale
                        expected = min(diameters, key=lambda item: abs(item - observed_mm))
                        error = abs(expected - observed_mm) / expected
                        if error <= tolerance and (expected not in chosen or error < chosen[expected][1]):
                            chosen[expected] = (observed, error)
                    matched = [value[0] for value in chosen.values()]
                    average_error = (sum(value[1] for value in chosen.values()) / len(chosen)
                                     if chosen else float("inf"))
                    if (len(matched), -average_error) > (len(best_group), -best_error):
                        best_group = matched
                        best_error = average_error
            group = best_group
            if not group:
                continue
            weights = [max(circle[2], 1.0) for circle in group]
            total = sum(weights)
            center_x = sum(circle[0] * weight for circle, weight in zip(group, weights)) / total
            center_y = sum(circle[1] * weight for circle, weight in zip(group, weights)) / total
            outer = max(group, key=lambda item: item[2])
            arc, error = self._arc_metrics(edges, excluded, center_x, center_y, outer[2])
            layers = len(group)
            geometry_quality = max(0.0, 1.0 - best_error / max(tolerance, 1e-6))
            quality = min(1.0, 0.45 * arc +
                          0.30 * min(1.0, layers / max(min_layers + 1, 1)) +
                          0.25 * geometry_quality)
            valid = layers >= min_layers and arc >= min_arc and error <= max_error
            reason = "" if valid else (f"圆弧/尺寸层质量不足 arc={arc:.2f} "
                                       f"layers={layers} err={error:.2f} geo={best_error:.3f}")
            candidates.append(RingCandidate(center_x, center_y, outer[2], arc, error,
                                            layers, valid=valid, quality=quality,
                                            reason=reason,
                                            geometry_error_ratio=best_error))
        return candidates

    def _circle_proposals(self, gray, edges):
        """从边缘圆弧拟合候选，并以 Hough 结果补充完整圆。"""
        cv2 = self.cv2
        minimum = float(self._config.get("min_radius_px", 18))
        maximum = float(self._config.get("max_radius_px", 120))
        min_axis_ratio = float(self._config.get("min_ellipse_axis_ratio", 0.65))
        min_arc = float(self._config.get("proposal_min_arc_ratio", 0.15))
        proposals = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 5:
                continue
            (x, y), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
            major = max(float(axis_a), float(axis_b))
            minor = min(float(axis_a), float(axis_b))
            radius = (major + minor) / 4.0
            if not minimum <= radius <= maximum or major <= 0 or minor / major < min_axis_ratio:
                continue
            coverage = cv2.arcLength(contour, False) / max(2.0 * math.pi * radius, 1.0)
            if coverage >= min_arc:
                proposals.append((x, y, radius))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=float(self._config.get("hough_dp", 1.2)),
            minDist=float(self._config.get("hough_min_distance_px", 80)),
            param1=float(self._config.get("canny_high", 160)),
            param2=float(self._config.get("hough_accumulator", 24)),
            minRadius=int(minimum), maxRadius=int(maximum),
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
            valid = sorted((item for item in candidates if item.valid),
                           key=lambda item: item.center_x_px)
            if len(valid) == 3:
                bound.extend(replace(item, position=index + 1)
                             for index, item in enumerate(valid))
            else:
                bound.extend(replace(item, valid=False,
                                     reason="未标定环位且不能唯一得到三个候选")
                             for item in candidates)
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

    def detect(self, frame) -> list[RingCandidate]:
        """检测并绑定 1..3 号环；配置未标定时仅输出候选，不判为可放置。"""
        cv2 = self.cv2
        valid_mask, excluded = self._masks(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        gray[valid_mask == 0] = 0
        edges = cv2.Canny(gray, int(self._config.get("canny_low", 60)),
                          int(self._config.get("canny_high", 160)))
        edges[valid_mask == 0] = 0
        raw = self._circle_proposals(gray, edges)
        result = self._bind_positions(self._group_concentric(raw, edges, excluded))
        if not bool(self._config.get("calibrated", False)):
            result = [replace(item, valid=False, stable=False,
                              reason="环检测参数尚未用实物标定") for item in result]
        return self._stabilize(result)

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
    for item in rings:
        color = (0, 220, 0) if item.valid and item.stable else ((0, 180, 255) if item.valid else (0, 0, 255))
        center = (round(item.center_x_px), round(item.center_y_px))
        cv2.circle(output, center, round(item.radius_px), color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 18, 2)
        label = (f"ring={item.position or '?'} arc={item.visible_arc_ratio:.2f} "
                 f"err={item.fit_error_px:.1f} geo={item.geometry_error_ratio:.3f} "
                 f"layers={item.concentric_layers} "
                 f"stable={item.stable_frames}")
        cv2.putText(output, label, (max(0, center[0] - round(item.radius_px)),
                                    max(18, center[1] - round(item.radius_px) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output
