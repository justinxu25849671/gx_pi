"""独立二维码读取与四颜色物料检测。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import math
import sys
from typing import Any

from .model import DetectedObject
from .motion_gate import point_in_regions, search_mask


def image_origin_px(x_px: float, y_px: float, width: int, height: int) -> tuple[int, int]:
    """把像素点换到画面中心坐标系。

    原点是画面中心 ``(0, 0)``。x 向右为正，y 向下为正，结果为四舍五入后的整数像素。
    """
    if width <= 0 or height <= 0:
        raise ValueError("画面宽高必须大于 0")
    return (
        _round_half_away(float(x_px) - width / 2.0),
        _round_half_away(float(y_px) - height / 2.0),
    )


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


MATERIAL_COLORS = ("red", "yellow", "blue", "green")
_DISPLAY_COLORS_BGR = {
    "red": (0, 0, 255), "yellow": (0, 255, 255),
    "blue": (255, 80, 0), "green": (0, 200, 0),
}


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


class ObjectDetector:
    """四颜色候选提取、形状过滤、同色目标选择与连续帧稳定判定。"""

    def __init__(self, camera_config: Mapping[str, Any], hsv_colors: Mapping[str, Any],
                 cv2_module=None) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as exc:
                raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2_module
        self._config = dict(camera_config.get("object_detection", camera_config))
        extra = set(hsv_colors) - set(MATERIAL_COLORS)
        missing = set(MATERIAL_COLORS) - set(hsv_colors)
        if extra or missing:
            raise ValueError("物料 HSV 只能且必须配置 red/yellow/blue/green 四种颜色")
        self._hsv_colors = hsv_colors
        self._tracks: list[dict[str, Any]] = []
        self._next_track_id = 1
        self.candidates = []
        self.color_mask = None
        self.edge_view = None

    def clear_tracks(self) -> None:
        """颜色范围刚学完时丢掉旧轨迹，避免沿用学习前的目标。"""
        self._tracks.clear()

    def set_detect_width(self, width: int) -> None:
        """运行中改检测图宽度，供画质和帧率自动调节。"""
        self._config["detect_max_width_px"] = max(160, int(width))

    def _valid_mask(self, shape):
        import numpy as np
        cv2 = self.cv2
        height, width = shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = _rect_px(self._config.get("roi", [0, 0, 1, 1]), width, height)
        cv2.rectangle(mask, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, -1)
        for rect in self._config.get("occlusion_rects", []):
            ax, ay, bx, by = _rect_px(rect, width, height)
            cv2.rectangle(mask, (ax, ay), (max(ax, bx - 1), max(ay, by - 1)), 0, -1)
        return mask

    def _reference_point(self, contour, bbox, centroid) -> tuple[float, float]:
        cv2 = self.cv2
        raw = self._config.get("reference_point", {})
        mode = raw.get("mode", "min_enclosing_circle")
        x, y, width, height = bbox
        if mode == "centroid":
            point = centroid
        elif mode == "bbox_center":
            point = (x + width / 2.0, y + height / 2.0)
        elif mode == "bottom_center":
            point = (x + width / 2.0, y + height)
        elif mode == "min_enclosing_circle":
            point, _ = cv2.minEnclosingCircle(contour)
        else:
            raise ValueError(f"未知物料参考点模式：{mode}")
        offset = raw.get("offset_px", [0.0, 0.0])
        return float(point[0]) + float(offset[0]), float(point[1]) + float(offset[1])

    @staticmethod
    def _size_ratio(new: float, old: float) -> float:
        small = min(new, old)
        if small <= 0:
            return float("inf")
        return max(new, old) / small

    def _match_tracks(self, objects: list[DetectedObject], tolerance: float) -> dict[int, int]:
        pairs = []
        for detection_index, item in enumerate(objects):
            point = item.reference_point_px
            for track_index, track in enumerate(self._tracks):
                if track["color"] != item.color:
                    continue
                distance = math.hypot(point[0] - track["x"], point[1] - track["y"])
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
        return matches

    def _stabilize(self, objects: list[DetectedObject], shape,
                   motion_regions=None) -> list[DetectedObject]:
        """连续出现才输出。面积突然变大或变小的那一帧不输出。"""
        del shape
        tolerance = float(self._config.get("stable_center_tolerance_px", 8.0))
        required = int(self._config.get("stable_frames", 4))
        max_ratio = float(self._config.get("max_area_change_ratio", 1.6))
        matches = self._match_tracks(objects, tolerance)
        next_tracks: list[dict[str, Any]] = []
        emitted: list[DetectedObject] = []
        for detection_index, item in enumerate(objects):
            point = item.reference_point_px
            track = None if detection_index not in matches else dict(self._tracks[matches[detection_index]])
            if track is None:
                if motion_regions is not None and not point_in_regions(
                        point[0], point[1], motion_regions):
                    continue
                track = {
                    "id": self._next_track_id, "color": item.color, "x": point[0], "y": point[1],
                    "area": item.area_px, "count": 1,
                    "pending_area": None, "pending_count": 0,
                }
                self._next_track_id += 1
                next_tracks.append(track)
                if required <= 1:
                    confidence = min(1.0, 0.45 * item.solidity + 0.25 * item.extent + 0.30)
                    emitted.append(replace(item, stable=True, stable_frames=1, confidence=confidence,
                                           track_id=int(track["id"])))
                continue
            alpha = min(1.0, max(0.05, float(self._config.get("track_smooth", 0.62))))
            track["x"] = point[0] * alpha + float(track["x"]) * (1.0 - alpha)
            track["y"] = point[1] * alpha + float(track["y"]) * (1.0 - alpha)
            if self._size_ratio(item.area_px, track["area"]) > max_ratio:
                pending = track["pending_area"]
                if pending is not None and self._size_ratio(item.area_px, pending) <= max_ratio:
                    track["pending_count"] += 1
                    track["pending_area"] = item.area_px
                else:
                    track["pending_area"] = item.area_px
                    track["pending_count"] = 1
                if track["pending_count"] < required:
                    next_tracks.append(track)
                    continue
                track["area"] = item.area_px
                track["count"] = track["pending_count"]
                track["pending_area"] = None
                track["pending_count"] = 0
            else:
                track["area"] = item.area_px
                track["count"] += 1
                track["pending_area"] = None
                track["pending_count"] = 0
            next_tracks.append(track)
            if track["count"] < required:
                continue
            confidence = min(1.0, 0.45 * item.solidity + 0.25 * item.extent +
                             0.30 * min(1.0, track["count"] / max(required, 1)))
            shift_x = float(track["x"]) - point[0]
            shift_y = float(track["y"]) - point[1]
            circle_x, circle_y = item.circle_center_px
            emitted.append(replace(
                item, stable=True, stable_frames=track["count"], confidence=confidence,

                reference_x_px=float(track["x"]), reference_y_px=float(track["y"]),
                circle_x_px=circle_x + shift_x, circle_y_px=circle_y + shift_y,
                track_id=int(track["id"]),
            ))
        self._tracks = next_tracks
        return emitted

    def detect(self, frame, colors=None, motion_regions=None) -> list[DetectedObject]:
        from .top_face import fit_top
        import numpy as np
        cv2 = self.cv2
        view, scale = self._detection_view(frame)
        factor = 1.0 / scale
        hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
        valid_mask = self._valid_mask(view.shape)
        minimum = float(self._config.get("min_area_px", 900))
        maximum = float(self._config.get("max_area_px", float("inf")))
        kernel_size = max(1, round(self._config.get("morphology_kernel_px", 5)*scale)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        chosen = MATERIAL_COLORS if not colors else tuple(c for c in colors if c in MATERIAL_COLORS)
        combined = np.zeros(view.shape[:2], np.uint8)
        self.edge_view = np.zeros(frame.shape[:2], np.uint8)
        candidates = []
        for color in chosen:
            mask = np.zeros(view.shape[:2], np.uint8)
            for bounds in self._hsv_colors[color]:
                h1, s1, v1, h2, s2, v2 = map(int, bounds)
                mask |= cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
            mask &= valid_mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            combined |= mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Ignore only tiny noise; retain rejected material for diagnosis.
            contours = [c for c in contours if cv2.contourArea(c) >= max(16, minimum*scale*scale*0.15)]
            if not contours:
                continue
            contour = np.rint(max(contours, key=cv2.contourArea)*factor).astype(np.int32)
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if not moments["m00"]:
                continue
            center = (moments["m10"]/moments["m00"], moments["m01"]/moments["m00"])
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity, extent = area/max(hull_area, 1), area/max(w*h, 1)
            ellipse, reason, metrics, edges = fit_top(frame, contour, self._config)
            if not minimum <= area <= maximum and not reason.startswith("目标不完整"):
                reason = "面积不符"
            # ROI and exclusion boundaries must not invent a complete candidate.
            boundary = cv2.dilate(255-valid_mask, np.ones((3, 3), np.uint8))
            small = np.rint(contour.reshape(-1, 2)*scale).astype(int)
            small[:, 0] = np.clip(small[:, 0], 0, boundary.shape[1]-1)
            small[:, 1] = np.clip(small[:, 1], 0, boundary.shape[0]-1)
            if np.any(boundary[small[:, 1], small[:, 0]]):
                reason = "目标不完整：接触 ROI 或遮挡区边界"
            if edges is not None:
                ex, ey, image = edges
                self.edge_view[ey:ey+image.shape[0], ex:ex+image.shape[1]] |= image
            top_center = ellipse[0] if ellipse else center
            candidates.append(DetectedObject(
                color=color, center_x_px=center[0], center_y_px=center[1], area_px=area,
                bbox=(x, y, w, h), contour=tuple(map(tuple, contour.reshape(-1, 2).tolist())),
                aspect_ratio=w/h, solidity=solidity, extent=extent,
                reference_x_px=top_center[0], reference_y_px=top_center[1],
                circle_x_px=top_center[0], circle_y_px=top_center[1],
                top_valid=ellipse is not None and not reason, top_ellipse=ellipse,
                top_reason=reason, top_metrics=tuple(metrics.items()),
            ))
        self.color_mask = cv2.resize(combined, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        stable = self._stabilize([c for c in candidates if c.top_valid], frame.shape, motion_regions)
        by_color = {c.color: c for c in stable}
        self.candidates = [by_color.get(c.color, c) for c in candidates]
        return stable

    def _detection_view(self, frame):
        """宽画面先缩小再分割。返回检测图和相对原图的缩放。"""
        cv2 = self.cv2
        _height, width = frame.shape[:2]
        limit = int(self._config.get("detect_max_width_px", 640))
        if limit <= 0 or width <= limit:
            return frame, 1.0
        scale = limit / float(width)
        size = (limit, max(1, int(round(frame.shape[0] * scale))))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale

    @staticmethod
    def best_target(objects: list[DetectedObject], color: str,
                    image_center_x_px: float) -> DetectedObject | None:
        candidates = [item for item in objects if item.color == color]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (
            not item.stable, abs(item.reference_point_px[0] - image_center_x_px),
            -item.confidence, -item.area_px,
        ))


def draw_shape_segments(frame, objects: list[DetectedObject], target: DetectedObject | None = None):
    """把每个颜色轮廓填成一块分割区域，并标出要发送的上层圆。"""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("未安装 OpenCV/numpy，请先执行 pip install -r requirements.txt") from exc
    output = frame.copy()
    for item in objects:
        if item.top_ellipse:
            cv2.ellipse(output, item.top_ellipse, (0, 255, 255) if item.top_valid else (0, 140, 255), 2)
            cv2.drawMarker(output, tuple(round(v) for v in item.top_ellipse[0]),
                           (0, 255, 255) if item.top_valid else (0, 140, 255), cv2.MARKER_CROSS, 12, 2)
        if not item.contour:
            continue
        contour = np.asarray(item.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(output, [contour], -1, _DISPLAY_COLORS_BGR[item.color], 2)
        cv2.drawMarker(output, (round(item.center_x_px), round(item.center_y_px)), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
    if target is not None:
        center = (round(target.circle_center_px[0]), round(target.circle_center_px[1]))
        cv2.circle(output, center, 10, (255, 255, 255), 2)
        cv2.putText(output, f"#{target.track_id} SEND" if target.track_id else "SEND",
                    (center[0] + 12, max(18, center[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def draw_object_debug_frame(frame, objects: list[DetectedObject], fps: float,
                            camera_index: int, camera_config: Mapping[str, Any] | None = None):
    """显示真实轮廓、抓取参考点、稳定帧数、ROI 和遮挡区域。"""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("未安装 OpenCV/numpy，请先执行 pip install -r requirements.txt") from exc
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    origin_x = width // 2
    origin_y = height // 2
    cv2.line(annotated, (0, origin_y), (max(0, width - 1), origin_y), (255, 220, 0), 1)
    cv2.line(annotated, (origin_x, 0), (origin_x, max(0, height - 1)), (255, 220, 0), 1)
    cv2.circle(annotated, (origin_x, origin_y), 4, (255, 220, 0), 1)
    cv2.putText(annotated, "origin (0,0)  +x right  +y down",
                (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA)
    detection = dict((camera_config or {}).get("object_detection", camera_config or {}))
    roi = _rect_px(detection.get("roi", [0, 0, 1, 1]), width, height)
    cv2.rectangle(annotated, (roi[0], roi[1]), (max(roi[0], roi[2] - 1), max(roi[1], roi[3] - 1)),
                  (220, 220, 220), 1)
    overlay = annotated.copy()
    for raw in detection.get("occlusion_rects", []):
        rect = _rect_px(raw, width, height)
        cv2.rectangle(overlay, (rect[0], rect[1]),
                      (max(rect[0], rect[2] - 1), max(rect[1], rect[3] - 1)),
                      (80, 80, 80), -1)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0.0, annotated)

    for item in objects:
        bgr = _DISPLAY_COLORS_BGR[item.color]
        center = (round(item.center_x_px), round(item.center_y_px))
        reference = (round(item.reference_point_px[0]), round(item.reference_point_px[1]))
        if item.contour:
            contour = np.asarray(item.contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(annotated, [contour], -1, bgr, 2)
        elif item.bbox:
            x, y, box_width, box_height = item.bbox
            cv2.rectangle(annotated, (x, y), (x + box_width, y + box_height), bgr, 2)
        else:
            radius = max(8, round(math.sqrt(max(item.area_px, 1.0) / math.pi)))
            cv2.circle(annotated, center, radius, bgr, 2)
        cv2.drawMarker(annotated, center, bgr, cv2.MARKER_CROSS, 10, 1)
        cv2.drawMarker(annotated, reference, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        origin = image_origin_px(item.circle_center_px[0], item.circle_center_px[1],
                                 width, height)
        label = (f"#{item.track_id or '?'} {item.color} circle=({origin[0]},{origin[1]}) "
                 f"A={item.area_px:.0f} stable={item.stable_frames} "
                 f"Q={item.confidence:.2f}")
        cv2.putText(annotated, label, (max(0, reference[0] - 80), max(18, reference[1] - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA)

    counts = Counter(item.color for item in objects)
    cv2.putText(annotated, f"Camera: {camera_index}  {width}x{height}  FPS: {fps:.1f}",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    detection_state = ("UNKNOWN" if not objects else
                       ("STABLE" if any(item.stable for item in objects) else "UNCERTAIN"))
    cv2.putText(annotated, f"Material state: {detection_state}", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 220, 0) if detection_state == "STABLE" else (0, 180, 255),
                2, cv2.LINE_AA)
    count_text = "  ".join(f"{color}: {counts[color]}" for color in MATERIAL_COLORS)
    cv2.putText(annotated, count_text, (16, height - 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "q/ESC: quit  s: save  p: print results",
                (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _camera_backend(cv2):
    """Windows 用 DirectShow。树莓派继续用 V4L2。"""
    if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraVision:
    """物料相机采帧封装；二维码相机应创建独立实例。"""

    def __init__(self, camera_config: Mapping[str, Any], hsv_colors: Mapping[str, Any]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2
        self._capture = cv2.VideoCapture(int(camera_config["index"]), _camera_backend(cv2))
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
        self._capture.set(cv2.CAP_PROP_FPS, int(camera_config["fps"]))
        if not self._capture.isOpened():
            raise RuntimeError("无法打开摄像头；检查 index、USB/CSI 接线及权限")
        self._detector = ObjectDetector(camera_config, hsv_colors, cv2)
        self._qr = cv2.QRCodeDetector()

    def read(self):
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("摄像头读帧失败")
        return frame

    def decode_task_code(self, frame) -> str | None:
        try:
            result = self._qr.detectAndDecodeMulti(frame)
            if len(result) == 4:
                ok, decoded, _, _ = result
                if ok:
                    return next((text.strip() for text in decoded if text), None)
        except (AttributeError, self.cv2.error):
            pass
        text, _, _ = self._qr.detectAndDecode(frame)
        return text.strip() if text else None

    def detect_objects(self, frame, colors=None, motion_regions=None) -> list[DetectedObject]:
        return self._detector.detect(frame, colors, motion_regions)

    def clear_tracks(self) -> None:
        self._detector.clear_tracks()

    def set_detect_width(self, width: int) -> None:
        self._detector.set_detect_width(width)

    best_target = staticmethod(ObjectDetector.best_target)

    def close(self) -> None:
        self._capture.release()
