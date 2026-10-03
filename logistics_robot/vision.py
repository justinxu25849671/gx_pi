"""独立二维码读取与四颜色物料检测。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

from .model import DetectedObject


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
        self._tracks: dict[str, tuple[float, float, int]] = {}

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

    def _stabilize(self, objects: list[DetectedObject], shape) -> list[DetectedObject]:
        height, width = shape[:2]
        reference = self._config.get("selection_reference_px", [width / 2.0, height / 2.0])
        reference = (float(reference[0]), float(reference[1]))
        tolerance = float(self._config.get("stable_center_tolerance_px", 8.0))
        required = int(self._config.get("stable_frames", 4))
        output = list(objects)
        visible_colors: set[str] = set()
        for color in MATERIAL_COLORS:
            indexed = [(index, item) for index, item in enumerate(output) if item.color == color]
            if not indexed:
                continue
            visible_colors.add(color)
            previous = self._tracks.get(color)
            if previous:
                selected = min(indexed, key=lambda pair: (
                    math.hypot(pair[1].reference_point_px[0] - previous[0],
                               pair[1].reference_point_px[1] - previous[1]),
                    math.hypot(pair[1].reference_point_px[0] - reference[0],
                               pair[1].reference_point_px[1] - reference[1]),
                    -pair[1].area_px,
                ))
            else:
                selected = min(indexed, key=lambda pair: (
                    math.hypot(pair[1].reference_point_px[0] - reference[0],
                               pair[1].reference_point_px[1] - reference[1]),
                    -pair[1].area_px,
                ))
            index, item = selected
            point = item.reference_point_px
            count = 1
            if previous and math.hypot(point[0] - previous[0], point[1] - previous[1]) <= tolerance:
                count = previous[2] + 1
            self._tracks[color] = (point[0], point[1], count)
            confidence = min(1.0, 0.45 * item.solidity + 0.25 * item.extent +
                             0.30 * min(1.0, count / max(required, 1)))
            output[index] = replace(item, stable=count >= required,
                                    stable_frames=count, confidence=confidence)
        self._tracks = {color: track for color, track in self._tracks.items()
                        if color in visible_colors}
        return output

    def detect(self, frame) -> list[DetectedObject]:
        cv2 = self.cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        valid_mask = self._valid_mask(frame.shape)
        minimum = float(self._config.get("min_area_px", self._config.get("min_object_area_px", 900)))
        maximum = float(self._config.get("max_area_px", float("inf")))
        min_aspect = float(self._config.get("min_aspect_ratio", 0.55))
        max_aspect = float(self._config.get("max_aspect_ratio", 1.80))
        min_solidity = float(self._config.get("min_solidity", 0.75))
        min_extent = float(self._config.get("min_extent", 0.45))
        kernel_size = max(1, int(self._config.get("morphology_kernel_px", 5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        objects: list[DetectedObject] = []
        for color in MATERIAL_COLORS:
            mask = None
            for bounds in self._hsv_colors[color]:
                h1, s1, v1, h2, s2, v2 = (int(value) for value in bounds)
                partial = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
                mask = partial if mask is None else cv2.bitwise_or(mask, partial)
            if mask is None:
                continue
            mask = cv2.bitwise_and(mask, valid_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not minimum <= area <= maximum:
                    continue
                x, y, box_width, box_height = cv2.boundingRect(contour)
                if box_width <= 0 or box_height <= 0:
                    continue
                aspect = box_width / box_height
                hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                solidity = area / hull_area if hull_area > 0 else 0.0
                extent = area / float(box_width * box_height)
                if not min_aspect <= aspect <= max_aspect:
                    continue
                if solidity < min_solidity or extent < min_extent:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                center = (moments["m10"] / moments["m00"],
                          moments["m01"] / moments["m00"])
                reference = self._reference_point(contour, (x, y, box_width, box_height), center)
                points = tuple((int(point[0][0]), int(point[0][1])) for point in contour)
                objects.append(DetectedObject(
                    color=color, center_x_px=center[0], center_y_px=center[1],
                    area_px=area, bbox=(x, y, box_width, box_height), contour=points,
                    aspect_ratio=aspect, solidity=solidity, extent=extent,
                    reference_x_px=reference[0], reference_y_px=reference[1],
                ))
        return self._stabilize(objects, frame.shape)

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
        label = (f"{item.color} ref=({item.reference_point_px[0]:.1f},"
                 f"{item.reference_point_px[1]:.1f}) A={item.area_px:.0f} "
                 f"stable={item.stable_frames} Q={item.confidence:.2f}")
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


class CameraVision:
    """物料相机采帧封装；二维码相机应创建独立实例。"""

    def __init__(self, camera_config: Mapping[str, Any], hsv_colors: Mapping[str, Any]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(int(camera_config["index"]), backend)
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

    def detect_objects(self, frame) -> list[DetectedObject]:
        return self._detector.detect(frame)

    best_target = staticmethod(ObjectDetector.best_target)

    def close(self) -> None:
        self._capture.release()
