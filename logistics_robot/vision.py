"""OpenCV 二维码与 HSV 颜色物料检测。

@File    : logistics_robot/vision.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 读取相机并输出任务码和颜色物料候选。
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import math
from typing import Any

from .model import DetectedObject


_DISPLAY_COLORS_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 80, 0),
    "green": (0, 200, 0),
    # 黑色物料使用白色标记，避免标记本身不可见。
    "black": (255, 255, 255),
    "cyan": (255, 255, 0),
}
_DEBUG_COLOR_ORDER = ("red", "yellow", "blue", "green", "black", "cyan")


def draw_object_debug_frame(frame, objects: list[DetectedObject], fps: float,
                            camera_index: int):
    """返回物料检测调试画面，不修改输入图像或检测结果。

    ``DetectedObject`` 只保存中心点和轮廓面积，因此调试界面以等面积圆和
    外接方框近似标出物体范围；HSV 轮廓提取仍只由 :meth:`detect_objects`
    负责。
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.line(annotated, (center_x, 0), (center_x, height - 1), (180, 180, 180), 1)
    cv2.line(annotated, (0, center_y), (width - 1, center_y), (180, 180, 180), 1)
    cv2.drawMarker(annotated, (center_x, center_y), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    for item in objects:
        bgr = _DISPLAY_COLORS_BGR.get(item.color, (255, 255, 255))
        point = (round(item.center_x_px), round(item.center_y_px))
        radius = max(8, round(math.sqrt(max(item.area_px, 1.0) / math.pi)))
        cv2.circle(annotated, point, radius, bgr, 2)
        cv2.rectangle(annotated, (point[0] - radius, point[1] - radius),
                      (point[0] + radius, point[1] + radius), bgr, 1)
        cv2.drawMarker(annotated, point, bgr, markerType=cv2.MARKER_CROSS,
                       markerSize=12, thickness=2)
        label = (f"{item.color} ({item.center_x_px:.1f}, {item.center_y_px:.1f}) "
                 f"A={item.area_px:.1f}")
        text_origin = (max(0, point[0] - radius), max(18, point[1] - radius - 6))
        cv2.putText(annotated, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, bgr, 2, cv2.LINE_AA)

    counts = Counter(item.color for item in objects)
    cv2.putText(annotated, f"Camera: {camera_index}  {width}x{height}  FPS: {fps:.1f}",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Objects: {len(objects)}", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    count_text = "  ".join(f"{color}: {counts[color]}" for color in _DEBUG_COLOR_ORDER)
    cv2.putText(annotated, count_text, (16, height - 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, "q/ESC: quit  s: save  p: print results",
                (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


class CameraVision:
    """封装摄像头采帧、二维码解码与 HSV 颜色物料检测。"""

    def __init__(self, camera_config: Mapping[str, Any], hsv_colors: Mapping[str, Any]) -> None:
        """打开并配置摄像头；失败立即报错，避免盲目执行取料。"""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("未安装 OpenCV，请先执行 pip install -r requirements.txt") from exc
        self.cv2 = cv2
        self._capture = cv2.VideoCapture(int(camera_config["index"]))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
        self._capture.set(cv2.CAP_PROP_FPS, int(camera_config["fps"]))
        if not self._capture.isOpened():
            raise RuntimeError("无法打开摄像头；检查 index、USB/CSI 接线及权限")
        # 二维码相机未来也会复用本类，但它不需要颜色检测面积阈值。
        self._min_area = float(camera_config.get("min_object_area_px", 0))
        self._hsv_colors = hsv_colors
        self._qr = cv2.QRCodeDetector()

    def read(self):
        """读取一帧 BGR 图像；设备断流时抛出异常供上层安全处理。"""
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("摄像头读帧失败")
        return frame

    def decode_task_code(self, frame) -> str | None:
        """优先使用多二维码 API，兼容不具备该 API 的 OpenCV 版本。"""
        try:
            result = self._qr.detectAndDecodeMulti(frame)
            if len(result) == 4:
                ok, decoded, _, _ = result
                if ok:
                    for text in decoded:
                        if text:
                            return text.strip()
        except (AttributeError, self.cv2.error):
            pass
        text, _, _ = self._qr.detectAndDecode(frame)
        return text.strip() if text else None

    def detect_objects(self, frame) -> list[DetectedObject]:
        """按 HSV 阈值分割物料，并输出满足最小面积的外轮廓中心。"""
        cv2 = self.cv2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        objects: list[DetectedObject] = []
        for color, ranges in self._hsv_colors.items():
            mask = None
            for bounds in ranges:
                h1, s1, v1, h2, s2, v2 = (int(value) for value in bounds)
                partial = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
                mask = partial if mask is None else cv2.bitwise_or(mask, partial)
            assert mask is not None
            # 开运算先腐蚀再膨胀，用于滤掉高光和传感器噪点形成的小斑块。
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self._min_area:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                objects.append(
                    DetectedObject(
                        color=color,
                        center_x_px=moments["m10"] / moments["m00"],
                        center_y_px=moments["m01"] / moments["m00"],
                        area_px=area,
                    )
                )
        return objects

    @staticmethod
    def best_target(objects: list[DetectedObject], color: str, image_center_x_px: float) -> DetectedObject | None:
        """优先选取面积大且最靠近取料中心线的指定颜色物料。"""
        candidates = [item for item in objects if item.color == color]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (abs(item.center_x_px - image_center_x_px), -item.area_px))

    def close(self) -> None:
        """释放摄像头设备句柄。"""
        self._capture.release()
