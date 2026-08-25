"""OpenCV 二维码与 HSV 颜色物料检测。

@File    : logistics_robot/vision.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 读取相机并输出任务码和颜色物料候选。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import DetectedObject


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
        self._min_area = float(camera_config["min_object_area_px"])
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
