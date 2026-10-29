"""按指定颜色识别圆环，并把圆心发给下位机。"""

from __future__ import annotations

import logging
from typing import Any

from .calibration import FrameRectifier
from .circle_select import CircleTracker, recognize_frame
from .model import DetectedObject
from .ring_detection import RingCandidate
from .vision import MATERIAL_COLORS, CameraVision, draw_shape_segments, image_origin_px


LOG = logging.getLogger(__name__)

_COLOR_ALIASES = {
    "red": "red", "1": "red", "红": "red",
    "yellow": "yellow", "2": "yellow", "黄": "yellow",
    "blue": "blue", "3": "blue", "蓝": "blue",
    "green": "green", "4": "green", "绿": "green",
}


def parse_locate_color(text: str) -> str:
    """解析打开视觉定位时要识别的颜色。"""
    color = _COLOR_ALIASES.get(str(text).strip().lower())
    if color not in MATERIAL_COLORS:
        raise ValueError("视觉定位颜色只能是 red/yellow/blue/green，或 1/2/3/4")
    return color


def retain_hsv_color(frame, color: str, hsv_colors: dict[str, Any], cv2):
    """只保留指定颜色的像素，其余清成黑色。"""
    if color not in hsv_colors:
        raise ValueError(f"未配置 {color} 的 HSV")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for bounds in hsv_colors[color]:
        h1, s1, v1, h2, s2, v2 = (int(value) for value in bounds)
        partial = cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
        mask = partial if mask is None else cv2.bitwise_or(mask, partial)
    if mask is None:
        raise ValueError(f"{color} 的 HSV 范围为空")
    return cv2.bitwise_and(frame, frame, mask=mask)


def select_upper_circle(objects: list[DetectedObject]) -> DetectedObject | None:
    """在已稳定的圆形里选画面最上层的一个。图像 y 越小越靠上。"""
    stable = [item for item in objects if item.stable and item.top_valid]
    if not stable:
        return None
    return min(stable, key=lambda item: (item.circle_center_px[1], item.circle_center_px[0]))


def upper_circle_coordinate(objects: list[DetectedObject], width: int,
                            height: int) -> tuple[int, int] | None:
    """上层稳定圆的画面中心坐标。没有稳定圆时不发。"""
    target = select_upper_circle(objects)
    if target is None:
        return None
    return image_origin_px(target.circle_center_px[0], target.circle_center_px[1], width, height)


def near_image_origin(point: tuple[int, int], tolerance_px: float = 40) -> bool:
    """圆心到画面中心的距离不超过容差时视为到位。容差单位是像素。"""
    limit = float(tolerance_px)
    if limit < 0 or limit != limit:
        raise ValueError("到位容差无效")
    x_px, y_px = int(point[0]), int(point[1])
    return x_px * x_px + y_px * y_px <= limit * limit


def locate_coordinate(rings: list[RingCandidate], width: int,
                      height: int) -> tuple[int, int] | None:
    """唯一稳定圆环才返回画面中心坐标。没有或不止一个时不发。"""
    stable = [item for item in rings if item.stable]
    if len(stable) != 1:
        return None
    item = stable[0]
    return image_origin_px(item.center_x_px, item.center_y_px, width, height)


def run_color_ring_locate(config: dict[str, Any], color: str,
                          camera_index: int | None, client) -> int:
    """打开视觉定位，分割指定颜色的圆形，并发送最上层那个的圆心。"""
    camera_config = dict(config["object_camera"])
    if camera_index is not None:
        camera_config["index"] = camera_index
    try:
        vision = CameraVision(camera_config, config["hsv_colors"])
    except RuntimeError as exc:
        LOG.error("无法打开物料识别相机：%s", exc)
        return 1
    rectifier = FrameRectifier(config["vision_calibration"].get("camera", {}), vision.cv2)
    tracker = CircleTracker(
        stable_frames=int(config["object_camera"]["object_detection"].get("stable_frames", 4)),
        tolerance_px=float(config["object_camera"]["object_detection"].get("stable_center_tolerance_px", 8)),
    )
    objects: list[DetectedObject] = []
    started = False
    window = True
    try:
        client.start_visual_positioning()
        started = True
        LOG.info("视觉定位已打开，分割并发送最上层的 %s 圆形", color)
        while True:
            frame = rectifier.apply(vision.read())
            objects = recognize_frame(
                frame, (color,), config["hsv_colors"],
                config["object_camera"]["object_detection"], tracker)
            target = select_upper_circle(objects)
            height, width = frame.shape[:2]
            point = upper_circle_coordinate(objects, width, height)
            tolerance = float(config["object_camera"]["object_detection"].get("align_tolerance_px", 40))
            if point is None:
                LOG.info("未锁定稳定的 %s 圆形，暂不发送坐标", color)
            elif near_image_origin(point, tolerance):
                LOG.info("%s 圆形圆心已进入画面中心 %.0f 像素内，结束视觉定位",
                         color, tolerance)
                client.stop_visual_positioning()
                started = False
                return 0
            else:
                client.send_visual_coordinate(point[0], point[1])
                LOG.info("发送上层 %s 圆形圆心 (%d, %d)", color, point[0], point[1])
            if window:
                try:
                    annotated = draw_shape_segments(frame, objects, target)
                    vision.cv2.imshow(f"Locate {color} circle", annotated)
                    key = vision.cv2.waitKey(1) & 0xFF
                except vision.cv2.error:
                    window = False
                    key = 255
                if key in (ord("q"), 27):
                    LOG.info("操作员退出视觉定位")
                    return 0
    except KeyboardInterrupt:
        LOG.info("操作员退出视觉定位")
        return 0
    finally:
        if started:
            try:
                client.stop_visual_positioning()
            except Exception as exc:
                LOG.error("停止视觉定位失败：%s", exc)
        vision.close()
        if window:
            try:
                vision.cv2.destroyAllWindows()
            except vision.cv2.error:
                pass
    return 0
