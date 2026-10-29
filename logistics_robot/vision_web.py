"""把处理后的物料相机画面转发到浏览器。"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import copy
import io
import zipfile
import math
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .calibration import FrameRectifier
from .direct_stream import FrameSlot, StreamBudget, normalize_stream_limits, stream_limits_from
from .material_learn import (
    crop_patch, hsv_bounds_from_bgr, hsv_bounds_from_rgb, rgb_sample_from_bounds, save_config,
)
from .model import DetectedObject
from .ring_detection import RingCandidate, RingDetector, draw_ring_debug
from .vision import MATERIAL_COLORS, CameraVision, draw_object_debug_frame, draw_shape_segments, image_origin_px
from .vision_locate import (
    near_image_origin, parse_locate_color, select_upper_circle, upper_circle_coordinate,
)


LOG = logging.getLogger(__name__)
_HTML_PATH = Path(__file__).resolve().parent / "web" / "index.html"
COLOR_VALUE_FLOOR = {"red": 165, "yellow": 145, "blue": 125, "green": 115}


class FrameHub:
    """保存最新一帧 JPEG 和坐标，供多个浏览器连接读取。"""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._jpeg = b""
        self._status: dict[str, Any] = {
            "ok": False, "error": "等待相机画面", "fps": 0.0,
            "width": 0, "height": 0, "camera_index": 0, "mode": "combined",
            "objects": [], "rings": [], "locate_color": "",
        }
        self._generation = 0
        self._locate_color = ""
        self._bgr = None
        self._reset_tracks = False
        self.closed = False
        self.view = "result"
        self.send_enabled = False
        self.revision = 0
        self.capture_frames = None
        self.stream_fps = 12.0
        self.stream_peak_kib = 512.0
        self.stream_quality = 70
        self.preview_width = 960
        self._viewers = 0


    def publish(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._status = status
            self._generation += 1
            self._lock.notify_all()

    def set_locate_color(self, color: str) -> None:
        with self._lock:
            self._locate_color = color
            self.send_enabled = False
            self._reset_tracks = True
            self.revision += 1

    def get_locate_color(self) -> str:
        with self._lock:
            return self._locate_color

    def set_stream_limits(self, fps, peak_kib_per_sec, quality=70) -> None:
        fps_value, peak_value, quality_value = normalize_stream_limits(
            fps, peak_kib_per_sec, quality)
        with self._lock:
            self.stream_fps = fps_value
            self.stream_peak_kib = peak_value
            self.stream_quality = quality_value

    def stream_limits(self) -> tuple[float, float, int]:
        with self._lock:
            return self.stream_fps, self.stream_peak_kib, self.stream_quality

    def set_preview_width(self, width: int) -> None:
        value = int(width)
        if not 320 <= value <= 1920:
            raise ValueError("预览宽度必须在 320..1920")
        with self._lock:
            self.preview_width = value

    def preview_width_px(self) -> int:
        with self._lock:
            return int(self.preview_width)

    def add_viewer(self) -> int:
        with self._lock:
            self._viewers += 1
            return self._viewers

    def remove_viewer(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    def viewers(self) -> int:
        with self._lock:
            return self._viewers

    def publish_video(self, jpeg: bytes, info: dict[str, Any]) -> None:
        """只更新原画 JPEG。不改识别结果。"""
        with self._lock:
            self._jpeg = jpeg
            tune = dict(self._status.get("tune") or {})
            tune.update(info.get("tune") or {})
            self._status.update(info)
            self._status["tune"] = tune
            self._generation += 1
            self._lock.notify_all()

    def publish_recognition(self, info: dict[str, Any]) -> None:
        """只更新识别结果，不推动视频流，避免两条线互相等待。"""
        with self._lock:
            tune = dict(self._status.get("tune") or {})
            tune.update(info.get("tune") or {})
            self._status.update(info)
            self._status["tune"] = tune

    def latest(self) -> tuple[bytes, dict[str, Any]]:
        with self._lock:
            status = dict(self._status)
            status["locate_color"] = self._locate_color
            status["send_enabled"] = self.send_enabled
            return self._jpeg, status

    def wait_newer(self, generation: int, timeout: float) -> tuple[int, bytes] | None:
        """等到比 ``generation`` 更新的帧；关闭时返回 ``None``。"""
        with self._lock:
            if self._generation <= generation and not self.closed:
                self._lock.wait(timeout)
            if self.closed:
                return None
            return self._generation, self._jpeg

    def remember_frame(self, frame) -> None:
        with self._lock:
            self._bgr = frame

    def consume_track_reset(self) -> bool:
        with self._lock:
            reset = self._reset_tracks
            self._reset_tracks = False
            return reset

    def learn_color(self, color: str, x: float, y: float, config: dict[str, Any],
                    config_path: Path | None) -> list[list[int]]:
        """按点击位置学习颜色，并写回正在使用的配置。"""
        with self._lock:
            frame = self._bgr
            if frame is None:
                raise RuntimeError("画面尚未就绪")
            patch = crop_patch(frame, x, y)
        bounds = hsv_bounds_from_bgr(patch)
        with self._lock:
            config["hsv_colors"][color] = bounds
            self._reset_tracks = True
            self.send_enabled = False
            self.revision += 1
            if config_path is not None:
                save_config(config_path, config)
        return bounds

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._lock.notify_all()


def vision_status(frame_shape, fps: float, camera_index: int, mode: str,
                  objects: list[DetectedObject], rings: list[RingCandidate],
                  error: str = "", locate_color: str = "",
                  target: DetectedObject | None = None,
                  send_state: str = "idle",
                  tune: dict[str, Any] | None = None,
                  motion: str = "update") -> dict[str, Any]:
    """整理网页侧栏使用的画面中心坐标。"""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    object_rows = []
    for item in objects:
        x_px, y_px = image_origin_px(item.circle_center_px[0], item.circle_center_px[1],
                                     width, height)
        object_rows.append({
            "color": item.color, "x": x_px, "y": y_px,
            "track_id": item.track_id,
            "stable": bool(item.stable), "stable_frames": int(item.stable_frames),
            "quality": round(float(item.confidence), 3),
            "material_center": [item.center_x_px, item.center_y_px],
            "top_center": list(item.top_ellipse[0]) if item.top_ellipse else None,
            "top_valid": item.top_valid, "reason": item.top_reason,
            "metrics": dict(item.top_metrics),
        })
    ring_rows = []
    for item in rings:
        x_px, y_px = image_origin_px(item.center_x_px, item.center_y_px, width, height)
        ring_rows.append({
            "position": item.position, "track_id": item.track_id, "x": x_px, "y": y_px,
            "radius_px": round(float(item.radius_px), 2),
            "valid": bool(item.valid), "stable": bool(item.stable),
            "stable_frames": int(item.stable_frames),
            "quality": round(float(item.quality), 3),
            "reason": item.reason,
        })
    return {
        "ok": not error, "error": error, "fps": round(float(fps), 2),
        "width": width, "height": height, "camera_index": int(camera_index),
        "mode": mode, "locate_color": locate_color, "motion": motion,
        "objects": object_rows, "rings": ring_rows,
        "target": _target_row(target, width, height),
        "send_state": send_state,
        "recognition_state": ("未发现物料" if not objects else
                              "可发送" if any(o.top_valid and o.stable for o in objects) else
                              "顶面已确认但未稳定" if any(o.top_valid for o in objects) else
                              "已找到物料但顶面未确认"),
        "tune": tune or {},
    }


def _target_row(target: DetectedObject | None, width: int, height: int) -> dict[str, Any] | None:
    if target is None:
        return None
    x_px, y_px = image_origin_px(target.circle_center_px[0], target.circle_center_px[1],
                                 width, height)
    return {
        "color": target.color, "track_id": target.track_id, "x": x_px, "y": y_px,
        "stable_frames": int(target.stable_frames),
    }


class _CircleSender:
    """选中颜色后，把上层稳定圆的圆心发给下位机。没选颜色时不打开串口。"""

    def __init__(self, serial_config: dict[str, Any]) -> None:
        self._config = dict(serial_config)
        self._link = None
        self._started = False
        self._retry_after = 0.0
        self.error = ""
        self.state = "idle"

    def update(self, color: str, point: tuple[int, int] | None,
               tolerance_px: float = 40) -> str:
        if not color:
            self.stop()
            self.state = "idle"
            return self.state
        if point is None:
            self.state = "waiting"
            return self.state
        if near_image_origin(point, tolerance_px):
            self.stop()
            self.state = "aligned"
            return self.state
        if not self._ensure_started():
            self.state = "error"
            return self.state
        try:
            self._link.path_client.send_visual_coordinate(point[0], point[1])
            self.error = ""
            self.state = "sending"
        except Exception as exc:
            self.error = str(exc)
            self._started = False
            self._retry_after = time.monotonic() + 2.0
            self.state = "error"
        return self.state

    def _ensure_started(self) -> bool:
        if self._started and self._link is not None:
            return True
        now = time.monotonic()
        if now < self._retry_after:
            return False
        try:
            if self._link is None:
                from .serial_link import SerialPathLink
                self._link = SerialPathLink(**self._config)
            self._link.path_client.start_visual_positioning()
            self._started = True
            self.error = ""
            return True
        except Exception as exc:
            self.error = str(exc)
            self._started = False
            self._retry_after = time.monotonic() + 2.0
            return False

    def stop(self) -> None:
        if self._started and self._link is not None:
            try:
                self._link.path_client.stop_visual_positioning()
            except Exception as exc:
                self.error = str(exc)
        self._started = False

    def close(self) -> None:
        self.stop()
        if self._link is not None:
            try:
                self._link.close()
            except Exception:
                pass
            self._link = None


def annotate_vision_frame(frame, objects: list[DetectedObject], rings: list[RingCandidate],
                          fps: float, camera_index: int, camera_config, mode: str):
    """叠加物料、圆环和画面中心坐标。返回新图像，不改原始帧。"""
    annotated = draw_object_debug_frame(
        frame, [] if mode == "rings" else objects, fps, camera_index, camera_config)
    if mode != "objects":
        annotated = draw_ring_debug(annotated, rings)
    return annotated


def preview_frame(frame, preview_width: int):
    """网页预览可以比检测图更窄，坐标仍按原始画面计算。"""
    import cv2
    height, width = frame.shape[:2]
    if preview_width <= 0 or width <= preview_width:
        return frame
    scale = preview_width / float(width)
    size = (preview_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def render_preview(frame, objects, rings, fps, camera_index, camera_config, mode,
                   preview_width, jpeg_quality, target) -> bytes:
    """每帧都按当前画面重画预览，避免网页停在旧图上。"""
    segmented = draw_shape_segments(frame, objects, target)
    annotated = annotate_vision_frame(segmented, objects, rings, fps, camera_index,
                                      camera_config, mode)
    return encode_jpeg(preview_frame(annotated, preview_width), jpeg_quality)


def black_white_mask(frame, hsv_colors, color: str = ""):
    """色域命中的像素为白，其余为黑。"""
    import cv2
    import numpy as np
    from .circle_select import color_mask
    names = (color,) if color and color in (hsv_colors or {}) else tuple(hsv_colors or {})
    mask = None
    for name in names:
        bounds = (hsv_colors or {}).get(name) or []
        if not bounds:
            continue
        partial = color_mask(frame, bounds, 3)
        mask = partial if mask is None else cv2.bitwise_or(mask, partial)
    if mask is None:
        mask = np.zeros(frame.shape[:2], np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def encode_jpeg(frame, quality: int = 80) -> bytes:
    import cv2
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return encoded.tobytes()


def _error_frame(message: str):
    import cv2
    import numpy as np
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(image, "CAMERA UNAVAILABLE", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 180, 255), 2, cv2.LINE_AA)
    y = 150
    for start in range(0, min(len(message), 240), 48):
        cv2.putText(image, message[start:start + 48], (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        y += 32
    return image


class VisionWebServer:
    """页面、状态、单帧、MJPEG，以及点击学习物料颜色。"""

    def __init__(self, hub: FrameHub, host: str, port: int,
                 config: dict[str, Any] | None = None,
                 config_path: Path | None = None) -> None:
        html = _HTML_PATH.read_bytes()
        handler = _handler(hub, html, config, config_path)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._httpd.daemon_threads = True
        self.host = host
        self.port = int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        shown = "127.0.0.1" if self.host in {"0.0.0.0", "::", ""} else self.host
        return f"http://{shown}:{self.port}/"

    def serve_forever(self) -> None:
        self._httpd.serve_forever(poll_interval=0.2)

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def _handler(hub: FrameHub, html: bytes, config: dict[str, Any] | None,
             config_path: Path | None):
    from .top_face import DEFAULTS
    setting_defaults = {**DEFAULTS, "min_area_px": 900, "max_area_px": 250000,
                        "min_circularity": 0.55, "stable_frames": 4,
                        "stable_center_tolerance_px": 8, "morphology_kernel_px": 5,
                        "align_tolerance_px": 40}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, "text/html; charset=utf-8", html)
            elif path == "/status.json":
                _, status = hub.latest()
                body = json.dumps(status, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/settings":
                with hub._lock:
                    current = {**setting_defaults, **(config or {}).get("object_camera", {}).get("object_detection", {})}
                self._send(200, "application/json", json.dumps({k: current[k] for k in setting_defaults}).encode())
            elif path == "/rgb":
                colors = {}
                with hub._lock:
                    hsv = dict((config or {}).get("hsv_colors") or {})
                for name, bounds in hsv.items():
                    red, green, blue = rgb_sample_from_bounds(bounds)
                    colors[name] = {
                        "r": red, "g": green, "b": blue, "tolerance": 48, "bounds": bounds,
                    }
                body = json.dumps({"ok": True, "colors": colors}, ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/stream-settings":
                fps, peak, quality = hub.stream_limits()
                body = json.dumps({
                    "fps": fps, "peak_kib_per_sec": peak, "jpeg_quality": quality,
                }).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body)
            elif path == "/capture.zip":
                with hub._lock:
                    capture = hub.capture_frames
                if capture is None:
                    self.send_error(503, "No frame")
                    return
                import cv2
                raw_frame, frame, mask, edges, diagnostic, status, settings = capture
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                    for name, image in (("camera-original.png", raw_frame), ("detection-original.png", frame),
                                        ("segmentation.png", mask), ("edges.png", edges), ("diagnostic.png", diagnostic)):
                        ok, encoded = cv2.imencode(".png", image)
                        if not ok:
                            raise RuntimeError("PNG encoding failed")
                        archive.writestr(name, encoded.tobytes())
                    archive.writestr("status.json", json.dumps(status, ensure_ascii=False, indent=2))
                    archive.writestr("config.json", json.dumps(settings, ensure_ascii=False, indent=2))
                self._send(200, "application/zip", output.getvalue())
            elif path == "/snapshot.jpg":
                jpeg, _ = hub.latest()
                if not jpeg:
                    self.send_error(503, "画面尚未就绪")
                    return
                self._send(200, "image/jpeg", jpeg)
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeError):
                body = json.dumps({"ok": False, "error": "请求无效"}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            if not isinstance(payload, dict):
                self._send(400, "application/json", b'{"ok":false}')
                return
            if path == "/controls":
                try:
                    with hub._lock:
                        if "view" in payload:
                            if payload["view"] not in ("raw", "mask", "edges", "result"):
                                raise ValueError("视图无效")
                            hub.view = payload["view"]
                        if "send_enabled" in payload:
                            if not isinstance(payload["send_enabled"], bool):
                                raise ValueError("发送开关无效")
                            hub.send_enabled = payload["send_enabled"] and bool(hub._locate_color)
                            hub.revision += 1
                    self._send(200, "application/json", b'{"ok":true}')
                except ValueError as exc:
                    self._send(400, "application/json", json.dumps({"error": str(exc)}).encode())
            elif path == "/stream":
                self._set_stream(payload)
            elif path == "/settings":
                try:
                    if config is None:
                        raise ValueError("无配置")
                    with hub._lock:
                        current = dict(config["object_camera"]["object_detection"])
                        updates = setting_defaults if payload.get("reset") else payload.get("values", {})
                        for key, value in updates.items():
                            if key not in setting_defaults or isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                                raise ValueError("参数无效")
                            current[key] = value
                        for key in ("top_min_axis_ratio", "top_min_area_ratio", "top_min_edge_support", "top_min_arc_coverage", "min_circularity"):
                            if not 0 < current.get(key, setting_defaults[key]) <= 1:
                                raise ValueError(key + " 必须在 (0,1]")
                        for key in setting_defaults:
                            if current.get(key, setting_defaults[key]) <= 0:
                                raise ValueError(key + " 必须大于 0")
                        for key in ("top_canny_low", "top_canny_high", "top_border_margin_px", "top_fit_trials", "stable_frames", "morphology_kernel_px", "align_tolerance_px"):
                            if int(current.get(key, setting_defaults[key])) != current.get(key, setting_defaults[key]):
                                raise ValueError(key + " 必须为整数")
                        if not (current.get("top_canny_low",45) < current.get("top_canny_high",120) <= 255
                                and current["min_area_px"] < current["max_area_px"]
                                and current.get("top_min_area_ratio",.12) < current.get("top_max_area_ratio",1.15) <= 2
                                and current.get("top_fit_trials",64) <= 256
                                and current.get("morphology_kernel_px",5) <= 31
                                and current.get("top_edge_tolerance_px",2.5) <= 10
                                and 1 <= current.get("align_tolerance_px", 40) <= 200):
                            raise ValueError("参数范围或上下限无效")
                        updated = copy.deepcopy(config)
                        updated["object_camera"]["object_detection"] = current
                        if payload.get("save"):
                            if config_path is None:
                                raise ValueError("未指定配置文件路径")
                            save_config(config_path, updated)
                        config["object_camera"]["object_detection"] = current
                        hub.send_enabled = False
                        hub._reset_tracks = True
                        hub.revision += 1
                    self._send(200, "application/json", b'{"ok":true}')
                except (ValueError, TypeError, OSError) as exc:
                    self._send(400, "application/json", json.dumps({"error": str(exc)}, ensure_ascii=False).encode())
            elif path == "/color":
                self._set_color(payload)
            elif path == "/learn":
                self._learn(payload)
            elif path == "/rgb":
                self._set_rgb(payload)
            else:
                self.send_error(404)

        def _set_stream(self, payload: dict[str, Any]) -> None:
            try:
                current_fps, current_peak, current_quality = hub.stream_limits()
                fps, peak, quality = normalize_stream_limits(
                    payload.get("fps", current_fps),
                    payload.get("peak_kib_per_sec", current_peak),
                    payload.get("jpeg_quality", current_quality),
                )
                hub.set_stream_limits(fps, peak, quality)
                if config is not None:
                    updated = copy.deepcopy(config)
                    previous = dict(updated.get("vision_stream") or {})
                    updated["vision_stream"] = {
                        "fps": fps, "peak_kib_per_sec": peak, "jpeg_quality": quality,
                        "preview_width": int(previous.get("preview_width", hub.preview_width_px())),
                    }
                    if payload.get("save"):
                        if config_path is None:
                            raise ValueError("未指定配置文件路径")
                        save_config(config_path, updated)
                    config["vision_stream"] = updated["vision_stream"]
            except (TypeError, ValueError, OSError) as exc:
                body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            self._send(200, "application/json; charset=utf-8", b'{"ok":true}')

        def _set_color(self, payload: dict[str, Any]) -> None:
            try:
                requested = str(payload.get("color", "")).strip()
                color = parse_locate_color(requested) if requested else ""
            except ValueError:
                body = json.dumps({"ok": False, "error": "颜色无效"}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            hub.set_locate_color(color)
            body = json.dumps({"ok": True, "color": color}, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        def _learn(self, payload: dict[str, Any]) -> None:
            try:
                color = parse_locate_color(str(payload.get("color", "")))
                if config is None or "hsv_colors" not in config:
                    raise RuntimeError("没有可写入的颜色配置")
                bounds = hub.learn_color(color, float(payload["x"]), float(payload["y"]),
                                         config, config_path)
            except (KeyError, TypeError, ValueError) as exc:
                message = str(exc) or "采样点无效"
                body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            except RuntimeError as exc:
                body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(409, "application/json; charset=utf-8", body)
                return
            body = json.dumps({"ok": True, "color": color, "bounds": bounds},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        def _set_rgb(self, payload: dict[str, Any]) -> None:
            try:
                color = parse_locate_color(str(payload.get("color", "")))
                if config is None or "hsv_colors" not in config:
                    raise RuntimeError("没有可写入的颜色配置")
                red = _rgb_channel(payload.get("r"))
                green = _rgb_channel(payload.get("g"))
                blue = _rgb_channel(payload.get("b"))
                tolerance = int(payload.get("tolerance", 48))
                if not 8 <= tolerance <= 120:
                    raise ValueError("容差必须在 8..120")
                bounds = hsv_bounds_from_rgb(
                    red, green, blue, tolerance, COLOR_VALUE_FLOOR.get(color, 0))
                with hub._lock:
                    config["hsv_colors"][color] = bounds
                    hub.view = "mask"
                    hub._locate_color = color
                    hub.send_enabled = False
                    hub._reset_tracks = True
                    hub.revision += 1
                    if payload.get("save"):
                        if config_path is None:
                            raise ValueError("未指定配置文件路径")
                        save_config(config_path, config)
            except (TypeError, ValueError, OSError) as exc:
                message = str(exc) or "颜色无效"
                body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            except RuntimeError as exc:
                body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(409, "application/json; charset=utf-8", body)
                return
            body = json.dumps({
                "ok": True, "color": color, "r": red, "g": green, "b": blue,
                "tolerance": tolerance, "bounds": bounds, "view": "mask",
            }, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            hub.add_viewer()
            generation = 0
            try:
                while True:
                    item = hub.wait_newer(generation, 0.5)
                    if item is None:
                        break
                    generation, jpeg = item
                    if not jpeg:
                        continue
                    chunk = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                             b"Content-Length: " + str(len(jpeg)).encode("ascii")
                             + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            finally:
                hub.remove_viewer()

        def log_message(self, fmt: str, *args) -> None:
            LOG.debug("视觉网页 %s", fmt % args)

    return Handler



def _recognize_loop(slot: FrameSlot, hub: FrameHub, config: dict[str, Any], mode: str,
                    sender: _CircleSender, stop: threading.Event, cv2_module,
                    camera_index: int) -> None:
    """原画识别。只读最新帧，不等待网页编码。"""
    cv2_module.setNumThreads(1)
    from .circle_select import CircleTracker, recognize_frame
    tracker = CircleTracker()
    ring_detector = RingDetector(config["ring_detection"], cv2_module) if mode != "objects" else None
    seen = 0
    ring_tick = 0
    cached_rings: list[RingCandidate] = []
    previous = time.perf_counter()
    while not stop.is_set():
        item = slot.wait_next(seen, 0.2)
        if item is None:
            break
        seen, frame = item
        if frame is None:
            continue
        try:
            with hub._lock:
                color = hub._locate_color
                revision = hub.revision
                reset = hub._reset_tracks
                hub._reset_tracks = False
                hsv = copy.deepcopy(config.get("hsv_colors", {}))
                detection = dict(config["object_camera"]["object_detection"])
                enabled = hub.send_enabled and revision == hub.revision
            if reset:
                tracker.clear()
                if mode != "objects":
                    ring_detector = RingDetector(config["ring_detection"], cv2_module)
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            recognition_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            colors = (color,) if color else MATERIAL_COLORS
            objects = recognize_frame(frame, colors, hsv, detection, tracker) if mode != "rings" else []
            rings = cached_rings
            if ring_detector is not None and (reset or ring_tick % 3 == 0):
                rings = ring_detector.detect(frame, None, color or None, hsv)
                cached_rings = rings
            ring_tick += 1
            target = select_upper_circle(objects) if color else None
            point = upper_circle_coordinate(objects, frame.shape[1], frame.shape[0]) if color else None
            tolerance = float(detection.get("align_tolerance_px", 40))
            send_state = sender.update(color if enabled else "", point, tolerance)
            if not enabled:
                send_state = "preview"
            fps_limit, peak_limit, quality_limit = hub.stream_limits()
            status = vision_status(frame.shape, recognition_fps, camera_index, mode, objects, rings,
                                   sender.error if send_state == "error" else "",
                                   color, target, send_state, {}, "recognize")
            status.pop("fps", None)
            status["tune"] = {
                "recognition_fps": round(recognition_fps, 2),
                "target_fps": fps_limit,
                "peak_kib_per_sec": peak_limit,
                "jpeg_quality": quality_limit,
            }
            status["actual_send_target"] = status["target"] if send_state == "sending" else None
            hub.publish_recognition(status)
        except Exception:
            LOG.exception("视觉识别中断，传输继续")


def _stream_loop(slot: FrameSlot, hub: FrameHub, config: dict[str, Any],
                 stop: threading.Event) -> None:
    """有人观看才编码一帧预览。没人看就丢弃，不占用识别。"""
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    fps, peak, quality = hub.stream_limits()
    budget = StreamBudget(fps, peak, quality)
    seen = 0
    while not stop.is_set():
        if hub.viewers() <= 0:
            if stop.wait(0.1):
                break
            continue
        fps, peak, quality = hub.stream_limits()
        budget.configure(fps, peak, quality)
        if not budget.wait_turn(stop):
            break
        item = slot.wait_next(seen, 0.2)
        if item is None:
            break
        seen, frame = item
        if frame is None:
            continue
        try:
            with hub._lock:
                view = hub.view
                color = hub._locate_color
                hsv = copy.deepcopy(config.get("hsv_colors", {})) if view == "mask" else None
            image = preview_frame(frame, hub.preview_width_px())
            if view == "mask":
                image = black_white_mask(image, hsv, color)
            jpeg = encode_jpeg(image, budget.encode_quality())
        except Exception:
            LOG.exception("预览编码失败")
            continue
        if not budget.accept(len(jpeg)):
            continue
        hub.publish_video(jpeg, {
            "ok": True,
            "fps": round(budget.measured_fps, 2),
            "tune": {
                "target_fps": budget.fps,
                "peak_kib_per_sec": budget.peak_kib,
                "jpeg_quality": budget.encode_quality(),
                "bitrate_kib_s": round(budget.measured_kib, 1),
                "dropped": budget.dropped,
                "preview_width": int(image.shape[1]),
                "viewers": hub.viewers(),
            },
        })


def _camera_loop(config: dict[str, Any], camera_index: int | None, mode: str,
                 hub: FrameHub, stop: threading.Event) -> None:
    camera_config = dict(config["object_camera"])
    if camera_index is not None:
        camera_config["index"] = camera_index
    selected_index = int(camera_config["index"])
    vision = None
    slot = FrameSlot()
    workers: list[threading.Thread] = []
    sender = _CircleSender(config.get("serial", {}))
    try:
        vision = CameraVision(camera_config, config["hsv_colors"])
        try:
            hub.set_stream_limits(*stream_limits_from(config))
            hub.set_preview_width(int((config.get("vision_stream") or {}).get("preview_width", 960)))
        except ValueError:
            hub.set_stream_limits(12, 512, 70)
            hub.set_preview_width(960)
        vision.cv2.setNumThreads(1)
        rectifier = FrameRectifier(config["vision_calibration"].get("camera", {}), vision.cv2)
        workers = [
            threading.Thread(
                target=_recognize_loop, name="vision-recognize", daemon=True,
                args=(slot, hub, config, mode, sender, stop, vision.cv2, selected_index),
            ),
            threading.Thread(
                target=_stream_loop, name="vision-stream", daemon=True,
                args=(slot, hub, config, stop),
            ),
        ]
        for worker in workers:
            worker.start()
        while not stop.is_set():
            raw_frame = vision.read()
            frame = rectifier.apply(raw_frame)
            slot.publish(frame.copy())
    except Exception as exc:
        LOG.error("视觉网页相机中断：%s", exc)
        try:
            image = _error_frame(str(exc))
            status = vision_status(image.shape, 0.0, selected_index, mode, [], [], str(exc),
                                   hub.get_locate_color())
            hub.publish(encode_jpeg(image), status)
        except Exception:
            LOG.exception("视觉网页无法生成错误画面")
        sender.stop()
        stop.wait()
    finally:
        stop.set()
        slot.close()
        for worker in workers:
            worker.join(timeout=2.0)
        sender.close()
        if vision is not None:
            vision.close()



def _rgb_channel(value) -> int:
    number = int(value)
    if not 0 <= number <= 255:
        raise ValueError("RGB 必须在 0..255")
    return number


def run_vision_web(config: dict[str, Any], camera_index: int | None, mode: str,
                   host: str, port: int, config_path: Path | None = None) -> int:
    """启动预览；显式打开网页发送开关后才发送已验证且稳定的顶面中心。"""
    if not 1 <= int(port) <= 65535:
        LOG.error("网页端口必须在 1..65535")
        return 2
    hub = FrameHub()
    try:
        server = VisionWebServer(hub, host, int(port), config, config_path)
    except OSError as exc:
        LOG.error("无法监听 %s:%s：%s", host, port, exc)
        return 1
    stop = threading.Event()
    worker = threading.Thread(target=_camera_loop, name="vision-web-camera",
                              args=(config, camera_index, mode, hub, stop), daemon=True)
    worker.start()
    LOG.info("视觉网页已启动：%s  局域网设备使用树莓派 IP 与端口 %s", server.url, server.port)
    try:
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        LOG.info("视觉网页已停止")
        return 0
    finally:
        stop.set()
        hub.close()
        server.shutdown()
        worker.join(timeout=2.0)
