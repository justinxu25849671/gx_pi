"""把处理后的物料相机画面转发到浏览器。"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .calibration import FrameRectifier
from .material_learn import crop_patch, hsv_bounds_from_bgr, save_config
from .model import DetectedObject
from .motion_gate import MotionGate
from .ring_detection import RingCandidate, RingDetector, draw_ring_debug
from .stream_tuner import StreamTuner
from .vision import CameraVision, draw_object_debug_frame, draw_shape_segments, image_origin_px
from .vision_locate import parse_locate_color, select_upper_circle, upper_circle_coordinate


LOG = logging.getLogger(__name__)
_HTML_PATH = Path(__file__).resolve().parent / "web" / "index.html"


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

    def publish(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._status = status
            self._generation += 1
            self._lock.notify_all()

    def set_locate_color(self, color: str) -> None:
        with self._lock:
            self._locate_color = color

    def get_locate_color(self) -> str:
        with self._lock:
            return self._locate_color

    def latest(self) -> tuple[bytes, dict[str, Any]]:
        with self._lock:
            status = dict(self._status)
            status["locate_color"] = self._locate_color
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
        config["hsv_colors"][color] = bounds
        with self._lock:
            self._reset_tracks = True
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

    def update(self, color: str, point: tuple[int, int] | None) -> str:
        if not color:
            self.stop()
            self.state = "idle"
            return self.state
        if point is None:
            self.state = "waiting"
            return self.state
        if point == (0, 0):
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
            if path == "/color":
                self._set_color(payload)
            elif path == "/learn":
                self._learn(payload)
            else:
                self.send_error(404)

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

        def log_message(self, fmt: str, *args) -> None:
            LOG.debug("视觉网页 %s", fmt % args)

    return Handler


def _camera_loop(config: dict[str, Any], camera_index: int | None, mode: str,
                 hub: FrameHub, stop: threading.Event) -> None:
    camera_config = dict(config["object_camera"])
    if camera_index is not None:
        camera_config["index"] = camera_index
    selected_index = int(camera_config["index"])
    vision = None
    try:
        vision = CameraVision(camera_config, config["hsv_colors"])
        ring_detector = RingDetector(config["ring_detection"], vision.cv2) if mode != "objects" else None
        rectifier = FrameRectifier(config["vision_calibration"].get("camera", {}), vision.cv2)
        sender = _CircleSender(config.get("serial", {}))
        tuner = StreamTuner(float(config.get("vision_stream", {}).get("target_fps", 12)))
        motion_gate = MotionGate()
        cached_rings: list[RingCandidate] = []
        have_cache = False
        force_detect = True
        active_color = hub.get_locate_color()
        previous = time.perf_counter()
        while not stop.is_set():
            color = hub.get_locate_color()
            if ring_detector is not None and color != active_color:
                ring_detector = RingDetector(config["ring_detection"], vision.cv2)
                active_color = color
                force_detect = True
            frame = rectifier.apply(vision.read())
            hub.remember_frame(frame)
            if hub.consume_track_reset():
                vision.clear_tracks()
                force_detect = True
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            level = tuner.observe(fps)
            vision.set_detect_width(level.detect_width)
            if ring_detector is not None:
                ring_detector.set_detect_width(level.detect_width)
            sample = motion_gate.update(frame)
            objects = (vision.detect_objects(frame, (color,) if color else None)
                       if mode != "rings" else [])
            if force_detect or not sample.reuse or not have_cache:
                regions = None if force_detect else sample.regions
                rings = (ring_detector.detect(frame, regions, color or None, config["hsv_colors"])
                         if ring_detector else [])
                cached_rings = rings
                have_cache = True
                force_detect = False
                motion_name = "update"
            else:
                rings = cached_rings
                motion_name = "hold"
            target = select_upper_circle(objects) if color else None
            point = upper_circle_coordinate(objects, frame.shape[1], frame.shape[0]) if color else None
            send_state = sender.update(color, point)
            tune = {
                "detect_width": level.detect_width,
                "preview_width": level.preview_width,
                "jpeg_quality": level.jpeg_quality,
                "target_fps": tuner.target_fps,
            }
            jpeg = render_preview(
                frame, objects, rings, fps, selected_index, camera_config, mode,
                level.preview_width, level.jpeg_quality, target,
            )
            status = vision_status(frame.shape, fps, selected_index, mode, objects, rings,
                                   sender.error if send_state == "error" else "",
                                   color, target, send_state, tune, motion_name)
            hub.publish(jpeg, status)
    except Exception as exc:
        LOG.error("视觉网页相机中断：%s", exc)
        try:
            image = _error_frame(str(exc))
            status = vision_status(image.shape, 0.0, selected_index, mode, [], [], str(exc),
                                   hub.get_locate_color())
            hub.publish(encode_jpeg(image), status)
        except Exception:
            LOG.exception("视觉网页无法生成错误画面")
        stop.wait()
    finally:
        if "sender" in locals():
            sender.close()
        if vision is not None:
            vision.close()


def run_vision_web(config: dict[str, Any], camera_index: int | None, mode: str,
                   host: str, port: int, config_path: Path | None = None) -> int:
    """启动浏览器预览。未选择颜色时不创建串口；选定颜色后才发送上层圆心。"""
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
