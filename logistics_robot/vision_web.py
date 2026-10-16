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
from .model import DetectedObject
from .ring_detection import RingCandidate, RingDetector, draw_ring_debug
from .vision import CameraVision, draw_object_debug_frame, image_origin_px
from .vision_locate import parse_locate_color, retain_hsv_color


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

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._lock.notify_all()


def vision_status(frame_shape, fps: float, camera_index: int, mode: str,
                  objects: list[DetectedObject], rings: list[RingCandidate],
                  error: str = "", locate_color: str = "") -> dict[str, Any]:
    """整理网页侧栏使用的画面中心坐标。"""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    object_rows = []
    for item in objects:
        x_px, y_px = image_origin_px(item.circle_center_px[0], item.circle_center_px[1],
                                     width, height)
        object_rows.append({
            "color": item.color, "x": x_px, "y": y_px,
            "stable": bool(item.stable), "stable_frames": int(item.stable_frames),
            "quality": round(float(item.confidence), 3),
        })
    ring_rows = []
    for item in rings:
        x_px, y_px = image_origin_px(item.center_x_px, item.center_y_px, width, height)
        ring_rows.append({
            "position": item.position, "x": x_px, "y": y_px,
            "radius_px": round(float(item.radius_px), 2),
            "valid": bool(item.valid), "stable": bool(item.stable),
            "stable_frames": int(item.stable_frames),
            "quality": round(float(item.quality), 3),
            "reason": item.reason,
        })
    return {
        "ok": not error, "error": error, "fps": round(float(fps), 2),
        "width": width, "height": height, "camera_index": int(camera_index),
        "mode": mode, "locate_color": locate_color,
        "objects": object_rows, "rings": ring_rows,
    }


def annotate_vision_frame(frame, objects: list[DetectedObject], rings: list[RingCandidate],
                          fps: float, camera_index: int, camera_config, mode: str):
    """叠加物料、圆环和画面中心坐标。返回新图像，不改原始帧。"""
    annotated = draw_object_debug_frame(
        frame, [] if mode == "rings" else objects, fps, camera_index, camera_config)
    if mode != "objects":
        annotated = draw_ring_debug(annotated, rings)
    return annotated


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
    """只读 HTTP 服务：页面、状态、单帧和 MJPEG。"""

    def __init__(self, hub: FrameHub, host: str, port: int) -> None:
        html = _HTML_PATH.read_bytes()
        handler = _handler(hub, html)
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


def _handler(hub: FrameHub, html: bytes):
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
            if path != "/color":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                requested = str(payload.get("color", "")).strip()
                color = parse_locate_color(requested) if requested else ""
            except (ValueError, json.JSONDecodeError, UnicodeError):
                body = json.dumps({"ok": False, "error": "颜色无效"}, ensure_ascii=False).encode("utf-8")
                self._send(400, "application/json; charset=utf-8", body)
                return
            hub.set_locate_color(color)
            body = json.dumps({"ok": True, "color": color}, ensure_ascii=False).encode("utf-8")
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
        active_color = hub.get_locate_color()
        previous = time.perf_counter()
        while not stop.is_set():
            color = hub.get_locate_color()
            if ring_detector is not None and color != active_color:
                ring_detector = RingDetector(config["ring_detection"], vision.cv2)
                active_color = color
            frame = rectifier.apply(vision.read())
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            objects = vision.detect_objects(frame) if mode != "rings" else []
            ring_frame = frame
            if color and ring_detector is not None:
                ring_frame = retain_hsv_color(frame, color, config["hsv_colors"], vision.cv2)
            if color:
                objects = [item for item in objects if item.color == color]
            rings = ring_detector.detect(ring_frame) if ring_detector else []
            annotated = annotate_vision_frame(frame, objects, rings, fps, selected_index,
                                              camera_config, mode)
            status = vision_status(annotated.shape, fps, selected_index, mode, objects, rings,
                                   locate_color=color)
            hub.publish(encode_jpeg(annotated), status)
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
        if vision is not None:
            vision.close()


def run_vision_web(config: dict[str, Any], camera_index: int | None, mode: str,
                   host: str, port: int) -> int:
    """启动浏览器预览。此函数不创建串口、路径或机构对象。"""
    if not 1 <= int(port) <= 65535:
        LOG.error("网页端口必须在 1..65535")
        return 2
    hub = FrameHub()
    try:
        server = VisionWebServer(hub, host, int(port))
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
