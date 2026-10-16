"""处理后相机画面的网页转发测试。"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import numpy as np
except ImportError:
    np = None

from logistics_robot import app
from logistics_robot.model import DetectedObject
from logistics_robot.ring_detection import RingCandidate
from logistics_robot.vision_web import (
    FrameHub, VisionWebServer, _CircleSender, annotate_vision_frame, encode_jpeg,
    vision_status,
)


@unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行网页画面测试")
class VisionWebTests(unittest.TestCase):
    def test_status_uses_image_center_and_drawing_keeps_source(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        original = frame.copy()
        material = DetectedObject("red", 90, 40, 1000, stable=True, stable_frames=4,
                                  confidence=0.91, circle_x_px=100, circle_y_px=40)
        ring = RingCandidate(80, 60, 18, 0.8, 1.2, 3, position=2, valid=True,
                             stable=True, stable_frames=4, quality=0.88)
        status = vision_status(frame.shape, 12.5, 0, "combined", [material], [ring])
        self.assertEqual(status["objects"][0]["x"], 20)
        self.assertEqual(status["objects"][0]["y"], -20)
        self.assertEqual((status["rings"][0]["x"], status["rings"][0]["y"],
                          status["rings"][0]["position"]), (0, 0, 2))
        annotated = annotate_vision_frame(frame, [material], [ring], 12.5, 0, {}, "combined")
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, original))
        self.assertFalse(np.array_equal(annotated, original))
        self.assertTrue(encode_jpeg(annotated).startswith(b"\xff\xd8"))

    def test_http_serves_page_snapshot_status_and_stream(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (20, 40, 60)
        hub = FrameHub()
        status = vision_status(frame.shape, 8.0, 1, "rings", [], [
            RingCandidate(40, 24, 10, 0.5, 1.0, 2, position=1),
        ])
        hub.publish(encode_jpeg(frame), status)
        server = VisionWebServer(hub, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            with urlopen(base + "/", timeout=2) as response:
                page = response.read().decode("utf-8")
            self.assertIn("/stream.mjpg", page)
            self.assertIn("圆环圆心", page)
            self.assertIn("发送目标", page)
            self.assertIn("识别颜色", page)
            self.assertIn('data-color="red"', page)
            with urlopen(base + "/snapshot.jpg", timeout=2) as response:
                jpeg = response.read()
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))
            with urlopen(base + "/status.json", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["rings"][0]["position"], 1)
            self.assertEqual((payload["rings"][0]["x"], payload["rings"][0]["y"]), (8, 0))
            with urlopen(base + "/stream.mjpg", timeout=2) as response:
                chunk = b""
                while b"\xff\xd8" not in chunk and len(chunk) < 65536:
                    piece = response.read(256)
                    if not piece:
                        break
                    chunk += piece
            self.assertIn(b"--frame", chunk)
            self.assertIn(b"\xff\xd8", chunk)
            request = Request(
                base + "/color",
                data=json.dumps({"color": "blue"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                changed = json.loads(response.read().decode("utf-8"))
            self.assertEqual(changed["color"], "blue")
            with urlopen(base + "/status.json", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["locate_color"], "blue")
            rejected = Request(
                base + "/color",
                data=json.dumps({"color": "purple"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as caught:
                urlopen(rejected, timeout=2)
            self.assertEqual(caught.exception.code, 400)
        finally:
            hub.close()
            server.shutdown()
            thread.join(timeout=2)

    def test_vision_web_main_does_not_open_serial(self):
        config = {
            "object_camera": {"index": 0},
            "hsv_colors": {},
            "ring_detection": {},
            "vision_calibration": {"camera": {}},
        }
        runner = Mock(return_value=0)
        with patch.object(app, "load_config", return_value=config), \
             patch.object(app, "run_vision_web", runner), \
             patch.object(app, "SerialPathLink") as serial_link, \
             patch.object(sys, "argv", ["main.py", "--vision-web", "--vision-web-port", "8091",
                                        "--vision-mode", "rings"]):
            self.assertEqual(app.main(), 0)
        runner.assert_called_once_with(config, None, "rings", "0.0.0.0", 8091)
        serial_link.assert_not_called()

    def test_sender_waits_without_serial_and_sends_locked_circle(self):
        sender = _CircleSender({})
        self.assertEqual(sender.update("", None), "idle")
        self.assertIsNone(sender._link)
        self.assertEqual(sender.update("blue", None), "waiting")
        self.assertIsNone(sender._link)
        client = Mock()
        sender._link = Mock(path_client=client)
        sender._started = True
        self.assertEqual(sender.update("blue", (-135, -122)), "sending")
        client.send_visual_coordinate.assert_called_once_with(-135, -122)
        self.assertEqual(sender.update("blue", (0, 0)), "aligned")
        client.stop_visual_positioning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
