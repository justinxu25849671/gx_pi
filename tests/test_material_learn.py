"""点击物料后学习 HSV 范围。"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from logistics_robot.material_learn import hsv_bounds_from_bgr
from logistics_robot.vision_web import FrameHub, VisionWebServer


@unittest.skipIf(np is None or cv2 is None, "未安装 numpy/OpenCV，无法运行物料学习测试")
class MaterialLearnTests(unittest.TestCase):
    def test_blue_patch_becomes_one_hue_range(self):
        hsv = np.zeros((36, 36, 3), dtype=np.uint8)
        hsv[:] = (110, 200, 180)
        bounds = hsv_bounds_from_bgr(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(len(bounds), 1)
        low, _s, _v, high, _s2, _v2 = bounds[0]
        self.assertLessEqual(low, 110)
        self.assertGreaterEqual(high, 110)

    def test_red_wrap_splits_into_two_ranges(self):
        hsv = np.zeros((36, 36, 3), dtype=np.uint8)
        hsv[:, :18] = (2, 180, 180)
        hsv[:, 18:] = (176, 180, 180)
        bounds = hsv_bounds_from_bgr(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        self.assertEqual(len(bounds), 2)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[1][3], 179)

    def test_pale_patch_is_rejected(self):
        gray = np.full((36, 36, 3), 40, dtype=np.uint8)
        with self.assertRaises(ValueError):
            hsv_bounds_from_bgr(gray)

    def test_http_learn_updates_config_file(self):
        hsv = np.zeros((80, 80, 3), dtype=np.uint8)
        hsv[:] = (110, 200, 180)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hub = FrameHub()
        hub.remember_frame(frame)
        config = {"hsv_colors": {"red": [], "yellow": [], "blue": [[0, 0, 0, 1, 1, 1]], "green": []}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{}\n", encoding="utf-8")
            server = VisionWebServer(hub, "127.0.0.1", 0, config, path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.port}/learn",
                    data=json.dumps({"color": "blue", "x": 40, "y": 40}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertEqual(config["hsv_colors"]["blue"], payload["bounds"])
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["hsv_colors"]["blue"], payload["bounds"])
                self.assertTrue(hub.consume_track_reset())
            finally:
                hub.close()
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
