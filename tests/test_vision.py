"""物料识别相机独立调试的无硬件单元测试。"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
except ImportError:  # 与 requirements.txt 一致；缺依赖的开发机仍可运行其余逻辑测试。
    np = None

from logistics_robot import app
from logistics_robot.model import DetectedObject
from logistics_robot.vision import CameraVision, draw_object_debug_frame


class VisionTests(unittest.TestCase):
    def test_best_target_prefers_center_then_area(self):
        objects = [
            DetectedObject("red", 430, 200, 9999),
            DetectedObject("red", 520, 200, 100),
            DetectedObject("red", 480, 200, 200),
            DetectedObject("red", 480, 300, 800),
        ]
        target = CameraVision.best_target(objects, "red", 500)
        self.assertEqual(target, objects[3])
        self.assertIsNone(CameraVision.best_target(objects, "yellow", 500))

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像绘制测试")
    def test_draw_debug_frame_does_not_mutate_inputs(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        original_frame = frame.copy()
        objects = [DetectedObject("black", 80.0, 60.0, 900.0)]
        rendered = draw_object_debug_frame(frame, objects, 30.0, 0)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, original_frame))
        self.assertEqual(objects, [DetectedObject("black", 80.0, 60.0, 900.0)])
        self.assertFalse(np.array_equal(rendered, original_frame))

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像绘制测试")
    def test_draw_debug_frame_handles_empty_objects(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        rendered = draw_object_debug_frame(frame, [], 0.0, 3)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(rendered.sum()), 0)

    def test_vision_debug_main_does_not_construct_serial_link(self):
        config = {
            "object_camera": {"index": 0, "width": 1920, "height": 1080,
                              "fps": 30, "min_object_area_px": 900},
            "hsv_colors": {"red": [[0, 0, 0, 1, 1, 1]]},
        }
        debug = Mock(return_value=0)
        with patch.object(app, "load_config", return_value=config), \
             patch.object(app, "_run_vision_debug", debug), \
             patch.object(app, "SerialPathLink") as serial_link, \
             patch.object(sys, "argv", ["main.py", "--vision-debug", "--vision-camera-index", "4"]):
            self.assertEqual(app.main(), 0)
        debug.assert_called_once_with(config, 4)
        serial_link.assert_not_called()


if __name__ == "__main__":
    unittest.main()
