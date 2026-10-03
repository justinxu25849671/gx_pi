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
from logistics_robot.vision import CameraVision, ObjectDetector, draw_object_debug_frame


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
        objects = [DetectedObject("green", 80.0, 60.0, 900.0)]
        rendered = draw_object_debug_frame(frame, objects, 30.0, 0)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, original_frame))
        self.assertEqual(objects, [DetectedObject("green", 80.0, 60.0, 900.0)])
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
                              "fps": 30, "object_detection": {"min_area_px": 900}},
            "hsv_colors": {color: [[0, 0, 0, 1, 1, 1]]
                           for color in ("red", "yellow", "blue", "green")},
            "ring_detection": {},
        }
        debug = Mock(return_value=0)
        with patch.object(app, "load_config", return_value=config), \
             patch.object(app, "_run_vision_debug", debug), \
             patch.object(app, "SerialPathLink") as serial_link, \
             patch.object(sys, "argv", ["main.py", "--vision-debug", "--vision-camera-index", "4"]):
            self.assertEqual(app.main(), 0)
        debug.assert_called_once_with(config, 4, "combined")
        serial_link.assert_not_called()

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像检测测试")
    def test_roi_shape_filter_and_continuous_stability(self):
        import cv2
        config = {
            "object_detection": {
                "roi": [0.25, 0.0, 1.0, 1.0], "occlusion_rects": [],
                "min_area_px": 200, "max_area_px": 2000,
                "min_aspect_ratio": 0.7, "max_aspect_ratio": 1.3,
                "min_solidity": 0.8, "min_extent": 0.6,
                "stable_frames": 3, "stable_center_tolerance_px": 3,
                "selection_reference_px": [100, 60],
                "reference_point": {"mode": "bbox_center", "offset_px": [2, -1]},
            }
        }
        hsv = {
            "red": [[0, 100, 100, 10, 255, 255]],
            "yellow": [[20, 100, 100, 35, 255, 255]],
            "blue": [[100, 100, 100, 130, 255, 255]],
            "green": [[40, 100, 100, 85, 255, 255]],
        }
        detector = ObjectDetector(config, hsv, cv2)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (4, 40), (32, 68), (0, 0, 255), -1)  # ROI 外，应屏蔽。
        cv2.rectangle(frame, (82, 42), (112, 72), (0, 0, 255), -1)
        for index in range(3):
            objects = detector.detect(frame)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0].stable, index == 2)
        self.assertAlmostEqual(objects[0].reference_point_px[0], 99.5, delta=1.0)


if __name__ == "__main__":
    unittest.main()
