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
from logistics_robot.vision import (
    CameraVision, ObjectDetector, draw_object_debug_frame, image_origin_px,
)


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

    def test_image_center_is_origin_and_y_increases_downward(self):
        self.assertEqual(image_origin_px(100, 80, 200, 160), (0, 0))
        self.assertEqual(image_origin_px(130, 50, 200, 160), (30, -30))
        self.assertEqual(image_origin_px(100.4, 79.6, 200, 160), (0, 0))
        self.assertEqual(image_origin_px(100.5, 80, 200, 160), (1, 0))
        self.assertEqual(image_origin_px(99.5, 80, 200, 160), (-1, 0))

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
        cv2.circle(frame, (97, 57), 15, (0, 0, 255), -1)
        for index in range(3):
            objects = detector.detect(frame)
            self.assertEqual(len(objects), 1 if index == 2 else 0)
        self.assertTrue(objects[0].stable)
        self.assertAlmostEqual(objects[0].reference_point_px[0], 97, delta=1.0)
        self.assertAlmostEqual(objects[0].circle_center_px[0], objects[0].center_x_px, delta=8)
        self.assertAlmostEqual(objects[0].circle_center_px[1], objects[0].center_y_px, delta=8)

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像检测测试")
    def test_circularity_keeps_disc_and_rejects_bar(self):
        import cv2
        config = {
            "object_detection": {
                "min_area_px": 200, "max_area_px": 20000,
                "min_aspect_ratio": 0.2, "max_aspect_ratio": 5.0,
                "min_solidity": 0.5, "min_extent": 0.3,
                "min_circularity": 0.80,
                "stable_frames": 1,
            }
        }
        hsv = {color: [[0, 0, 0, 180, 255, 255]]
               for color in ("red", "yellow", "blue", "green")}
        hsv["red"] = [[0, 80, 80, 10, 255, 255]]
        detector = ObjectDetector(config, hsv, cv2)
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        cv2.circle(frame, (60, 80), 24, (0, 0, 255), -1)
        cv2.rectangle(frame, (120, 70), (190, 90), (0, 0, 255), -1)
        objects = detector.detect(frame)
        self.assertEqual(len(objects), 1)
        self.assertLess(abs(objects[0].center_x_px - 60), 8)

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像检测测试")
    def test_side_view_cylinder_top_is_recognized(self):
        import cv2
        config = {
            "object_detection": {
                "min_area_px": 200, "max_area_px": 20000,
                "min_aspect_ratio": 0.4, "max_aspect_ratio": 1.8,
                "min_solidity": 0.7, "min_extent": 0.4,
                "min_circularity": 0.8,
                "stable_frames": 1,
            }
        }
        hsv = {color: [[0, 0, 0, 1, 1, 1]]
               for color in ("red", "yellow", "blue", "green")}
        hsv["red"] = [[0, 80, 80, 15, 255, 255]]
        detector = ObjectDetector(config, hsv, cv2)
        frame = np.zeros((180, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (58, 78), (102, 150), (0, 45, 140), -1)
        cv2.circle(frame, (80, 70), 28, (0, 0, 230), -1)
        objects = detector.detect(frame)
        self.assertEqual(len(objects), 1)
        self.assertLess(abs(objects[0].circle_center_px[0] - 80), 8)
        self.assertLess(abs(objects[0].circle_center_px[1] - 70), 10)

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行图像检测测试")
    def test_largest_color_blob_supplies_top_candidate(self):
        import cv2
        config = {
            "object_detection": {
                "min_area_px": 100, "max_area_px": 20000,
                "min_aspect_ratio": 0.4, "max_aspect_ratio": 1.8,
                "min_solidity": 0.7, "min_extent": 0.4,
                "min_circularity": 0.55,
                "stable_frames": 1,
            }
        }
        hsv = {color: [[0, 0, 0, 1, 1, 1]]
               for color in ("red", "yellow", "blue", "green")}
        hsv["red"] = [[0, 80, 80, 12, 255, 255]]
        detector = ObjectDetector(config, hsv, cv2)
        frame = np.zeros((180, 220, 3), dtype=np.uint8)
        cv2.circle(frame, (40, 40), 14, (0, 0, 255), -1)
        cv2.circle(frame, (140, 90), 32, (0, 0, 255), -1)
        objects = detector.detect(frame)
        self.assertEqual(len(objects), 1)
        self.assertLess(abs(objects[0].circle_center_px[0] - 140), 8)
        self.assertLess(abs(objects[0].circle_center_px[1] - 90), 8)

    def test_brief_hit_and_area_jump_are_hidden(self):
        import cv2
        config = {"object_detection": {
            "stable_frames": 3, "stable_center_tolerance_px": 8,
            "max_area_change_ratio": 1.5,
        }}
        hsv = {color: [[0, 0, 0, 1, 1, 1]]
               for color in ("red", "yellow", "blue", "green")}
        detector = ObjectDetector(config, hsv, cv2)

        def seen(area):
            return detector._stabilize(
                [DetectedObject("red", 50, 50, area, reference_x_px=50, reference_y_px=50,
                                solidity=1.0, extent=1.0)],
                (100, 100))

        self.assertEqual(seen(1000), [])
        self.assertEqual(seen(1000), [])
        locked = seen(1000)
        self.assertEqual(len(locked), 1)
        self.assertEqual(locked[0].track_id, 1)
        self.assertEqual(seen(4000), [])
        kept = seen(1000)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].track_id, 1)
        self.assertAlmostEqual(kept[0].area_px, 1000)
        outside = DetectedObject("red", 50, 50, 1000, reference_x_px=50, reference_y_px=50,
                                  solidity=1.0, extent=1.0)
        fresh = ObjectDetector(config, hsv, cv2)
        self.assertEqual(fresh._stabilize([outside], (100, 100), [(0, 0, 10, 10)]), [])


if __name__ == "__main__":
    unittest.main()
