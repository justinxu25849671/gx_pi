"""指定颜色圆环定位：只认该颜色，并只在唯一稳定时给出坐标。"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:
    np = None

from logistics_robot.model import DetectedObject
from logistics_robot.ring_detection import RingCandidate, RingDetector
from logistics_robot.vision_locate import (
    locate_coordinate, parse_locate_color, retain_hsv_color,
    select_upper_circle, upper_circle_coordinate,
)


class VisionLocateTests(unittest.TestCase):
    def test_parse_color_names_and_task_digits(self):
        self.assertEqual(parse_locate_color("Red"), "red")
        self.assertEqual(parse_locate_color("2"), "yellow")
        self.assertEqual(parse_locate_color("蓝"), "blue")
        with self.assertRaises(ValueError):
            parse_locate_color("purple")

    def test_upper_stable_circle_is_sent(self):
        lower = DetectedObject("blue", 80, 140, 4000, top_valid=True, stable=True, stable_frames=8,
                               circle_x_px=80, circle_y_px=140)
        upper = DetectedObject("blue", 100, 40, 5000, top_valid=True, stable=True, stable_frames=12,
                               circle_x_px=100, circle_y_px=40)
        brief = DetectedObject("blue", 90, 10, 3000, stable=False,
                               circle_x_px=90, circle_y_px=10)
        self.assertIs(select_upper_circle([lower, brief, upper]), upper)
        self.assertEqual(upper_circle_coordinate([lower, upper], 200, 160), (0, -40))
        self.assertIsNone(upper_circle_coordinate([brief], 200, 160))

    def test_coordinate_requires_exactly_one_stable_ring(self):
        ring = RingCandidate(130, 50, 40, 0.9, 1.0, 2, stable=True)
        self.assertEqual(locate_coordinate([ring], 200, 160), (30, -30))
        self.assertEqual(locate_coordinate([ring, ring], 200, 160), None)
        lost = RingCandidate(130, 50, 40, 0.9, 1.0, 2, stable=False)
        self.assertIsNone(locate_coordinate([lost], 200, 160))

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行颜色圆环测试")
    def test_only_requested_color_ring_remains(self):
        import cv2
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 120), 36, (0, 0, 255), 3)
        cv2.circle(frame, (230, 120), 36, (255, 80, 0), 3)
        hsv = {
            "red": [[0, 80, 80, 10, 255, 255], [170, 80, 80, 180, 255, 255]],
            "yellow": [[18, 80, 80, 40, 255, 255]],
            "blue": [[95, 80, 60, 130, 255, 255]],
            "green": [[42, 70, 50, 85, 255, 255]],
        }
        masked = retain_hsv_color(frame, "red", hsv, cv2)
        self.assertGreater(int(masked[120, 80 + 36].sum()) + int(masked[120, 80 - 36].sum()), 0)
        self.assertEqual(int(masked[120, 230].sum()), 0)
        detector = RingDetector({
            "min_radius_px": 18, "max_radius_px": 80,
            "min_visible_arc_ratio": 0.30, "hough_accumulator": 40,
            "proposal_min_arc_ratio": 0.40, "min_ellipse_axis_ratio": 0.75,
            "detect_max_width_px": 640, "stable_frames": 1,
            "calibrated": True,
        }, cv2)
        found = detector.detect(masked)
        self.assertGreaterEqual(len(found), 1)
        nearest = min(found, key=lambda item: (item.center_x_px - 80) ** 2
                      + (item.center_y_px - 120) ** 2)
        self.assertLess(abs(nearest.center_x_px - 80), 15)
        self.assertGreater(abs(nearest.center_x_px - 230), 40)

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行颜色圆环测试")
    def test_color_filter_uses_the_downscaled_frame(self):
        import cv2
        frame = np.zeros((300, 800, 3), dtype=np.uint8)
        cv2.circle(frame, (180, 150), 48, (0, 0, 255), 8)
        cv2.circle(frame, (620, 150), 48, (255, 80, 0), 8)
        hsv = {
            "red": [[0, 80, 80, 10, 255, 255], [170, 80, 80, 180, 255, 255]],
            "yellow": [[18, 80, 80, 40, 255, 255]],
            "blue": [[95, 80, 60, 130, 255, 255]],
            "green": [[42, 70, 50, 85, 255, 255]],
        }
        detector = RingDetector({
            "min_radius_px": 18, "max_radius_px": 80,
            "min_visible_arc_ratio": 0.30, "hough_accumulator": 40,
            "proposal_min_arc_ratio": 0.40, "min_ellipse_axis_ratio": 0.75,
            "detect_max_width_px": 400, "stable_frames": 1,
            "calibrated": True,
        }, cv2)
        found = detector.detect(frame, color="red", hsv_colors=hsv)
        self.assertGreaterEqual(len(found), 1)
        nearest = min(found, key=lambda item: (item.center_x_px - 180) ** 2
                      + (item.center_y_px - 150) ** 2)
        self.assertLess(abs(nearest.center_x_px - 180), 24)
        self.assertLess(abs(nearest.center_y_px - 150), 24)
        self.assertGreater(abs(nearest.center_x_px - 620), 80)


if __name__ == "__main__":
    unittest.main()
