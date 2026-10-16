"""环位绑定、候选唯一性和连续帧稳定测试（无 OpenCV 硬件依赖）。"""

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:
    np = None

from logistics_robot.ring_detection import RingCandidate, RingDetector


class RingDetectionTests(unittest.TestCase):
    @staticmethod
    def candidate(x, y):
        return RingCandidate(x, y, 40, 0.8, 1.0, 4, valid=True, quality=0.9)

    def test_expected_positions_bind_nearest_candidate(self):
        detector = RingDetector({
            "position_centers_px": {"1": [100, 100], "2": [300, 100], "3": [500, 100]},
            "position_match_tolerance_px": 30,
        }, cv2_module=object())
        bound = detector._bind_positions([
            self.candidate(496, 102), self.candidate(105, 98), self.candidate(298, 101),
        ])
        self.assertEqual([(item.position, round(item.center_x_px)) for item in bound],
                         [(1, 105), (2, 298), (3, 496)])
        self.assertTrue(all(item.valid for item in bound))

    def test_unconfigured_layout_keeps_any_count(self):
        detector = RingDetector({}, cv2_module=object())
        bound = detector._bind_positions([self.candidate(10, 10), self.candidate(20, 10)])
        self.assertEqual(len(bound), 2)
        self.assertTrue(all(item.valid for item in bound))

    def test_center_must_remain_stable_for_required_frames(self):
        detector = RingDetector({"stable_frames": 3,
                                 "stable_center_tolerance_px": 2}, cv2_module=object())
        for index, x in enumerate((100, 101, 100.5)):
            item = RingCandidate(x, 100, 40, 0.8, 1, 4, position=1,
                                 valid=True, quality=0.9)
            output = detector._stabilize([item])[0]
            self.assertEqual(output.stable, index == 2)

    @unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行圆形检测测试")
    def test_clean_circle_is_kept_and_line_clutter_is_not(self):
        import cv2
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 42, (255, 255, 255), 2)
        cv2.line(frame, (8, 8), (120, 30), (255, 255, 255), 2)
        cv2.rectangle(frame, (20, 180), (90, 220), (255, 255, 255), 2)
        detector = RingDetector({
            "min_radius_px": 18, "max_radius_px": 80,
            "min_visible_arc_ratio": 0.30,
            "hough_accumulator": 48,
            "proposal_min_arc_ratio": 0.45,
            "min_ellipse_axis_ratio": 0.80,
            "detect_max_width_px": 640,
            "stable_frames": 1,
        }, cv2)
        found = detector.detect(frame)
        self.assertGreaterEqual(len(found), 1)
        self.assertLessEqual(len(found), 3)
        nearest = min(found, key=lambda item: (item.center_x_px - 160) ** 2
                      + (item.center_y_px - 120) ** 2)
        self.assertLess(abs(nearest.center_x_px - 160), 12)
        self.assertLess(abs(nearest.center_y_px - 120), 12)
        self.assertTrue(nearest.valid or nearest.reason == "环检测参数尚未用实物标定")

    def test_brief_circle_and_radius_jump_are_hidden(self):
        detector = RingDetector({
            "stable_frames": 3, "stable_center_tolerance_px": 6,
            "max_radius_change_ratio": 1.4,
        }, cv2_module=object())

        def circle(radius):
            return RingCandidate(80, 60, radius, 0.9, 1.0, 2, valid=True, quality=0.8)

        self.assertEqual(detector._persist([circle(40)]), [])
        self.assertEqual(detector._persist([circle(40)]), [])
        self.assertEqual(len(detector._persist([circle(40)])), 1)
        self.assertEqual(detector._persist([circle(90)]), [])
        kept = detector._persist([circle(40)])
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].radius_px, 40)


if __name__ == "__main__":
    unittest.main()
