"""环位绑定、候选唯一性和连续帧稳定测试（无 OpenCV 硬件依赖）。"""

from __future__ import annotations

import unittest

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

    def test_unconfigured_layout_requires_exactly_three_candidates(self):
        detector = RingDetector({}, cv2_module=object())
        bound = detector._bind_positions([self.candidate(10, 10), self.candidate(20, 10)])
        self.assertTrue(all(not item.valid for item in bound))

    def test_center_must_remain_stable_for_required_frames(self):
        detector = RingDetector({"stable_frames": 3,
                                 "stable_center_tolerance_px": 2}, cv2_module=object())
        for index, x in enumerate((100, 101, 100.5)):
            item = RingCandidate(x, 100, 40, 0.8, 1, 4, position=1,
                                 valid=True, quality=0.9)
            output = detector._stabilize([item])[0]
            self.assertEqual(output.stable, index == 2)


if __name__ == "__main__":
    unittest.main()
