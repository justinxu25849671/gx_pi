"""静止画面少算、运动区域覆盖移动物体。"""

from __future__ import annotations

import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from logistics_robot.motion_gate import MotionGate, point_in_regions


@unittest.skipIf(np is None, "未安装 numpy/OpenCV，无法运行运动门控测试")
class MotionGateTests(unittest.TestCase):
    def test_still_frames_are_reused_until_the_refresh(self):
        gate = MotionGate(refresh_every=3)
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        self.assertFalse(gate.update(frame).reuse)
        self.assertTrue(gate.update(frame).reuse)
        self.assertTrue(gate.update(frame).reuse)
        self.assertFalse(gate.update(frame).reuse)

    def test_moving_block_produces_a_region(self):
        gate = MotionGate(min_ratio=0.01)
        still = np.zeros((90, 160, 3), dtype=np.uint8)
        gate.update(still)
        moved = still.copy()
        cv2.rectangle(moved, (100, 20), (140, 60), (255, 255, 255), -1)
        sample = gate.update(moved)
        self.assertFalse(sample.reuse)
        self.assertIsNotNone(sample.regions)
        self.assertTrue(point_in_regions(120, 40, sample.regions))
        self.assertFalse(point_in_regions(10, 80, sample.regions))


if __name__ == "__main__":
    unittest.main()
