"""画质和帧率自动调节。"""

from __future__ import annotations

import unittest

from logistics_robot.stream_tuner import StreamTuner


class StreamTunerTests(unittest.TestCase):
    def test_slow_frames_step_down_and_fast_frames_step_up(self):
        tuner = StreamTuner(target_fps=12)
        self.assertEqual(tuner.level.detect_width, 640)
        for _ in range(5):
            tuner.observe(4)
        self.assertEqual(tuner.level.detect_width, 640)
        tuner.observe(4)
        self.assertEqual(tuner.level.detect_width, 480)
        self.assertLess(tuner.level.jpeg_quality, 72)
        for _ in range(10):
            tuner.observe(20)
        self.assertEqual(tuner.level.detect_width, 640)


if __name__ == "__main__":
    unittest.main()
