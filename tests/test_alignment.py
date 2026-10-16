"""画面中心坐标系视觉定位。到位后结束，不下降、不松爪、不撤离。"""

from __future__ import annotations

import unittest

from logistics_robot.alignment import (
    ActionResult, AlignmentObservation, AlignmentState, PlacementAlignmentController,
)


class FakeChassis:
    def __init__(self):
        self.moves = []
        self.stopped = 0
        self.poll_result = ActionResult.SUCCEEDED
        self.fail_move = False
        self.fail_stop = False

    def move(self, kind, distance_mm):
        self.moves.append((kind, distance_mm))
        if self.fail_move:
            raise RuntimeError("底盘指令失败")

    def poll(self):
        return self.poll_result

    def stop(self):
        self.stopped += 1
        if self.fail_stop:
            raise RuntimeError("停止失败")


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.link = FakeChassis()
        self.config = {
            "min_target_quality": 0.7,
            "total_timeout_s": 20.0,
            "lateral_tolerance_mm": 1.0,
            "longitudinal_tolerance_mm": 1.0,
            "max_correction_step_mm": 5.0,
            "max_corrections": 8,
            "required_aligned_frames": 2,
            "mm_per_px_x": 1.0,
            "mm_per_px_y": 1.0,
            "x_positive": "right",
            "y_positive": "forward",
        }

    def observation(self, x, y, valid=True, stable=True, quality=0.9,
                    width=200, height=160):
        return AlignmentObservation(
            x, y, valid, stable, quality, "material_circle",
            image_width=width, image_height=height)

    def controller(self, now=None):
        return PlacementAlignmentController(self.link, self.config, now=now)

    def test_centered_target_finishes_without_mechanism(self):
        controller = self.controller()
        first = controller.tick(self.observation(100, 80))
        self.assertEqual(first.state, AlignmentState.ACQUIRE)
        self.assertEqual(self.link.moves, [])
        self.assertEqual(self.link.stopped, 0)

        status = controller.tick(self.observation(100, 80))
        self.assertEqual(status.state, AlignmentState.COMPLETE)
        self.assertTrue(status.plane_locked)
        self.assertFalse(status.release_started)
        self.assertFalse(status.released)
        self.assertFalse(controller.needs_observation)
        self.assertEqual(self.link.moves, [])
        self.assertEqual(self.link.stopped, 1)

        later = controller.tick(self.observation(180, 20))
        self.assertEqual(later.state, AlignmentState.COMPLETE)
        self.assertEqual(self.link.moves, [])
        self.assertEqual(self.link.stopped, 1)

    def test_offset_moves_chassis_and_does_not_finish(self):
        controller = self.controller()
        status = controller.tick(self.observation(130, 80))
        self.assertEqual(status.state, AlignmentState.ACQUIRE)
        self.assertEqual(self.link.moves, [("rightmm", 5)])
        self.assertFalse(status.plane_locked)

    def test_unstable_target_does_not_move(self):
        controller = self.controller()
        controller.tick(self.observation(130, 80, stable=False))
        controller.tick(self.observation(130, 80, quality=0.2))
        controller.tick(None)
        self.assertEqual(self.link.moves, [])
        self.assertEqual(controller.status().state, AlignmentState.ACQUIRE)

    def test_timeout_faults_without_mechanism(self):
        clock = {"now": 0.0}
        controller = self.controller(now=lambda: clock["now"])
        controller.tick(self.observation(130, 80))
        clock["now"] = 21.0
        status = controller.tick(self.observation(110, 80))
        self.assertEqual(status.state, AlignmentState.FAULT)
        self.assertIn("视觉定位超时", status.message)
        self.assertFalse(status.released)


if __name__ == "__main__":
    unittest.main()
