"""松爪前对准、动作反馈和一次性放置约束测试。"""

from __future__ import annotations

import unittest

from logistics_robot.alignment import (
    ActionResult, AlignmentAction, AlignmentObservation, AlignmentState,
    PlacementAlignmentController,
)
from logistics_robot.calibration import PoseCalibration


class FakeActuator:
    def __init__(self):
        self.started = []
        self.results = []

    def start(self, action, value=None):
        self.started.append((action, value))

    def poll(self, action):
        return self.results.pop(0) if self.results else ActionResult.BUSY


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.actuator = FakeActuator()
        self.calibration = PoseCalibration(
            "placement_layer1", True, 100.0, 0.0, (100.0, 100.0),
            ((1.0, 0.0), (0.0, 1.0)),
        )
        self.config = {
            "lateral_tolerance_mm": 1.0,
            "longitudinal_tolerance_mm": 1.0,
            "max_correction_step_mm": 5.0,
            "max_corrections": 4,
            "required_aligned_frames": 2,
            "min_target_quality": 0.7,
            "total_timeout_s": 20.0,
        }

    @staticmethod
    def observation(x, y, valid=True, stable=True, quality=0.9):
        return AlignmentObservation(x, y, valid, stable, quality, "ring")

    def test_adjusts_one_axis_then_remeasures_and_releases_once(self):
        controller = PlacementAlignmentController(self.calibration, self.actuator,
                                                  self.config)
        controller.tick(self.observation(110, 120))
        self.assertEqual(self.actuator.started, [(AlignmentAction.LATERAL, 5.0)])
        self.assertFalse(controller.needs_observation)
        self.actuator.results.append(ActionResult.SUCCEEDED)
        controller.tick(self.observation(0, 0))  # 动作等待期的画面不得用于修正。
        self.assertTrue(controller.needs_observation)

        controller.tick(self.observation(100, 120))
        self.assertEqual(self.actuator.started[-1], (AlignmentAction.LONGITUDINAL, 5.0))
        self.actuator.results.append(ActionResult.SUCCEEDED)
        controller.tick()
        controller.tick(self.observation(100.5, 100.5))
        controller.tick(self.observation(100.4, 100.4))
        self.assertEqual(self.actuator.started[-1][0], AlignmentAction.DESCEND)
        self.assertTrue(controller.status().plane_locked)
        self.assertFalse(controller.needs_observation)

        for expected in (AlignmentAction.RELEASE, AlignmentAction.RETREAT):
            self.actuator.results.append(ActionResult.SUCCEEDED)
            controller.tick(self.observation(9999, 9999))
            controller.tick(self.observation(9999, 9999))
            self.assertEqual(self.actuator.started[-1][0], expected)
        self.actuator.results.append(ActionResult.SUCCEEDED)
        status = controller.tick(self.observation(-9999, -9999))
        self.assertEqual(status.state, AlignmentState.COMPLETE)
        self.assertEqual([action for action, _ in self.actuator.started].count(
            AlignmentAction.RELEASE), 1)

    def test_unstable_or_low_quality_target_never_moves(self):
        controller = PlacementAlignmentController(self.calibration, self.actuator,
                                                  self.config)
        controller.tick(self.observation(100, 100, stable=False))
        controller.tick(self.observation(100, 100, quality=0.2))
        self.assertEqual(self.actuator.started, [])

    def test_failed_action_faults_without_retry(self):
        controller = PlacementAlignmentController(self.calibration, self.actuator,
                                                  self.config)
        controller.tick(self.observation(110, 100))
        self.actuator.results.append(ActionResult.FAILED)
        status = controller.tick()
        self.assertEqual(status.state, AlignmentState.FAULT)
        self.assertEqual(len(self.actuator.started), 1)

    def test_uncalibrated_pose_is_rejected(self):
        uncalibrated = PoseCalibration("bad", False, 0, 0, (0, 0),
                                     ((0, 0), (0, 0)))
        with self.assertRaisesRegex(RuntimeError, "未标定"):
            PlacementAlignmentController(uncalibrated, self.actuator, self.config)


if __name__ == "__main__":
    unittest.main()
