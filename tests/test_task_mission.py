"""四颜色任务码和第二层同色承接位置测试。"""

from __future__ import annotations

import unittest

from logistics_robot.mission import LogisticsMission
from logistics_robot.model import DetectedObject, MechanismAction, MissionStep
from logistics_robot.placement_target import resolve_placement_target
from logistics_robot.ring_detection import RingCandidate
from logistics_robot.task_code import TaskCode


class TaskMissionTests(unittest.TestCase):
    def test_four_color_task_code(self):
        task = TaskCode.parse("123+123+321+231")
        self.assertEqual(task.first_colors, ("red", "yellow", "blue"))
        for invalid in ("125+123+521+231", "112+123+211+231", "123+123+234+231"):
            with self.assertRaises(ValueError):
                TaskCode.parse(invalid)

    def test_second_layer_uses_first_batch_same_color_position(self):
        task = TaskCode.parse("123+123+321+231")
        steps = LogisticsMission._steps_for(task)
        stack = [step for step in steps if step.action == MechanismAction.STACK_TO_TEMPORARY]
        self.assertEqual([(step.color, step.position) for step in stack],
                         [("blue", 3), ("yellow", 2), ("red", 1)])

    def test_first_layer_uses_ring_second_layer_uses_stable_same_color(self):
        ring = RingCandidate(10, 20, 30, 0.8, 1.0, 4, position=2,
                             valid=True, stable=True, stable_frames=4, quality=0.9)
        first = MissionStep("temporary", MechanismAction.PLACE_TO_TEMPORARY,
                            "red", 2, 1)
        target = resolve_placement_target(first, [ring], [])
        self.assertEqual((target.source, target.target_x_px), ("ring", 10))

        support = DetectedObject("red", 31, 42, 1000, reference_x_px=32,
                                 reference_y_px=43, stable=True, stable_frames=4,
                                 confidence=0.88)
        second = MissionStep("temporary", MechanismAction.STACK_TO_TEMPORARY,
                             "red", 2, 2)
        target = resolve_placement_target(second, [ring], [support])
        self.assertEqual((target.source, target.target_x_px, target.target_y_px),
                         ("support_object", 32, 43))


if __name__ == "__main__":
    unittest.main()
