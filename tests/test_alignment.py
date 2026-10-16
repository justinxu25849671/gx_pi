"""画面中心坐标系视觉定位，以及定位结束后的一次性放置。"""

from __future__ import annotations

import unittest

from logistics_robot.alignment import (
    ActionResult, AlignmentAction, AlignmentObservation, AlignmentState,
    PlacementAlignmentController,
)


class FakeLocateLink:
    def __init__(self):
        self.events = []
        self.fail_start = False
        self.fail_send = False
        self.fail_stop = False

    def start(self):
        self.events.append("start")
        if self.fail_start:
            raise RuntimeError("定位指令无回复")

    def send_coordinate(self, x_px, y_px):
        self.events.append(("coord", x_px, y_px))
        if self.fail_send:
            raise RuntimeError("坐标发送失败")

    def stop(self):
        self.events.append("stop")
        if self.fail_stop:
            raise RuntimeError("停止失败")


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
        self.link = FakeLocateLink()
        self.actuator = FakeActuator()
        self.config = {
            "min_target_quality": 0.7,
            "total_timeout_s": 20.0,
        }

    def observation(self, x, y, valid=True, stable=True, quality=0.9,
                    width=200, height=160):
        return AlignmentObservation(
            x, y, valid, stable, quality, "material_circle",
            image_width=width, image_height=height)

    def controller(self, now=None):
        return PlacementAlignmentController(self.link, self.actuator, self.config, now=now)

    def test_start_once_then_stream_until_origin_and_release_once(self):
        controller = self.controller()
        controller.tick(self.observation(130, 50))
        controller.tick(self.observation(110, 90))
        self.assertEqual(self.link.events, ["start", ("coord", 30, -30), ("coord", 10, 10)])
        self.assertEqual(self.actuator.started, [])

        controller.tick(self.observation(100, 80))
        self.assertNotIn(("coord", 0, 0), self.link.events)
        self.assertEqual(self.link.events[-1], "stop")
        self.assertEqual(self.actuator.started, [(AlignmentAction.DESCEND, None)])
        self.assertTrue(controller.status().plane_locked)
        self.assertFalse(controller.needs_observation)

        controller.tick(self.observation(180, 20))
        self.assertEqual(self.link.events.count("start"), 1)
        self.assertEqual(self.link.events.count("stop"), 1)
        self.assertEqual([item for item in self.link.events if item != "start" and item != "stop"],
                         [("coord", 30, -30), ("coord", 10, 10)])

        for expected in (AlignmentAction.RELEASE, AlignmentAction.RETREAT):
            self.actuator.results.append(ActionResult.SUCCEEDED)
            controller.tick(self.observation(9999, 9999))
            controller.tick(self.observation(9999, 9999))
            self.assertEqual(self.actuator.started[-1][0], expected)
        self.actuator.results.append(ActionResult.SUCCEEDED)
        status = controller.tick(self.observation(0, 0))
        self.assertEqual(status.state, AlignmentState.COMPLETE)
        self.assertEqual([action for action, _ in self.actuator.started].count(
            AlignmentAction.RELEASE), 1)

    def test_unstable_target_does_not_send_coordinates_or_zero(self):
        controller = self.controller()
        controller.tick(self.observation(130, 50, stable=False))
        controller.tick(self.observation(130, 50, quality=0.2))
        controller.tick(None)
        self.assertEqual(self.link.events, ["start"])
        self.assertEqual(self.actuator.started, [])

    def test_already_centered_stops_without_coordinate_frame(self):
        controller = self.controller()
        controller.tick(self.observation(100.4, 79.6))
        self.assertEqual(self.link.events, ["start", "stop"])
        self.assertEqual(self.actuator.started[0][0], AlignmentAction.DESCEND)

    def test_start_failure_is_not_retried_and_does_not_stream(self):
        self.link.fail_start = True
        controller = self.controller()
        first = controller.tick(self.observation(130, 50))
        second = controller.tick(self.observation(130, 50))
        self.assertEqual(first.state, AlignmentState.FAULT)
        self.assertEqual(second.state, AlignmentState.FAULT)
        self.assertEqual(self.link.events, ["start"])

    def test_coordinate_failure_stops_positioning(self):
        self.link.fail_send = True
        controller = self.controller()
        status = controller.tick(self.observation(130, 50))
        self.assertEqual(status.state, AlignmentState.FAULT)
        self.assertEqual(self.link.events, ["start", ("coord", 30, -30), "stop"])
        self.assertEqual(self.actuator.started, [])

    def test_timeout_after_start_sends_stop(self):
        clock = {"now": 0.0}
        controller = self.controller(now=lambda: clock["now"])
        controller.tick(self.observation(130, 50))
        clock["now"] = 21.0
        status = controller.tick(self.observation(110, 90))
        self.assertEqual(status.state, AlignmentState.FAULT)
        self.assertEqual(self.link.events, ["start", ("coord", 30, -30), "stop"])

    def test_failed_descend_does_not_retry(self):
        controller = self.controller()
        controller.tick(self.observation(100, 80))
        self.actuator.results.append(ActionResult.FAILED)
        status = controller.tick()
        self.assertEqual(status.state, AlignmentState.FAULT)
        self.assertEqual(len(self.actuator.started), 1)


if __name__ == "__main__":
    unittest.main()
