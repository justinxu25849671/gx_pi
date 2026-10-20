"""色块多圆拟合后只留一个圆，并和传输线程分开。"""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from logistics_robot.circle_select import CircleTracker, fit_blob_circles, recognize_frame
from logistics_robot.direct_stream import FrameSlot, StreamBudget
from logistics_robot.vision_locate import upper_circle_coordinate
from logistics_robot.vision_web import _CircleSender


HSV = {
    "red": [[0, 80, 50, 12, 255, 255]],
    "yellow": [[20, 80, 50, 35, 255, 255]],
    "blue": [[100, 80, 50, 130, 255, 255]],
    "green": [[40, 80, 50, 85, 255, 255]],
}
DETECTION = {"min_area_px": 150, "max_area_px": 100000, "stable_frames": 1,
             "stable_center_tolerance_px": 8, "morphology_kernel_px": 5}


def disk(inner=False):
    frame = np.zeros((300, 400, 3), np.uint8)
    cv2.circle(frame, (190, 115), 45, (0, 0, 240), -1)
    if inner:
        cv2.circle(frame, (190, 115), 22, (40, 40, 255), 3)
    return frame


class CircleSelectTests(unittest.TestCase):
    def test_largest_blob_keeps_one_circle_after_several_fits(self):
        frame = disk(inner=True)
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (0, 80, 50), (12, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        proposals, winner = fit_blob_circles(frame, contour)
        self.assertGreaterEqual(len(proposals), 2)
        self.assertIsNotNone(winner)
        self.assertLess(np.hypot(winner.x - 190, winner.y - 115), 8)

    def test_tracker_sends_only_after_stable_frames_and_resets_when_lost(self):
        tracker = CircleTracker(stable_frames=2)
        detection = dict(DETECTION, stable_frames=2)
        first = recognize_frame(disk(), ("red",), HSV, detection, tracker)
        self.assertEqual(len(first), 1)
        self.assertTrue(first[0].top_valid)
        self.assertFalse(first[0].stable)
        self.assertIsNone(upper_circle_coordinate(first, 400, 300))
        second = recognize_frame(disk(), ("red",), HSV, detection, tracker)
        point = upper_circle_coordinate(second, 400, 300)
        self.assertIsNotNone(point)
        self.assertLess(abs(point[0] - (190 - 200)), 8)
        self.assertLess(abs(point[1] - (115 - 150)), 8)
        sender = _CircleSender({})
        client = Mock()
        sender._link = Mock(path_client=client)
        sender._started = True
        sender.update("red", point)
        client.send_visual_coordinate.assert_called_once()
        blank = np.zeros((300, 400, 3), np.uint8)
        held = recognize_frame(blank, ("red",), HSV, detection, tracker)
        self.assertEqual(held[0].track_id, second[0].track_id)
        self.assertIsNotNone(upper_circle_coordinate(held, 400, 300))
        last = held
        for _ in range(tracker.hold_frames):
            last = recognize_frame(blank, ("red",), HSV, detection, tracker)
        self.assertIsNone(upper_circle_coordinate(last, 400, 300))
        sender.update("red", None)
        self.assertEqual(client.send_visual_coordinate.call_count, 1)

    def test_locked_circle_is_followed_in_a_small_window(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        cv2.circle(frame, (640, 360), 70, (0, 0, 240), -1)
        detection = dict(DETECTION, min_area_px=400, stable_frames=1, detect_max_width_px=2000,
                         track_motion_px=48)
        tracker = CircleTracker(stable_frames=1)
        widths: list[int] = []
        import logistics_robot.circle_select as circle_select
        original = circle_select.color_mask

        def counting(image, bounds, kernel_px=5):
            widths.append(int(image.shape[1]))
            return original(image, bounds, kernel_px)

        with patch.object(circle_select, "color_mask", counting):
            first = recognize_frame(frame, ("red",), HSV, detection, tracker)
            moved = frame.copy()
            moved[:] = 0
            cv2.circle(moved, (670, 360), 70, (0, 0, 240), -1)
            second = recognize_frame(moved, ("red",), HSV, detection, tracker)
        self.assertTrue(first[0].top_valid)
        self.assertEqual(second[0].track_id, first[0].track_id)
        self.assertTrue(second[0].stable)
        self.assertGreater(widths[0], 1000)
        self.assertLess(widths[-1], 500)
        self.assertGreater(second[0].circle_center_px[0], 655)
        lost = recognize_frame(np.zeros((720, 1280, 3), np.uint8), ("red",), HSV, detection, tracker)
        self.assertEqual(lost[0].track_id, first[0].track_id)
        back = recognize_frame(moved, ("red",), HSV, detection, tracker)
        self.assertEqual(back[0].track_id, first[0].track_id)
        self.assertTrue(back[0].stable)

    def test_fast_move_stays_on_the_same_track(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        cv2.circle(frame, (400, 360), 70, (0, 0, 240), -1)
        detection = dict(DETECTION, min_area_px=400, stable_frames=1)
        tracker = CircleTracker(stable_frames=1)
        first = recognize_frame(frame, ("red",), HSV, detection, tracker)
        moved = np.zeros_like(frame)
        cv2.circle(moved, (560, 390), 70, (0, 0, 240), -1)
        jumped = recognize_frame(moved, ("red",), HSV, detection, tracker)
        self.assertEqual(jumped[0].track_id, first[0].track_id)
        self.assertLess(abs(jumped[0].circle_center_px[0] - 400), 24)
        second = recognize_frame(moved, ("red",), HSV, detection, tracker)
        self.assertEqual(second[0].track_id, first[0].track_id)
        self.assertGreater(second[0].circle_center_px[0], 540)
        self.assertGreater(second[0].circle_center_px[1], 375)

    def test_area_jump_keeps_the_previous_circle_for_one_frame(self):
        frame = np.zeros((320, 400, 3), np.uint8)
        cv2.circle(frame, (200, 150), 60, (0, 0, 240), -1)
        detection = dict(DETECTION, min_area_px=80, stable_frames=1)
        tracker = CircleTracker(stable_frames=1)
        first = recognize_frame(frame, ("red",), HSV, detection, tracker)
        first_radius = dict(first[0].top_metrics)["radius_px"]
        shrunk = np.zeros_like(frame)
        cv2.circle(shrunk, (200, 150), 18, (0, 0, 240), -1)
        jumped = recognize_frame(shrunk, ("red",), HSV, detection, tracker)
        jumped_radius = dict(jumped[0].top_metrics)["radius_px"]
        self.assertEqual(jumped[0].track_id, first[0].track_id)
        self.assertGreater(jumped_radius, first_radius * 0.7)
        kept = recognize_frame(shrunk, ("red",), HSV, detection, tracker)
        kept_radius = dict(kept[0].top_metrics)["radius_px"]
        self.assertEqual(kept[0].track_id, first[0].track_id)
        self.assertLess(kept_radius, 30)
        mild = np.zeros_like(frame)
        cv2.circle(mild, (200, 150), 24, (0, 0, 240), -1)
        followed = recognize_frame(mild, ("red",), HSV, detection, tracker)
        followed_radius = dict(followed[0].top_metrics)["radius_px"]
        self.assertEqual(followed[0].track_id, first[0].track_id)
        self.assertLess(abs(followed_radius - 24), 10)

    def test_lighter_smaller_circle_is_kept(self):
        frame = np.zeros((320, 400, 3), np.uint8)
        cv2.circle(frame, (200, 160), 50, (0, 0, 170), -1)
        cv2.circle(frame, (200, 145), 20, (80, 80, 255), -1)
        detection = dict(DETECTION, min_area_px=100, stable_frames=1)
        found = recognize_frame(frame, ("red",), HSV, detection, CircleTracker(stable_frames=1))
        x_px, y_px = found[0].circle_center_px
        radius = dict(found[0].top_metrics)["radius_px"]
        self.assertLess(abs(x_px - 200), 8)
        self.assertLess(abs(y_px - 145), 12)
        self.assertLess(radius, 30)

    def test_gamut_circle_stays_on_the_round_top(self):
        frame = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(frame, (200, 80), 36, (0, 0, 230), -1)
        cv2.rectangle(frame, (176, 80), (224, 180), (0, 0, 210), -1)
        detection = dict(DETECTION, min_area_px=100, stable_frames=1)
        found = recognize_frame(frame, ("red",), HSV, detection, CircleTracker(stable_frames=1))
        x_px, y_px = found[0].circle_center_px
        radius = dict(found[0].top_metrics)["radius_px"]
        self.assertLess(abs(x_px - 200), 10)
        self.assertLess(y_px, 100)
        self.assertLess(radius, 48)

    def test_top_layer_ignores_the_larger_body(self):
        frame = np.zeros((360, 400, 3), np.uint8)
        cv2.rectangle(frame, (160, 90), (240, 280), (0, 0, 210), -1)
        cv2.ellipse(frame, (200, 90), (58, 22), 0, 0, 360, (0, 0, 230), -1)
        detection = dict(DETECTION, min_area_px=100, stable_frames=1)
        found = recognize_frame(frame, ("red",), HSV, detection, CircleTracker(stable_frames=1))
        x_px, y_px = found[0].circle_center_px
        radius = dict(found[0].top_metrics)["radius_px"]
        self.assertLess(abs(x_px - 200), 12)
        self.assertLess(abs(y_px - 90), 18)
        self.assertGreater(radius, 45)

    def test_stream_budget_drops_the_frame_that_exceeds_the_peak(self):
        budget = StreamBudget(20, 32, 70)
        self.assertTrue(budget.accept(8 * 1024))
        self.assertFalse(budget.accept(40 * 1024))
        self.assertGreaterEqual(budget.dropped, 1)
        self.assertLess(budget.encode_quality(), 70)

    def test_slow_reader_does_not_block_a_newer_frame(self):
        slot = FrameSlot()
        started = threading.Event()

        def slow():
            started.set()
            time.sleep(0.2)
            item = slot.wait_next(0, 0.2)
            self.assertIsNotNone(item)
            self.assertEqual(item[0], 2)

        worker = threading.Thread(target=slow)
        worker.start()
        self.assertTrue(started.wait(0.5))
        began = time.perf_counter()
        slot.publish(np.zeros((8, 8, 3), np.uint8))
        slot.publish(np.zeros((8, 8, 3), np.uint8))
        self.assertLess(time.perf_counter() - began, 0.1)
        worker.join(1)
        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
