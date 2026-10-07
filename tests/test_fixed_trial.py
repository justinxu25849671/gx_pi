"""固定单件试验的预检、顺序和记录测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys

from logistics_robot import app
from logistics_robot.fixed_trial import FixedTrialProfile, FixedTrialRunner
from logistics_robot.path_protocol import AckEvent, ArmDoneEvent, ArmState, DoneEvent, PathResult
from logistics_robot.trial_recorder import TrialRecorder


def complete_profile() -> dict:
    axis = lambda maximum, single, rpm: {
        "positive_motion": "confirmed positive",
        "negative_motion": "confirmed negative",
        "min_position_pulses": -maximum,
        "max_position_pulses": maximum,
        "max_single_move_pulses": single,
        "rpm": rpm,
        "acceleration": 100,
    }
    return {
        "version": 1,
        "name": "test_trial",
        "target_ring_id": 1,
        "material": {"table_top_height_mm": 140, "top_diameter_mm": 30,
                     "bottom_diameter_mm": 50, "height_mm": 60},
        "gripper": {"open_angle_deg": 110, "closed_angle_deg": 180, "wait_s": 0.01},
        "axes": {"5": axis(1200, 1200, 5), "6": axis(800, 1000, 30),
                 "7": axis(1200, 2000, 60)},
        "poses": {
            "observation": {"5": 0, "6": 0, "7": 0},
            "platform_pick": {"6": 200, "7": 300},
            "transport": {"6": 0, "7": 0},
            "rear_place": {"5": 400, "6": 150, "7": 200},
            "rear_pick": {"5": 400, "6": 160, "7": 210},
            "ring_place": {"5": -400, "6": 300, "7": 500},
            "ring_pick": {"5": -400, "6": 310, "7": 510},
        },
        "chassis_segments": [
            {"target_x_mm": 100, "target_y_mm": 0, "rpm": 20, "acceleration": 10},
            {"target_x_mm": 100, "target_y_mm": 200, "rpm": 20, "acceleration": 10},
        ],
    }


BASE_CONFIG = {"servos": {"rear": {"enabled": False}, "gripper": {"enabled": True}}}


class FakeClient:
    def __init__(self) -> None:
        self.commands = []
        self.sequence = 0
        self.encoder = {5: 1000, 6: 2000, 7: 3000}
        self.active_axis = None
        self.current_path = None
        self.last_path_start_sequence = None

    def _ack(self, item_id):
        self.sequence += 1
        return AckEvent(self.sequence, item_id, PathResult.OK, True)

    def arm_status(self, axis):
        return SimpleNamespace(encoder_count=self.encoder[axis], reverse=False,
                               state=ArmState.IDLE, result=PathResult.OK)

    def arm_move(self, axis, direction, pulses, rpm, acceleration):
        self.commands.append(("arm", axis, direction, pulses, rpm, acceleration))
        self.active_axis = axis
        self.encoder[axis] += direction * pulses
        return self._ack(axis)

    def wait_arm_done(self, axis, timeout_s):
        self.active_axis = None
        return ArmDoneEvent(self.sequence, axis, PathResult.OK, self.encoder[axis], False)

    def arm_stop(self, axis):
        self.commands.append(("arm_stop", axis))

    def set_servo_angle(self, servo_id, angle_tenths):
        self.commands.append(("servo", servo_id, angle_tenths))
        return self._ack(servo_id)

    def upload_and_start(self, path_id, points, reset_origin=True):
        self.commands.append(("path", path_id, tuple(points), reset_origin))
        self.current_path = (path_id, tuple(points)[0])
        self.last_path_start_sequence = self._ack(path_id).request_sequence

    def wait_until_terminal_with_keepalive(self):
        path_id, point = self.current_path
        return DoneEvent(path_id, 1, point.x_mm, point.y_mm)

    def stop(self):
        self.commands.append(("path_stop",))


class FixedTrialTests(unittest.TestCase):
    def _load(self, root: Path, raw: dict) -> FixedTrialProfile:
        path = root / "profile.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return FixedTrialProfile.load(path, BASE_CONFIG)

    def test_incomplete_profile_lists_steps_and_refuses_run(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = self._load(Path(directory), {
                "version": 1, "name": "incomplete", "target_ring_id": 1,
                "material": {"table_top_height_mm": 140, "top_diameter_mm": 30,
                             "bottom_diameter_mm": 50, "height_mm": 60},
                "gripper": {"open_angle_deg": 110, "closed_angle_deg": 180, "wait_s": None},
                "axes": {}, "poses": {}, "chassis_segments": [],
            })
            self.assertEqual(len(profile.steps), 39)
            self.assertTrue(profile.issues)
            with self.assertRaisesRegex(ValueError, "禁止运行"):
                profile.ensure_ready()

    def test_cli_incomplete_run_never_opens_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            raw = complete_profile()
            raw["poses"]["ring_place"]["7"] = None
            profile_path.write_text(json.dumps(raw), encoding="utf-8")
            argv = ["main.py", "--fixed-trial-run", "--profile", str(profile_path)]
            with patch.object(sys, "argv", argv), \
                 patch.object(app, "load_config", return_value={"serial": {}, **BASE_CONFIG}), \
                 patch.object(app, "SerialPathLink") as serial_link:
                self.assertEqual(app.main(), 2)
                serial_link.assert_not_called()

    def test_whole_table_checks_cumulative_target_and_single_move(self):
        raw = complete_profile()
        raw["axes"]["5"]["max_single_move_pulses"] = 500
        with tempfile.TemporaryDirectory() as directory:
            profile = self._load(Path(directory), raw)
            self.assertTrue(any("单次 800 脉冲超限" in issue for issue in profile.issues))

    def test_one_run_records_prepare_before_each_real_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = self._load(root, complete_profile())
            self.assertEqual(profile.issues, [])
            client = FakeClient()
            recorder = TrialRecorder(profile.path, profile.name, root / "records")
            runner = FixedTrialRunner(profile, client, recorder, input_fn=lambda _: "RUN",
                                      sleep_fn=lambda _: None)
            self.assertEqual(runner.run(), 0)
            lines = [json.loads(line) for line in
                     (recorder.output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            prepare = [line for line in lines if line["event"] == "准备执行"]
            results = [line for line in lines if line["event"] == "执行结果"]
            self.assertEqual(len(prepare), len(client.commands))
            self.assertEqual(len(results), len(client.commands))
            self.assertEqual([item[3] for item in client.commands if item[0] == "path"], [True, False])
            summary = json.loads((recorder.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["success"])
            self.assertFalse(summary["physical_pick_or_place_confirmed"])


if __name__ == "__main__":
    unittest.main()
