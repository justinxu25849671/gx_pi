"""视觉标定矩阵与安全配置测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

from logistics_robot.calibration import FrameRectifier, PoseCalibration, VisualCalibration
from logistics_robot.config import load_config


class CalibrationConfigTests(unittest.TestCase):
    def test_cross_axis_mapping_is_explicit(self):
        profile = PoseCalibration("test", True, 10, 5, (100, 200),
                                  ((0.0, 0.5), (-0.25, 0.0)))
        self.assertEqual(profile.correction_mm((108, 204)), (2.0, -2.0))

    def test_unconfigured_profile_cannot_be_required(self):
        calibration = VisualCalibration.from_config({"poses": {
            "placement_layer1": {
                "calibrated": False, "landing_reference_px": [0, 0],
                "pixel_to_motion_mm": [[0, 0], [0, 0]],
            }
        }})
        with self.assertRaisesRegex(RuntimeError, "未标定"):
            calibration.require("placement_layer1")

    def test_disabled_frame_rectifier_is_identity_without_opencv(self):
        marker = object()
        self.assertIs(FrameRectifier({"apply": False}).apply(marker), marker)

    def test_enabled_uncalibrated_frame_rectifier_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "尚未标定"):
            FrameRectifier({"apply": True, "calibrated": False})

    def test_project_configs_are_valid_and_locked(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("config.json", "config.example.json"):
            config = load_config(root / name)
            self.assertEqual(set(config["hsv_colors"]),
                             {"red", "yellow", "blue", "green"})
            self.assertFalse(config["ring_detection"]["calibrated"])
            self.assertTrue(config["mechanism"]["production_feedback_required"])


if __name__ == "__main__":
    unittest.main()
