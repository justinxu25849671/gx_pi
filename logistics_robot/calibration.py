"""视觉像素误差到机构小范围修正量的显式标定模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PoseCalibration:
    """一个固定观察姿态下的二维局部线性标定。"""

    name: str
    calibrated: bool
    observation_height_mm: float
    rotation_deg: float
    landing_reference_px: tuple[float, float]
    pixel_to_motion_mm: tuple[tuple[float, float], tuple[float, float]]

    @classmethod
    def from_config(cls, name: str, raw: Mapping[str, Any]) -> "PoseCalibration":
        reference = raw.get("landing_reference_px", [0.0, 0.0])
        matrix = raw.get("pixel_to_motion_mm", [[0.0, 0.0], [0.0, 0.0]])
        if len(reference) != 2 or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError(f"标定姿态 {name} 的参考点或二维映射格式错误")
        return cls(
            name=name,
            calibrated=bool(raw.get("calibrated", False)),
            observation_height_mm=float(raw.get("observation_height_mm", 0.0)),
            rotation_deg=float(raw.get("rotation_deg", 0.0)),
            landing_reference_px=(float(reference[0]), float(reference[1])),
            pixel_to_motion_mm=(
                (float(matrix[0][0]), float(matrix[0][1])),
                (float(matrix[1][0]), float(matrix[1][1])),
            ),
        )

    def correction_mm(self, target_px: tuple[float, float]) -> tuple[float, float]:
        """输出 ``(整车横移, 机械臂伸缩)`` 修正量；未标定时拒绝计算。"""
        if not self.calibrated:
            raise RuntimeError(f"视觉姿态 {self.name} 尚未标定，禁止生成机构修正量")
        dx = float(target_px[0]) - self.landing_reference_px[0]
        dy = float(target_px[1]) - self.landing_reference_px[1]
        matrix = self.pixel_to_motion_mm
        lateral = matrix[0][0] * dx + matrix[0][1] * dy
        longitudinal = matrix[1][0] * dx + matrix[1][1] * dy
        return lateral, longitudinal


class VisualCalibration:
    """按取料、第一层放置和第二层码垛分别保存姿态标定。"""

    def __init__(self, profiles: Mapping[str, PoseCalibration]) -> None:
        self._profiles = dict(profiles)

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "VisualCalibration":
        profiles = raw.get("poses", {})
        return cls({name: PoseCalibration.from_config(name, value)
                    for name, value in profiles.items()})

    def require(self, name: str) -> PoseCalibration:
        try:
            profile = self._profiles[name]
        except KeyError as exc:
            raise ValueError(f"缺少视觉姿态标定：{name}") from exc
        if not profile.calibrated:
            raise RuntimeError(f"视觉姿态 {name} 尚未标定")
        return profile


class FrameRectifier:
    """可选的镜头去畸变和目标平面单应映射。"""

    def __init__(self, raw: Mapping[str, Any], cv2_module=None) -> None:
        self._raw = dict(raw)
        self._enabled = bool(raw.get("apply", False))
        self._calibrated = bool(raw.get("calibrated", False))
        self._cv2 = cv2_module
        if self._enabled and not self._calibrated:
            raise RuntimeError("相机校正被启用但尚未标定")

    def apply(self, frame):
        """按配置先去畸变，再映射到标定目标平面。"""
        if not self._enabled:
            return frame
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("相机校正需要 OpenCV") from exc
        else:
            cv2 = self._cv2
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("相机校正需要 numpy") from exc
        matrix = np.asarray(self._raw["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(self._raw["distortion_coefficients"], dtype=np.float64)
        homography = np.asarray(self._raw["plane_homography"], dtype=np.float64)
        if matrix.shape != (3, 3) or homography.shape != (3, 3):
            raise ValueError("camera_matrix 和 plane_homography 必须是 3x3")
        rectified = cv2.undistort(frame, matrix, distortion)
        height, width = rectified.shape[:2]
        output_size = self._raw.get("output_size", [width, height])
        return cv2.warpPerspective(rectified, homography,
                                   (int(output_size[0]), int(output_size[1])))
