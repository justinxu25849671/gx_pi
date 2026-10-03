"""JSON 配置读取。默认配置不会允许真实车辆运动。

@File    : logistics_robot/config.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 读取并做基础结构校验的运行配置。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .vision import MATERIAL_COLORS


def _validate_vision(config: dict[str, Any]) -> None:
    if set(config.get("hsv_colors", {})) != set(MATERIAL_COLORS):
        raise ValueError("hsv_colors 必须且只能包含 red/yellow/blue/green")
    camera = config.get("object_camera", {})
    detection = camera.get("object_detection", {})
    if not detection:
        raise ValueError("object_camera 缺少 object_detection 配置")
    if float(detection.get("min_area_px", 0)) <= 0:
        raise ValueError("min_area_px 必须大于 0")
    if float(detection.get("max_area_px", 0)) <= float(detection["min_area_px"]):
        raise ValueError("max_area_px 必须大于 min_area_px")
    ring = config.get("ring_detection", {})
    if ring.get("material_base_diameter_mm") != 50:
        raise ValueError("当前物料底径配置应为已确认的 50 mm")
    if ring.get("ring_outer_diameters_mm") != [53, 58, 65, 75, 85, 95]:
        raise ValueError("环外径配置必须完整保留 53/58/65/75/85/95 mm")
    alignment = config.get("placement_alignment", {})
    for name in ("lateral_tolerance_mm", "longitudinal_tolerance_mm",
                 "max_correction_step_mm", "total_timeout_s"):
        if float(alignment.get(name, 0)) <= 0:
            raise ValueError(f"placement_alignment.{name} 必须大于 0")
    for name in ("max_corrections", "required_aligned_frames"):
        if int(alignment.get(name, 0)) <= 0:
            raise ValueError(f"placement_alignment.{name} 必须大于 0")
    camera_calibration = config.get("vision_calibration", {}).get("camera", {})
    if camera_calibration.get("apply") and not camera_calibration.get("calibrated"):
        raise ValueError("vision_calibration.camera 未标定时不能启用 apply")


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置，并检查应用启动所需的顶层字段。

    这里只校验结构完整性；坐标、HSV、串口等数值是否正确需在各硬件阶段
    单独标定和验证。
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"serial", "node_route", "object_camera", "qr_camera",
                "hsv_colors", "ring_detection", "vision_calibration",
                "placement_alignment"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(sorted(missing))}")
    _validate_vision(config)
    return config
