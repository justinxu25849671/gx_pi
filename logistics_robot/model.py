"""项目中共享的无硬件依赖数据类型。

@File    : logistics_robot/model.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 定义位姿、速度、视觉和任务等共享模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class DetectedObject:
    """视觉模块输出的单个颜色物料候选。"""
    color: str
    center_x_px: float
    center_y_px: float
    area_px: float
    bbox: tuple[int, int, int, int] | None = None
    contour: tuple[tuple[int, int], ...] = ()
    aspect_ratio: float = 0.0
    solidity: float = 0.0
    extent: float = 0.0
    reference_x_px: float | None = None
    reference_y_px: float | None = None
    stable: bool = False
    stable_frames: int = 0
    confidence: float = 0.0

    @property
    def reference_point_px(self) -> tuple[float, float]:
        """返回经过配置偏移修正的抓取参考点，而不是默认使用颜色重心。"""
        return (
            self.center_x_px if self.reference_x_px is None else self.reference_x_px,
            self.center_y_px if self.reference_y_px is None else self.reference_y_px,
        )


class MechanismAction(str, Enum):
    """赛题中允许的机构动作类型。"""
    PICK_FROM_RAW = "PICK_FROM_RAW"
    PLACE_TO_COARSE = "PLACE_TO_COARSE"
    PICK_FROM_COARSE = "PICK_FROM_COARSE"
    PLACE_TO_TEMPORARY = "PLACE_TO_TEMPORARY"
    STACK_TO_TEMPORARY = "STACK_TO_TEMPORARY"


@dataclass(frozen=True)
class MissionStep:
    """一项不可拆分的导航到区、对准和机构取/放任务。"""
    zone: str
    action: MechanismAction
    color: str
    position: int
    batch: int
