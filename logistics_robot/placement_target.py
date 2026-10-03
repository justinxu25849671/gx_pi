"""第一层环心与第二层同色承接物料的放置目标选择。"""

from __future__ import annotations

from .alignment import AlignmentObservation
from .model import DetectedObject, MechanismAction, MissionStep
from .ring_detection import RingCandidate


def resolve_placement_target(step: MissionStep, rings: list[RingCandidate],
                             objects: list[DetectedObject]) -> AlignmentObservation:
    """在本次松爪前解析目标；返回无效结果时状态机保持夹持。"""
    if step.action not in {
        MechanismAction.PLACE_TO_COARSE,
        MechanismAction.PLACE_TO_TEMPORARY,
        MechanismAction.STACK_TO_TEMPORARY,
    }:
        return AlignmentObservation(0.0, 0.0, False, False, 0.0, "none",
                                    "当前任务不是放置动作")

    if step.action == MechanismAction.STACK_TO_TEMPORARY:
        # 第二层位置必须来自当前画面中的第一层同色物料，不照搬第二批粗加工环位。
        candidates = [item for item in objects if item.color == step.color]
        if not candidates:
            return AlignmentObservation(0.0, 0.0, False, False, 0.0,
                                        "support_object", "未识别到第一层同色承接物料")
        stable = [item for item in candidates if item.stable]
        if len(stable) != 1:
            return AlignmentObservation(0.0, 0.0, False, False, 0.0,
                                        "support_object", "同色承接物料不稳定或候选不唯一")
        item = stable[0]
        x, y = item.reference_point_px
        return AlignmentObservation(x, y, True, True, item.confidence,
                                    "support_object")

    candidates = [item for item in rings if item.position == step.position]
    if len(candidates) != 1:
        return AlignmentObservation(0.0, 0.0, False, False, 0.0,
                                    "ring", "目标环位缺失或候选不唯一")
    ring = candidates[0]
    return AlignmentObservation(ring.center_x_px, ring.center_y_px,
                                ring.valid, ring.stable, ring.quality,
                                "ring", ring.reason)
