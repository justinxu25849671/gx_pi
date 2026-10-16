"""按当前任务颜色选择物料圆心，供画面中心坐标系定位使用。"""

from __future__ import annotations

from .alignment import AlignmentObservation
from .model import DetectedObject, MechanismAction, MissionStep
from .ring_detection import RingCandidate


def resolve_placement_target(step: MissionStep, rings: list[RingCandidate],
                             objects: list[DetectedObject],
                             image_size: tuple[int, int] = (0, 0)) -> AlignmentObservation:
    """解析本次定位使用的物料圆心。

    返回的像素是画面绝对坐标，控制器再把它换成以画面中心为原点的整数坐标。
    ``rings`` 保留调用签名，定位坐标不取环心。
    """
    width, height = int(image_size[0]), int(image_size[1])
    if step.action not in {
        MechanismAction.PLACE_TO_COARSE,
        MechanismAction.PLACE_TO_TEMPORARY,
        MechanismAction.STACK_TO_TEMPORARY,
    }:
        return AlignmentObservation(0.0, 0.0, False, False, 0.0, "none",
                                    "当前任务不是放置动作", width, height)

    candidates = [item for item in objects if item.color == step.color]
    if not candidates:
        return AlignmentObservation(0.0, 0.0, False, False, 0.0, "material_circle",
                                    "未识别到目标颜色物料", width, height)
    stable = [item for item in candidates if item.stable]
    if len(stable) != 1:
        return AlignmentObservation(0.0, 0.0, False, False, 0.0, "material_circle",
                                    "目标颜色物料不稳定或候选不唯一", width, height)
    item = stable[0]
    x, y = item.circle_center_px
    return AlignmentObservation(x, y, True, True, item.confidence, "material_circle",
                                "", width, height)
