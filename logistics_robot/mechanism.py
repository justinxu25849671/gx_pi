"""取放机构动作的统一接口。

@File    : logistics_robot/mechanism.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 管理取放动作及其联调阶段的完成判定。
"""

from __future__ import annotations

import time
from typing import Mapping

from .alignment import ActionResult, AlignmentAction
from .model import MechanismAction, MissionStep


class FeedbackMechanism:
    """放置对准需要实现的真实机构接口。

    子类必须把横移、伸缩、下降、松爪和撤离映射到已有 STM32/机构协议，并从
    下位机完成/失败事件返回结果。当前仓库没有这些协议，基类因此默认拒绝执行。
    """

    # 真实适配器完成 start/poll 后必须显式覆盖为 True。
    feedback_capable = False

    def start(self, action: AlignmentAction, value: float | None = None) -> None:
        raise RuntimeError(f"尚未接入 {action.value} 的真实机构命令与完成反馈")

    def poll(self, action: AlignmentAction) -> ActionResult:
        raise RuntimeError(f"尚未接入 {action.value} 的真实机构完成反馈")


class TimedMechanism:
    """旧联调定时器；禁止作为自动对准放置的完成反馈。"""

    feedback_capable = False

    def __init__(self, config: Mapping[str, float | bool]) -> None:
        self._config = config
        self._active: MissionStep | None = None
        self._finish_at = 0.0

    def start(self, step: MissionStep) -> None:
        """开始当前机构动作。"""
        if self._active is not None:
            raise RuntimeError("机构尚有未完成动作")
        duration_key = {
            MechanismAction.PICK_FROM_RAW: "pick_duration_s",
            MechanismAction.PICK_FROM_COARSE: "transfer_duration_s",
            MechanismAction.PLACE_TO_COARSE: "place_duration_s",
            MechanismAction.PLACE_TO_TEMPORARY: "place_duration_s",
            MechanismAction.STACK_TO_TEMPORARY: "place_duration_s",
        }[step.action]
        self._active = step
        self._finish_at = time.monotonic() + float(self._config[duration_key])

    def complete(self) -> bool:
        """仅返回占位计时结束；它不等于传感器或下位机动作完成反馈。"""
        if self._active is None or time.monotonic() < self._finish_at:
            return False
        self._active = None
        return True

    @property
    def busy(self) -> bool:
        """动作尚未完成时为真，供后续反馈型机构实现复用。"""
        return self._active is not None
