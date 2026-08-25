"""取放机构动作的统一接口。

@File    : logistics_robot/mechanism.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 管理取放动作及其联调阶段的完成判定。
"""

from __future__ import annotations

import time
from typing import Mapping

from .model import MechanismAction, MissionStep


class TimedMechanism:
    """联调阶段的动作定时器；正式比赛必须替换为真实反馈确认。"""

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
        """动作结束即返回成功；接入传感器后在这里返回真实复核结果。"""
        if self._active is None or time.monotonic() < self._finish_at:
            return False
        self._active = None
        return True

    @property
    def busy(self) -> bool:
        """动作尚未完成时为真，供后续反馈型机构实现复用。"""
        return self._active is not None
