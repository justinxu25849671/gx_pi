"""松爪前闭环对准和一次性放置状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Mapping, Protocol

from .calibration import PoseCalibration


class AlignmentAction(str, Enum):
    LATERAL = "LATERAL"
    LONGITUDINAL = "LONGITUDINAL"
    DESCEND = "DESCEND"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"


class ActionResult(str, Enum):
    BUSY = "BUSY"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class AlignmentState(str, Enum):
    ACQUIRE = "ACQUIRE"
    WAIT_ACTION = "WAIT_ACTION"
    DESCEND = "DESCEND"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    COMPLETE = "COMPLETE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class AlignmentObservation:
    """当前帧的放置目标；第二层应来自已有同色物料。"""

    target_x_px: float
    target_y_px: float
    valid: bool
    stable: bool
    quality: float
    source: str
    reason: str = ""


@dataclass(frozen=True)
class AlignmentStatus:
    state: AlignmentState
    pending_action: AlignmentAction | None
    correction_count: int
    aligned_frames: int
    plane_locked: bool
    release_started: bool
    released: bool
    message: str


class FeedbackActuator(Protocol):
    """真实机构适配器；每个动作必须返回独立完成/失败反馈。"""

    def start(self, action: AlignmentAction, value: float | None = None) -> None: ...

    def poll(self, action: AlignmentAction) -> ActionResult: ...


class PlacementAlignmentController:
    """只在松爪前使用视觉，且一次只驱动一个执行机构。"""

    def __init__(self, calibration: PoseCalibration, actuator: FeedbackActuator,
                 config: Mapping[str, Any], now=None) -> None:
        if not calibration.calibrated:
            raise RuntimeError("放置姿态未标定，禁止启动自动放置")
        self._calibration = calibration
        self._actuator = actuator
        self._lateral_tolerance = float(config["lateral_tolerance_mm"])
        self._longitudinal_tolerance = float(config["longitudinal_tolerance_mm"])
        self._max_step = float(config["max_correction_step_mm"])
        self._max_corrections = int(config["max_corrections"])
        self._required_frames = int(config["required_aligned_frames"])
        self._min_quality = float(config["min_target_quality"])
        self._timeout = float(config["total_timeout_s"])
        self._now = now or time.monotonic
        self._started_at = self._now()
        self._state = AlignmentState.ACQUIRE
        self._pending: AlignmentAction | None = None
        self._corrections = 0
        self._aligned_frames = 0
        self._plane_locked = False
        self._release_started = False
        self._released = False
        self._message = "等待稳定目标"

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @property
    def needs_observation(self) -> bool:
        """平面锁定后始终为假，防止下降/松爪后继续视觉复查。"""
        return self._state == AlignmentState.ACQUIRE and not self._plane_locked

    def _fault(self, message: str) -> None:
        self._state = AlignmentState.FAULT
        self._pending = None
        self._message = message

    def _start(self, action: AlignmentAction, value: float | None = None) -> None:
        if self._pending is not None:
            raise RuntimeError("已有动作未完成，禁止并行启动另一个执行机构")
        if action == AlignmentAction.RELEASE:
            if self._release_started:
                raise RuntimeError("本次放置已发出过松爪命令，禁止重试")
            self._release_started = True
        try:
            self._actuator.start(action, value)
        except Exception as exc:
            self._fault(f"{action.value} 启动失败：{exc}")
            return
        self._pending = action
        self._state = AlignmentState.WAIT_ACTION
        self._message = f"等待 {action.value} 完成反馈"

    def _after_action(self, action: AlignmentAction) -> None:
        if action in {AlignmentAction.LATERAL, AlignmentAction.LONGITUDINAL}:
            self._aligned_frames = 0
            self._state = AlignmentState.ACQUIRE
            self._message = "动作完成，重新识别目标"
        elif action == AlignmentAction.DESCEND:
            self._state = AlignmentState.RELEASE
            self._message = "下降完成，准备松爪一次"
        elif action == AlignmentAction.RELEASE:
            self._released = True
            self._state = AlignmentState.RETREAT
            self._message = "已松爪，后续仅允许撤离"
        elif action == AlignmentAction.RETREAT:
            self._state = AlignmentState.COMPLETE
            self._message = "一次性放置完成"

    def tick(self, observation: AlignmentObservation | None = None) -> AlignmentStatus:
        """推进一个事件周期；任何机构动作均等待显式完成反馈。"""
        if self._state in {AlignmentState.COMPLETE, AlignmentState.FAULT}:
            return self.status()
        if self._now() - self._started_at > self._timeout:
            self._fault("放置流程总超时" if not self._released else "松爪后撤离超时")
            return self.status()

        if self._pending is not None:
            action = self._pending
            try:
                result = self._actuator.poll(action)
            except Exception as exc:
                self._fault(f"{action.value} 完成反馈读取失败：{exc}")
                return self.status()
            if result == ActionResult.BUSY:
                return self.status()
            self._pending = None
            if result == ActionResult.FAILED:
                self._fault(f"{action.value} 动作失败；禁止自动重试")
                return self.status()
            self._after_action(action)
            return self.status()

        if self._state == AlignmentState.RELEASE:
            self._start(AlignmentAction.RELEASE)
            return self.status()
        if self._state == AlignmentState.RETREAT:
            self._start(AlignmentAction.RETREAT)
            return self.status()
        if self._state != AlignmentState.ACQUIRE:
            self._fault("非法对准状态")
            return self.status()

        if observation is None or not observation.valid or not observation.stable:
            self._aligned_frames = 0
            self._message = "目标丢失或尚未稳定，保持夹持"
            return self.status()
        if observation.quality < self._min_quality:
            self._aligned_frames = 0
            self._message = "目标质量不足，保持夹持"
            return self.status()
        if not math.isfinite(observation.target_x_px + observation.target_y_px):
            self._fault("目标坐标非法")
            return self.status()

        lateral, longitudinal = self._calibration.correction_mm(
            (observation.target_x_px, observation.target_y_px))
        if abs(lateral) > self._lateral_tolerance:
            if self._corrections >= self._max_corrections:
                self._fault("横移/伸缩修正次数已达上限")
            else:
                self._corrections += 1
                self._start(AlignmentAction.LATERAL, self._clamp(lateral, self._max_step))
            return self.status()
        if abs(longitudinal) > self._longitudinal_tolerance:
            if self._corrections >= self._max_corrections:
                self._fault("横移/伸缩修正次数已达上限")
            else:
                self._corrections += 1
                self._start(AlignmentAction.LONGITUDINAL,
                            self._clamp(longitudinal, self._max_step))
            return self.status()

        self._aligned_frames += 1
        self._message = f"对准稳定 {self._aligned_frames}/{self._required_frames} 帧"
        if self._aligned_frames >= self._required_frames:
            self._plane_locked = True
            self._state = AlignmentState.DESCEND
            self._start(AlignmentAction.DESCEND)
        return self.status()

    def status(self) -> AlignmentStatus:
        return AlignmentStatus(
            state=self._state,
            pending_action=self._pending,
            correction_count=self._corrections,
            aligned_frames=self._aligned_frames,
            plane_locked=self._plane_locked,
            release_started=self._release_started,
            released=self._released,
            message=self._message,
        )
