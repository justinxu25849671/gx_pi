"""画面中心坐标系下的视觉定位。

定位阶段只发送地面站同款底盘位置命令：前进、后退、左移、右移。
这个阶段不发送机械臂命令。底盘进入容差后定位结束，不下降、不松爪、不撤离。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Any, Mapping, Protocol

from .vision import image_origin_px


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


_KIND_TEXT = {
    "fwdmm": "前进",
    "bwdmm": "后退",
    "leftmm": "左移",
    "rightmm": "右移",
}


@dataclass(frozen=True)
class AlignmentObservation:
    """当前帧物料圆心。``target_*`` 是画面绝对像素，不是中心坐标系。"""

    target_x_px: float
    target_y_px: float
    valid: bool
    stable: bool
    quality: float
    source: str
    reason: str = ""
    image_width: int = 0
    image_height: int = 0


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


class ChassisMotionLink(Protocol):
    """底盘前后左右。实现不得在这些方法里发送机械臂命令。"""

    def move(self, kind: str, distance_mm: int) -> None: ...

    def poll(self) -> ActionResult: ...

    def stop(self) -> None: ...


class PlacementAlignmentController:
    """用底盘位置步进把物料圆心收进容差，到位后结束定位。"""

    def __init__(self, link: ChassisMotionLink, config: Mapping[str, Any], now=None) -> None:
        self._link = link
        self._min_quality = float(config["min_target_quality"])
        self._timeout = float(config["total_timeout_s"])
        self._lateral_tolerance = float(config["lateral_tolerance_mm"])
        self._longitudinal_tolerance = float(config["longitudinal_tolerance_mm"])
        self._max_step = float(config["max_correction_step_mm"])
        self._max_corrections = int(config["max_corrections"])
        self._required_frames = int(config["required_aligned_frames"])
        self._mm_per_px_x = float(config["mm_per_px_x"])
        self._mm_per_px_y = float(config["mm_per_px_y"])
        if config["x_positive"] not in ("left", "right"):
            raise ValueError("x_positive 只能是 left 或 right")
        if config["y_positive"] not in ("forward", "backward"):
            raise ValueError("y_positive 只能是 forward 或 backward")
        self._x_sign = 1.0 if config["x_positive"] == "right" else -1.0
        self._y_sign = 1.0 if config["y_positive"] == "forward" else -1.0
        self._now = now or time.monotonic
        self._started_at = self._now()
        self._state = AlignmentState.ACQUIRE
        self._chassis_busy = False
        self._corrections = 0
        self._aligned_frames = 0
        self._plane_locked = False
        self._message = "等待物料圆心"

    @property
    def needs_observation(self) -> bool:
        """定位结束后为假，不再根据画面继续调底盘。"""
        return self._state == AlignmentState.ACQUIRE and not self._plane_locked

    def _fault(self, message: str, *, stop: bool = True) -> None:
        if stop:
            extra = self._stop_chassis()
            if extra:
                message = f"{message}；{extra}"
        self._state = AlignmentState.FAULT
        self._chassis_busy = False
        self._message = message

    def _stop_chassis(self) -> str | None:
        try:
            self._link.stop()
        except Exception as exc:
            return f"停止底盘失败：{exc}"
        return None

    def _offsets_mm(self, x_px: int, y_px: int) -> tuple[float, float]:
        """返回 ``(向右毫米, 向前毫米)``。符号由配置的画面正方向决定。"""
        right = x_px * self._mm_per_px_x * self._x_sign
        forward = y_px * self._mm_per_px_y * self._y_sign
        return right, forward

    def _chassis_command(self, right_mm: float, forward_mm: float) -> tuple[str, int] | None:
        lateral_out = abs(right_mm) > self._lateral_tolerance
        longitudinal_out = abs(forward_mm) > self._longitudinal_tolerance
        if not lateral_out and not longitudinal_out:
            return None
        if lateral_out and (not longitudinal_out or abs(right_mm) >= abs(forward_mm)):
            kind = "rightmm" if right_mm > 0 else "leftmm"
            error = abs(right_mm)
        else:
            kind = "fwdmm" if forward_mm > 0 else "bwdmm"
            error = abs(forward_mm)
        distance = int(math.floor(min(error, self._max_step) + 0.5))
        return kind, max(1, distance)

    def _finish(self) -> None:
        error = self._stop_chassis()
        if error:
            self._fault(error, stop=False)
            return
        self._plane_locked = True
        self._state = AlignmentState.COMPLETE
        self._message = "物料圆心已在底盘容差内，视觉定位结束"

    def _seek(self, observation: AlignmentObservation) -> None:
        if not observation.valid or not observation.stable:
            self._aligned_frames = 0
            self._message = "目标丢失或尚未稳定，底盘保持不动"
            return
        if observation.quality < self._min_quality:
            self._aligned_frames = 0
            self._message = "目标质量不足，底盘保持不动"
            return
        if not math.isfinite(observation.target_x_px + observation.target_y_px):
            self._fault("物料圆心坐标非法")
            return
        try:
            x_px, y_px = image_origin_px(
                observation.target_x_px, observation.target_y_px,
                observation.image_width, observation.image_height)
        except ValueError as exc:
            self._fault(str(exc))
            return
        right_mm, forward_mm = self._offsets_mm(x_px, y_px)
        if not math.isfinite(right_mm + forward_mm):
            self._fault("底盘修正量非法")
            return
        command = self._chassis_command(right_mm, forward_mm)
        if command is None:
            self._aligned_frames += 1
            if self._aligned_frames >= self._required_frames:
                self._finish()
                return
            self._message = f"底盘已在容差内 {self._aligned_frames}/{self._required_frames}"
            return
        self._aligned_frames = 0
        if self._corrections >= self._max_corrections:
            self._fault("底盘修正次数达到上限")
            return
        kind, distance_mm = command
        try:
            self._link.move(kind, distance_mm)
        except Exception as exc:
            self._fault(f"底盘定位指令失败：{exc}")
            return
        self._chassis_busy = True
        self._corrections += 1
        self._message = f"{_KIND_TEXT[kind]} {distance_mm} mm"

    def tick(self, observation: AlignmentObservation | None = None) -> AlignmentStatus:
        """推进一个周期。定位阶段只查询或驱动底盘。"""
        if self._state in {AlignmentState.COMPLETE, AlignmentState.FAULT}:
            return self.status()
        if self._now() - self._started_at > self._timeout:
            self._fault("视觉定位超时")
            return self.status()

        if self._chassis_busy:
            try:
                result = self._link.poll()
            except Exception as exc:
                self._fault(f"底盘状态读取失败：{exc}")
                return self.status()
            if result == ActionResult.BUSY:
                return self.status()
            self._chassis_busy = False
            if result == ActionResult.FAILED:
                self._fault("底盘动作失败；禁止自动重试")
                return self.status()
            self._message = "底盘步进完成，等待下一帧"
            return self.status()

        if self._state != AlignmentState.ACQUIRE:
            self._fault("非法对准状态")
            return self.status()
        if observation is None:
            self._message = "尚未收到物料圆心，底盘保持不动"
            return self.status()
        self._seek(observation)
        return self.status()

    def status(self) -> AlignmentStatus:
        return AlignmentStatus(
            state=self._state,
            pending_action=None,
            correction_count=self._corrections,
            aligned_frames=self._aligned_frames,
            plane_locked=self._plane_locked,
            release_started=False,
            released=False,
            message=self._message,
        )
