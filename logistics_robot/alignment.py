"""画面中心坐标系下的视觉定位，以及定位结束后的一次性放置。"""

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


class VisionLocateLink(Protocol):
    """下位机视觉定位：一条开始指令、圆心坐标流、一条停止指令。"""

    def start(self) -> None: ...

    def send_coordinate(self, x_px: int, y_px: int) -> None: ...

    def stop(self) -> None: ...


class FeedbackActuator(Protocol):
    """定位完成后的下降、松爪和撤离。横移与伸缩不再由上位机逐轴下发。"""

    def start(self, action: AlignmentAction, value: float | None = None) -> None: ...

    def poll(self, action: AlignmentAction) -> ActionResult: ...


class SerialVisionLocateLink:
    """把路径客户端适配为视觉定位链路。"""

    def __init__(self, client) -> None:
        self._client = client

    def start(self) -> None:
        self._client.start_visual_positioning()

    def send_coordinate(self, x_px: int, y_px: int) -> None:
        self._client.send_visual_coordinate(x_px, y_px)

    def stop(self) -> None:
        self._client.stop_visual_positioning()


class PlacementAlignmentController:
    """视觉定位只发一条开始指令；收到回复后发送物料圆心，到原点后停止。"""

    def __init__(self, link: VisionLocateLink, actuator: FeedbackActuator,
                 config: Mapping[str, Any], now=None) -> None:
        self._link = link
        self._actuator = actuator
        self._min_quality = float(config["min_target_quality"])
        self._timeout = float(config["total_timeout_s"])
        self._now = now or time.monotonic
        self._started_at = self._now()
        self._state = AlignmentState.ACQUIRE
        self._pending: AlignmentAction | None = None
        self._plane_locked = False
        self._release_started = False
        self._released = False
        self._locate_started = False
        self._locate_active = False
        self._message = "等待发送视觉定位指令"

    @property
    def needs_observation(self) -> bool:
        """平面锁定后始终为假，定位结束后不再发送圆心坐标。"""
        return self._state == AlignmentState.ACQUIRE and not self._plane_locked

    def _fault(self, message: str) -> None:
        extra = self._stop_locating()
        if extra:
            message = f"{message}；{extra}"
        self._state = AlignmentState.FAULT
        self._pending = None
        self._message = message

    def _stop_locating(self) -> str | None:
        if not self._locate_active:
            return None
        self._locate_active = False
        try:
            self._link.stop()
        except Exception as exc:
            return f"停止视觉定位失败：{exc}"
        return None

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
        if action == AlignmentAction.DESCEND:
            self._state = AlignmentState.RELEASE
            self._message = "下降完成，准备松爪一次"
        elif action == AlignmentAction.RELEASE:
            self._released = True
            self._state = AlignmentState.RETREAT
            self._message = "已松爪，后续仅允许撤离"
        elif action == AlignmentAction.RETREAT:
            self._state = AlignmentState.COMPLETE
            self._message = "一次性放置完成"
        else:
            self._fault(f"视觉定位完成后不允许再执行 {action.value}")

    def _begin_locating(self) -> bool:
        """整段定位只发送一次开始指令。成功收到回复后返回真。"""
        if self._locate_started:
            return self._locate_active
        self._locate_started = True
        try:
            self._link.start()
        except Exception as exc:
            self._fault(f"视觉定位指令失败：{exc}")
            return False
        self._locate_active = True
        self._message = "已收到视觉定位回复，等待物料圆心"
        return True

    def _stream_circle(self, observation: AlignmentObservation) -> None:
        if not observation.valid or not observation.stable:
            self._message = "目标丢失或尚未稳定，暂不发送圆心坐标"
            return
        if observation.quality < self._min_quality:
            self._message = "目标质量不足，暂不发送圆心坐标"
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
        if (x_px, y_px) == (0, 0):
            error = self._stop_locating()
            if error:
                self._fault(error)
                return
            self._plane_locked = True
            self._state = AlignmentState.DESCEND
            self._message = "物料圆心已在画面原点，视觉定位结束"
            self._start(AlignmentAction.DESCEND)
            return
        try:
            self._link.send_coordinate(x_px, y_px)
        except Exception as exc:
            self._fault(f"发送物料圆心失败：{exc}")
            return
        self._message = f"发送物料圆心 ({x_px}, {y_px})"

    def tick(self, observation: AlignmentObservation | None = None) -> AlignmentStatus:
        """推进一个周期。定位阶段不驱动横移或伸缩。"""
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
        if not self._begin_locating():
            return self.status()
        if observation is None:
            self._message = "尚未收到物料圆心，保持视觉定位"
            return self.status()
        self._stream_circle(observation)
        return self.status()

    def status(self) -> AlignmentStatus:
        return AlignmentStatus(
            state=self._state,
            pending_action=self._pending,
            correction_count=0,
            aligned_frames=0,
            plane_locked=self._plane_locked,
            release_started=self._release_started,
            released=self._released,
            message=self._message,
        )
