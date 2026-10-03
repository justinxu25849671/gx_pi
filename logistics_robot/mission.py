"""严格按赛题顺序生成两批物料的搬运动作。

@File    : logistics_robot/mission.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 定义搬运任务状态机与步骤序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .alignment import AlignmentObservation, AlignmentState, PlacementAlignmentController
from .model import MechanismAction, MissionStep
from .task_code import TaskCode


class MissionState(str, Enum):
    """任务状态机的有限状态集合。"""
    WAIT_START = "WAIT_START"
    SCAN_TASK = "SCAN_TASK"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAULT = "FAULT"


@dataclass(frozen=True)
class MissionStatus:
    """供显示层和日志层读取的不可变任务快照。"""
    state: MissionState
    task_code: str | None
    current_step: MissionStep | None
    completed_steps: int
    total_steps: int
    correct_picks: int
    correct_places: int
    message: str


class LogisticsMission:
    """把赛题任务码展开为严格有序的 24 个取放步骤。"""

    def __init__(self) -> None:
        self._state = MissionState.WAIT_START
        self._task: TaskCode | None = None
        self._steps: list[MissionStep] = []
        self._index = 0
        self._correct_picks = 0
        self._correct_places = 0
        self._fault_message = ""

    @staticmethod
    def _steps_for(task: TaskCode) -> list[MissionStep]:
        """按比赛规则生成两批物料的取放步骤，不访问任何硬件。"""
        steps: list[MissionStep] = []
        # 第一批：原料区取料 -> 粗加工区放置 -> 粗加工区取回 -> 暂存区平放。
        for color, position in zip(task.first_colors, task.first_positions):
            steps.append(MissionStep("raw", MechanismAction.PICK_FROM_RAW, color, position, 1))
        for color, position in zip(task.first_colors, task.first_positions):
            steps.append(MissionStep("coarse", MechanismAction.PLACE_TO_COARSE, color, position, 1))
        for color, position in zip(task.first_colors, task.first_positions):
            steps.append(MissionStep("coarse", MechanismAction.PICK_FROM_COARSE, color, position, 1))
        for color, position in zip(task.first_colors, task.first_positions):
            steps.append(MissionStep("temporary", MechanismAction.PLACE_TO_TEMPORARY, color, position, 1))

        # 第二批：同样经过粗加工区，最后在第一批同色物料的实际承接位置码垛。
        for color, position in zip(task.second_colors, task.second_positions):
            steps.append(MissionStep("raw", MechanismAction.PICK_FROM_RAW, color, position, 2))
        for color, position in zip(task.second_colors, task.second_positions):
            steps.append(MissionStep("coarse", MechanismAction.PLACE_TO_COARSE, color, position, 2))
        for color, position in zip(task.second_colors, task.second_positions):
            steps.append(MissionStep("coarse", MechanismAction.PICK_FROM_COARSE, color, position, 2))
        first_temporary_position = dict(zip(task.first_colors, task.first_positions))
        for color in task.second_colors:
            if color not in first_temporary_position:
                raise ValueError(f"第二批 {color} 在第一批中没有同色承接物料，无法码垛")
            steps.append(MissionStep("temporary", MechanismAction.STACK_TO_TEMPORARY, color,
                                     first_temporary_position[color], 2))
        return steps

    def press_start(self) -> None:
        """响应物理按钮；只有等待状态允许进入扫码阶段。"""
        if self._state == MissionState.WAIT_START:
            self._state = MissionState.SCAN_TASK

    def set_task(self, task: TaskCode) -> None:
        """在成功读码后锁定任务，并把状态切换到执行中。"""
        if self._state != MissionState.SCAN_TASK:
            raise RuntimeError("只能在读任务码阶段设置任务")
        self._task = task
        self._steps = self._steps_for(task)
        self._index = 0
        self._state = MissionState.RUNNING

    def current_step(self) -> MissionStep | None:
        """返回待执行步骤；非运行状态或全部完成时返回 ``None``。"""
        if self._state != MissionState.RUNNING or self._index >= len(self._steps):
            return None
        return self._steps[self._index]

    def complete_current_step(self, verified: bool) -> None:
        """提交动作完成反馈；放置步不得把松爪后视觉复查当成 ``verified``。"""
        step = self.current_step()
        if step is None:
            raise RuntimeError("当前没有可完成的任务步骤")
        if not verified:
            self._state = MissionState.FAULT
            self._fault_message = f"{step.action.value} 未通过松爪前条件或机构完成反馈"
            return
        if step.action in {MechanismAction.PICK_FROM_RAW, MechanismAction.PICK_FROM_COARSE}:
            self._correct_picks += 1
        else:
            self._correct_places += 1
        self._index += 1
        if self._index == len(self._steps):
            self._state = MissionState.COMPLETE

    def complete_aligned_placement(self, *, completed: bool, released: bool,
                                   release_started: bool) -> None:
        """只接受一次性放置状态机的终态，不允许松爪后视觉补救。"""
        step = self.current_step()
        if step is None or step.action in {
            MechanismAction.PICK_FROM_RAW, MechanismAction.PICK_FROM_COARSE,
        }:
            raise RuntimeError("当前步骤不是放置动作")
        if not (completed and released and release_started):
            self.complete_current_step(False)
            return
        self.complete_current_step(True)

    def fail(self, message: str) -> None:
        """从任意运行阶段显式切换到故障态。"""
        self._state = MissionState.FAULT
        self._fault_message = message

    def status(self) -> MissionStatus:
        """生成当前状态快照，不改变任务进度。"""
        step = self.current_step()
        if self._state == MissionState.WAIT_START:
            message = "等待车体一键启动"
        elif self._state == MissionState.SCAN_TASK:
            message = "正在寻找并读取任务二维码"
        elif self._state == MissionState.COMPLETE:
            message = "搬运完成，请返回启停区"
        elif self._state == MissionState.FAULT:
            message = self._fault_message
        elif step is not None:
            message = f"第 {self._index + 1}/{len(self._steps)} 步：{step.action.value} {step.color}，环位 {step.position}"
        else:
            message = "状态异常"
        return MissionStatus(
            state=self._state,
            task_code=self._task.raw if self._task else None,
            current_step=step,
            completed_steps=self._index,
            total_steps=len(self._steps),
            correct_picks=self._correct_picks,
            correct_places=self._correct_places,
            message=message,
        )


class MissionPlacementCoordinator:
    """把单次放置控制器的终态安全提交给总任务，不做松爪后视觉复查。"""

    def __init__(self, mission: LogisticsMission,
                 controller: PlacementAlignmentController) -> None:
        self._mission = mission
        self._controller = controller
        self._submitted = False

    def tick(self, observation: AlignmentObservation | None = None):
        # 平面锁定后显式丢弃上层仍传入的画面，确保下降、松爪和撤离不复查。
        current = observation if self._controller.needs_observation else None
        status = self._controller.tick(current)
        if self._submitted:
            return status
        if status.state == AlignmentState.COMPLETE:
            self._mission.complete_aligned_placement(
                completed=True,
                released=status.released,
                release_started=status.release_started,
            )
            self._submitted = True
        elif status.state == AlignmentState.FAULT:
            self._mission.fail(status.message)
            self._submitted = True
        return status
