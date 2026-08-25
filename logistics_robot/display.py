"""高亮车载显示屏：只显示，不读取任何用户输入。

@File    : logistics_robot/display.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 渲染任务状态供现场观察。
"""

from __future__ import annotations

from typing import Any

from .mission import MissionStatus


class StatusDisplay:
    """比赛屏幕的只读状态视图，不接受任何触摸或键盘控制。"""

    def __init__(self, config: dict[str, Any]) -> None:
        """按配置创建 OpenCV 窗口；依赖缺失时静默降级为无显示模式。"""
        self._enabled = bool(config["enabled"])
        self._cv2 = None
        self._np = None
        if not self._enabled:
            return
        try:
            import cv2
            import numpy as np
        except ImportError:
            self._enabled = False
            return
        self._cv2 = cv2
        self._np = np
        cv2.namedWindow("物流搬运任务", cv2.WINDOW_NORMAL)
        if bool(config.get("fullscreen", True)):
            cv2.setWindowProperty("物流搬运任务", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, status: MissionStatus) -> None:
        """把当前任务快照渲染为一帧，供裁判和操作人员观察。"""
        if not self._enabled:
            return
        assert self._cv2 is not None and self._np is not None
        image = self._np.zeros((480, 800, 3), dtype=self._np.uint8)
        image[:] = (245, 245, 245)
        # OpenCV 的内置 Hershey 字体不支持中文；显示内容保持 ASCII，确保比赛屏幕可读。
        step_text = "--"
        if status.current_step is not None:
            step = status.current_step
            step_text = f"{step.action.value} {step.color.upper()} P{step.position} B{step.batch}"
        lines = [
            f"STATE: {status.state.value}",
            f"TASK: {status.task_code or '--'}",
            f"PROGRESS: {status.completed_steps}/{status.total_steps}",
            f"PICK OK: {status.correct_picks}   PLACE OK: {status.correct_places}",
            f"NEXT: {step_text}",
        ]
        y = 70
        for index, line in enumerate(lines):
            self._cv2.putText(image, line, (25, y), self._cv2.FONT_HERSHEY_SIMPLEX, 0.62 if index == 4 else 0.9, (0, 0, 0), 2)
            y += 78
        self._cv2.imshow("物流搬运任务", image)
        self._cv2.waitKey(1)  # 仅刷新窗口，不处理控制输入。

    def close(self) -> None:
        """关闭本模块创建的 OpenCV 窗口。"""
        if self._enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()
