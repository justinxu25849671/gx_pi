"""比赛二维码任务码的解析与中文可读表示。

@File    : logistics_robot/task_code.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 校验并解析比赛二维码任务码。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


COLOR_BY_DIGIT = {
    "1": "red",
    "2": "yellow",
    "3": "blue",
    "4": "green",
}

CHINESE_COLOR = {
    "red": "红色",
    "yellow": "黄色",
    "blue": "蓝色",
    "green": "绿色",
}

_PATTERN = re.compile(r"^([1-4]{3})\+([1-3]{3})\+([1-4]{3})\+([1-3]{3})$")


@dataclass(frozen=True)
class TaskCode:
    """二维码中四组三位数对应的物流搬运任务。"""

    raw: str
    first_colors: tuple[str, str, str]
    first_positions: tuple[int, int, int]
    second_colors: tuple[str, str, str]
    second_positions: tuple[int, int, int]

    @classmethod
    def parse(cls, text: str) -> "TaskCode":
        """校验并解析 ``颜色+环位+颜色+环位`` 格式的二维码任务码。"""
        normalized = text.strip().replace(" ", "")
        match = _PATTERN.fullmatch(normalized)
        if not match:
            raise ValueError("任务码应为 123+123+321+231 这样的四组三位格式；颜色仅允许 1..4")

        first_color_digits, first_positions, second_color_digits, second_positions = match.groups()
        if len(set(first_color_digits)) != 3 or len(set(second_color_digits)) != 3:
            raise ValueError("每一批必须是三种不同颜色的物料")
        if set(first_color_digits) != set(second_color_digits):
            raise ValueError("第二批每种颜色必须在第一批有同色物料，才能按实际承接位置码垛")

        return cls(
            raw=normalized,
            first_colors=tuple(COLOR_BY_DIGIT[digit] for digit in first_color_digits),
            first_positions=tuple(int(digit) for digit in first_positions),
            second_colors=tuple(COLOR_BY_DIGIT[digit] for digit in second_color_digits),
            second_positions=tuple(int(digit) for digit in second_positions),
        )

    def describe(self) -> str:
        """返回适合日志显示的中文颜色摘要。"""
        first = "、".join(CHINESE_COLOR[color] for color in self.first_colors)
        second = "、".join(CHINESE_COLOR[color] for color in self.second_colors)
        return f"第一批：{first}；第二批：{second}"
