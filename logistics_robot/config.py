"""JSON 配置读取。默认配置不会允许真实车辆运动。

@File    : logistics_robot/config.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 读取并做基础结构校验的运行配置。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置，并检查应用启动所需的顶层字段。

    这里只校验结构完整性；坐标、HSV、串口等数值是否正确需在各硬件阶段
    单独标定和验证。
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {"serial", "node_route"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"配置缺少字段：{', '.join(sorted(missing))}")
    return config
