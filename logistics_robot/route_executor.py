"""树莓派逐段执行节点路线；STM32 只控制当前一个绝对毫米目标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .node_route import NineNodeRouteMap, RoutePlan
from .path_client import PathProtocolError
from .path_protocol import DoneEvent, ErrorEvent, PathPoint


@dataclass(frozen=True)
class SegmentCompletion:
    start_node: int
    end_node: int
    path_id: int
    done: DoneEvent


class RouteExecutor:
    """每个节点段单独上传、启动并等待 DONE，失败时绝不推进下一段。"""

    def __init__(self, path_client, route_map: NineNodeRouteMap,
                 rpm: int, acceleration: int,
                 on_complete: Callable[[SegmentCompletion], None] | None = None,
                 initial_exit_mm: tuple[int, int] = (0, 0)) -> None:
        if not 1 <= rpm <= 5000 or not 0 <= acceleration <= 255:
            raise ValueError("rpm 或 acceleration 超出协议范围")
        if len(initial_exit_mm) != 2 or any(type(value) is not int or value < 0 for value in initial_exit_mm):
            raise ValueError("initial_exit_mm 必须是非负整数毫米 (前进, 左移)")
        self._client = path_client
        self._map = route_map
        self._rpm = rpm
        self._acceleration = acceleration
        self._on_complete = on_complete
        self._initial_exit_mm = initial_exit_mm

    def execute(self, plan: RoutePlan, first_path_id: int = 1) -> tuple[SegmentCompletion, ...]:
        if not 1 <= first_path_id <= 255:
            raise ValueError("first_path_id 必须在 1..255")
        completed: list[SegmentCompletion] = []
        path_id = first_path_id
        # 在首次运动前验证整条路线，避免后续段缺失边长时车已离开起点。
        targets = self._map.segment_targets_mm(plan)
        # 只有从初始节点 1 出发，才执行驶出停放位置的动作。
        exit_x, exit_y = self._initial_exit_mm if plan.nodes[0] == 1 else (0, 0)
        exit_targets: list[tuple[int, int]] = []
        if exit_x:
            exit_targets.append((exit_x, 0))
        if exit_y:
            exit_targets.append((exit_x, exit_y))
        exit_points = tuple(PathPoint(x, y, self._rpm, self._acceleration)
                            for x, y in exit_targets)
        points = tuple(PathPoint(x_mm + exit_x, y_mm + exit_y, self._rpm, self._acceleration)
                       for x_mm, y_mm in targets)
        for point in (*exit_points, *points):
            point.validate()
        for index, point in enumerate(exit_points):
            self._client.upload_and_start(path_id, (point,), reset_origin=(index == 0))
            terminal = self._client.wait_until_terminal_with_keepalive()
            if isinstance(terminal, ErrorEvent):
                raise PathProtocolError(f"驶出初始位失败：{terminal.error.name}")
            if not isinstance(terminal, DoneEvent) or terminal.path_id != path_id:
                raise PathProtocolError("驶出初始位完成事件不匹配")
            path_id = path_id % 255 + 1
        for segment, point in zip(plan.segments, points):
            # 只有第一段建立 F4 局部原点；后续目标累计本次路线的实测边长。
            self._client.upload_and_start(path_id, (point,),
                reset_origin=(not exit_points and not completed))
            terminal = self._client.wait_until_terminal_with_keepalive()
            if isinstance(terminal, ErrorEvent):
                raise PathProtocolError(
                    f"{segment.start}->{segment.end} 失败：{terminal.error.name}")
            if not isinstance(terminal, DoneEvent) or terminal.path_id != path_id:
                raise PathProtocolError(f"{segment.start}->{segment.end} 完成事件不匹配")
            item = SegmentCompletion(segment.start, segment.end, path_id, terminal)
            completed.append(item)
            if self._on_complete is not None:
                self._on_complete(item)
            path_id = path_id % 255 + 1
        return tuple(completed)
