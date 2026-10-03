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
                 on_complete: Callable[[SegmentCompletion], None] | None = None) -> None:
        if not 1 <= rpm <= 5000 or not 0 <= acceleration <= 255:
            raise ValueError("rpm 或 acceleration 超出协议范围")
        self._client = path_client
        self._map = route_map
        self._rpm = rpm
        self._acceleration = acceleration
        self._on_complete = on_complete

    def execute(self, plan: RoutePlan, first_path_id: int = 1) -> tuple[SegmentCompletion, ...]:
        if not 1 <= first_path_id <= 255:
            raise ValueError("first_path_id 必须在 1..255")
        completed: list[SegmentCompletion] = []
        path_id = first_path_id
        # PATH_RESET_ORIGIN 把 STM32 的当前位置设为 (0, 0)。地图坐标则以节点 1
        # 为参考；因此每个任务开始时都必须相对本次起点平移，才能从任意节点启动。
        origin_x_mm, origin_y_mm = self._map.mm_coordinate(plan.nodes[0])
        for segment in plan.segments:
            x_mm, y_mm = self._map.mm_coordinate(segment.end)
            # 只有第一段建立 STM32 的局部原点；后续坐标保持同一毫米参考系。
            self._client.upload_and_start(path_id, (PathPoint(
                x_mm - origin_x_mm, y_mm - origin_y_mm,
                self._rpm, self._acceleration),),
                reset_origin=(not completed))
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
