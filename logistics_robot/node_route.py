"""九点路网的路径校验、规划与连续直线压缩。

@File    : logistics_robot/node_route.py
@Author  : justinxu25849671
@Date    : 2026-08-18
@Brief   : 校验九点路径并压缩连续直行段。

这个模块只处理逻辑节点预览，不控制电机，也不假设节点间的实际距离已经标定。
真实下发使用 ``path_protocol.PathPoint`` 的绝对毫米坐标，不能直接发送节点号。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import isfinite
from typing import Any, Iterable, Mapping


DEFAULT_NODE_COORDINATES: dict[int, tuple[int, int]] = {
    # 与场地图一致：1/4/7 在右侧，3/6/9 在左侧。
    1: (2, 0), 2: (1, 0), 3: (0, 0),
    4: (2, 1), 5: (1, 1), 6: (0, 1),
    7: (2, 2), 8: (1, 2), 9: (0, 2),
}

DEFAULT_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9),
    (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9),
)


@dataclass(frozen=True)
class RouteEdge:
    """一条无向相邻通道；距离为 ``None`` 表示尚未实测标定。"""

    start: int
    end: int
    distance_mm: float | None


@dataclass(frozen=True)
class RouteSegment:
    """一个连续执行段；``via`` 中的节点会经过但不停。"""

    start: int
    end: int
    via: tuple[int, ...]
    distance_mm: float | None


@dataclass(frozen=True)
class RoutePlan:
    """完整节点序列及其可连续执行的直线段。"""

    nodes: tuple[int, ...]
    segments: tuple[RouteSegment, ...]

    def path_text(self) -> str:
        """返回可直接用于日志或 ``PATH`` 协议的连字符节点串。"""
        return "-".join(str(node) for node in self.nodes)


class NineNodeRouteMap:
    """九点正交路网；执行距离取实测边长，毫米坐标只确定轴向符号。"""

    def __init__(self, coordinates: Mapping[int, tuple[int, int]], edges: Iterable[RouteEdge],
                 mm_coordinates: Mapping[int, tuple[int, int]] | None = None) -> None:
        """构造路网并拒绝重复、非正距离或引用不存在节点的边。"""
        self._coordinates = dict(coordinates)
        self._mm_coordinates = dict(mm_coordinates or {})
        unknown_mm_nodes = set(self._mm_coordinates) - set(self._coordinates)
        if unknown_mm_nodes:
            raise ValueError(f"毫米坐标引用未知节点：{sorted(unknown_mm_nodes)}")
        self._edges: dict[frozenset[int], RouteEdge] = {}
        for edge in edges:
            if edge.start == edge.end or edge.start not in self._coordinates or edge.end not in self._coordinates:
                raise ValueError("路径边引用了无效节点")
            if edge.distance_mm is not None and (not isfinite(edge.distance_mm) or edge.distance_mm <= 0.0):
                raise ValueError("节点间距离必须为正数或 null（待标定）")
            key = frozenset((edge.start, edge.end))
            if key in self._edges:
                raise ValueError("节点边重复定义")
            self._edges[key] = edge

    @classmethod
    def default(cls) -> "NineNodeRouteMap":
        """创建默认场地拓扑，距离均留待场地实测。"""
        return cls(DEFAULT_NODE_COORDINATES, (RouteEdge(a, b, None) for a, b in DEFAULT_EDGES))

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "NineNodeRouteMap":
        """从配置创建路网；缺省的坐标或边分别回退到默认定义。"""
        if not config:
            return cls.default()
        raw_coordinates = config.get("node_coordinates", {})
        coordinates = {
            int(node): (int(value[0]), int(value[1]))
            for node, value in raw_coordinates.items()
        } or DEFAULT_NODE_COORDINATES
        raw_mm_coordinates = config.get("node_mm_coordinates", {})
        mm_coordinates = {
            int(node): (int(value[0]), int(value[1]))
            for node, value in raw_mm_coordinates.items()
        }
        raw_edges = config.get("edges", [])
        edges = [
            RouteEdge(int(item["from"]), int(item["to"]),
                      None if item.get("distance_mm") is None else float(item["distance_mm"]))
            for item in raw_edges
        ]
        return cls(coordinates, edges or (RouteEdge(a, b, None) for a, b in DEFAULT_EDGES),
                   mm_coordinates)

    def mm_coordinate(self, node: int) -> tuple[int, int]:
        """返回实测场地毫米坐标，绝不把逻辑网格坐标作为毫米下发。"""
        if node not in self._coordinates:
            raise ValueError(f"未知节点：{node}")
        try:
            return self._mm_coordinates[node]
        except KeyError as exc:
            raise ValueError(
                f"节点 {node} 缺少 node_mm_coordinates 实测值，禁止下发运动") from exc

    @staticmethod
    def parse_path(text: str) -> tuple[int, ...]:
        """将 ``1-2-3`` 形式的人工输入拆成节点元组。"""
        parts = [part.strip() for part in text.strip().split("-")]
        if len(parts) < 2 or any(not part.isdigit() for part in parts):
            raise ValueError("路径应为至少两个节点，例如 1-2-3")
        return tuple(int(part) for part in parts)

    def _edge(self, start: int, end: int) -> RouteEdge:
        """查找两节点之间的直接边；不存在时给出可读错误。"""
        try:
            return self._edges[frozenset((start, end))]
        except KeyError as exc:
            raise ValueError(f"节点 {start} 与 {end} 之间没有已定义的通道") from exc

    def validate_path(self, nodes: Iterable[int]) -> tuple[int, ...]:
        """确认路径节点存在，且每一对相邻节点都有已定义通道。"""
        result = tuple(nodes)
        if len(result) < 2:
            raise ValueError("路径至少应包含起点和终点")
        for node in result:
            if node not in self._coordinates:
                raise ValueError(f"未知节点：{node}")
        for start, end in zip(result, result[1:]):
            self._edge(start, end)
        return result

    def shortest_path(self, start: int, end: int, blocked: Iterable[int] = ()) -> tuple[int, ...]:
        """使用 Dijkstra 搜索避开屏蔽节点和未标定边的最低距离路径。"""
        if start not in self._coordinates or end not in self._coordinates:
            raise ValueError("起点或终点不在节点图中")
        blocked_set = set(blocked)
        if start in blocked_set or end in blocked_set:
            raise ValueError("起点或终点不能被屏蔽")
        queue: list[tuple[float, int, tuple[int, ...]]] = [(0.0, start, (start,))]
        best = {start: 0.0}
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == end:
                return path
            if cost != best[node]:
                continue
            for edge in self._edges.values():
                if edge.start == node:
                    neighbor = edge.end
                elif edge.end == node:
                    neighbor = edge.start
                else:
                    continue
                if neighbor in blocked_set:
                    continue
                if edge.distance_mm is None:
                    continue
                new_cost = cost + edge.distance_mm
                if new_cost < best.get(neighbor, float("inf")):
                    best[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + (neighbor,)))
        raise ValueError(f"从节点 {start} 无法到达节点 {end}")

    def _direction(self, start: int, end: int) -> tuple[int, int]:
        """取得正交相邻边的单位方向，用于识别是否可以不停点直行。"""
        x1, y1 = self._coordinates[start]
        x2, y2 = self._coordinates[end]
        dx, dy = x2 - x1, y2 - y1
        # 本项目路网只允许正交相邻边；这也阻止把 1→3 或 1→5 当作直接边。
        if abs(dx) + abs(dy) != 1:
            raise ValueError(f"节点 {start} 到 {end} 不是正交相邻路径")
        return dx, dy

    def plan(self, nodes: Iterable[int]) -> RoutePlan:
        """校验路径后合并同方向连续边，转弯点始终保留为分段边界。"""
        path = self.validate_path(nodes)
        segments: list[RouteSegment] = []
        index = 0
        while index < len(path) - 1:
            start = path[index]
            direction = self._direction(path[index], path[index + 1])
            end_index = index + 1
            distance: float | None = self._edge(path[index], path[index + 1]).distance_mm
            while end_index < len(path) - 1 and self._direction(path[end_index], path[end_index + 1]) == direction:
                next_distance = self._edge(path[end_index], path[end_index + 1]).distance_mm
                distance = None if distance is None or next_distance is None else distance + next_distance
                end_index += 1
            segments.append(RouteSegment(start, path[end_index], path[index + 1:end_index], distance))
            index = end_index
        return RoutePlan(path, tuple(segments))

    def segment_targets_mm(self, plan: RoutePlan) -> tuple[tuple[int, int], ...]:
        """按选定路径的边长累计 F4 局部绝对目标，执行前验证全部输入。"""
        x_mm = 0.0
        y_mm = 0.0
        targets: list[tuple[int, int]] = []
        for segment in plan.segments:
            route = (segment.start, *segment.via, segment.end)
            for start, end in zip(route, route[1:]):
                distance = self._edge(start, end).distance_mm
                if distance is None:
                    raise ValueError(f"节点 {start}->{end} 缺少实测 distance_mm，禁止下发运动")
                grid_dx, grid_dy = self._direction(start, end)
                start_x, start_y = self.mm_coordinate(start)
                end_x, end_y = self.mm_coordinate(end)
                if grid_dx:
                    axis_delta = end_x - start_x
                    if axis_delta == 0:
                        raise ValueError(f"节点 {start}->{end} 缺少 X 运动方向，禁止下发运动")
                    x_mm += distance if axis_delta > 0 else -distance
                elif grid_dy:
                    axis_delta = end_y - start_y
                    if axis_delta == 0:
                        raise ValueError(f"节点 {start}->{end} 缺少 Y 运动方向，禁止下发运动")
                    y_mm += distance if axis_delta > 0 else -distance
            # F4 协议使用整数毫米；半毫米按远离零点的方向取整。
            targets.append((int(x_mm + 0.5) if x_mm >= 0 else -int(-x_mm + 0.5),
                            int(y_mm + 0.5) if y_mm >= 0 else -int(-y_mm + 0.5)))
        return tuple(targets)
