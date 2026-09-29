"""验证树莓派只在当前段 DONE 后发送下一段。"""

from __future__ import annotations

import unittest

from logistics_robot.node_route import NineNodeRouteMap, RouteEdge
from logistics_robot.path_protocol import DoneEvent, ErrorEvent, PathPoint, PathResult, PathSegment
from logistics_robot.route_executor import RouteExecutor


class FakeClient:
    def __init__(self, fail_on: int | None = None) -> None:
        self.started: list[tuple[int, tuple[PathPoint, ...], bool]] = []
        self.fail_on = fail_on
        self._current_id = 0

    def upload_and_start(self, path_id, points, reset_origin=True):
        values = tuple(points)
        self.started.append((path_id, values, reset_origin))
        self._current_id = path_id

    def wait_until_terminal(self):
        return self.wait_until_terminal_with_keepalive()

    def wait_until_terminal_with_keepalive(self):
        if self._current_id == self.fail_on:
            return ErrorEvent(self._current_id, 0, PathSegment.X,
                              PathResult.MOTION_TIMEOUT, 0, 0)
        point = self.started[-1][1][0]
        return DoneEvent(self._current_id, 1, point.x_mm, point.y_mm)


class RouteExecutorTests(unittest.TestCase):
    def _map(self, mm=True):
        logical = {1: (0, 0), 2: (1, 0), 3: (2, 0)}
        coords = {1: (100, 200), 2: (600, 200), 3: (1100, 200)} if mm else {}
        return NineNodeRouteMap(logical, (RouteEdge(1, 2, 500), RouteEdge(2, 3, 500)), coords)

    def test_each_segment_waits_for_done_and_keeps_mm_reference(self):
        route_map = self._map()
        client = FakeClient()
        result = RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 3)), 7)
        self.assertEqual([(item.start_node, item.end_node) for item in result], [(1, 3)])
        # 同一直线先合并为一段，绝不下发逻辑网格值 (2, 0)。
        self.assertEqual(client.started, [(7, (PathPoint(1100, 200, 40, 10),), True)])

    def test_turn_boundary_creates_next_command_after_prior_done(self):
        logical = {1: (0, 0), 2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(1, 2, 500), RouteEdge(2, 5, 600)),
                                     {1: (0, 0), 2: (500, 0), 5: (500, 600)})
        client = FakeClient()
        RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 5)), 9)
        self.assertEqual([item[0] for item in client.started], [9, 10])
        self.assertEqual([item[2] for item in client.started], [True, False])

    def test_missing_mm_coordinate_prevents_any_motion(self):
        route_map = self._map(mm=False)
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "node_mm_coordinates"):
            RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2)))
        self.assertEqual(client.started, [])

    def test_error_does_not_advance_next_segment(self):
        logical = {1: (0, 0), 2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(1, 2, 500), RouteEdge(2, 5, 600)),
                                     {1: (0, 0), 2: (500, 0), 5: (500, 600)})
        client = FakeClient(fail_on=11)
        with self.assertRaisesRegex(RuntimeError, "MOTION_TIMEOUT"):
            RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 5)), 11)
        self.assertEqual([item[0] for item in client.started], [11])


if __name__ == "__main__":
    unittest.main()
