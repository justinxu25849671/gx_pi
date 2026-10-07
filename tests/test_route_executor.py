"""验证树莓派只在当前段 DONE 后发送下一段。"""

from __future__ import annotations

import unittest
import json
from pathlib import Path

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

    def test_each_segment_waits_for_done_and_translates_start_node_to_origin(self):
        route_map = self._map()
        client = FakeClient()
        result = RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 3)), 7)
        self.assertEqual([(item.start_node, item.end_node) for item in result], [(1, 3)])
        # 同一直线先合并为一段，目标是本次起点的局部累计位移。
        self.assertEqual(client.started, [(7, (PathPoint(1000, 0, 40, 10),), True)])

    def test_turn_boundary_creates_next_command_after_prior_done(self):
        logical = {1: (0, 0), 2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(1, 2, 500), RouteEdge(2, 5, 600)),
                                     {1: (0, 0), 2: (500, 0), 5: (500, 600)})
        client = FakeClient()
        RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 5)), 9)
        self.assertEqual([item[0] for item in client.started], [9, 10])
        self.assertEqual([item[2] for item in client.started], [True, False])
        self.assertEqual([item[1][0] for item in client.started],
                         [PathPoint(500, 0, 40, 10), PathPoint(500, 600, 40, 10)])

    def test_vertical_edge_is_a_valid_direct_route(self):
        logical = {2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(2, 5, 1100),),
                                     {2: (950, 0), 5: (950, 1100)})
        client = FakeClient()
        RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((2, 5)), 10)
        self.assertEqual(client.started, [(10, (PathPoint(0, 1100, 40, 10),), True)])

    def test_missing_mm_coordinate_prevents_any_motion(self):
        route_map = self._map(mm=False)
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "node_mm_coordinates"):
            RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2)))
        self.assertEqual(client.started, [])

    def test_vertical_edge_uses_measured_length_without_cross_axis_drift(self):
        logical = {3: (0, 0), 6: (0, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(3, 6, 1101.1),),
                                     {3: (1900, 0), 6: (1850, 1100)})
        client = FakeClient()
        RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((3, 6)))
        self.assertEqual(client.started, [(1, (PathPoint(0, 1101, 40, 10),), True)])

    def test_missing_later_edge_length_prevents_any_motion(self):
        logical = {1: (0, 0), 2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical,
                                     (RouteEdge(1, 2, 500), RouteEdge(2, 5, None)),
                                     {1: (0, 0), 2: (500, 0), 5: (500, 600)})
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "distance_mm"):
            RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 5)))
        self.assertEqual(client.started, [])

    def test_shortest_path_skips_unmeasured_edge(self):
        logical = {1: (0, 0), 2: (1, 0), 4: (0, 1), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical,
                                     (RouteEdge(1, 2, None), RouteEdge(1, 4, 1100),
                                      RouteEdge(4, 5, 900), RouteEdge(5, 2, 1101)))
        self.assertEqual(route_map.shortest_path(1, 2), (1, 4, 5, 2))

    def test_error_does_not_advance_next_segment(self):
        logical = {1: (0, 0), 2: (1, 0), 5: (1, 1)}
        route_map = NineNodeRouteMap(logical, (RouteEdge(1, 2, 500), RouteEdge(2, 5, 600)),
                                     {1: (0, 0), 2: (500, 0), 5: (500, 600)})
        client = FakeClient(fail_on=11)
        with self.assertRaisesRegex(RuntimeError, "MOTION_TIMEOUT"):
            RouteExecutor(client, route_map, 40, 10).execute(route_map.plan((1, 2, 5)), 11)
        self.assertEqual([item[0] for item in client.started], [11])

    def test_initial_exit_finishes_before_node_route_and_offsets_targets(self):
        route_map = self._map()
        client = FakeClient()
        result = RouteExecutor(client, route_map, 40, 10,
                               initial_exit_mm=(100, 80)).execute(route_map.plan((1, 2)), 254)
        self.assertEqual([(path_id, points[0], reset) for path_id, points, reset in client.started],
                         [(254, PathPoint(100, 0, 40, 10), True),
                          (255, PathPoint(100, 80, 40, 10), False),
                          (1, PathPoint(600, 80, 40, 10), False)])
        self.assertEqual([item.path_id for item in result], [1])

    def test_initial_exit_error_prevents_route_motion(self):
        route_map = self._map()
        client = FakeClient(fail_on=2)
        with self.assertRaisesRegex(RuntimeError, "驶出初始位失败"):
            RouteExecutor(client, route_map, 40, 10,
                          initial_exit_mm=(100, 80)).execute(route_map.plan((1, 2)), 1)
        self.assertEqual([item[0] for item in client.started], [1, 2])

    def test_initial_exit_only_applies_at_node_one(self):
        route_map = self._map()
        client = FakeClient()
        RouteExecutor(client, route_map, 40, 10,
                      initial_exit_mm=(100, 80)).execute(route_map.plan((2, 3)))
        self.assertEqual(client.started, [(1, (PathPoint(500, 0, 40, 10),), True)])

    def test_current_measured_route_lengths(self):
        config_path = Path(__file__).resolve().parents[1] / "config.json"
        route_map = NineNodeRouteMap.from_config(json.loads(config_path.read_text())["node_route"])
        plan = route_map.plan((1, 2, 3, 6, 5, 4, 7, 8, 9))
        self.assertEqual([segment.distance_mm for segment in plan.segments],
                         [1850, 950, 1900, 900, 1850])


if __name__ == "__main__":
    unittest.main()
