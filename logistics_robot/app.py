"""九点绝对航点路径协议的命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from .config import load_config
from .node_route import NineNodeRouteMap
from .path_client import PathProtocolError
from .path_protocol import DoneEvent, ErrorEvent, PathPoint
from .route_executor import RouteExecutor
from .serial_link import SerialPathLink
from .task_code import TaskCode
from .vision import CameraVision, draw_object_debug_frame


LOG = logging.getLogger(__name__)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.json"


def _parse_absolute_points(text: str, rpm: int, acceleration: int) -> tuple[PathPoint, ...]:
    """解析 ``x,y;x,y``，坐标为 X 前正、Y 左正的绝对毫米航点。"""
    points: list[PathPoint] = []
    for raw_point in text.split(";"):
        values = [value.strip() for value in raw_point.split(",")]
        if len(values) != 2:
            raise ValueError("绝对航点格式应为 x,y;x,y，例如 0,500;500,500")
        try:
            point = PathPoint(int(values[0]), int(values[1]), rpm, acceleration)
        except ValueError as exc:
            raise ValueError(f"非法航点 {raw_point!r}") from exc
        point.validate()
        points.append(point)
    if not 1 <= len(points) <= 9:
        raise ValueError("绝对路径必须包含 1..9 个航点")
    return tuple(points)


def _log_path_event(event) -> None:
    if hasattr(event, "point_index"):
        LOG.info("路径事件 %s：path=%s point=%s", type(event).__name__, event.path_id,
                 event.point_index + 1)
    else:
        LOG.info("路径事件 %s：path=%s", type(event).__name__, getattr(event, "path_id", 0))


def _print_objects(objects) -> None:
    """将当前帧检测结果输出为便于复制的像素测量记录。"""
    if not objects:
        LOG.info("当前画面未识别到有效物料")
        return
    for item in objects:
        LOG.info("color=%s, center_x_px=%.1f, center_y_px=%.1f, area_px=%.1f",
                 item.color, item.center_x_px, item.center_y_px, item.area_px)


def _run_vision_debug(config, camera_index: int | None) -> int:
    """独立运行物料相机调试；此函数不访问串口或路径对象。"""
    camera_config = dict(config["object_camera"])
    if camera_index is not None:
        camera_config["index"] = camera_index
    selected_index = int(camera_config["index"])
    try:
        vision = CameraVision(camera_config, config["hsv_colors"])
    except RuntimeError as exc:
        LOG.error("无法打开物料识别相机：请检查 object_camera.index、USB/CSI 接线和权限。(%s)", exc)
        return 1

    try:
        cv2 = vision.cv2
        previous_time = time.perf_counter()
        while True:
            try:
                frame = vision.read()
            except RuntimeError as exc:
                LOG.error("物料识别相机读帧失败：%s", exc)
                return 1
            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            objects = vision.detect_objects(frame)
            annotated = draw_object_debug_frame(frame, objects, fps, selected_index)
            try:
                cv2.imshow("Object Camera Debug", annotated)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                LOG.error("当前环境无法打开 OpenCV 窗口；请在树莓派桌面、VNC 或接入显示器的环境中运行 --vision-debug。(%s)", exc)
                return 1
            if key in (ord("q"), 27):
                return 0
            if key == ord("p"):
                _print_objects(objects)
            elif key == ord("s"):
                output_dir = Path("logs") / "vision"
                output_path = output_dir / time.strftime("object_%Y%m%d_%H%M%S.jpg")
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(output_path), annotated):
                        LOG.info("已保存物料识别调试画面：%s", output_path.resolve())
                    else:
                        LOG.error("保存物料识别调试画面失败：%s", output_path.resolve())
                except OSError as exc:
                    LOG.error("保存物料识别调试画面失败：%s (%s)", output_path.resolve(), exc)
    finally:
        vision.close()
        try:
            vision.cv2.destroyAllWindows()
        except vision.cv2.error:
            pass


def main() -> int:
    """只提供路径预览、任务码校验和已由 F4 实现的路径命令。"""
    parser = argparse.ArgumentParser(description="智能搬运机器人九点路径控制")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--dry-run", action="store_true", help="仅校验任务码，不访问硬件")
    parser.add_argument("--task", help="与 --dry-run 同用，校验任务码，例如 156+123+516+231")
    parser.add_argument("--plan-path", help="只校验/规划九点路径，例如 1-2-3 或 1-2-3-6-9")
    parser.add_argument("--execute-path", help="下发 1..9 个绝对毫米航点，例如 '0,500;500,500'")
    parser.add_argument("--execute-node-route", help="按节点图逐段执行，例如 1-2-3-6；必须配置 node_mm_coordinates")
    parser.add_argument("--path-id", type=int, default=1, help="路径 ID，范围 1..255")
    parser.add_argument("--path-rpm", type=int, default=20, help="各航点 RPM，范围 1..5000")
    parser.add_argument("--path-acceleration", type=int, default=10, help="各航点加速度，范围 0..255")
    parser.add_argument("--keep-path-origin", action="store_true", help="启动前不重置 STM32 局部原点")
    parser.add_argument("--path-status", action="store_true", help="查询 STM32 当前路径状态后退出")
    parser.add_argument("--path-stop", action="store_true", help="停止 STM32 当前路径后退出")
    parser.add_argument("--vision-debug", action="store_true", help="独立调试物料识别相机，不访问 STM32")
    parser.add_argument("--vision-camera-index", type=int,
                        help="仅本次运行覆盖 object_camera.index")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)

    if arguments.vision_debug:
        return _run_vision_debug(config, arguments.vision_camera_index)
    if arguments.vision_camera_index is not None:
        parser.error("--vision-camera-index 必须与 --vision-debug 同用")

    if arguments.dry_run:
        if not arguments.task:
            parser.error("--dry-run 必须同时提供 --task")
        LOG.info("任务码有效：%s", TaskCode.parse(arguments.task).describe())
        return 0
    if arguments.task:
        parser.error("--task 只能与 --dry-run 同用")
    if arguments.plan_path:
        node_route = NineNodeRouteMap.from_config(config.get("node_route"))
        plan = node_route.plan(node_route.parse_path(arguments.plan_path))
        LOG.info("节点路径：%s", plan.path_text())
        for segment in plan.segments:
            passed = "-".join(str(node) for node in segment.via) or "无"
            distance = "待标定" if segment.distance_mm is None else f"{segment.distance_mm:.1f} mm"
            LOG.info("连续段 %s→%s；中间不停节点：%s；距离：%s",
                     segment.start, segment.end, passed, distance)
        return 0

    path_actions = sum(bool(value) for value in
                       (arguments.execute_path, arguments.execute_node_route,
                        arguments.path_status, arguments.path_stop))
    if path_actions != 1:
        parser.error("必须选择 --execute-path、--path-status 或 --path-stop 之一")
    link = SerialPathLink(**config["serial"])
    link.path_client.add_listener(_log_path_event)
    try:
        if arguments.path_status:
            status = link.query_path_status()
            point_text = "-" if status.count == 0 else f"{status.point_index + 1}/{status.count}"
            LOG.info("路径状态：path=%d state=%s point=%s segment=%s estimated=(%d,%d) target=(%d,%d)",
                     status.path_id, status.state.name, point_text, status.segment.name,
                     status.estimated_x_mm, status.estimated_y_mm,
                     status.active_target_x_mm, status.active_target_y_mm)
        elif arguments.path_stop:
            link.stop_path()
            LOG.info("PATH_STOP 已获 ACK")
        elif arguments.execute_path:
            points = _parse_absolute_points(arguments.execute_path, arguments.path_rpm,
                                            arguments.path_acceleration)
            if len(points) != 1:
                parser.error("--execute-path 现在只允许一个单段绝对毫米目标；多段请用 --execute-node-route")
            link.start_path(arguments.path_id, points, not arguments.keep_path_origin)
            terminal = link.path_client.wait_until_terminal()
            if isinstance(terminal, ErrorEvent):
                raise PathProtocolError(f"路径失败：{terminal.error.name}")
            assert isinstance(terminal, DoneEvent)
            LOG.info("单段完成：估算坐标 (%d,%d)", terminal.estimated_x_mm,
                     terminal.estimated_y_mm)
        else:
            route_map = NineNodeRouteMap.from_config(config.get("node_route"))
            plan = route_map.plan(route_map.parse_path(arguments.execute_node_route))
            executor = RouteExecutor(link.path_client, route_map, arguments.path_rpm,
                                     arguments.path_acceleration)
            completions = executor.execute(plan, arguments.path_id)
            LOG.info("节点路线完成：%d 段；最终节点 %d", len(completions), plan.nodes[-1])
    except KeyboardInterrupt:
        link.stop_path()
        LOG.warning("用户取消，PATH_STOP 已获 ACK")
        return 130
    finally:
        link.close()
    return 0
