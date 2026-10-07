"""九点绝对航点路径协议的命令行入口。"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import time

from .config import load_config
from .fixed_trial import FixedTrialProfile, FixedTrialRunner
from .trial_recorder import TrialRecorder
from .calibration import FrameRectifier
from .node_route import NineNodeRouteMap
from .path_client import PathProtocolError
from .path_protocol import DoneEvent, ErrorEvent, PathPoint, PathResult, encode_arm_move
from .route_executor import RouteExecutor
from .ring_detection import RingDetector, draw_ring_debug
from .serial_link import SerialPathLink
from .task_code import TaskCode
from .vision import CameraVision, ObjectDetector, draw_object_debug_frame


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
        LOG.info("color=%s, center=(%.1f,%.1f), reference=(%.1f,%.1f), area=%.1f, "
                 "solidity=%.2f, stable=%s/%d, quality=%.2f",
                 item.color, item.center_x_px, item.center_y_px,
                 item.reference_point_px[0], item.reference_point_px[1], item.area_px,
                 item.solidity, item.stable, item.stable_frames, item.confidence)


def _run_vision_debug(config, camera_index: int | None, mode: str = "combined") -> int:
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
        ring_detector = RingDetector(config["ring_detection"], vision.cv2) if mode != "objects" else None
        rectifier = FrameRectifier(config["vision_calibration"].get("camera", {}), vision.cv2)
        cv2 = vision.cv2
        previous_time = time.perf_counter()
        while True:
            try:
                frame = rectifier.apply(vision.read())
            except RuntimeError as exc:
                LOG.error("物料识别相机读帧失败：%s", exc)
                return 1
            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            objects = vision.detect_objects(frame) if mode != "rings" else []
            rings = ring_detector.detect(frame) if ring_detector else []
            annotated = draw_object_debug_frame(frame, objects, fps, selected_index, camera_config)
            if rings:
                annotated = draw_ring_debug(annotated, rings)
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
                for item in rings:
                    LOG.info("ring=%s center=(%.1f,%.1f) arc=%.2f error=%.2f layers=%d "
                             "valid=%s stable=%s/%d reason=%s", item.position,
                             item.center_x_px, item.center_y_px, item.visible_arc_ratio,
                             item.fit_error_px, item.concentric_layers, item.valid,
                             item.stable, item.stable_frames, item.reason or "-")
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


def _run_vision_image(config, image_path: Path, output_path: Path | None,
                      mode: str = "combined") -> int:
    """对离线照片运行同一套检测；不打开任何相机、串口或机构接口。"""
    try:
        import cv2
    except ImportError:
        LOG.error("离线图片验证需要 OpenCV，请先安装 requirements.txt")
        return 1
    frame = cv2.imread(str(image_path))
    if frame is None:
        LOG.error("无法读取离线图片：%s", image_path)
        return 1
    frame = FrameRectifier(config["vision_calibration"].get("camera", {}), cv2).apply(frame)
    objects = ObjectDetector(config["object_camera"], config["hsv_colors"], cv2).detect(frame) \
        if mode != "rings" else []
    rings = RingDetector(config["ring_detection"], cv2).detect(frame) \
        if mode != "objects" else []
    annotated = draw_object_debug_frame(frame, objects, 0.0,
                                        int(config["object_camera"]["index"]),
                                        config["object_camera"])
    if rings:
        annotated = draw_ring_debug(annotated, rings)
    _print_objects(objects)
    for item in rings:
        LOG.info("ring=%s center=(%.1f,%.1f) valid=%s reason=%s", item.position,
                 item.center_x_px, item.center_y_px, item.valid, item.reason or "-")
    destination = output_path or image_path.with_name(f"{image_path.stem}_detected.jpg")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOG.error("无法创建离线检测输出目录：%s", exc)
        return 1
    if not cv2.imwrite(str(destination), annotated):
        LOG.error("无法保存离线检测结果：%s", destination)
        return 1
    LOG.info("离线检测结果：%s", destination.resolve())
    return 0


def main() -> int:
    """只提供路径预览、任务码校验和已由 F4 实现的路径命令。"""
    parser = argparse.ArgumentParser(description="智能搬运机器人九点路径控制")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--dry-run", action="store_true", help="仅校验任务码，不访问硬件")
    parser.add_argument("--task", help="与 --dry-run 同用，校验任务码，例如 123+123+321+231")
    parser.add_argument("--plan-path", help="只校验/规划九点路径，例如 1-2-3 或 1-2-3-6-9")
    parser.add_argument("--execute-path", help="下发 1..9 个绝对毫米航点，例如 '0,500;500,500'")
    parser.add_argument("--execute-node-route", help="按节点图逐段执行，例如 1-2-3-6；必须配置 node_mm_coordinates")
    parser.add_argument("--path-id", type=int, default=1, help="路径 ID，范围 1..255")
    parser.add_argument("--path-rpm", type=int, default=20, help="各航点 RPM，范围 1..5000")
    parser.add_argument("--path-acceleration", type=int, default=10, help="各航点加速度，范围 0..255")
    parser.add_argument("--keep-path-origin", action="store_true", help="启动前不重置 STM32 局部原点")
    parser.add_argument("--path-status", action="store_true", help="查询 STM32 当前路径状态后退出")
    parser.add_argument("--path-stop", action="store_true", help="停止 STM32 当前路径后退出")
    parser.add_argument("--arm-jog", type=int, metavar="ID",
                        help="手动点动机械臂电机 5/6/7；每次最多 32 脉冲")
    parser.add_argument("--arm-move", type=int, metavar="ID",
                        help="5/6/7 号一次相对位置运动；须显式指定方向和脉冲数")
    parser.add_argument("--arm-direction", choices=("+", "-"),
                        help="原始方向；7 号 + 为下降、- 为上升")
    parser.add_argument("--arm-pulses", type=int,
                        help="点动默认 8；ARM_MOVE 上限：5/6/7 号 3200/1600/6400")
    parser.add_argument("--arm-rpm", type=int,
                        help="仅 ARM_MOVE 使用；5/6/7 号默认 5/30/120 rpm")
    parser.add_argument("--arm-acceleration", type=int,
                        help="仅长行程使用；Emm_V5 加速度档 0..255，默认 250")
    parser.add_argument("--arm-stop", type=int, metavar="ID",
                        help="立即停止 5/6/7 号中的指定电机")
    parser.add_argument("--arm-status", type=int, metavar="ID",
                        help="查询指定机械臂电机的状态和编码器计数")
    parser.add_argument("--servo", nargs=2, metavar=("NAME", "ANGLE"),
                        help="设置单个舵机角度：rear 0..270 或 gripper 0..180")
    parser.add_argument("--fixed-trial-check", action="store_true",
                        help="预检固定单件流程，不连接机构")
    parser.add_argument("--fixed-trial-run", action="store_true",
                        help="从人工确认的参考位开始执行一轮固定流程")
    parser.add_argument("--profile", type=Path, help="固定单件流程 JSON 配置")
    parser.add_argument("--step", action="store_true", help="固定流程每步由操作员放行")
    parser.add_argument("--record", action="store_true", help="记录固定流程的命令、反馈和摘要")
    parser.add_argument("--vision-debug", action="store_true", help="独立调试物料识别相机，不访问 STM32")
    parser.add_argument("--vision-camera-index", type=int,
                        help="仅本次运行覆盖 object_camera.index")
    parser.add_argument("--vision-mode", choices=("objects", "rings", "combined"),
                        default="combined", help="物料/环/组合调试画面")
    parser.add_argument("--vision-image", type=Path,
                        help="验证单张离线图片，不打开相机或串口")
    parser.add_argument("--vision-output", type=Path,
                        help="--vision-image 的标注图片输出路径")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)

    fixed_selected = arguments.fixed_trial_check or arguments.fixed_trial_run
    if arguments.fixed_trial_check and arguments.fixed_trial_run:
        parser.error("--fixed-trial-check 与 --fixed-trial-run 不能同用")
    if fixed_selected:
        if arguments.profile is None:
            parser.error("固定流程必须提供 --profile")
        if arguments.fixed_trial_check and (arguments.step or arguments.record):
            parser.error("--step/--record 只能与 --fixed-trial-run 同用")
        conflicting = any((arguments.dry_run, arguments.task, arguments.plan_path,
                           arguments.execute_path, arguments.execute_node_route,
                           arguments.path_status, arguments.path_stop,
                           arguments.arm_jog, arguments.arm_move, arguments.arm_stop,
                           arguments.arm_status, arguments.servo,
                           arguments.vision_debug, arguments.vision_image))
        if conflicting:
            parser.error("固定流程不能与其他运行模式同用")
        try:
            profile = FixedTrialProfile.load(arguments.profile, config)
        except (OSError, ValueError) as exc:
            LOG.error("固定流程预检失败：%s", exc)
            return 2
        for line in profile.describe_steps():
            LOG.info("%s", line)
        if profile.issues:
            for issue in profile.issues:
                LOG.error("未就绪：%s", issue)
            LOG.error("固定流程共有 %d 项未完成；禁止连接机构或运行",
                      len(profile.issues))
            return 2
        if arguments.fixed_trial_check:
            LOG.info("固定流程预检通过：%d 步；未连接串口或机构", len(profile.steps))
            return 0
        recorder = TrialRecorder(arguments.profile, profile.name) if arguments.record else None
        try:
            link = SerialPathLink(**config["serial"])
        except Exception as exc:
            LOG.error("固定流程串口连接失败：%s", exc)
            if recorder is not None:
                recorder.event("连接失败", error_type=type(exc).__name__, error=str(exc))
                recorder.write_summary({"profile_name": profile.name, "success": False,
                                        "completed_steps": 0, "completed_commands": 0,
                                        "last_step": None,
                                        "stop_reason": f"串口连接失败：{exc}",
                                        "initial_encoders": {}, "final_encoders": {},
                                        "stage_parking_positions": []})
                recorder.close()
                LOG.info("本次记录：%s", recorder.output_dir.resolve())
            return 1
        try:
            return FixedTrialRunner(profile, link.path_client, recorder,
                                    step_mode=arguments.step).run()
        finally:
            link.close()
            if recorder is not None:
                LOG.info("本次记录：%s", recorder.output_dir.resolve())
    if arguments.profile is not None or arguments.step or arguments.record:
        parser.error("--profile/--step/--record 必须与固定流程模式同用")

    arm_selected = any(motor_id is not None for motor_id in
                       (arguments.arm_jog, arguments.arm_move,
                        arguments.arm_stop, arguments.arm_status))
    mechanism_selected = arm_selected or arguments.servo is not None
    if mechanism_selected and (arguments.vision_image or arguments.vision_debug or
                               arguments.dry_run or arguments.plan_path or arguments.task):
        parser.error("机构手动命令不能与视觉、任务码或路径预览模式同用")

    if arguments.vision_image:
        if arguments.vision_debug or arguments.vision_camera_index is not None:
            parser.error("--vision-image 不与相机调试参数同用")
        return _run_vision_image(config, arguments.vision_image,
                                 arguments.vision_output, arguments.vision_mode)
    if arguments.vision_output is not None:
        parser.error("--vision-output 必须与 --vision-image 同用")
    if arguments.vision_debug:
        return _run_vision_debug(config, arguments.vision_camera_index,
                                 arguments.vision_mode)
    if arguments.vision_camera_index is not None or arguments.vision_mode != "combined":
        parser.error("--vision-camera-index/--vision-mode 必须与视觉调试参数同用")

    if arguments.dry_run:
        if not arguments.task:
            parser.error("--dry-run 必须同时提供 --task")
        try:
            task = TaskCode.parse(arguments.task)
        except ValueError as exc:
            LOG.error("任务码无效：%s", exc)
            return 2
        LOG.info("任务码有效：%s", task.describe())
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
                        arguments.path_status, arguments.path_stop,
                        arguments.arm_jog, arguments.arm_move,
                        arguments.arm_stop, arguments.arm_status,
                        arguments.servo))
    if path_actions != 1:
        parser.error("必须选择一个路径或机械臂操作")
    arm_motion_id = arguments.arm_move if arguments.arm_move is not None else arguments.arm_jog
    if ((arguments.arm_direction is not None or arguments.arm_pulses is not None) and
            arm_motion_id is None):
        parser.error("--arm-direction/--arm-pulses 必须与 --arm-jog 或 --arm-move 同用")
    if ((arguments.arm_rpm is not None or arguments.arm_acceleration is not None) and
            arguments.arm_move is None):
        parser.error("--arm-rpm/--arm-acceleration 只能与 --arm-move 同用")
    for motor_id in (arguments.arm_jog, arguments.arm_move,
                     arguments.arm_stop, arguments.arm_status):
        if motor_id is not None and motor_id not in (5, 6, 7):
            parser.error("机械臂电机编号只能为 5、6、7")
    jog_pulses = arguments.arm_pulses if arguments.arm_pulses is not None else 8
    if arguments.arm_jog is not None and not 1 <= jog_pulses <= 32:
        parser.error("--arm-pulses 必须在 1..32")
    move_rpm = None
    move_acceleration = None
    servo_name = None
    servo_angle_tenths = None
    servo_config = None
    if arguments.servo is not None:
        servo_name, raw_angle = arguments.servo
        if servo_name not in config["servos"]:
            parser.error("--servo NAME 只能为 rear 或 gripper")
        try:
            angle = float(raw_angle)
        except ValueError:
            parser.error("--servo ANGLE 必须是数字")
        if not math.isfinite(angle):
            parser.error("--servo ANGLE 必须是有限数字")
        servo_config = config["servos"][servo_name]
        if not servo_config["enabled"]:
            parser.error(f"舵机 {servo_name} 当前配置为 enabled=false")
        servo_angle_tenths = round(angle * 10)
        if abs(angle * 10 - servo_angle_tenths) > 1e-6:
            parser.error("--servo ANGLE 最多保留一位小数")
        if not 0 <= angle <= float(servo_config["max_angle_deg"]):
            parser.error(
                f"{servo_name} 角度必须在 0..{servo_config['max_angle_deg']}°")
    if arguments.arm_move is not None:
        if arguments.arm_move not in (5, 6, 7):
            parser.error("--arm-move 只支持 5、6、7 号电机")
        if arguments.arm_direction is None or arguments.arm_pulses is None:
            parser.error("--arm-move 必须显式指定 --arm-direction 和 --arm-pulses")
        move_rpm = arguments.arm_rpm if arguments.arm_rpm is not None else {5: 5, 6: 30, 7: 120}[arguments.arm_move]
        move_acceleration = (arguments.arm_acceleration if arguments.arm_acceleration is not None
                             else 250)
        try:
            encode_arm_move(arguments.arm_move,
                            1 if arguments.arm_direction == "+" else -1,
                            arguments.arm_pulses, move_rpm, move_acceleration)
        except ValueError as exc:
            parser.error(str(exc))
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
        elif arguments.arm_status is not None:
            status = link.path_client.arm_status(arguments.arm_status)
            LOG.info("机械臂 %d 号：%s，结果=%s，编码器计数=%d，反向标志=%s",
                     status.motor_id, status.state.name, status.result.name,
                     status.encoder_count, status.reverse)
        elif arguments.arm_stop is not None:
            link.path_client.arm_stop(arguments.arm_stop)
            LOG.info("机械臂 %d 号停止命令已获 ACK", arguments.arm_stop)
        elif servo_config is not None:
            assert servo_name is not None and servo_angle_tenths is not None
            link.path_client.set_servo_angle(int(servo_config["id"]), servo_angle_tenths)
            LOG.info("舵机 %s（ID=%d）目标 %.1f° 已获 ACK；等待 %.2fs 仅用于调试时序",
                     servo_name, servo_config["id"], servo_angle_tenths / 10.0,
                     servo_config["wait_s"])
            time.sleep(float(servo_config["wait_s"]))
        elif arguments.arm_jog is not None:
            jog_direction = arguments.arm_direction or "+"
            direction = 1 if jog_direction == "+" else -1
            LOG.warning("即将点动 %d 号电机：原始方向 %s，%d 脉冲；现场观察实际方向",
                        arguments.arm_jog, jog_direction, jog_pulses)
            try:
                link.path_client.arm_jog(arguments.arm_jog, direction, jog_pulses)
                terminal = link.path_client.wait_arm_done(arguments.arm_jog)
                if terminal.result != PathResult.OK:
                    raise PathProtocolError(f"机械臂 {terminal.motor_id} 号未完成：{terminal.result.name}")
            except Exception:
                try:
                    link.path_client.arm_stop(arguments.arm_jog)
                except Exception as stop_error:
                    LOG.error("点动失败后停止命令未确认：%s；请检查实物并切断电机电源", stop_error)
                raise
            LOG.info("机械臂 %d 号点动完成：编码器计数=%d，反向标志=%s",
                     terminal.motor_id, terminal.encoder_count, terminal.reverse)
        elif arguments.arm_move is not None:
            direction = 1 if arguments.arm_direction == "+" else -1
            assert move_rpm is not None and move_acceleration is not None
            LOG.warning("即将移动 %d 号电机：方向 %s，%d 脉冲，%d rpm，加速度档 %d",
                        arguments.arm_move, arguments.arm_direction, arguments.arm_pulses,
                        move_rpm, move_acceleration)
            try:
                link.path_client.arm_move(arguments.arm_move, direction, arguments.arm_pulses,
                                          move_rpm, move_acceleration)
                nominal_s = arguments.arm_pulses * 60.0 / (3200.0 * move_rpm)
                terminal = link.path_client.wait_arm_done(arguments.arm_move,
                                                          timeout_s=8.0 + 2.0 * nominal_s)
                if terminal.result != PathResult.OK:
                    raise PathProtocolError(f"机械臂 {terminal.motor_id} 号未完成：{terminal.result.name}")
            except Exception:
                try:
                    link.path_client.arm_stop(arguments.arm_move)
                except Exception as stop_error:
                    LOG.error("移动失败后停止命令未确认：%s；请检查实物并切断电机电源", stop_error)
                raise
            LOG.info("机械臂 %d 号移动完成：编码器计数=%d，反向标志=%s",
                     terminal.motor_id, terminal.encoder_count, terminal.reverse)
        elif arguments.path_stop:
            link.stop_path()
            LOG.info("PATH_STOP 已获 ACK")
        elif arguments.execute_path:
            points = _parse_absolute_points(arguments.execute_path, arguments.path_rpm,
                                            arguments.path_acceleration)
            if len(points) != 1:
                parser.error("--execute-path 现在只允许一个单段绝对毫米目标；多段请用 --execute-node-route")
            link.start_path(arguments.path_id, points, not arguments.keep_path_origin)
            terminal = link.path_client.wait_until_terminal_with_keepalive()
            if isinstance(terminal, ErrorEvent):
                raise PathProtocolError(f"路径失败：{terminal.error.name}")
            assert isinstance(terminal, DoneEvent)
            if terminal.path_id != arguments.path_id:
                raise PathProtocolError(
                    f"单段完成事件路径 ID 不匹配：期望 {arguments.path_id}，实际 {terminal.path_id}")
            LOG.info("单段完成：估算坐标 (%d,%d)", terminal.estimated_x_mm,
                     terminal.estimated_y_mm)
        else:
            route_config = config.get("node_route") or {}
            route_map = NineNodeRouteMap.from_config(route_config)
            plan = route_map.plan(route_map.parse_path(arguments.execute_node_route))
            initial_exit = route_config.get("initial_exit_mm", {})
            executor = RouteExecutor(link.path_client, route_map, arguments.path_rpm,
                                     arguments.path_acceleration,
                                     initial_exit_mm=(initial_exit.get("forward", 0),
                                                      initial_exit.get("left", 0)))
            completions = executor.execute(plan, arguments.path_id)
            LOG.info("节点路线完成：%d 段；最终节点 %d", len(completions), plan.nodes[-1])
    except KeyboardInterrupt:
        if arm_motion_id is not None:
            link.path_client.arm_stop(arm_motion_id)
            LOG.warning("用户取消，机械臂 %d 号停止命令已获 ACK", arm_motion_id)
        elif servo_config is not None:
            LOG.warning("用户取消等待；舵机 PWM 保持最后一次已确认的目标")
        else:
            link.stop_path()
            LOG.warning("用户取消，PATH_STOP 已获 ACK")
        return 130
    finally:
        link.close()
    return 0
