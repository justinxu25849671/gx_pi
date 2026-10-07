"""无视觉、无规划的固定单件取放试验。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable

from .path_client import PathProtocolError
from .path_protocol import DoneEvent, ErrorEvent, PathPoint, PathResult
from .trial_recorder import TrialRecorder


LOG = logging.getLogger(__name__)
PROTOCOL_MAX_PULSES = {5: 3200, 6: 1600, 7: 6400}
PROTOCOL_MAX_RPM = {5: 10, 6: 60, 7: 180}


@dataclass(frozen=True)
class AxisConfig:
    minimum: int | None
    maximum: int | None
    max_single_move: int | None
    rpm: int | None
    acceleration: int | None
    positive_motion: str | None
    negative_motion: str | None


@dataclass(frozen=True)
class TrialStep:
    stage: str
    name: str
    kind: str
    axis: int | None = None
    target: int | None = None
    angle_deg: float | None = None
    wait_s: float | None = None
    path_point: PathPoint | None = None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class FixedTrialProfile:
    """解析动作参数，并在连接硬件前完成全表累计行程检查。"""

    REQUIRED_POSES = {
        "observation": (5, 6, 7),
        "platform_pick": (6, 7),
        "transport": (6, 7),
        "rear_place": (5, 6, 7),
        "rear_pick": (5, 6, 7),
        "ring_place": (5, 6, 7),
        "ring_pick": (5, 6, 7),
    }

    def __init__(self, path: Path, raw: dict[str, Any], base_config: dict[str, Any]) -> None:
        self.path = path
        self.raw = raw
        self.name = str(raw.get("name") or "fixed_trial")
        self.issues: list[str] = []
        self.axes = self._parse_axes(raw.get("axes"))
        self.poses = self._parse_poses(raw.get("poses"))
        self.open_angle, self.closed_angle, self.gripper_wait_s = self._parse_gripper(
            raw.get("gripper"))
        self.path_points = self._parse_path(raw.get("chassis_segments"))
        self._validate_metadata(raw, base_config)
        self.steps = self._build_steps()
        self._validate_motion_table()

    @classmethod
    def load(cls, path: str | Path, base_config: dict[str, Any]) -> "FixedTrialProfile":
        source = Path(path)
        with source.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("固定流程配置顶层必须是 JSON 对象")
        return cls(source, raw, base_config)

    def _parse_axes(self, value: Any) -> dict[int, AxisConfig]:
        axes: dict[int, AxisConfig] = {}
        if not isinstance(value, dict):
            self.issues.append("缺少 axes 配置")
            value = {}
        for axis in (5, 6, 7):
            item = value.get(str(axis), {})
            if not isinstance(item, dict):
                item = {}
            axes[axis] = AxisConfig(
                _optional_int(item.get("min_position_pulses")),
                _optional_int(item.get("max_position_pulses")),
                _optional_int(item.get("max_single_move_pulses")),
                _optional_int(item.get("rpm")),
                _optional_int(item.get("acceleration")),
                item.get("positive_motion") if isinstance(item.get("positive_motion"), str) else None,
                item.get("negative_motion") if isinstance(item.get("negative_motion"), str) else None,
            )
        return axes

    def _parse_poses(self, value: Any) -> dict[str, dict[int, int | None]]:
        poses: dict[str, dict[int, int | None]] = {}
        if not isinstance(value, dict):
            self.issues.append("缺少 poses 配置")
            value = {}
        for name, required_axes in self.REQUIRED_POSES.items():
            item = value.get(name, {})
            if not isinstance(item, dict):
                item = {}
            poses[name] = {axis: _optional_int(item.get(str(axis))) for axis in required_axes}
        return poses

    def _parse_gripper(self, value: Any) -> tuple[float | None, float | None, float | None]:
        item = value if isinstance(value, dict) else {}
        def number(name: str) -> float | None:
            raw = item.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
            return None
        return number("open_angle_deg"), number("closed_angle_deg"), number("wait_s")

    def _parse_path(self, value: Any) -> tuple[PathPoint | None, ...]:
        if not isinstance(value, list):
            self.issues.append("chassis_segments 必须是数组；台架试验可使用空数组")
            return ()
        result: list[PathPoint | None] = []
        for index, item in enumerate(value, 1):
            if not isinstance(item, dict):
                self.issues.append(f"底盘第 {index} 段必须是对象")
                result.append(None)
                continue
            values = [_optional_int(item.get(key)) for key in
                      ("target_x_mm", "target_y_mm", "rpm", "acceleration")]
            if any(part is None for part in values):
                self.issues.append(f"底盘第 {index} 段存在未测量字段")
                result.append(None)
                continue
            point = PathPoint(values[0], values[1], values[2], values[3])
            try:
                point.validate()
            except ValueError as exc:
                self.issues.append(f"底盘第 {index} 段：{exc}")
                result.append(None)
            else:
                result.append(point)
        return tuple(result)

    def _validate_metadata(self, raw: dict[str, Any], base_config: dict[str, Any]) -> None:
        if raw.get("version") != 1:
            self.issues.append("version 必须为 1")
        ring_id = _optional_int(raw.get("target_ring_id"))
        if ring_id is None or ring_id < 1:
            self.issues.append("target_ring_id 必须是已确认的正整数")
        material = raw.get("material", {})
        if not isinstance(material, dict) or material.get("top_diameter_mm") != 30 or \
                material.get("bottom_diameter_mm") != 50 or material.get("height_mm") != 60:
            self.issues.append("物料尺寸必须保留上径 30、底径 50、高 60 mm")
        if not isinstance(material, dict) or material.get("table_top_height_mm") != 140:
            self.issues.append("物料台顶点必须记录为 140 mm；抓取脉冲仍需实测")
        servos = base_config.get("servos", {})
        if servos.get("rear", {}).get("enabled") is not False:
            self.issues.append("config.json 中 servos.rear.enabled 必须为 false")
        if servos.get("gripper", {}).get("enabled") is not True:
            self.issues.append("config.json 中 servos.gripper.enabled 必须为 true")
        if self.open_angle != 110.0 or self.closed_angle != 180.0:
            self.issues.append("夹爪张开/闭合角度必须分别为 110°/180°")
        if self.gripper_wait_s is None or self.gripper_wait_s <= 0:
            self.issues.append("夹爪 wait_s 必须是经实测的正数")
        for axis, cfg in self.axes.items():
            values = (cfg.minimum, cfg.maximum, cfg.max_single_move, cfg.rpm, cfg.acceleration)
            if any(value is None for value in values):
                self.issues.append(f"{axis} 号轴行程/单次脉冲/RPM/加速度存在未测量值")
            if not cfg.positive_motion or not cfg.negative_motion:
                self.issues.append(f"{axis} 号轴正反方向说明未实测填写")
            if any(value is None for value in values):
                continue
            assert cfg.minimum is not None and cfg.maximum is not None
            assert cfg.max_single_move is not None and cfg.rpm is not None
            assert cfg.acceleration is not None
            if not cfg.minimum <= 0 <= cfg.maximum or cfg.minimum >= cfg.maximum:
                self.issues.append(f"{axis} 号轴安全范围必须包含人工参考位 0")
            if not 1 <= cfg.max_single_move <= PROTOCOL_MAX_PULSES[axis]:
                self.issues.append(f"{axis} 号轴 max_single_move_pulses 超出协议上限")
            if not 1 <= cfg.rpm <= PROTOCOL_MAX_RPM[axis]:
                self.issues.append(f"{axis} 号轴 rpm 超出协议上限")
            if not 0 <= cfg.acceleration <= 255:
                self.issues.append(f"{axis} 号轴 acceleration 必须在 0..255")

    def _pose_steps(self, stage: str, pose: str, axes: tuple[int, ...], prefix: str) -> list[TrialStep]:
        return [TrialStep(stage, f"{prefix}：{axis} 号轴到 {pose}", "arm",
                          axis=axis, target=self.poses[pose].get(axis)) for axis in axes]

    def _gripper(self, stage: str, name: str, closed: bool) -> TrialStep:
        return TrialStep(stage, name, "gripper",
                         angle_deg=self.closed_angle if closed else self.open_angle,
                         wait_s=self.gripper_wait_s)

    def _build_steps(self) -> tuple[TrialStep, ...]:
        steps: list[TrialStep] = []
        steps += self._pose_steps("物料台取料", "observation", (5, 6, 7), "到观察位")
        steps.append(self._gripper("物料台取料", "夹爪张开到 110°", False))
        steps += self._pose_steps("物料台取料", "platform_pick", (6,), "伸缩到抓取位")
        steps += self._pose_steps("物料台取料", "platform_pick", (7,), "下降到抓取高度")
        steps.append(self._gripper("物料台取料", "夹爪闭合到 180°", True))
        steps += self._pose_steps("物料台取料", "transport", (7, 6), "撤回运输位")

        steps += self._pose_steps("车后放料", "rear_place", (5, 6, 7), "到车后放料位")
        steps.append(self._gripper("车后放料", "夹爪张开松开", False))
        steps += self._pose_steps("车后放料", "transport", (7, 6), "撤回运输位")

        steps += self._pose_steps("车后取料", "rear_pick", (5, 6, 7), "到车后取料位")
        steps.append(self._gripper("车后取料", "夹爪闭合", True))
        steps += self._pose_steps("车后取料", "transport", (7, 6), "撤回运输位")

        for index, point in enumerate(self.path_points, 1):
            steps.append(TrialStep("地面色环放料", f"底盘固定行驶段 {index}", "path",
                                   path_point=point))
        steps += self._pose_steps("地面色环放料", "ring_place", (5, 6, 7), "到色环放料位")
        steps.append(self._gripper("地面色环放料", "夹爪张开松开一次", False))
        steps += self._pose_steps("地面色环放料", "transport", (7, 6), "撤回运输位")

        steps += self._pose_steps("地面取回", "ring_pick", (5, 6, 7), "到色环取料位")
        steps.append(self._gripper("地面取回", "夹爪闭合", True))
        steps += self._pose_steps("地面取回", "transport", (7, 6), "撤回运输位")

        steps += self._pose_steps("放回车后", "rear_place", (5, 6, 7), "到车后放料位")
        steps.append(self._gripper("放回车后", "夹爪张开松开", False))
        steps += self._pose_steps("放回车后", "transport", (7, 6), "撤回运输位")
        return tuple(steps)

    def _validate_motion_table(self) -> None:
        positions = {5: 0, 6: 0, 7: 0}
        for pose, required_axes in self.REQUIRED_POSES.items():
            for axis in required_axes:
                if self.poses[pose].get(axis) is None:
                    self.issues.append(f"poses.{pose}.{axis} 未实测填写")
        for index, step in enumerate(self.steps, 1):
            if step.kind != "arm" or step.axis is None or step.target is None:
                continue
            cfg = self.axes[step.axis]
            if cfg.minimum is not None and cfg.maximum is not None and not (
                    cfg.minimum <= step.target <= cfg.maximum):
                self.issues.append(f"第 {index} 步 {step.name} 目标 {step.target} 超出轴安全范围")
            delta = abs(step.target - positions[step.axis])
            if cfg.max_single_move is not None and delta > cfg.max_single_move:
                self.issues.append(f"第 {index} 步 {step.name} 单次 {delta} 脉冲超限")
            positions[step.axis] = step.target

    def ensure_ready(self) -> None:
        if self.issues:
            raise ValueError("配置未完成，禁止运行：" + "；".join(self.issues))

    def describe_steps(self) -> tuple[str, ...]:
        lines = []
        positions = {5: 0, 6: 0, 7: 0}
        for index, step in enumerate(self.steps, 1):
            if step.kind == "arm":
                if step.target is None or step.axis is None:
                    detail = f"{step.axis} 号轴目标=待测"
                else:
                    delta = step.target - positions[step.axis]
                    detail = f"{step.axis} 号轴目标={step.target} 脉冲，本步变化={delta:+d}"
                    positions[step.axis] = step.target
            elif step.kind == "gripper":
                detail = f"夹爪={step.angle_deg if step.angle_deg is not None else '待测'}°"
            else:
                point = step.path_point
                detail = (f"累计目标=({point.x_mm},{point.y_mm}) mm"
                          if point is not None else "底盘目标=待测")
            lines.append(f"{index:02d}. [{step.stage}] {step.name}；{detail}")
        return tuple(lines)


class FixedTrialRunner:
    def __init__(self, profile: FixedTrialProfile, client: Any,
                 recorder: TrialRecorder | None, step_mode: bool = False,
                 input_fn: Callable[[str], str] = input,
                 sleep_fn: Callable[[float], None] = time.sleep) -> None:
        self.profile = profile
        self.client = client
        self.recorder = recorder
        self.step_mode = step_mode
        self.input_fn = input_fn
        self.sleep_fn = sleep_fn
        self.positions = {5: 0, 6: 0, 7: 0}
        self.initial_encoders: dict[str, Any] = {}
        self.final_encoders: dict[str, Any] = {}
        self.stage_parking_positions: list[dict[str, Any]] = []
        self.completed_steps = 0
        self.completed_commands = 0
        self.last_step: str | None = None
        self._active_kind: str | None = None
        self._active_axis: int | None = None
        self._active_command_number: int | None = None
        self._next_path_id = 1
        self._path_started = False

    @staticmethod
    def _status_dict(status: Any) -> dict[str, Any]:
        return {"encoder_count": status.encoder_count, "reverse": status.reverse,
                "state": status.state.name, "result": status.result.name}

    def _read_encoders(self) -> dict[str, Any]:
        return {str(axis): self._status_dict(self.client.arm_status(axis)) for axis in (5, 6, 7)}

    def _record_prepare(self, **values: Any) -> int:
        if self.recorder is None:
            number = self.completed_commands + 1
        else:
            number = self.recorder.prepare(**values)
        self._active_command_number = number
        return number

    def _record_result(self, command_number: int, **values: Any) -> None:
        if self.recorder is not None:
            self.recorder.result(command_number, **values)
        if self._active_command_number == command_number:
            self._active_command_number = None

    def _record_active_error(self, exc: BaseException) -> None:
        if self._active_command_number is not None:
            self._record_result(self._active_command_number, status="error",
                                error_type=type(exc).__name__, error=str(exc))

    def _execute_arm(self, step_index: int, step: TrialStep) -> None:
        assert step.axis is not None and step.target is not None
        delta = step.target - self.positions[step.axis]
        before = self.client.arm_status(step.axis)
        common = {"step_index": step_index, "stage": step.stage, "step_name": step.name,
                  "kind": "arm", "axis": step.axis, "target_position_pulses": step.target,
                  "direction": "+" if delta > 0 else "-" if delta < 0 else None,
                  "target_pulses": abs(delta), "encoder_before": self._status_dict(before)}
        if delta == 0:
            if self.recorder is not None:
                self.recorder.event("步骤跳过", **common, status="already_at_target",
                                    encoder_after=self._status_dict(before))
            return
        cfg = self.profile.axes[step.axis]
        assert cfg.rpm is not None and cfg.acceleration is not None
        common.update({"rpm": cfg.rpm, "acceleration": cfg.acceleration})
        number = self._record_prepare(**common)
        started = time.monotonic()
        self._active_kind, self._active_axis = "arm", step.axis
        ack = self.client.arm_move(step.axis, 1 if delta > 0 else -1, abs(delta),
                                   cfg.rpm, cfg.acceleration)
        nominal_s = abs(delta) * 60.0 / (3200.0 * cfg.rpm)
        done = self.client.wait_arm_done(step.axis, timeout_s=8.0 + 2.0 * nominal_s)
        if done.result != PathResult.OK:
            raise PathProtocolError(f"{step.name} 未完成：{done.result.name}")
        self._active_kind, self._active_axis = None, None
        self.positions[step.axis] = step.target
        self._record_result(number, status="command_completed",
                            ack={"accepted": ack.accepted, "result": ack.result.name,
                                 "request_sequence": ack.request_sequence},
                            arm_done={"result": done.result.name,
                                      "request_sequence": done.request_sequence},
                            encoder_after={"encoder_count": done.encoder_count,
                                           "reverse": done.reverse},
                            elapsed_s=round(time.monotonic() - started, 3))
        self.completed_commands += 1

    def _execute_gripper(self, step_index: int, step: TrialStep) -> None:
        assert step.angle_deg is not None and step.wait_s is not None
        number = self._record_prepare(step_index=step_index, stage=step.stage,
                                      step_name=step.name, kind="gripper",
                                      servo_id=1, gripper_angle_deg=step.angle_deg,
                                      wait_s=step.wait_s)
        started = time.monotonic()
        ack = self.client.set_servo_angle(1, round(step.angle_deg * 10))
        self.sleep_fn(step.wait_s)
        self._record_result(number, status="angle_command_acknowledged",
                            ack={"accepted": ack.accepted, "result": ack.result.name,
                                 "request_sequence": ack.request_sequence},
                            physical_grip_confirmed=False,
                            elapsed_s=round(time.monotonic() - started, 3))
        self.completed_commands += 1

    def _execute_path(self, step_index: int, step: TrialStep) -> None:
        assert step.path_point is not None
        path_id = self._next_path_id
        point = step.path_point
        number = self._record_prepare(
            step_index=step_index, stage=step.stage, step_name=step.name, kind="path",
            path_id=path_id, target_x_mm=point.x_mm, target_y_mm=point.y_mm,
            rpm=point.rpm, acceleration=point.acceleration,
            reset_origin=not self._path_started)
        started = time.monotonic()
        self._active_kind, self._active_axis = "path", None
        self.client.upload_and_start(path_id, (point,), reset_origin=not self._path_started)
        terminal = self.client.wait_until_terminal_with_keepalive()
        if isinstance(terminal, ErrorEvent):
            raise PathProtocolError(f"{step.name} 失败：{terminal.error.name}")
        if not isinstance(terminal, DoneEvent) or terminal.path_id != path_id:
            raise PathProtocolError(f"{step.name} DONE 与 path_id 不匹配")
        self._active_kind = None
        self._path_started = True
        self._record_result(number, status="command_completed",
                            protocol_sequence=self.client.last_path_start_sequence,
                            path_done={"path_id": terminal.path_id,
                                       "estimated_x_mm": terminal.estimated_x_mm,
                                       "estimated_y_mm": terminal.estimated_y_mm},
                            elapsed_s=round(time.monotonic() - started, 3))
        self.stage_parking_positions.append({
            "stage": step.stage,
            "step_name": step.name,
            "path_id": terminal.path_id,
            "target_x_mm": point.x_mm,
            "target_y_mm": point.y_mm,
            "estimated_x_mm": terminal.estimated_x_mm,
            "estimated_y_mm": terminal.estimated_y_mm,
        })
        self.completed_commands += 1
        self._next_path_id = path_id % 255 + 1

    def _stop_active(self) -> str | None:
        try:
            if self._active_kind == "arm" and self._active_axis is not None:
                self.client.arm_stop(self._active_axis)
                return f"ARM_STOP({self._active_axis}) ACK"
            if self._active_kind == "path":
                self.client.stop()
                return "PATH_STOP ACK"
        except Exception as exc:
            return f"停止命令未确认：{exc}"
        return None

    def _finish(self, success: bool, reason: str) -> None:
        try:
            self.final_encoders = self._read_encoders()
        except Exception as exc:
            self.final_encoders = {"read_error": str(exc)}
        if self.recorder is not None:
            self.recorder.write_summary({
                "profile_name": self.profile.name,
                "success": success,
                "completed_steps": self.completed_steps,
                "completed_commands": self.completed_commands,
                "last_step": self.last_step,
                "stop_reason": reason,
                "initial_encoders": self.initial_encoders,
                "final_encoders": self.final_encoders,
                "logical_final_positions_pulses": {str(k): v for k, v in self.positions.items()},
                "stage_parking_positions": self.stage_parking_positions,
                "physical_pick_or_place_confirmed": False,
            })
            self.recorder.close()

    def run(self) -> int:
        try:
            self.profile.ensure_ready()
        except ValueError as exc:
            LOG.error("%s", exc)
            self._finish(False, str(exc))
            return 2
        answer = self.input_fn(
            "请确认 5/6/7 号轴已人工低速移到启动标记，周围无人、无干涉。输入 RUN 开始：")
        if answer.strip() != "RUN":
            self._finish(False, "操作员未确认人工参考位")
            return 2
        try:
            self.initial_encoders = self._read_encoders()
            if self.recorder is not None:
                self.recorder.event("建立本次参考位", encoders=self.initial_encoders)
            for index, step in enumerate(self.profile.steps, 1):
                self.last_step = step.name
                if self.step_mode:
                    answer = self.input_fn(f"{index:02d}/{len(self.profile.steps)} {step.name}；输入 GO 放行：")
                    if answer.strip() != "GO":
                        raise KeyboardInterrupt("操作员拒绝放行当前步骤")
                LOG.info("%02d/%02d [%s] %s", index, len(self.profile.steps), step.stage, step.name)
                if step.kind == "arm":
                    self._execute_arm(index, step)
                elif step.kind == "gripper":
                    self._execute_gripper(index, step)
                else:
                    self._execute_path(index, step)
                self.completed_steps += 1
        except KeyboardInterrupt as exc:
            self._record_active_error(exc)
            stop = self._stop_active()
            reason = str(exc) or "用户中断"
            if stop:
                reason += f"；{stop}"
            LOG.warning("固定流程已停止：%s", reason)
            self._finish(False, reason)
            return 130
        except Exception as exc:
            self._record_active_error(exc)
            stop = self._stop_active()
            reason = f"{type(exc).__name__}: {exc}"
            if stop:
                reason += f"；{stop}"
            LOG.error("固定流程失败：%s", reason)
            self._finish(False, reason)
            return 1
        self._finish(True, "固定单件流程完成")
        LOG.info("固定单件流程完成；仅确认命令与回包，实物结果请填写 field_notes.md")
        return 0
