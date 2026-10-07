"""固定单件试验的可追溯记录。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


class TrialRecorder:
    """先落盘准备事件，再允许调用方下发真实命令。"""

    def __init__(self, profile_path: Path, profile_name: str,
                 root: Path | None = None) -> None:
        now = datetime.now().astimezone()
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", profile_name).strip("_") or "trial"
        parent = root or Path("logs") / "fixed_trial"
        self.output_dir = parent / f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{safe_name}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(profile_path, self.output_dir / "profile_used.json")
        self._events = (self.output_dir / "events.jsonl").open("a", encoding="utf-8")
        self._command_number = 0
        (self.output_dir / "field_notes.md").write_text(
            "# 现场观察\n\n"
            "- 实际是否夹住：\n"
            "- 是否碰撞或卡滞：\n"
            "- 夹爪是否偏心：\n"
            "- 地面落点偏差：\n"
            "- 需要增减脉冲的步骤：\n",
            encoding="utf-8")

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    @property
    def next_command_number(self) -> int:
        return self._command_number + 1

    def event(self, event_type: str, **values: Any) -> None:
        item = {"time": self._now(), "event": event_type, **values}
        self._events.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())

    def prepare(self, **values: Any) -> int:
        self._command_number += 1
        self.event("准备执行", command_number=self._command_number, **values)
        return self._command_number

    def result(self, command_number: int, **values: Any) -> None:
        self.event("执行结果", command_number=command_number, **values)

    def write_summary(self, summary: dict[str, Any]) -> None:
        destination = self.output_dir / "summary.json"
        temporary = self.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps({"written_at": self._now(), **summary},
                                        ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(destination)

    def close(self) -> None:
        if not self._events.closed:
            self._events.close()

