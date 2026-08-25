"""路径任务严格顺序、请求关联、错误和超时确认测试。"""

import struct
import unittest

from logistics_robot.path_client import PathClient, PathNackError
from logistics_robot.path_protocol import (
    FrameDecoder, PathCommand, PathEventCode, PathPoint, PathResult, PathState,
    encode_frame,
)


class FakeStm32:
    def __init__(self) -> None:
        self.client = PathClient(self.write, request_timeout_s=0.01)
        self.commands = []
        self.nack_command = None
        self.drop_start_ack = False
        self.output_sequence = 0

    def _send(self, command, payload) -> None:
        self.output_sequence = self.output_sequence % 255 + 1
        self.client.feed(encode_frame(command, self.output_sequence, payload))

    def write(self, raw: bytes) -> None:
        frame = FrameDecoder().feed(raw)[0]
        command = PathCommand(frame.command)
        self.commands.append((command, frame.sequence, frame.payload))
        if command == PathCommand.STATUS_REQ:
            state = PathState.RUN_X if self.drop_start_ack else PathState.IDLE
            payload = struct.pack("<BBBBBiiii", 3, state, 0, 0, 2, 0, 0, 100, 200)
            self._send(PathEventCode.STATUS, payload)
            return
        if command == self.nack_command:
            self._send(PathEventCode.NACK,
                       bytes((frame.sequence, 0, PathResult.BAD_PARAM)))
            return
        if command == PathCommand.START and self.drop_start_ack:
            return
        path_id = frame.payload[0] if command in (PathCommand.UPLOAD, PathCommand.START) else 0
        self._send(PathEventCode.ACK, bytes((frame.sequence, path_id, PathResult.OK)))


class PathClientTests(unittest.TestCase):
    def test_normal_flow_waits_in_exact_order(self) -> None:
        fake = FakeStm32()
        points = (PathPoint(0, 500), PathPoint(500, 500))
        fake.client.upload_and_start(3, points)
        self.assertEqual([item[0] for item in fake.commands], [
            PathCommand.CLEAR, PathCommand.RESET_ORIGIN,
            PathCommand.UPLOAD, PathCommand.START,
        ])
        self.assertEqual(fake.commands[2][2][:2], b"\x03\x02")

    def test_nack_stops_following_commands(self) -> None:
        fake = FakeStm32()
        fake.nack_command = PathCommand.RESET_ORIGIN
        with self.assertRaises(PathNackError):
            fake.client.upload_and_start(3, (PathPoint(0, 500),))
        self.assertEqual([item[0] for item in fake.commands],
                         [PathCommand.CLEAR, PathCommand.RESET_ORIGIN])

    def test_start_ack_loss_queries_status_without_retry(self) -> None:
        fake = FakeStm32()
        fake.drop_start_ack = True
        status = fake.client.upload_and_start(3, (PathPoint(100, 200),))
        self.assertEqual(status.state, PathState.RUN_X)
        self.assertEqual([item[0] for item in fake.commands].count(PathCommand.START), 1)
        self.assertEqual(fake.commands[-1][0], PathCommand.STATUS_REQ)

    def test_point_done_done_and_error_update_task(self) -> None:
        fake = FakeStm32()
        fake.client.upload_and_start(3, (PathPoint(100, 200),))
        point_payload = struct.pack("<BBiiii", 3, 0, 100, 200, 100, 200)
        fake._send(PathEventCode.POINT_DONE, point_payload)
        self.assertTrue(fake.client.task_active)
        done_payload = struct.pack("<BBii", 3, 1, 100, 200)
        fake._send(PathEventCode.DONE, done_payload)
        terminal = fake.client.wait_until_terminal(0.01)
        self.assertEqual(terminal.path_id, 3)
        self.assertFalse(fake.client.task_active)

    def test_stop_uses_protocol_ack(self) -> None:
        fake = FakeStm32()
        fake.client.stop()
        self.assertEqual(fake.commands[-1][0], PathCommand.STOP)

    def test_reconnect_clears_partial_decoder(self) -> None:
        fake = FakeStm32()
        fake.client.feed(b"\xAA\x55\x01")
        fake.client.connection_lost()
        fake.client.connection_restored()
        status = fake.client.query_status()
        self.assertEqual(status.state, PathState.IDLE)


if __name__ == "__main__":
    unittest.main()
