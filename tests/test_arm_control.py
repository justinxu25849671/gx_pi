"""机械臂手动点动的上位机编解码与请求关联。"""

import struct
import unittest

from logistics_robot.path_client import PathClient, PathNackError, PathRequestTimeout
from logistics_robot.path_protocol import (
    ArmState, FrameDecoder, PathCommand, PathEventCode, PathResult,
    encode_arm_jog, encode_frame,
)


class FakeArmStm32:
    def __init__(self) -> None:
        self.commands = []
        self.nack = False
        self.client = PathClient(self.write, request_timeout_s=0.01)

    def send(self, code, payload):
        self.client.feed(encode_frame(code, 1, payload))

    def write(self, raw):
        frame = FrameDecoder().feed(raw)[0]
        self.commands.append(frame)
        motor_id = frame.payload[0]
        if self.nack:
            self.send(PathEventCode.NACK,
                      bytes((frame.sequence, motor_id, PathResult.BUSY)))
        elif frame.command == PathCommand.ARM_STATUS_REQ:
            self.send(PathEventCode.ARM_STATUS,
                      struct.pack("<BBBBIB", frame.sequence, motor_id,
                                  ArmState.MOVING, PathResult.OK, 1234, 0))
        else:
            self.send(PathEventCode.ACK,
                      bytes((frame.sequence, motor_id, PathResult.OK)))
            if frame.command == PathCommand.ARM_JOG:
                self.send(PathEventCode.ARM_DONE,
                          struct.pack("<BBBIB", frame.sequence, motor_id,
                                      PathResult.OK, 1456, 0))


class ArmControlTests(unittest.TestCase):
    def test_bounded_jog_payload_and_matching_done(self):
        fake = FakeArmStm32()
        fake.client.arm_jog(7, -1, 8)
        done = fake.client.wait_arm_done(7, 0.01)
        self.assertEqual(fake.commands[0].payload, b"\x07\x01\x08\x00")
        self.assertEqual(done.encoder_count, 1456)
        self.assertEqual(done.request_sequence, fake.commands[0].sequence)

    def test_status_and_stop(self):
        fake = FakeArmStm32()
        self.assertEqual(fake.client.arm_status(6).encoder_count, 1234)
        fake.client.arm_stop(6)
        self.assertEqual(fake.commands[-1].command, PathCommand.ARM_STOP)

    def test_nack_is_not_mistaken_for_status_timeout(self):
        fake = FakeArmStm32()
        fake.nack = True
        with self.assertRaises(PathNackError):
            fake.client.arm_status(5)

    def test_invalid_motor_direction_and_pulse_budget(self):
        for args in ((4, 1, 8), (7, 0, 8), (7, 1, 0), (7, 1, 33)):
            with self.assertRaises(ValueError):
                encode_arm_jog(*args)

    def test_stale_status_sequence_is_ignored(self):
        fake = FakeArmStm32()

        def stale(raw):
            frame = FrameDecoder().feed(raw)[0]
            fake.send(PathEventCode.ARM_STATUS,
                      struct.pack("<BBBBIB", frame.sequence + 1,
                                  5, ArmState.IDLE, PathResult.OK, 999, 0))

        fake.client._write = stale
        with self.assertRaises(PathRequestTimeout):
            fake.client.arm_status(5)


if __name__ == "__main__":
    unittest.main()
