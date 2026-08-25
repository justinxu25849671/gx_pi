"""九点路径二进制协议的已知帧、边界与增量解析测试。"""

import struct
import unittest

from logistics_robot.path_protocol import (
    AckEvent, DoneEvent, ErrorEvent, FrameDecoder, MAX_PAYLOAD, PathCommand,
    PathEventCode, PathPoint, PathResult, PathSegment, PathState,
    PointDoneEvent, StatusEvent, crc16_modbus, encode_frame,
    encode_upload, parse_event,
)


class PathProtocolTests(unittest.TestCase):
    def test_manual_known_frames(self) -> None:
        self.assertEqual(encode_frame(PathCommand.STATUS_REQ, 1).hex(" ").upper(),
                         "AA 55 01 14 01 00 00 4D F0")
        self.assertEqual(encode_frame(PathCommand.CLEAR, 2).hex(" ").upper(),
                         "AA 55 01 10 02 00 00 BC C0")
        self.assertEqual(encode_frame(PathCommand.RESET_ORIGIN, 3).hex(" ").upper(),
                         "AA 55 01 15 03 00 00 ED CC")
        self.assertEqual(encode_frame(PathCommand.START, 5, b"\x01").hex(" ").upper(),
                         "AA 55 01 12 05 01 00 01 29 05")

    def test_crc_known_vector(self) -> None:
        self.assertEqual(crc16_modbus(b"123456789"), 0x4B37)

    def test_payload_boundaries(self) -> None:
        for payload in (b"", b"x", bytes(MAX_PAYLOAD)):
            frame = encode_frame(PathCommand.UPLOAD, 1, payload)
            decoded = FrameDecoder().feed(frame)
            self.assertEqual(decoded[0].payload, payload)
        with self.assertRaises(ValueError):
            encode_frame(PathCommand.UPLOAD, 1, bytes(MAX_PAYLOAD + 1))

    def test_negative_coordinates_are_little_endian(self) -> None:
        payload = encode_upload(7, (PathPoint(-1, -10000, 20, 10),))
        self.assertEqual(payload[:2], b"\x07\x01")
        self.assertEqual(payload[2:6], b"\xFF\xFF\xFF\xFF")
        self.assertEqual(struct.unpack("<i", payload[6:10])[0], -10000)

    def test_fragmented_and_multiple_frames(self) -> None:
        first = encode_frame(PathCommand.CLEAR, 1)
        second = encode_frame(PathCommand.STATUS_REQ, 2)
        decoder = FrameDecoder()
        self.assertEqual(decoder.feed(first[:4]), [])
        frames = decoder.feed(first[4:] + second)
        self.assertEqual([frame.command for frame in frames],
                         [PathCommand.CLEAR, PathCommand.STATUS_REQ])

    def test_noise_bad_crc_version_and_oversize_recover(self) -> None:
        valid = encode_frame(PathCommand.CLEAR, 4)
        bad_crc = bytearray(encode_frame(PathCommand.CLEAR, 2))
        bad_crc[-1] ^= 0xFF
        bad_version = bytearray(encode_frame(PathCommand.CLEAR, 3))
        bad_version[2] = 2
        oversize = b"\xAA\x55\x01\x10\x01\xFF\x00"
        decoder = FrameDecoder()
        frames = decoder.feed(b"noise" + bytes(bad_crc) + bytes(bad_version) + oversize + valid)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].sequence, 4)
        self.assertGreaterEqual(decoder.invalid_frames, 3)

    def test_truncated_frame_is_retained(self) -> None:
        frame = encode_frame(PathCommand.START, 5, b"\x01")
        decoder = FrameDecoder()
        self.assertEqual(decoder.feed(frame[:-1]), [])
        self.assertEqual(decoder.feed(frame[-1:])[0].payload, b"\x01")

    def test_truncated_long_frame_does_not_hide_following_valid_frame(self) -> None:
        truncated = b"\xAA\x55\x01\x11\x01\x6E\x00\x01\x02"
        valid = encode_frame(PathCommand.STATUS_REQ, 8)
        frames = FrameDecoder().feed(truncated + valid)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].sequence, 8)

    def test_parse_ack_and_status_strict_lengths(self) -> None:
        decoder = FrameDecoder()
        ack_frame = decoder.feed(encode_frame(PathEventCode.ACK, 9, b"\x05\x01\x00"))[0]
        self.assertEqual(parse_event(ack_frame), AckEvent(5, 1, 0, True))
        status_payload = struct.pack("<BBBBBiiii", 1, PathState.RUN_Y, 2,
                                     PathSegment.Y, 3, 500, -100, 500, 1000)
        status_frame = decoder.feed(encode_frame(PathEventCode.STATUS, 10, status_payload))[0]
        event = parse_event(status_frame)
        self.assertIsInstance(event, StatusEvent)
        self.assertEqual(event.active_target_y_mm, 1000)
        short = decoder.feed(encode_frame(PathEventCode.STATUS, 11, status_payload[:-1]))[0]
        with self.assertRaises(ValueError):
            parse_event(short)

    def test_parse_all_event_payload_models(self) -> None:
        decoder = FrameDecoder()

        def event(code, payload):
            return parse_event(decoder.feed(encode_frame(code, 1, payload))[0])

        nack = event(PathEventCode.NACK, bytes((4, 2, PathResult.BUSY)))
        self.assertEqual(nack, AckEvent(4, 2, PathResult.BUSY, False))
        point = event(PathEventCode.POINT_DONE,
                      struct.pack("<BBiiii", 2, 1, -1, 2, -3, 4))
        self.assertEqual(point, PointDoneEvent(2, 1, -1, 2, -3, 4))
        done = event(PathEventCode.DONE, struct.pack("<BBii", 2, 3, -5, 6))
        self.assertEqual(done, DoneEvent(2, 3, -5, 6))
        error = event(PathEventCode.ERROR,
                      struct.pack("<BBBBii", 2, 1, PathSegment.X,
                                  PathResult.MOTION_TIMEOUT, 7, -8))
        self.assertEqual(error, ErrorEvent(2, 1, PathSegment.X,
                                           PathResult.MOTION_TIMEOUT, 7, -8))

    def test_point_validation(self) -> None:
        for point in (PathPoint(10001, 0), PathPoint(0, 0, 0),
                      PathPoint(0, 0, 20, 256), PathPoint(0, 0, flags=1)):
            with self.assertRaises(ValueError):
                point.validate()


if __name__ == "__main__":
    unittest.main()
