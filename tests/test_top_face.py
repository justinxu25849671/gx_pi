"""Synthetic geometry and fail-closed integration; not real-camera acceptance."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import zipfile

import cv2
import numpy as np

from logistics_robot.model import DetectedObject
from logistics_robot.vision import ObjectDetector
from logistics_robot.vision_locate import upper_circle_coordinate
from logistics_robot.vision_web import FrameHub, VisionWebServer, _CircleSender, _camera_loop, vision_status

HSV = {"red": [[0, 80, 50, 10, 255, 255]], "yellow": [[20,80,50,35,255,255]],
       "blue": [[100,80,50,130,255,255]], "green": [[40,80,50,85,255,255]]}
BGR = {"red": (0,0,240), "yellow": (0,240,240), "blue": (240,0,0), "green": (0,240,0)}


def scene(color="red", radius=45, angle=0):
    frame = np.zeros((300, 400, 3), np.uint8)
    cv2.ellipse(frame, (190, 115), (radius, round(radius*.6)), angle, 0, 360, BGR[color], -1)
    return frame


def detector(**kwargs):
    return ObjectDetector({"min_area_px": 150, "max_area_px": 100000,
                           "stable_frames": 3, **kwargs}, HSV)


class TopFaceTests(unittest.TestCase):
    def test_four_colors_sizes_angles_center_accuracy(self):
        for color in BGR:
            for radius in (22, 45, 75):
                for angle in (-25, 0, 25):
                    with self.subTest(color=color, radius=radius, angle=angle):
                        d = detector(stable_frames=1)
                        found = d.detect(scene(color, radius, angle), [color])
                        self.assertEqual(len(found), 1, d.candidates)
                        self.assertLess(np.linalg.norm(np.array(found[0].circle_center_px)-(190,115)), 2)

    def test_failure_immediately_stops_new_coordinates_and_resets_stability(self):
        d = detector()
        sender = _CircleSender({})
        client = Mock()
        sender._link = Mock(path_client=client)
        sender._started = True
        good = scene()
        for _ in range(3):
            found = d.detect(good, ["red"])
        sender.update("red", upper_circle_coordinate(found, 400, 300))
        client.send_visual_coordinate.assert_called_once()
        failures = [np.zeros_like(good), np.full_like(good, (0,0,240))]
        clipped = np.zeros_like(good)
        cv2.ellipse(clipped, (5,100), (45,30), 0, 0, 360, BGR["red"], -1)
        failures.append(clipped)
        occluded = good.copy(); occluded[110:160, 180:250] = 0
        failures.append(occluded)
        for bad in failures:
            with self.subTest(kind=int(bad.sum())):
                found = d.detect(bad, ["red"])
                self.assertIsNone(upper_circle_coordinate(found,400,300))
                sender.update("red", upper_circle_coordinate(found,400,300))
                self.assertEqual(client.send_visual_coordinate.call_count, 1)
        self.assertEqual(d.detect(good,["red"]), [])

    def test_material_center_differs_from_top_and_threshold_is_not_capped(self):
        f = scene()
        cv2.rectangle(f,(150,115),(230,220),(0,0,110),-1)
        cv2.ellipse(f,(190,115),(45,27),0,0,360,BGR["red"],-1)
        d = detector(stable_frames=1)
        found = d.detect(f,["red"])
        self.assertEqual(len(found),1)
        self.assertGreater(found[0].center_y_px-found[0].circle_center_px[1],20)
        self.assertLess(abs(found[0].circle_center_px[1]-115),2)
        strict = detector(stable_frames=1,min_circularity=1.0)
        self.assertEqual(strict.detect(f,["red"]), [])
        self.assertFalse(strict.candidates[0].top_valid)

    def test_roi_truncation_and_unverified_legacy_object_cannot_send(self):
        d=detector(stable_frames=1,roi=[0.45,0,1,1])
        self.assertEqual(d.detect(scene(),["red"]),[])
        self.assertIn("不完整",d.candidates[0].top_reason)
        self.assertIsNone(upper_circle_coordinate([DetectedObject("red",1,2,300,stable=True)],400,300))

    def test_four_status_states(self):
        d=detector()
        def state():
            return vision_status((300,400),10,0,"objects",d.candidates,[])["recognition_state"]
        d.detect(np.zeros((300,400,3),np.uint8)); self.assertEqual(state(),"未发现物料")
        d.detect(np.full((300,400,3),(0,0,240),np.uint8),["red"])
        self.assertEqual(state(),"已找到物料但顶面未确认")
        d.detect(scene(),["red"]); self.assertEqual(state(),"顶面已确认但未稳定")
        d.detect(scene(),["red"]); d.detect(scene(),["red"]); self.assertEqual(state(),"可发送")

    def test_camera_loop_gates_every_frame_and_preview_never_sends(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
                config["hsv_colors"] = HSV
                config["object_camera"]["object_detection"]["stable_frames"] = 1
                stop = threading.Event()
                hub = FrameHub(); hub.set_locate_color("red"); hub.send_enabled = enabled
                sender = _CircleSender({})
                client = Mock(); sender._link = Mock(path_client=client)
                statuses = []
                original_publish = hub.publish
                def publish(jpeg, status):
                    statuses.append(status)
                    original_publish(jpeg,status)
                hub.publish = publish
                class Camera:
                    cv2 = __import__("cv2")
                    def __init__(self, camera_config, hsv):
                        self._detector = ObjectDetector(camera_config,hsv)
                        self.count = 0
                    def read(self):
                        self.count += 1
                        if self.count == 3: stop.set()
                        return scene() if self.count == 1 else np.zeros((300,400,3),np.uint8)
                    def clear_tracks(self): self._detector.clear_tracks()
                    def set_detect_width(self,width): self._detector.set_detect_width(width)
                    def detect_objects(self,frame,colors): return self._detector.detect(frame,colors)
                    def close(self): pass
                with patch("logistics_robot.vision_web.CameraVision",Camera), patch("logistics_robot.vision_web._CircleSender",return_value=sender):
                    _camera_loop(config,None,"objects",hub,stop)
                self.assertEqual(len(statuses),3)
                self.assertTrue(all(s["ok"] for s in statuses),statuses)
                self.assertEqual(client.send_visual_coordinate.call_count,1 if enabled else 0)
                self.assertIsNone(statuses[1]["actual_send_target"])
                self.assertIsNone(statuses[2]["actual_send_target"])
                if not enabled:
                    client.start_visual_positioning.assert_not_called()
                    self.assertEqual(statuses[0]["send_state"],"preview")

    def test_offline_cli_reports_geometry_without_claiming_temporal_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ok, image=cv2.imencode(".png",scene())
            self.assertTrue(ok)
            source=root/"original.png";source.write_bytes(image.tobytes())
            run=subprocess.run([sys.executable,"-B","-m","logistics_robot.evaluate_top_face",
                                str(source),"--config","config.example.json","--output",str(root/"result")],
                               capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((root/"result"/"results.json").read_text(encoding="utf-8"))
            self.assertFalse(report[0]["temporal_stability_tested"])
            self.assertTrue(report[0]["objects"][0]["top_valid"])
            self.assertFalse(report[0]["objects"][0]["stable"])
            self.assertTrue((root/"result"/"000-original-result.png").exists())

    def test_http_preview_controls_settings_and_lossless_capture(self):
        config = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
        original_hsv = copy.deepcopy(config["hsv_colors"])
        hub=FrameHub()
        f=scene(); mask=np.zeros(f.shape[:2],np.uint8)
        hub.capture_frames=(f,f,mask,mask,f,{"frame_time_ns":123},config)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"config.json"
            server=VisionWebServer(hub,"127.0.0.1",0,config,path)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            def post(route,data):
                request=Request(server.url.rstrip("/")+route,json.dumps(data).encode(),{"Content-Type":"application/json"})
                with urlopen(request,timeout=2) as response:return json.load(response)
            try:
                self.assertFalse(hub.send_enabled)
                post("/color",{"color":"red"});self.assertFalse(hub.send_enabled)
                post("/controls",{"send_enabled":True});self.assertTrue(hub.send_enabled)
                for view in ("raw","mask","edges","result"):
                    post("/controls",{"view":view});self.assertEqual(hub.view,view)
                post("/settings",{"values":{"min_circularity":.93},"save":True})
                self.assertFalse(hub.send_enabled)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["object_camera"]["object_detection"]["min_circularity"],.93)
                for invalid in (float("nan"),1.2,-1):
                    with self.assertRaises(HTTPError) as caught:post("/settings",{"values":{"min_circularity":invalid}})
                    caught.exception.close()
                post("/settings",{"reset":True})
                self.assertEqual(config["hsv_colors"],original_hsv)
                with urlopen(server.url+"capture.zip",timeout=2) as response:
                    with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                        raw=cv2.imdecode(np.frombuffer(archive.read("camera-original.png"),np.uint8),cv2.IMREAD_COLOR)
                        self.assertTrue(np.array_equal(raw,f))
                        self.assertEqual(json.loads(archive.read("status.json"))["frame_time_ns"],123)
            finally:
                hub.close();server.shutdown();worker.join(2)
