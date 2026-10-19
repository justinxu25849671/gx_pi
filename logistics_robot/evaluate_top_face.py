"""Offline evaluation of saved detection-original.png; never opens serial/camera."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .vision import ObjectDetector, draw_shape_segments
from .vision_web import vision_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(args.images):
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        detector = ObjectDetector(config["object_camera"], config["hsv_colors"])
        start = time.perf_counter()
        detector.detect(frame)
        elapsed = time.perf_counter()-start
        # One still image validates geometry only, never temporal stability.
        status = vision_status(frame.shape, 0, 0, "objects", detector.candidates, [])
        status.update(source=str(path.resolve()), detection_ms=elapsed*1000,
                      temporal_stability_tested=False)
        results.append(status)
        diagnostic = draw_shape_segments(frame, detector.candidates)
        for name, image in (("result", diagnostic), ("edges", detector.edge_view), ("mask", detector.color_mask)):
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError("PNG encoding failed")
            (args.output/f"{index:03}-{path.stem}-{name}.png").write_bytes(encoded.tobytes())
    report = args.output/"results.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.resolve())


if __name__ == "__main__":
    main()
