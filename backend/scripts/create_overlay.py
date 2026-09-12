from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.yolo_overlay import create_yolo_overlay_video


def main() -> None:
    parser = ArgumentParser(description="Create a YOLO annotated overlay video.")
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Output mp4 path")
    parser.add_argument("--seconds", type=float, default=30, help="Max seconds to process. Use 0 for full video.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or local model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or Path("data/results") / f"{input_path.stem}_yolo_overlay.mp4"
    max_seconds = None if args.seconds <= 0 else args.seconds

    result = create_yolo_overlay_video(
        input_path=input_path,
        output_path=output_path,
        model_name=args.model,
        max_seconds=max_seconds,
        confidence=args.conf,
    )
    print(f"output={result.output_path}")
    print(f"frames={result.frames_processed}")
    print(f"detections={result.detections}")
    print(f"duration={result.duration_seconds}s")


if __name__ == "__main__":
    main()


