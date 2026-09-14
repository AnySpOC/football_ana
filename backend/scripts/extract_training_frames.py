from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser(description="Extract train/val frames from match videos for YOLO training.")
    parser.add_argument("videos", nargs="+", type=Path, help="Input video files")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/futsal"), help="Dataset root")
    parser.add_argument("--fps", type=float, default=2, help="Frames to extract per second")
    parser.add_argument("--max-seconds", type=float, default=0, help="Limit seconds per video. Use 0 for full video.")
    parser.add_argument("--val-every", type=int, default=5, help="Send every Nth frame to validation")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install backend/requirements-vision.txt.") from exc

    train_dir = args.dataset / "images" / "train"
    val_dir = args.dataset / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for video_path in args.videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"skip={video_path} reason=not_opened")
            continue

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, int(round(source_fps / args.fps)))
        frame_limit = total_frames
        if args.max_seconds > 0:
            frame_limit = min(total_frames, int(source_fps * args.max_seconds)) if total_frames else int(source_fps * args.max_seconds)

        saved_for_video = 0
        frame_index = 0
        while frame_index < frame_limit:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break

            split = "val" if args.val_every > 0 and total_saved % args.val_every == 0 else "train"
            out_dir = val_dir if split == "val" else train_dir
            out_path = out_dir / f"{video_path.stem}_f{frame_index:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            total_saved += 1
            saved_for_video += 1
            frame_index += frame_step

        capture.release()
        print(f"video={video_path} saved={saved_for_video}")

    print(f"dataset={args.dataset}")
    print(f"total_saved={total_saved}")


if __name__ == "__main__":
    main()
