# save as tools/hstack_videos.py
import argparse
from pathlib import Path
import cv2

def resolve_video(p: str) -> str:
    path = Path(p)
    if path.is_dir():
        vids = sorted(path.glob("*.mp4"))
        if not vids:
            raise FileNotFoundError(f"No mp4 in dir: {p}")
        return str(vids[0])
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return str(path)

def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 0 else 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True, help="left video file or dir")
    ap.add_argument("--right", required=True, help="right video file or dir")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--fps", type=float, default=0, help="override fps (optional)")
    args = ap.parse_args()

    left = resolve_video(args.left)
    right = resolve_video(args.right)

    cap1 = cv2.VideoCapture(left)
    cap2 = cv2.VideoCapture(right)
    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Failed to open input video(s)")

    # read one frame to get size
    ok1, f1 = cap1.read()
    ok2, f2 = cap2.read()
    if not ok1 or not ok2:
        raise RuntimeError("Failed to read first frame")

    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    target_h = min(h1, h2)

    w1r = int(round(w1 * target_h / h1))
    w2r = int(round(w2 * target_h / h2))

    fps = args.fps if args.fps > 0 else get_fps(cap1)
    if fps <= 0:
        fps = get_fps(cap2)

    # reset to start
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_w = w1r + w2r
    out_h = target_h
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    while True:
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        if not ok1 or not ok2:
            break
        f1 = cv2.resize(f1, (w1r, target_h), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (w2r, target_h), interpolation=cv2.INTER_AREA)
        stacked = cv2.hconcat([f1, f2])
        writer.write(stacked)

    cap1.release()
    cap2.release()
    writer.release()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
