"""Capture a real-world Rock-Paper-Scissors dataset from the webcam, one photo per second.

    python realworld/capture_rps.py                       # new session, 60 photos per class
    python realworld/capture_rps.py --per-class 40 --session kitchen_evening

The window shows the live camera and tells you which class to show. Press SPACE to start a
class; it then saves one full 640x480 frame every --interval seconds until the class has
--per-class photos, and moves on to the next class (SPACE again to start it).
While a class is capturing, keep changing what the camera sees: move the hand around inside
the box, nearer and farther, rotate it, palm and back, left and right hand, other backgrounds
and lighting. The "none" class is everything that is NOT a gesture — empty background, a
relaxed or moving hand, your face or arm, other objects — so a model can learn to say so.

Keys: SPACE start / pause   N skip to next class   Q or Esc quit (everything saved so far is kept)

Photos go to data/raw/rps_real/{session}/{class}/ (gitignored). Each session is meant to be a
different place or time — sessions are kept whole when splitting into train / validation /
test, so the test photos really are from a setting the model never trained on. Re-running with
the same --session continues numbering, so a class can be topped up.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["rock", "paper", "scissors", "none"]
HINT = {
    "rock": "make a FIST - move it around, rotate, near/far",
    "paper": "OPEN HAND, fingers together - move, rotate, palm/back",
    "scissors": "TWO FINGERS (V) - move, rotate, palm/back",
    "none": "NO gesture - background, relaxed hand, face, objects",
}
COLOUR = {"rock": (60, 160, 255), "paper": (80, 220, 120), "scissors": (230, 120, 255), "none": (200, 200, 200)}
WINDOW = "RPS capture - SPACE start/pause, N next, Q quit"


def existing(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".jpg")) if os.path.isdir(folder) else []


def overlay(view, cls, n, target, state, until_next, flash, mirror):
    h, w = view.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    if mirror:
        view = cv2.flip(view, 1)
    cv2.rectangle(view, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 255, 255), 1, cv2.LINE_AA)
    bar = view.copy()
    cv2.rectangle(bar, (0, 0), (w, 92), (0, 0, 0), -1)
    view = cv2.addWeighted(bar, 0.6, view, 0.4, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(view, cls.upper(), (12, 44), font, 1.4, COLOUR[cls], 3, cv2.LINE_AA)
    cv2.putText(view, f"{n}/{target}", (w - 150, 44), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, HINT[cls], (12, 78), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(view, (0, 88), (int(w * min(n / target, 1.0)), 92), COLOUR[cls], -1)  # progress
    if state == "ready":
        msg = "press SPACE to start"
    elif state == "paused":
        msg = "PAUSED - SPACE to resume"
    else:
        msg = f"next photo in {until_next:.1f} s"
    (tw, _), _ = cv2.getTextSize(msg, font, 0.8, 2)
    cv2.putText(view, msg, ((w - tw) // 2, h - 20), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, msg, ((w - tw) // 2, h - 20), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if flash:  # photo just taken
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), (255, 255, 255), 10)
    return view


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=time.strftime("s%m%d_%H%M"), help="name of this capture session")
    p.add_argument("--out", default=os.path.join(REPO, "data", "raw", "rps_real"))
    p.add_argument("--classes", default=",".join(CLASSES))
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--interval", type=float, default=1.0, help="seconds between photos")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--no-mirror", action="store_true", help="show the preview unmirrored (photos are never mirrored)")
    p.add_argument("--auto-start", action="store_true", help=argparse.SUPPRESS)  # for testing without a keyboard
    args = p.parse_args()
    classes = [c for c in args.classes.split(",") if c]
    mirror = not args.no_mirror

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    for _ in range(10):  # let auto-exposure settle
        cap.read()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    session_dir = os.path.join(args.out, args.session)
    print(f"session '{args.session}' -> {session_dir}", flush=True)

    log, quit_all = [], False
    for cls in classes:
        folder = os.path.join(session_dir, cls)
        os.makedirs(folder, exist_ok=True)
        n = len(existing(folder))
        state, next_shot, flash_until = ("capturing" if args.auto_start else "ready"), 0.0, 0.0
        print(f"[{cls}] {n} already saved, target {args.per_class}", flush=True)
        while n < args.per_class:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.time()
            if state == "capturing" and now >= next_shot:
                path = os.path.join(folder, f"{cls}_{args.session}_{n:04d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                log.append({"file": os.path.relpath(path, args.out).replace("\\", "/"), "class": cls,
                            "session": args.session, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                n += 1
                next_shot, flash_until = now + args.interval, now + 0.12
            cv2.imshow(WINDOW, overlay(frame, cls, n, args.per_class, state, max(next_shot - now, 0.0),
                                       now < flash_until, mirror))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                quit_all = True
                break
            if key == ord(" "):
                if state == "capturing":
                    state = "paused"
                else:
                    state, next_shot = "capturing", now + (0.0 if n == 0 else args.interval)
            elif key == ord("n"):
                break
        print(f"[{cls}] {n} photos", flush=True)
        if quit_all:
            break
    cap.release()
    cv2.destroyAllWindows()

    if log:
        with open(os.path.join(session_dir, "captures.jsonl"), "a") as f:
            for row in log:
                f.write(json.dumps(row) + "\n")
    counts = {c: len(existing(os.path.join(session_dir, c))) for c in classes}
    print(f"session '{args.session}': {counts} -> {session_dir}", flush=True)


if __name__ == "__main__":
    main()
