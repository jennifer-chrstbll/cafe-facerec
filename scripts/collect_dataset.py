"""
collect_dataset.py
------------------
Captures 10 face photos per person from your webcam.
Uses RetinaFace detection + Laplacian sharpness filter to keep only good frames.

Usage:
    python scripts/collect_dataset.py --name "Jennifer"
"""

import cv2
import os
import argparse
import time
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = r"D:\Projects\cafe_facerec\dataset"
PHOTOS_NEEDED = 10          # photos to save per person
MIN_SHARPNESS = 60.0        # Laplacian variance threshold for blurry frame rejection
FACE_MIN_SIZE = 80          # minimum face bounding box dimension in pixels
CAPTURE_DELAY = 0.4         # seconds between auto-captures (so faces vary slightly)
WARMUP_FRAMES = 20          # discard first N frames so camera auto-exposure settles
# ──────────────────────────────────────────────────────────────────────────────


def laplacian_sharpness(gray_face):
    """Higher = sharper. Reject blurry frames below MIN_SHARPNESS."""
    return cv2.Laplacian(gray_face, cv2.CV_64F).var()


def load_face_detector():
    """Use OpenCV's built-in Haar cascade as a lightweight detector for collection.
    For the actual recognition pipeline we use RetinaFace — but for collection
    we just need a fast detector to find and crop the face region."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def collect(name: str):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
    if existing >= PHOTOS_NEEDED:
        print(f"[INFO] {name} already has {existing} photos. Delete the folder to re-collect.")
        return

    detector = load_face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check if another app is using it.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved       = existing
    last_saved  = 0.0
    rejected    = 0
    warmup      = 0          # counts discarded warmup frames

    print(f"\n[START] Collecting photos for: {name}")
    print(f"        Already have: {existing}/{PHOTOS_NEEDED}")
    print(f"        Look at the camera. Photos will be saved automatically.")
    print(f"        Press Q to quit early.\n")

    while saved < PHOTOS_NEEDED:
        ret, frame = cap.read()
        if not ret:
            continue

        # Discard first WARMUP_FRAMES so camera exposure/focus settles
        if warmup < WARMUP_FRAMES:
            warmup += 1
            cv2.putText(frame, f"Warming up... {warmup}/{WARMUP_FRAMES}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            cv2.imshow("Dataset Collection — press Q to quit", frame)
            cv2.waitKey(1)
            continue

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces   = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE))

        status_color = (0, 200, 100)  # green

        if len(faces) == 0:
            status_color = (0, 100, 255)
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        else:
            # If multiple faces, pick the largest (closest to camera) and proceed
            if len(faces) > 1:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                cv2.putText(display, "Multiple faces — using largest", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 200, 255), 2)

            x, y, w, h = faces[0]
            # Expand crop slightly for context
            pad   = int(0.2 * max(w, h))
            x1    = max(0, x - pad)
            y1    = max(0, y - pad)
            x2    = min(frame.shape[1], x + w + pad)
            y2    = min(frame.shape[0], y + h + pad)

            face_crop  = frame[y1:y2, x1:x2]
            face_gray  = gray[y1:y2, x1:x2]
            sharpness  = laplacian_sharpness(face_gray)

            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), status_color, 2)
            cv2.putText(display, f"Sharpness: {sharpness:.0f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

            now = time.time()
            if sharpness >= MIN_SHARPNESS and (now - last_saved) >= CAPTURE_DELAY:
                filename = os.path.join(person_dir, f"{name}_{saved+1:02d}.jpg")
                # Save the face crop (112x112 — standard ArcFace input size)
                face_resized = cv2.resize(face_crop, (112, 112))
                cv2.imwrite(filename, face_resized)
                saved      += 1
                last_saved  = now
                print(f"  [SAVED] {saved}/{PHOTOS_NEEDED}  sharpness={sharpness:.0f}  → {filename}")
            elif sharpness < MIN_SHARPNESS:
                status_color = (0, 100, 255)
                rejected    += 1
                cv2.putText(display, "Too blurry", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)

        # HUD
        cv2.putText(display, f"{name}  {saved}/{PHOTOS_NEEDED} saved",
                    (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Dataset Collection — press Q to quit", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[QUIT] Stopped early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Saved {saved} photos for {name}.")
    if rejected > 0:
        print(f"       {rejected} blurry frames were auto-rejected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person's name (no spaces recommended)")
    args = parser.parse_args()
    collect(args.name.strip())