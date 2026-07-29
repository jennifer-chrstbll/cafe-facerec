"""
collect_dataset.py
------------------
Captures 10 face photos per person from your webcam.

Pipeline:
  1. SCRFD detection (InsightFace buffalo_l/det_10g.onnx)
     → detects face bounding box + 5 facial landmarks
     Note: SCRFD = Sample and Computation Redistribution for Efficient Face Detection
     (Guo et al. 2021). InsightFace buffalo_l uses SCRFD, not RetinaFace.
  2. norm_crop (affine warp to canonical 112×112 pose)
     → both eyes at fixed coords, face upright regardless of head tilt
  3. Laplacian sharpness filter — reject blurry frames
  4. Save aligned 112×112 crop (same format models were trained on)

Usage:
    python scripts/collect_dataset.py --name "Jennifer"
    python scripts/collect_dataset.py --name "Jennifer" --photos 20
"""

import cv2
import os
import sys
import argparse
import time
import numpy as np

PROJECT_ROOT = r"D:\Projects\cafe_facerec"
sys.path.insert(0, PROJECT_ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(PROJECT_ROOT, "dataset")
PHOTOS_NEEDED = 10          # photos to save per person
MIN_SHARPNESS = 45.0        # Laplacian variance threshold (lower OK since alignment normalizes pose)
FACE_MIN_SIZE = 60          # minimum face bbox width/height in pixels
CAPTURE_DELAY = 0.5         # seconds between auto-captures (for pose variety)
WARMUP_FRAMES = 20          # discard first N frames for camera auto-exposure
DET_THRESH    = 0.5         # SCRFD detection confidence threshold
DET_SIZE      = (320, 320)  # SCRFD input size (lowered from 640 for faster CPU inference)
# ──────────────────────────────────────────────────────────────────────────────

# ── Colors ────────────────────────────────────────────────────────────────────
COL_GREEN  = (80,  210,  80)
COL_AMBER  = (30,  190, 255)
COL_RED    = (60,   60, 220)
COL_WHITE  = (230, 230, 230)
COL_CYAN   = (230, 200,   0)
# ──────────────────────────────────────────────────────────────────────────────


def laplacian_sharpness(gray_img: np.ndarray) -> float:
    """Return Laplacian variance — higher = sharper. Reject frames below MIN_SHARPNESS."""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()


def load_face_detector():
    """
    Load InsightFace FaceAnalysis with detection-only module.
    Uses buffalo_l pack → det_10g.onnx (SCRFD-10GF detector).
    Outputs bounding boxes + 5 facial landmarks per face.
    """
    from insightface.app import FaceAnalysis
    print("[DETECTOR] Loading SCRFD (buffalo_l / det_10g.onnx) ...")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print("[DETECTOR] Ready.\n")
    return app


def draw_landmarks(display: np.ndarray, kps: np.ndarray):
    """Draw 5 facial landmarks on display frame."""
    for pt in kps.astype(int):
        cv2.circle(display, tuple(pt), 4, COL_CYAN, -1)


def collect(name: str, photos_needed: int):
    from insightface.utils import face_align

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")])
    if existing >= photos_needed:
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
    warmup      = 0

    print(f"[START] Collecting aligned photos for: {name}")
    print(f"        Already have : {existing}/{photos_needed}")
    print(f"        Move your head slightly between captures for variety.")
    print(f"        Press Q to quit early.\n")

    while saved < photos_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        # ── Warmup ────────────────────────────────────────────────────────────
        if warmup < WARMUP_FRAMES:
            warmup += 1
            cv2.putText(frame, f"Warming up... {warmup}/{WARMUP_FRAMES}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)
            cv2.imshow("Dataset Collection — press Q to quit", frame)
            cv2.waitKey(1)
            continue

        display = frame.copy()
        faces   = detector.get(frame)

        # Filter by detection threshold
        faces = [f for f in faces if f.det_score >= DET_THRESH]

        if not faces:
            cv2.putText(display, "No face detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_AMBER, 2)

        else:
            # Use highest-confidence face
            face  = max(faces, key=lambda f: f.det_score)
            bbox  = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            w, h  = x2 - x1, y2 - y1

            if len(faces) > 1:
                cv2.putText(display, "Multiple faces — using best", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_AMBER, 2)

            if min(w, h) < FACE_MIN_SIZE:
                cv2.putText(display, "Move closer to the camera", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_AMBER, 2)
                cv2.rectangle(display, (x1, y1), (x2, y2), COL_AMBER, 2)

            else:
                # ── norm_crop: landmark-based affine alignment ──────────────
                try:
                    aligned = face_align.norm_crop(frame, face.kps, image_size=112)
                except Exception as e:
                    cv2.putText(display, f"Alignment error: {e}", (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_RED, 1)
                    cv2.imshow("Dataset Collection — press Q to quit", display)
                    cv2.waitKey(1)
                    continue

                gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                sharpness    = laplacian_sharpness(gray_aligned)
                col          = COL_GREEN if sharpness >= MIN_SHARPNESS else COL_RED

                # Bounding box + sharpness label
                cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                cv2.putText(display, f"Score:{face.det_score:.2f}  Sharp:{sharpness:.0f}",
                            (x1, max(y1 - 10, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                # Facial landmarks
                draw_landmarks(display, face.kps)

                # Aligned face preview (top-right corner)
                ph = 90
                preview = cv2.resize(aligned, (ph, ph))
                px_off  = display.shape[1] - ph - 10
                display[10:10+ph, px_off:px_off+ph] = preview
                cv2.rectangle(display, (px_off-1, 9), (px_off+ph, 10+ph), COL_CYAN, 1)
                cv2.putText(display, "aligned", (px_off, 10+ph+13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_CYAN, 1)

                now = time.time()
                if sharpness >= MIN_SHARPNESS and (now - last_saved) >= CAPTURE_DELAY:
                    fname = os.path.join(person_dir, f"{name}_{saved+1:02d}.jpg")
                    cv2.imwrite(fname, aligned)   # save aligned 112×112 BGR crop
                    saved     += 1
                    last_saved = now
                    print(f"  [SAVED] {saved}/{photos_needed}  "
                          f"sharp={sharpness:.0f}  det={face.det_score:.2f}  → {fname}")
                elif sharpness < MIN_SHARPNESS:
                    rejected += 1
                    cv2.putText(display, "Too blurry — hold still", (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_RED, 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        h_frame = display.shape[0]
        cv2.putText(display,
                    f"{name}  |  {saved}/{photos_needed} saved  |  {rejected} rejected",
                    (20, h_frame - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 2)
        cv2.putText(display, "Detector: SCRFD + norm_crop alignment",
                    (20, h_frame - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        # Progress bar
        bar_w    = int((saved / photos_needed) * 400)
        cv2.rectangle(display, (20, h_frame - 65), (420, h_frame - 55), (60, 60, 60), -1)
        cv2.rectangle(display, (20, h_frame - 65), (20 + bar_w, h_frame - 55), COL_GREEN, -1)

        cv2.imshow("Dataset Collection — press Q to quit", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[QUIT] Stopped early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Saved {saved}/{photos_needed} aligned photos for '{name}'.")
    if rejected > 0:
        print(f"       {rejected} blurry frames were auto-rejected.")
    if saved == photos_needed:
        print(f"\n  Next step:")
        print(f"    python scripts/extract_embeddings.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect aligned face photos using SCRFD detector + norm_crop."
    )
    parser.add_argument("--name",   required=True,
                        help="Person's name (no spaces recommended, e.g. Jennifer)")
    parser.add_argument("--photos", type=int, default=PHOTOS_NEEDED,
                        help=f"Number of photos to collect (default: {PHOTOS_NEEDED})")
    args = parser.parse_args()
    collect(args.name.strip(), args.photos)