"""
live_demo.py
------------
Live webcam demo for Cafe FaceRec Fase 1.

Pipeline: SCRFD detection → norm_crop alignment → ArcFace embedding → FAISS 1:N search
          (all via FaceRecognitionModule)

Recognition modes:
  - Known customer  → name + confidence + visit count
  - Unknown         → "Customer Baru" prompt + enroll option

Controls:
  Q — quit
  E — enroll current face as new customer (look at camera, press E, type name, press E again)
  D — delete last live-enrolled person

Usage:
  python scripts/live_demo.py
  python scripts/live_demo.py --threshold 0.30
  python scripts/live_demo.py --det-size 640
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_recognition_module import FaceRecognitionModule

# ── Paths ──────────────────────────────────────────────────────────────────────
DEMO_DB_PATH   = PROJECT_ROOT / "demo_db.json"
ATTENDANCE_CSV = PROJECT_ROOT / "attendance.csv"

# ── Display config ─────────────────────────────────────────────────────────────
WINDOW_W = 1080
PANEL_W  = 340
CAM_W    = WINDOW_W - PANEL_W
FONT     = cv2.FONT_HERSHEY_SIMPLEX

COL_GREEN  = (70,  210,  90)
COL_AMBER  = (30,  190, 255)
COL_RED    = (60,   60, 220)
COL_WHITE  = (230, 230, 230)
COL_DARK   = (25,   25,  25)
COL_PANEL  = (18,   18,  18)
COL_CYAN   = (230, 200,   0)
COL_GRAY   = (120, 120, 120)


# ── Demo DB (local enrollment store) ──────────────────────────────────────────

def load_demo_db() -> dict:
    if DEMO_DB_PATH.exists():
        with open(DEMO_DB_PATH) as f:
            return json.load(f)
    return {}


def save_demo_db(db: dict):
    with open(DEMO_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ── Attendance log ────────────────────────────────────────────────────────────

def log_attendance(name: str, score: float):
    is_new = not ATTENDANCE_CSV.exists()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "name", "confidence"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{score:.4f}"])


# ── Panel drawing ──────────────────────────────────────────────────────────────

def draw_panel(canvas: np.ndarray, result: dict, visit_count: int,
               threshold: float, fps: float, last_logged: float):
    h, w = canvas.shape[:2]
    px   = WINDOW_W - PANEL_W

    cv2.rectangle(canvas, (px, 0), (w, h), COL_PANEL, -1)
    cv2.line(canvas, (px, 0), (px, h), (55, 55, 55), 1)

    y = 28

    def txt(text, size=0.52, color=COL_WHITE, bold=False):
        nonlocal y
        cv2.putText(canvas, text, (px + 14, y), FONT, size,
                    color, 2 if bold else 1, cv2.LINE_AA)
        y += max(18, int(size * 36))

    def divider(gap=10):
        nonlocal y
        y += gap // 2
        cv2.line(canvas, (px + 10, y), (w - 10, y), (50, 50, 50), 1)
        y += gap // 2 + 6

    status      = result.get("status", "no_face")
    customer_id = result.get("customer_id")
    score       = result.get("score", 0.0)

    txt("CAFE FACEREC", 0.62, COL_GREEN, bold=True)
    txt("SCRFD + ArcFace + FAISS", 0.36, COL_GRAY)
    txt(f"Threshold: {threshold:.3f}", 0.38, COL_GRAY)
    divider()

    if status == "known" and customer_id:
        txt(f"Halo, Kak {customer_id}!", 0.68, COL_GREEN, bold=True)
        txt(f"Confidence : {score:.3f}", 0.48, COL_WHITE)
        txt(f"Visits     : {visit_count}", 0.48, COL_WHITE)
        divider()

        # Cooldown indicator
        elapsed  = time.time() - last_logged
        cooldown = 30.0
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            txt(f"  (log in {remaining:.0f}s)", 0.36, COL_GRAY)

    elif status == "unknown":
        txt("Customer Baru", 0.62, COL_AMBER, bold=True)
        txt(f"Best score : {score:.3f}", 0.46, COL_GRAY)
        divider()
        txt("Tekan E untuk", 0.48, COL_WHITE)
        txt("enroll wajah baru.", 0.48, COL_WHITE)

    else:
        txt("Mendeteksi...", 0.52, COL_GRAY)

    divider()
    txt(f"FPS: {fps:.1f}", 0.42, COL_GRAY)
    txt("Gallery: FAISS IndexFlatIP", 0.36, COL_GRAY)
    divider()
    txt("Q=quit  E=enroll  D=delete", 0.38, COL_GRAY)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(threshold: float | None, det_size: int):
    print(f"\n[INIT] Loading FaceRecognitionModule (SCRFD + ArcFace + FAISS) ...")
    module = FaceRecognitionModule(
        det_size=(det_size, det_size),
        threshold=threshold,
    )

    demo_db   = load_demo_db()
    threshold = module.threshold  # use (possibly EER-loaded) value

    stats = module.get_gallery_stats()
    print(f"[INIT] Gallery: {stats['n_embeddings']} embeddings | {stats['n_people']} people → {stats['people']}")
    print(f"[INIT] Threshold τ = {threshold:.4f}")
    print(f"\n  Controls: Q=quit  E=enroll new person  D=delete last enrolled\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    RECOG_INTERVAL    = 4
    frame_count       = 0
    last_result       = {"status": "no_face", "customer_id": None, "score": 0.0, "det_score": None}
    fps_time          = time.time()
    fps               = 0.0
    last_logged_time: dict[str, float] = {}
    LOG_COOLDOWN      = 30.0

    # Enrollment state
    enrolling      = False
    enroll_frames  = 0
    enroll_embeddings: list[list[float]] = []
    enroll_name    = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        cam_h     = int(frame.shape[0] * CAM_W / frame.shape[1])
        cam_frame = cv2.resize(frame.copy(), (CAM_W, cam_h))

        # ── Recognition every N frames ─────────────────────────────────────────
        if frame_count % RECOG_INTERVAL == 0:
            result      = module.recognize_face(frame)
            last_result = result

            cid   = result.get("customer_id")
            score = result.get("score", 0.0)

            # Attendance logging with cooldown
            if result["status"] == "known" and cid:
                now   = time.time()
                last_t = last_logged_time.get(cid, 0.0)
                if (now - last_t) >= LOG_COOLDOWN:
                    log_attendance(cid, score)
                    last_logged_time[cid] = now
                    demo_db.setdefault(cid, {"visit_count": 0})
                    demo_db[cid]["visit_count"] = demo_db[cid].get("visit_count", 0) + 1
                    save_demo_db(demo_db)

            # Collect frame for ongoing enrollment
            if enrolling:
                emb_result = module.extract_embedding_from_frame(frame)
                if emb_result["success"]:
                    enroll_embeddings.append(emb_result["embedding"])
                    enroll_frames += 1

        # ── Draw face overlay on cam_frame ─────────────────────────────────────
        annotated = module.annotate_frame(cam_frame, last_result)

        # ── Build canvas ───────────────────────────────────────────────────────
        canvas = np.full((max(cam_h, 520), WINDOW_W, 3), COL_DARK, dtype=np.uint8)
        canvas[:cam_h, :CAM_W] = annotated

        # ── Panel ──────────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - fps_time, 0.001)
        fps_time = now

        cid_panel    = last_result.get("customer_id")
        visit_count  = demo_db.get(cid_panel, {}).get("visit_count", 0) if cid_panel else 0
        last_log_t   = last_logged_time.get(cid_panel, 0.0) if cid_panel else 0.0

        draw_panel(canvas, last_result, visit_count, threshold, fps, last_log_t)

        # ── Enrollment status overlay ──────────────────────────────────────────
        if enrolling:
            msg = f"ENROLLING... {enroll_frames} frames. Press E again to save."
            cv2.putText(canvas, msg, (10, cam_h - 12), FONT, 0.55, COL_AMBER, 2, cv2.LINE_AA)

        cv2.imshow("Cafe FaceRec — Live Demo", canvas)
        key = cv2.waitKey(1) & 0xFF

        # ── Key controls ───────────────────────────────────────────────────────
        if key == ord("q"):
            break

        elif key == ord("e"):
            if not enrolling:
                enrolling         = True
                enroll_frames     = 0
                enroll_embeddings = []
                print("\n[ENROLL] Started. Look at camera ~3 seconds. Press E again to save.")

            else:
                # Second E press — save
                if len(enroll_embeddings) >= 3:
                    print("[ENROLL] Enter customer name/ID: ", end="", flush=True)
                    enroll_name = input().strip()
                    if enroll_name:
                        # Save each embedding to embeddings/arcface/<name>.npy
                        from face_recognition_module import EMBEDDINGS_DIR, _l2_normalize
                        import numpy as _np
                        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
                        npy_path = EMBEDDINGS_DIR / f"{enroll_name}.npy"

                        new_vecs = _np.array(
                            [_l2_normalize(_np.array(e, dtype=_np.float32)) for e in enroll_embeddings],
                            dtype=_np.float32,
                        )

                        if npy_path.exists():
                            existing = _np.load(npy_path).astype(_np.float32)
                            if existing.ndim == 1:
                                existing = existing[_np.newaxis, :]
                            combined = _np.vstack([existing, new_vecs])
                        else:
                            combined = new_vecs

                        _np.save(npy_path, combined)
                        module.reload_gallery()

                        demo_db.setdefault(enroll_name, {"visit_count": 0})
                        save_demo_db(demo_db)

                        updated_stats = module.get_gallery_stats()
                        print(f"[ENROLL] Saved {len(enroll_embeddings)} embeddings for '{enroll_name}'.")
                        print(f"[GALLERY] Now {updated_stats['n_people']} people: {updated_stats['people']}")
                    else:
                        print("[ENROLL] Cancelled.")
                else:
                    print(f"[ENROLL] Only {enroll_frames} frames — need ≥ 3. Try again.")

                enrolling         = False
                enroll_frames     = 0
                enroll_embeddings = []

        elif key == ord("d"):
            from face_recognition_module import EMBEDDINGS_DIR
            if demo_db:
                last_key = list(demo_db.keys())[-1]
                npy_path = EMBEDDINGS_DIR / f"{last_key}.npy"
                if npy_path.exists():
                    npy_path.unlink()
                del demo_db[last_key]
                save_demo_db(demo_db)
                module.reload_gallery()
                print(f"[DELETE] Removed '{last_key}' from gallery and demo_db.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Attendance log saved to: {ATTENDANCE_CSV}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cafe FaceRec Live Demo — SCRFD + ArcFace + FAISS"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Cosine similarity threshold τ (default: auto from EER results, τ=0.258)"
    )
    parser.add_argument(
        "--det-size", type=int, default=320,
        help="SCRFD detector input size: 320 (fast) or 640 (accurate). Default: 320"
    )
    args = parser.parse_args()
    run(threshold=args.threshold, det_size=args.det_size)