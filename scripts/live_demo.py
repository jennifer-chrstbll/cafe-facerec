"""
live_demo.py
------------
Live webcam demo using the best model from evaluation.
Simulates the cafe cashier display:
    - Known customer  → show name + visit count + top-3 menu recs
    - Unknown         → show "New Customer" prompt

Usage:
    python scripts/live_demo.py --model arcface --threshold 0.35
    python scripts/live_demo.py --model arcface   (uses threshold from evaluate results)

Controls:
    Q — quit
    E — enroll current face as new person (type name in terminal)
    D — delete last enrollment
"""

import cv2
import os
import sys
import argparse
import time
import json
import numpy as np

PROJECT_ROOT   = r"D:\Projects\cafe_facerec"
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
DEMO_DB_PATH   = os.path.join(PROJECT_ROOT, "demo_db.json")
sys.path.insert(0, PROJECT_ROOT)

# ── Display config ────────────────────────────────────────────────────────────
WINDOW_W       = 1000
PANEL_W        = 340          # right-side info panel width
CAM_W          = WINDOW_W - PANEL_W
FONT           = cv2.FONT_HERSHEY_SIMPLEX

COL_GREEN  = (80,  210, 100)
COL_AMBER  = (30,  190, 255)
COL_RED    = (60,   60, 220)
COL_WHITE  = (230, 230, 230)
COL_DARK   = (30,   30,  30)
COL_PANEL  = (20,   20,  20)

# Simulated order history (replace with real DB later in Phase 3)
MOCK_ORDERS = {
    "Jennifer": ["Matcha Latte", "Croissant", "Matcha Latte", "Iced Americano", "Matcha Latte"],
    "Teo":      ["Espresso", "Espresso", "Blueberry Muffin", "Espresso"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_threshold(model_key: str) -> float:
    """Load EER threshold saved by evaluate.py."""
    path = os.path.join(RESULTS_DIR, f"{model_key}_results.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.startswith("eer_threshold"):
                    return float(line.split(":")[1].strip())
    return 0.35   # sensible default


def top3_recs(name: str) -> list:
    """Return top-3 recommended menu items based on order frequency."""
    orders = MOCK_ORDERS.get(name, [])
    if not orders:
        return ["Espresso", "Latte", "Croissant"]
    from collections import Counter
    counts = Counter(orders)
    return [item for item, _ in counts.most_common(3)]


def load_demo_db():
    if os.path.exists(DEMO_DB_PATH):
        with open(DEMO_DB_PATH) as f:
            db = json.load(f)
        # Convert lists back to np arrays
        for k in db:
            db[k]["embeddings"] = [np.array(e, dtype=np.float32)
                                   for e in db[k]["embeddings"]]
        return db
    return {}


def save_demo_db(db):
    serialisable = {}
    for k, v in db.items():
        serialisable[k] = {
            "visit_count": v["visit_count"],
            "embeddings":  [e.tolist() for e in v["embeddings"]],
        }
    with open(DEMO_DB_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)


def build_gallery(model_key: str, demo_db: dict):
    """
    Build gallery from:
        1. Saved .npy embeddings (from extract_embeddings.py)
        2. Live enrollments stored in demo_db
    Returns: (names_list, embs_array)
    """
    names, embs = [], []

    # From .npy files
    model_dir = os.path.join(EMBEDDINGS_DIR, model_key)
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".npy"):
                continue
            person = fname[:-4]
            arr    = np.load(os.path.join(model_dir, fname))
            # Use mean embedding as representative (more stable than any single photo)
            mean_emb = arr.mean(axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
            names.append(person)
            embs.append(mean_emb)

    # From demo_db live enrollments
    for person, data in demo_db.items():
        if person in names:
            continue   # already loaded from .npy
        arr      = np.stack(data["embeddings"])
        mean_emb = arr.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        names.append(person)
        embs.append(mean_emb)

    if not embs:
        return [], np.zeros((0, 512), dtype=np.float32)
    return names, np.stack(embs).astype(np.float32)


def identify(probe_emb, gallery_names, gallery_embs, threshold):
    """
    Returns (name, score) if above threshold, else (None, score).
    """
    if len(gallery_names) == 0:
        return None, 0.0
    sims    = gallery_embs @ probe_emb          # cosine (already L2-normed)
    idx     = int(np.argmax(sims))
    score   = float(sims[idx])
    if score >= threshold:
        return gallery_names[idx], score
    return None, score


# ── Panel drawing ──────────────────────────────────────────────────────────────

def draw_panel(canvas, name, score, visit_count, recs, threshold, fps):
    h, w = canvas.shape[:2]
    px   = WINDOW_W - PANEL_W   # panel starts here

    # Panel background
    cv2.rectangle(canvas, (px, 0), (w, h), COL_PANEL, -1)
    cv2.line(canvas, (px, 0), (px, h), (60, 60, 60), 1)

    y = 30

    def txt(text, size=0.55, color=COL_WHITE, bold=False):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(canvas, text, (px + 14, y), FONT, size, color, thickness, cv2.LINE_AA)
        y += int(size * 38)

    def divider():
        nonlocal y
        cv2.line(canvas, (px + 10, y), (w - 10, y), (60, 60, 60), 1)
        y += 14

    txt("CAFE FACEREC", 0.65, COL_GREEN, bold=True)
    txt(f"Model threshold: {threshold:.3f}", 0.42, (130, 130, 130))
    divider()

    if name:
        txt(f"Halo Kak {name}!", 0.7, COL_GREEN, bold=True)
        txt(f"Confidence : {score:.3f}", 0.5, COL_WHITE)
        txt(f"Visits     : {visit_count}", 0.5, COL_WHITE)
        divider()
        txt("Rekomendasi:", 0.55, COL_AMBER, bold=True)
        y += 4
        for i, item in enumerate(recs, 1):
            txt(f"  {i}. {item}", 0.52, COL_WHITE)
    else:
        txt("Customer Baru", 0.65, COL_AMBER, bold=True)
        txt(f"Best score : {score:.3f}", 0.5, (160, 160, 160))
        divider()
        txt("Tekan E untuk", 0.5, COL_WHITE)
        txt("enroll wajah baru", 0.5, COL_WHITE)

    divider()
    txt(f"FPS: {fps:.1f}", 0.45, (120, 120, 120))


def draw_face_box(frame, x1, y1, x2, y2, name, score, threshold):
    color = COL_GREEN if name else COL_AMBER
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} | {score:.2f}" if name else f"Unknown | {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, COL_DARK, 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(model_key: str, threshold: float):
    from models import load_model

    print(f"\n[INIT] Loading model: {model_key}")
    model = load_model(model_key)

    # Face detector (Haar cascade for speed in live demo)
    detector    = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    demo_db     = load_demo_db()
    g_names, g_embs = build_gallery(model_key, demo_db)
    print(f"[INIT] Gallery loaded: {len(g_names)} people → {g_names}")
    print(f"[INIT] Threshold: {threshold:.4f}")
    print(f"\n  Controls: Q=quit  E=enroll new person  D=delete last enrolled\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Recognition state (update every N frames for performance)
    RECOG_INTERVAL = 5    # run recognition every 5 frames
    frame_count    = 0
    last_name      = None
    last_score     = 0.0
    fps_time       = time.time()
    fps            = 0.0

    # Enroll buffer
    enroll_buffer  = []
    enrolling      = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        display      = frame.copy()
        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame area to CAM_W
        cam_h = int(frame.shape[0] * CAM_W / frame.shape[1])
        cam_frame = cv2.resize(display, (CAM_W, cam_h))

        # Detect faces
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 1 and frame_count % RECOG_INTERVAL == 0:
            x, y, w, h = faces[0]
            pad  = int(0.15 * max(w, h))
            x1   = max(0, x - pad);  y1 = max(0, y - pad)
            x2   = min(frame.shape[1], x + w + pad)
            y2   = min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]

            t0        = time.perf_counter()
            probe_emb = model.get_embedding(crop)
            lat_ms    = (time.perf_counter() - t0) * 1000

            if probe_emb is not None:
                last_name, last_score = identify(probe_emb, g_names, g_embs, threshold)

                if enrolling:
                    enroll_buffer.append(probe_emb)

            # Scale face box to cam_frame coords
            scale_x = CAM_W / frame.shape[1]
            scale_y = cam_h / frame.shape[0]
            sx1 = int(x1 * scale_x); sy1 = int(y1 * scale_y)
            sx2 = int(x2 * scale_x); sy2 = int(y2 * scale_y)
            draw_face_box(cam_frame, sx1, sy1, sx2, sy2, last_name, last_score, threshold)

        # Build canvas
        canvas = np.zeros((max(cam_h, 500), WINDOW_W, 3), dtype=np.uint8)
        canvas[:cam_h, :CAM_W] = cam_frame

        # Visit count + recs
        visit_count = demo_db.get(last_name, {}).get("visit_count", 1) if last_name else 0
        recs        = top3_recs(last_name) if last_name else []

        # FPS
        now      = time.time()
        fps      = 1.0 / max(now - fps_time, 0.001)
        fps_time = now

        draw_panel(canvas, last_name, last_score, visit_count, recs, threshold, fps)

        if enrolling:
            msg = f"Enrolling... {len(enroll_buffer)} frames collected. Press E again to finish."
            cv2.putText(canvas, msg, (10, 30), FONT, 0.55, COL_AMBER, 2, cv2.LINE_AA)

        cv2.imshow("Cafe FaceRec — Live Demo", canvas)
        key = cv2.waitKey(1) & 0xFF

        # ── Key controls ──
        if key == ord("q"):
            break

        elif key == ord("e"):
            if not enrolling:
                # Start enrollment
                enrolling     = True
                enroll_buffer = []
                print("\n[ENROLL] Started. Look at camera for 3-4 seconds. Press E again to save.")

            else:
                # Finish enrollment
                if len(enroll_buffer) >= 3:
                    print("[ENROLL] Enter name: ", end="", flush=True)
                    name = input().strip()
                    if name:
                        if name not in demo_db:
                            demo_db[name] = {"visit_count": 0, "embeddings": []}
                        demo_db[name]["embeddings"].extend(enroll_buffer)
                        demo_db[name]["visit_count"] += 1
                        save_demo_db(demo_db)
                        # Rebuild gallery
                        g_names, g_embs = build_gallery(model_key, demo_db)
                        print(f"[ENROLL] Saved {len(enroll_buffer)} embeddings for '{name}'.")
                        print(f"[GALLERY] Now {len(g_names)} people: {g_names}")
                    else:
                        print("[ENROLL] Cancelled — empty name.")
                else:
                    print(f"[ENROLL] Only {len(enroll_buffer)} frames — need at least 3. Try again.")
                enrolling     = False
                enroll_buffer = []

        elif key == ord("d"):
            if demo_db:
                last_key = list(demo_db.keys())[-1]
                del demo_db[last_key]
                save_demo_db(demo_db)
                g_names, g_embs = build_gallery(model_key, demo_db)
                print(f"[DELETE] Removed '{last_key}' from demo DB.")

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="arcface",
                        help="Model to use: arcface, facenet, insightface, adaface, magface")
    parser.add_argument("--threshold", default=None, type=float,
                        help="Cosine similarity threshold (default: load from evaluate results)")
    args = parser.parse_args()

    threshold = args.threshold if args.threshold else load_threshold(args.model)
    run(args.model, threshold)