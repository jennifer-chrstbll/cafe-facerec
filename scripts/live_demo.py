"""
live_demo.py
------------
Live webcam demo using the best model from evaluation.
Pipeline: RetinaFace detection → norm_crop alignment → embedding → FAISS 1:N search

Recognition modes:
  - Known customer  → name + confidence + visit count + top-3 menu recs
  - New customer    → "Customer Baru" prompt + enroll option

Controls:
  Q — quit
  E — enroll current face as new person (look at camera, press E, type name)
  D — delete last live-enrolled person

Usage:
  python scripts/live_demo.py --model arcface --threshold 0.38
  python scripts/live_demo.py --model magface   (auto-loads threshold from evaluate results)
"""

import cv2
import os
import sys
import argparse
import time
import json
import csv
import requests
import numpy as np
from datetime import datetime

PROJECT_ROOT   = r"D:\Projects\cafe_facerec"
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
DEMO_DB_PATH   = os.path.join(PROJECT_ROOT, "demo_db.json")
ATTENDANCE_CSV = os.path.join(PROJECT_ROOT, "attendance.csv")
CRM_API_URL = "http://127.0.0.1:8000/recognition/search"
sys.path.insert(0, PROJECT_ROOT)

# ── FAISS (optional — graceful fallback) ──────────────────────────────────────
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False
    print("[WARN] faiss not installed. Falling back to numpy cosine search.")
    print("       Install with: pip install faiss-cpu")

# ── Display config ────────────────────────────────────────────────────────────
WINDOW_W  = 1080
PANEL_W   = 340
CAM_W     = WINDOW_W - PANEL_W
FONT      = cv2.FONT_HERSHEY_SIMPLEX

COL_GREEN  = (70,  210,  90)
COL_AMBER  = (30,  190, 255)
COL_RED    = (60,   60, 220)
COL_WHITE  = (230, 230, 230)
COL_DARK   = (25,   25,  25)
COL_PANEL  = (18,   18,  18)
COL_CYAN   = (230, 200,   0)
COL_GRAY   = (120, 120, 120)

# ── Mock order history (replace with real DB in production) ───────────────────
MOCK_ORDERS: dict[str, list[str]] = {
    "Jennifer": ["Matcha Latte", "Croissant", "Matcha Latte", "Iced Americano",
                 "Matcha Latte", "Croissant"],
    "Mama":     ["Cappuccino", "Blueberry Muffin", "Cappuccino", "Chocolate Cake"],
    "Koko":     ["Espresso", "Espresso", "Americano", "Espresso", "Croissant"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_threshold(model_key: str) -> float:
    path = os.path.join(RESULTS_DIR, f"{model_key}_results.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.startswith("eer_threshold"):
                    return float(line.split(":")[1].strip())
    return 0.35


def top3_recs(name: str) -> list[str]:
    from collections import Counter
    orders = MOCK_ORDERS.get(name, [])
    if not orders:
        return ["Espresso", "Latte", "Croissant"]
    return [item for item, _ in Counter(orders).most_common(3)]


def load_demo_db() -> dict:
    if os.path.exists(DEMO_DB_PATH):
        with open(DEMO_DB_PATH) as f:
            db = json.load(f)
        for k in db:
            db[k]["embeddings"] = [np.array(e, dtype=np.float32)
                                   for e in db[k]["embeddings"]]
        return db
    return {}


def save_demo_db(db: dict):
    serialisable = {
        k: {"visit_count": v["visit_count"],
            "embeddings":  [e.tolist() for e in v["embeddings"]]}
        for k, v in db.items()
    }
    with open(DEMO_DB_PATH, "w") as f:
        json.dump(serialisable, f, indent=2)


def log_attendance(name: str, score: float):
    """Append a visit record to attendance.csv."""
    is_new = not os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "name", "confidence"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{score:.4f}"])


# ── Gallery ───────────────────────────────────────────────────────────────────

def build_gallery(model_key: str, demo_db: dict):
    """
    Build gallery from:
      1. Saved .npy embeddings (from extract_embeddings.py) — ALL 10 per person
      2. Live enrollments in demo_db

    Returns: (labels_list, embs_array, faiss_index_or_None)
      labels_list : list[str]      — one name per embedding row
      embs_array  : np.ndarray     — shape (N_total, 512)
      index       : faiss index or None
    """
    labels: list[str] = []
    embs:   list[np.ndarray] = []

    # Load from .npy files (individual embeddings — NOT mean)
    model_dir = os.path.join(EMBEDDINGS_DIR, model_key)
    if os.path.isdir(model_dir):
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".npy"):
                continue
            person = fname[:-4]
            arr = np.load(os.path.join(model_dir, fname))   # (n_photos, 512)
            for emb in arr:
                labels.append(person)
                embs.append(emb.astype(np.float32))

    # Live-enrolled customers from demo_db
    for person, data in demo_db.items():
        if person in set(labels):
            continue  # already loaded from .npy
        for emb in data["embeddings"]:
            labels.append(person)
            embs.append(np.array(emb, dtype=np.float32))

    if not embs:
        return [], np.zeros((0, 512), dtype=np.float32), None

    embs_np = np.stack(embs).astype(np.float32)   # (N_total, 512)

    # Build FAISS index
    index = None
    if USE_FAISS and embs_np.shape[0] > 0:
        index = faiss.IndexFlatIP(embs_np.shape[1])
        index.add(embs_np)

    return labels, embs_np, index


def identify(probe_emb: np.ndarray,
             g_labels: list,
             gallery_embs: np.ndarray,
             index,
             threshold: float):
    """
    1:N nearest-neighbor identification.
    Uses FAISS if available, else numpy dot-product.
    Returns (name_or_None, score).
    """
    if not g_labels:
        return None, 0.0

    if USE_FAISS and index is not None:
        D, I = index.search(probe_emb.reshape(1, -1), k=1)
        score = float(D[0][0])
        idx   = int(I[0][0])
    else:
        sims  = gallery_embs @ probe_emb
        idx   = int(np.argmax(sims))
        score = float(sims[idx])

    if score >= threshold:
        return g_labels[idx], score
    return None, score


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_panel(canvas: np.ndarray, name, score: float, visit_count: int,
               recs: list, threshold: float, fps: float, model_key: str,
               last_logged: float):
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

    txt("CAFE FACEREC", 0.62, COL_GREEN, bold=True)
    txt(f"Model: {model_key.upper()}", 0.38, COL_GRAY)
    txt(f"Threshold: {threshold:.3f}", 0.38, COL_GRAY)
    divider()

    if name:
        txt(f"Halo, Kak {name}!", 0.68, COL_GREEN, bold=True)
        txt(f"Confidence : {score:.3f}", 0.48, COL_WHITE)
        txt(f"Visits     : {visit_count}", 0.48, COL_WHITE)
        divider()
        txt("Top Rekomendasi:", 0.52, COL_AMBER, bold=True)
        y += 4
        icons = ["1.", "2.", "3."]
        for i, item in enumerate(recs[:3]):
            txt(f"  {icons[i]} {item}", 0.48, COL_WHITE)

        # Cooldown indicator
        elapsed = time.time() - last_logged
        cooldown = 30.0  # seconds before logging same person again
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            txt(f"  (log in {remaining:.0f}s)", 0.36, COL_GRAY)
    else:
        txt("Customer Baru", 0.62, COL_AMBER, bold=True)
        txt(f"Best score : {score:.3f}", 0.46, COL_GRAY)
        divider()
        txt("Tekan E untuk", 0.48, COL_WHITE)
        txt("enroll wajah baru.", 0.48, COL_WHITE)

    divider()
    txt(f"FPS: {fps:.1f}", 0.42, COL_GRAY)
    txt(f"Gallery: FAISS" if USE_FAISS else "Gallery: numpy", 0.38, COL_GRAY)


def draw_face_box(cam_frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  name, score: float, kps: np.ndarray = None):
    color = COL_GREEN if name else COL_AMBER
    cv2.rectangle(cam_frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} {score:.2f}" if name else f"Unknown {score:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
    cv2.rectangle(cam_frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(cam_frame, label, (x1 + 4, y1 - 5), FONT, 0.52, COL_DARK, 1, cv2.LINE_AA)
    # Draw landmarks
    if kps is not None:
        for pt in kps.astype(int):
            # Scale landmark to cam_frame coords
            cv2.circle(cam_frame, tuple(pt), 3, COL_CYAN, -1)

def search_customer_api(
    embedding: np.ndarray
):
    try:

        response = requests.post(
            CRM_API_URL,
            json={
                "embedding": embedding.tolist()
            },
            timeout=5
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:

        print(
            "[API ERROR]",
            e
        )

        return None

# ── Main loop ─────────────────────────────────────────────────────────────────

def run(model_key: str, threshold: float):
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from models import load_model

    print(f"\n[INIT] Loading recognition model: {model_key}")
    rec_model = load_model(model_key)

    print("[INIT] Loading RetinaFace detector (buffalo_l/det_10g.onnx) ...")
    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    print("[INIT] Detector ready.")

    demo_db = load_demo_db()
    g_labels, g_embs, g_index = build_gallery(model_key, demo_db)
    people_set = sorted(set(g_labels))
    print(f"[INIT] Gallery: {len(g_labels)} embeddings | {len(people_set)} people → {people_set}")
    print(f"[INIT] Search : {'FAISS IndexFlatIP' if USE_FAISS else 'numpy dot-product'}")
    print(f"[INIT] Threshold: {threshold:.4f}")
    print(f"\n  Controls: Q=quit  E=enroll new person  D=delete last enrolled\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    RECOG_INTERVAL = 4      # run recognition every N frames
    frame_count    = 0
    last_name      = None
    last_score     = 0.0
    last_kps       = None
    last_bbox      = None
    fps_time       = time.time()
    fps            = 0.0

    # Attendance dedup: track last log time per person
    last_logged_time: dict[str, float] = {}
    LOG_COOLDOWN = 30.0   # minimum seconds between logging same person

    # Enrollment
    enroll_buffer: list[np.ndarray] = []
    enrolling = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        cam_h = int(frame.shape[0] * CAM_W / frame.shape[1])
        cam_frame = cv2.resize(frame.copy(), (CAM_W, cam_h))

        scale_x = CAM_W / frame.shape[1]
        scale_y = cam_h / frame.shape[0]

        # ── Detection + Recognition every RECOG_INTERVAL frames ───────────────
        if frame_count % RECOG_INTERVAL == 0:
            faces = detector.get(frame)
            faces = [f for f in faces if f.det_score >= 0.5]

            if faces:
                face   = max(faces, key=lambda f: f.det_score)
                bbox   = face.bbox.astype(int)
                last_bbox = bbox

                try:
                    aligned   = face_align.norm_crop(frame, face.kps, image_size=112)
                    probe_emb = rec_model.get_embedding(aligned)
                    last_kps  = (face.kps * np.array([scale_x, scale_y])).astype(int)
                except Exception:
                    probe_emb = None

                if probe_emb is not None:
                    result = search_customer_api(
                        probe_emb
                    )

                    if result:

                        if result["recognized"]:

                            last_name = result["customer_name"]

                            last_score = result["score"]

                        else:

                            last_name = None

                            last_score = result["score"]

                    # Attendance logging with cooldown
                    if last_name:
                        now = time.time()
                        last_t = last_logged_time.get(last_name, 0.0)
                        if (now - last_t) >= LOG_COOLDOWN:
                            log_attendance(last_name, last_score)
                            last_logged_time[last_name] = now
                            demo_db.setdefault(last_name, {"visit_count": 0, "embeddings": []})
                            demo_db[last_name]["visit_count"] += 1
                            save_demo_db(demo_db)

                    if enrolling:
                        enroll_buffer.append(probe_emb)
            else:
                last_bbox = None
                last_kps  = None
                last_name = None
                last_score = 0.0

        # ── Draw face box on cam_frame ─────────────────────────────────────────
        if last_bbox is not None:
            x1 = int(last_bbox[0] * scale_x)
            y1 = int(last_bbox[1] * scale_y)
            x2 = int(last_bbox[2] * scale_x)
            y2 = int(last_bbox[3] * scale_y)
            draw_face_box(cam_frame, x1, y1, x2, y2, last_name, last_score, last_kps)

        # ── Build canvas ──────────────────────────────────────────────────────
        canvas = np.full((max(cam_h, 520), WINDOW_W, 3), COL_DARK, dtype=np.uint8)
        canvas[:cam_h, :CAM_W] = cam_frame

        # ── Panel ─────────────────────────────────────────────────────────────
        visit_count  = demo_db.get(last_name, {}).get("visit_count", 1) if last_name else 0
        recs         = top3_recs(last_name) if last_name else []
        last_log_t   = last_logged_time.get(last_name, 0.0) if last_name else 0.0

        now      = time.time()
        fps      = 1.0 / max(now - fps_time, 0.001)
        fps_time = now

        draw_panel(canvas, last_name, last_score, visit_count, recs,
                   threshold, fps, model_key, last_log_t)

        # ── Enrollment status overlay ──────────────────────────────────────────
        if enrolling:
            msg = f"ENROLLING... {len(enroll_buffer)} frames. Press E to finish."
            cv2.putText(canvas, msg, (10, cam_h - 12), FONT, 0.55, COL_AMBER, 2, cv2.LINE_AA)

        cv2.imshow("Cafe FaceRec — Live Demo", canvas)
        key = cv2.waitKey(1) & 0xFF

        # ── Key controls ──────────────────────────────────────────────────────
        if key == ord("q"):
            break

        elif key == ord("e"):
            if not enrolling:
                enrolling     = True
                enroll_buffer = []
                print("\n[ENROLL] Started. Look at camera ~3 seconds. Press E again to save.")
            else:
                if len(enroll_buffer) >= 3:
                    print("[ENROLL] Enter name: ", end="", flush=True)
                    enroll_name = input().strip()
                    if enroll_name:
                        if enroll_name not in demo_db:
                            demo_db[enroll_name] = {"visit_count": 0, "embeddings": []}
                        demo_db[enroll_name]["embeddings"].extend(enroll_buffer)
                        demo_db[enroll_name]["visit_count"] += 1
                        save_demo_db(demo_db)
                        # Rebuild gallery
                        g_labels, g_embs, g_index = build_gallery(model_key, demo_db)
                        people_set = sorted(set(g_labels))
                        print(f"[ENROLL] Saved {len(enroll_buffer)} embeddings for '{enroll_name}'.")
                        print(f"[GALLERY] Now {len(people_set)} people: {people_set}")
                    else:
                        print("[ENROLL] Cancelled.")
                else:
                    print(f"[ENROLL] Only {len(enroll_buffer)} frames — need ≥ 3. Try again.")
                enrolling     = False
                enroll_buffer = []

        elif key == ord("d"):
            if demo_db:
                last_key = list(demo_db.keys())[-1]
                del demo_db[last_key]
                save_demo_db(demo_db)
                g_labels, g_embs, g_index = build_gallery(model_key, demo_db)
                print(f"[DELETE] Removed '{last_key}' from demo_db.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Attendance log saved to: {ATTENDANCE_CSV}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="magface",
                        help="Model: arcface, facenet_vgg, facenet_casia, adaface, magface")
    parser.add_argument("--threshold", default=None, type=float,
                        help="Cosine similarity threshold (default: load from evaluate results)")
    args = parser.parse_args()
    threshold = args.threshold if args.threshold else load_threshold(args.model)
    run(args.model, threshold)