"""
camera_agent.py -- 100% True AI POV Stream Sync + 10x Fast MobileFaceNet
-------------------------------------------------------------------------
- Stream is directly synchronized 1:1 with AI processing (True AI POV)
- Zero box lag when moving
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv

# Use all 4 CPU Cores
cv2.setNumThreads(4)

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

sys.path.insert(0, str(_HERE))
from face_recognition_module import FaceRecognitionModule

DATABASE_URL      = os.getenv("DATABASE_URL")
CAMERA_ID         = os.getenv("CAMERA_ID",         "uno-q-001")
COOLDOWN_SECONDS  = int(os.getenv("COOLDOWN",       "10"))
RELOAD_INTERVAL   = int(os.getenv("RELOAD_INTERVAL","300"))
STREAM_PORT       = 5001

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

latest_jpeg_frame = None
frame_lock = threading.Lock()


def _parse_vector(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            return json.loads(v)
        return [float(x) for x in v.split(",")]
    return list(v)


def connect_db():
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def load_gallery(module, conn) -> int:
    cur = conn.cursor()
    cur.execute("""
        SELECT e.customer_id::text, e.embedding_vector::text
        FROM   embeddings e
        JOIN   customers  c ON e.customer_id = c.customer_id
        WHERE  c.is_active = TRUE
    """)
    rows = cur.fetchall()
    cur.close()

    if not rows:
        print("[Gallery] WARNING: No embeddings in Supabase yet!")
        return 0

    gallery = [(cid, _parse_vector(vec)) for cid, vec in rows]
    module.load_gallery_from_rows(gallery)
    n_people = len({r[0] for r in gallery})
    print(f"[Gallery] Loaded {len(gallery)} embeddings for {n_people} customers")
    return len(gallery)


def log_recognition(conn, customer_id, score: float, recognized: bool):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO recognition_logs
            (customer_id, similarity_score, model_used, camera_id, recognized)
        VALUES (%s, %s, %s, %s, %s)
    """, (customer_id, round(score, 6), "arcface_mbf", CAMERA_ID, recognized))
    conn.commit()
    cur.close()


def create_visit(conn, customer_id: str):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO visits (customer_id, entry_time)
        VALUES (%s::uuid, NOW())
    """, (customer_id,))
    conn.commit()
    cur.close()


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/', '/video_feed']:
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
            self.end_headers()
            while True:
                with frame_lock:
                    frame = latest_jpeg_frame
                if frame is not None:
                    self.wfile.write(b'--jpgboundary\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                time.sleep(0.04)
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


def start_stream_server():
    server = HTTPServer(('0.0.0.0', STREAM_PORT), StreamingHandler)
    server.serve_forever()


def open_camera():
    for index in [0, 1, 2, 3, 4]:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[Camera] ✓ Opened webcam on /dev/video{index}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                return cap
            cap.release()
    return None


def main():
    global latest_jpeg_frame

    print("\n" + "=" * 55)
    print("  Cafe FaceRec -- True AI POV Sync & Fast MobileFaceNet")
    print("=" * 55)

    print("[DB] Connecting to Supabase...")
    conn = connect_db()
    print("[DB] Connected")

    print("[Model] Loading SCRFD + MobileFaceNet...")
    module = FaceRecognitionModule()
    print("[Model] Ready (Using 4 CPU Cores)")

    load_gallery(module, conn)

    print("[Camera] Auto-detecting webcam...")
    cap = open_camera()

    if cap is None:
        print("[ERROR] Cannot open webcam. Please plug webcam into Arduino Uno Q!")
        return

    # Start HTTP Stream Server Thread (port 5001)
    t_stream = threading.Thread(target=start_stream_server, daemon=True)
    t_stream.start()
    print(f"[Stream] 🎥 True AI POV Stream active at http://192.168.18.80:{STREAM_PORT}/video_feed")

    print(f"[Ready] Cooldown={COOLDOWN_SECONDS}s | Processing 1:1 Synchronized AI POV Frames")
    print("        Press Ctrl+C to stop.\n")

    last_log_time  = {}
    last_reload_t  = time.time()

    fps_start_time = time.time()
    fps_counter    = 0
    current_fps    = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            now = time.time()
            if now - last_reload_t > RELOAD_INTERVAL:
                print("[Gallery] Reloading from Supabase...")
                try:
                    load_gallery(module, conn)
                except Exception as e:
                    print(f"[Gallery] Reload failed: {e}")
                last_reload_t = now

            # Process Frame synchronously with AI engine (100% True AI POV)
            result      = module.recognize_face(frame)
            status      = result["status"]
            customer_id = result["customer_id"]
            score       = result["score"]
            lat_ms      = result.get("latency_ms", 0.0)
            bbox        = result.get("bbox", None)
            recognized  = (status == "known")

            if status != "no_face":
                log_key = customer_id if customer_id else "unknown"
                if now - last_log_time.get(log_key, 0) >= COOLDOWN_SECONDS:
                    last_log_time[log_key] = now
                    try:
                        log_recognition(conn, customer_id, score, recognized)
                        ts = time.strftime("%H:%M:%S")
                        if recognized and customer_id:
                            create_visit(conn, customer_id)
                            print(f"  [{ts}] 🎉 KNOWN    id={customer_id[:8]}...  score={score:.3f} | AI Latency: {lat_ms:.0f}ms | FPS: {current_fps:.1f}")
                        else:
                            print(f"  [{ts}] ❓ UNKNOWN  score={score:.3f} | AI Latency: {lat_ms:.0f}ms | FPS: {current_fps:.1f}")
                    except Exception as e:
                        print(f"  [DB Error] {e}")

            # Draw 1:1 In-Sync AI POV Bounding Box & HUD
            display_frame = frame.copy()
            cv2.putText(display_frame, f"AI Latency: {lat_ms:.0f}ms | FPS: {current_fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            if status != "no_face" and bbox:
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if status == "known" else (0, 165, 255)
                cid_str = customer_id[:8] if customer_id else "UNKNOWN"
                label = f"{cid_str}... ({score:.2f})" if status == "known" else f"UNKNOWN ({score:.2f})"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Update Stream Buffer with exact AI POV frame
            ret_jpg, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret_jpg:
                with frame_lock:
                    latest_jpeg_frame = jpeg.tobytes()

    except KeyboardInterrupt:
        print("\n[Stopped] Camera agent shut down.")
    finally:
        if cap:
            cap.release()
        conn.close()


if __name__ == "__main__":
    main()
