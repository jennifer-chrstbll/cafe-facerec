"""
api_server.py
-------------
Cafe FaceRec CCTV & API Server

Endpoints
---------
    GET  /video_feed          — MJPEG livestream with recognition overlay
    GET  /latest-embedding    — Latest embedding from live camera (for CRM enrollment)
    POST /extract-embedding   — Extract embedding from uploaded image
    POST /recognize           — Run recognition on uploaded image or current camera frame
    GET  /health              — Health check + gallery stats

Architecture
------------
    - Uses FaceRecognitionModule (SCRFD + ArcFace + FAISS) for all inference.
    - Recognition is LOCAL — does not depend on CRM being online.
    - After local recognition, results are forwarded to CRM for logging + visit tracking.
    - CRM URL configured via env var: FACEREC_CRM_URL (default: http://127.0.0.1:8001)
    - If CRM is down, recognition still works — error is logged, not silently swallowed.

Usage
-----
    python api_server.py
    # Runs on port 5001 by default (set PORT env var to override)
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_recognition_module import FaceRecognitionModule

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api_server")

# ── Config ────────────────────────────────────────────────────────────────────

CRM_API_URL     = os.getenv("FACEREC_CRM_URL",    "http://127.0.0.1:8001")
CRM_SEARCH_URL  = f"{CRM_API_URL}/recognition/search"
PORT            = int(os.getenv("PORT", "5001"))
RECOG_INTERVAL  = int(os.getenv("RECOG_INTERVAL", "4"))   # frames between inferences
CRM_TIMEOUT     = float(os.getenv("CRM_TIMEOUT",  "2.0")) # seconds

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cafe FaceRec — CCTV & Embedding API",
    version="1.0.0",
    description=(
        "Face recognition API using SCRFD (detector) + ArcFace (embedding) + FAISS. "
        "Recognition is local — works offline. "
        "Results are forwarded to Cafe CRM for logging."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Module init (runs once at startup) ───────────────────────────────────────

logger.info("[INIT] Loading FaceRecognitionModule (SCRFD + ArcFace + FAISS) ...")
face_module = FaceRecognitionModule()
logger.info("[INIT] FaceRecognitionModule ready.")

# ── Global camera state ────────────────────────────────────────────────────────

camera:            cv2.VideoCapture | None = None
latest_frame:      bytes | None            = None
latest_embedding:  list[float] | None      = None
latest_result:     dict | None             = None
last_face_time:    float                   = 0.0
_camera_lock                               = threading.Lock()


# ── CRM forwarding ────────────────────────────────────────────────────────────

def _forward_to_crm(embedding: list[float]) -> dict | None:
    """
    POST embedding to CRM /recognition/search.
    Returns CRM response dict or None if CRM is unreachable.
    Does NOT raise — errors are logged so camera loop is never blocked.
    """
    try:
        res = requests.post(
            CRM_SEARCH_URL,
            json={"embedding": embedding},
            timeout=CRM_TIMEOUT,
        )
        if res.status_code == 200:
            return res.json()
        else:
            logger.warning(
                f"[CRM] /recognition/search returned HTTP {res.status_code}: {res.text[:200]}"
            )
            return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"[CRM] Cannot reach {CRM_SEARCH_URL} — CRM may be offline. Recognition still local.")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"[CRM] Request timeout after {CRM_TIMEOUT}s — CRM too slow.")
        return None
    except Exception as e:
        logger.error(f"[CRM] Unexpected error forwarding to CRM: {e}")
        return None


# ── Camera loop ────────────────────────────────────────────────────────────────

def camera_loop():
    global latest_frame, latest_embedding, latest_result, last_face_time, camera

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    last_result_local = {"status": "no_face", "customer_id": None, "score": 0.0, "det_score": None}

    logger.info("[Camera] Loop started.")

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        frame_count += 1

        # ── Recognition every N frames ────────────────────────────────────────
        if frame_count % RECOG_INTERVAL == 0:
            result = face_module.recognize_face(frame)
            last_result_local = result

            if result["status"] != "no_face":
                emb_result = face_module.extract_embedding_from_frame(frame)
                if emb_result["success"]:
                    with _camera_lock:
                        latest_embedding = emb_result["embedding"]
                        latest_result    = result
                        last_face_time   = time.time()

                    # Forward to CRM (non-blocking path — failure is logged, not fatal)
                    _forward_to_crm(emb_result["embedding"])
            else:
                with _camera_lock:
                    latest_result = result

        # ── Annotate frame ────────────────────────────────────────────────────
        annotated = face_module.annotate_frame(frame, last_result_local)

        # ── MJPEG encode ──────────────────────────────────────────────────────
        ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with _camera_lock:
                latest_frame = buffer.tobytes()

        # ~30 FPS cap
        time.sleep(0.03)

        if frame_count % 300 == 0:
            gc.collect()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    logger.info(f"[Server] Camera thread started. CRM target: {CRM_API_URL}")


@app.on_event("shutdown")
def shutdown_event():
    if camera is not None:
        camera.release()
    logger.info("[Server] Shutdown complete.")


# ── MJPEG stream ──────────────────────────────────────────────────────────────

def _generate_mjpeg():
    while True:
        with _camera_lock:
            frame = latest_frame
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.033)


@app.get("/video_feed", summary="MJPEG livestream with recognition overlay")
def video_feed():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check + gallery stats")
def health():
    stats = face_module.get_gallery_stats()
    with _camera_lock:
        face_present = (
            latest_embedding is not None and (time.time() - last_face_time) < 5.0
        )
    return {
        "status":       "ok",
        "gallery":      stats,
        "threshold":    face_module.threshold,
        "face_present": face_present,
        "crm_url":      CRM_API_URL,
    }


# ── Latest embedding (for CRM enrollment) ─────────────────────────────────────

@app.get(
    "/latest-embedding",
    summary="Get embedding of the face currently in front of the camera",
)
def get_latest_embedding():
    """
    Returns the 512-d ArcFace embedding of the most recent face seen by the camera.
    Useful for enrollment: CRM calls this right after customer consent.
    Returns 404 if no face has been seen in the last 5 seconds.
    """
    with _camera_lock:
        emb   = latest_embedding
        t     = last_face_time
        result = latest_result

    if emb is None:
        raise HTTPException(status_code=404, detail="No face detected yet. Please look at the camera.")

    if time.time() - t > 5.0:
        raise HTTPException(
            status_code=404,
            detail="No face detected in the last 5 seconds. Please look at the camera.",
        )

    return {
        "embedding":   emb,
        "recognition": result,
    }


# ── Extract embedding from uploaded image ─────────────────────────────────────

@app.post(
    "/extract-embedding",
    summary="Extract ArcFace embedding from an uploaded image",
)
async def extract_embedding(file: UploadFile = File(...)):
    """
    Upload a face photo → get back the 512-d ArcFace embedding.
    Used for offline enrollment or testing.
    Privacy-preserving: the photo is NOT stored server-side.
    """
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    result = face_module.extract_embedding_from_frame(img)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    return {
        "embedding": result["embedding"],
        "det_score": result["det_score"],
    }


# ── Recognize from uploaded image ─────────────────────────────────────────────

@app.post(
    "/recognize",
    summary="Run face recognition on an uploaded image",
)
async def recognize_from_image(file: UploadFile = File(...)):
    """
    Upload a photo → get back {status, customer_id, score}.
    Searches local FAISS gallery (does NOT call CRM).
    This is the Fase 1 deliverable endpoint.
    """
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    result = face_module.recognize_face(img)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"[Server] Starting Cafe FaceRec API on port {PORT} ...")
    logger.info(f"[Server] CRM URL: {CRM_API_URL}")
    logger.info(f"[Server] Threshold τ = {face_module.threshold:.4f}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
