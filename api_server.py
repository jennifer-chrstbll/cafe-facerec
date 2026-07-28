import os
import cv2
import time
import requests
import numpy as np
import threading
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

from models import load_model
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FastAPI(title="Cafe FaceRec CCTV & API")

# Configuration
CRM_API_URL = "http://127.0.0.1:8001/recognition/search"
MODEL_KEY = os.getenv("FACEREC_MODEL", "arcface")
RECOG_INTERVAL = 4 # Run recognition every N frames

# Globals for camera and state
camera = None
latest_frame = None
latest_embedding = None
last_face_time = 0.0

# Initialize Models
print(f"[INIT] Loading Face Recognition model: {MODEL_KEY}...")
rec_model = load_model(MODEL_KEY)

print("[INIT] Loading RetinaFace detector...")
detector = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=0, det_size=(320, 320))

import gc

def camera_loop():
    global latest_frame, latest_embedding, last_face_time, camera
    
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    last_bbox = None
    last_name = None
    last_score = 0.0
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
            
        frame_count += 1
        
        # Recognition logic every N frames
        if frame_count % RECOG_INTERVAL == 0:
            faces = detector.get(frame)
            faces = [f for f in faces if f.det_score >= 0.5]
            
            if faces:
                face = max(faces, key=lambda f: f.det_score)
                last_bbox = face.bbox.astype(int)
                
                try:
                    aligned = face_align.norm_crop(frame, face.kps, image_size=112)
                    emb = rec_model.get_embedding(aligned)
                    
                    if emb is not None:
                        # Save latest embedding for enrollment
                        latest_embedding = emb
                        last_face_time = time.time()
                        
                        # Post to CRM
                        try:
                            res = requests.post(
                                CRM_API_URL, 
                                json={"embedding": emb.tolist()},
                                timeout=2
                            )
                            if res.status_code == 200:
                                data = res.json()
                                if data.get("recognized"):
                                    last_name = data.get("customer_name")
                                    last_score = data.get("score")
                                else:
                                    last_name = "Unknown"
                                    last_score = data.get("score")
                        except Exception as e:
                            pass # API might be down, ignore
                except Exception as e:
                    print("[WARN] Alignment/Embedding error:", e)
            else:
                last_bbox = None
                last_name = None
                
        # Draw bounding box on frame for streaming
        out_frame = frame.copy()
        if last_bbox is not None:
            x1, y1, x2, y2 = last_bbox
            color = (70, 210, 90) if last_name and last_name != "Unknown" else (30, 190, 255)
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{last_name} ({last_score:.2f})" if last_name else "Detecting..."
            cv2.putText(out_frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Update latest frame
        ret, buffer = cv2.imencode('.jpg', out_frame)
        if ret:
            latest_frame = buffer.tobytes()
            
        # Cap frame rate to ~30 FPS to prevent OOM / CPU starvation
        time.sleep(0.03)
        
        # Force garbage collection every 30 frames to prevent OOM
        if frame_count % 30 == 0:
            gc.collect()

@app.on_event("startup")
def startup_event():
    # Start the camera loop in a background thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@app.on_event("shutdown")
def shutdown_event():
    if camera:
        camera.release()

def generate_mjpeg():
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.03) # ~30 fps

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/latest-embedding")
def get_latest_embedding():
    """Returns the embedding of the face currently in front of the camera (within last 5 seconds)"""
    if latest_embedding is None:
        raise HTTPException(status_code=404, detail="No face detected yet")
        
    if time.time() - last_face_time > 5.0:
        raise HTTPException(status_code=404, detail="No face detected in the last 5 seconds. Please look at the camera.")
        
    return {"embedding": latest_embedding.tolist()}

@app.post("/extract-embedding")
async def extract_embedding(file: UploadFile = File(...)):
    """Fallback: extract from uploaded image"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
        
    faces = detector.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")
        
    face = max(faces, key=lambda f: f.det_score)
    aligned = face_align.norm_crop(img, face.kps, image_size=112)
    embedding = rec_model.get_embedding(aligned)
    
    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    print("[SERVER] Starting Cafe FaceRec API + CCTV Server on port 5001...")
    uvicorn.run(app, host="0.0.0.0", port=5001)
