"""
facerec_uno_q.py
----------------
Cashier face-recognition agent for the Arduino UNO Q (Linux side) + USB webcam.
This single file is everything that runs on the device.

Pipeline (every camera frame)
    webcam frame (640x480)
      -> SCRFD-0.5G            face boxes + 5 landmarks          (det_500m.onnx)
      -> IoU tracker           same face keeps the same track id across frames
      -> quality gate          size, pose (yaw/pitch), sharpness, detector score
      -> 5-point alignment     similarity transform to the 112x112 ArcFace template
      -> MobileFaceNet         512-d embedding, L2-normalised     (w600k_mbf.onnx)
      -> FAISS IndexFlatIP     1:N cosine search over the Supabase gallery
      -> open-set threshold    score >= tau -> "known", else "unknown" vote
      -> temporal voting       >= VOTE_MIN agreeing votes out of VOTE_WINDOW decide
      -> Supabase              recognition_logs + visits (background thread)

HTTP API (port 5001, used by the CRM)
    GET  /video_feed          MJPEG stream with overlay
    GET  /latest-embedding    averaged embedding of the face in front of the camera (enrollment)
    POST /extract-embedding   embedding from an uploaded photo (multipart field "file" or raw bytes)
    POST /recognize           1:N recognition of an uploaded photo
    POST /reload-gallery      reload embeddings from Supabase now
    GET  /health              status, gallery size, timings, thresholds

Run
    python3 facerec_uno_q.py            # reads .env next to this file
"""

from __future__ import annotations

import collections
import email.parser
import email.policy
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

HERE = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(HERE / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("facerec")

MODEL_TAG = "scrfd500m+mobilefacenet"    # written to recognition_logs.model_used
EMBED_DIM = 512


# ── Configuration ─────────────────────────────────────────────────────────────

def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return cast(raw.strip())


@dataclass
class Config:
    database_url: str = _env("DATABASE_URL", "")
    camera_id: str = _env("CAMERA_ID", "uno-q-001")
    camera_index: int = _env("CAMERA_INDEX", -1, int)          # -1 = auto-detect /dev/video0..4
    frame_w: int = _env("FRAME_WIDTH", 640, int)
    frame_h: int = _env("FRAME_HEIGHT", 480, int)
    port: int = _env("STREAM_PORT", 5001, int)

    # Models
    det_model: str = _env("DET_MODEL_PATH", "")
    rec_model: str = _env("REC_MODEL_PATH", "")                # point at an INT8 model to use it
    num_threads: int = _env("NUM_THREADS", os.cpu_count() or 4, int)
    det_size: int = _env("DET_SIZE", 320, int)                 # SCRFD input width (multiple of 32)
    det_thresh: float = _env("DET_THRESH", 0.50, float)

    # Recognition
    match_threshold: float = _env("MATCH_THRESHOLD", 0.40, float)   # cosine tau (see scripts/evaluate.py)
    vote_window: int = _env("VOTE_WINDOW", 5, int)
    vote_min: int = _env("VOTE_MIN", 3, int)
    reverify_sec: float = _env("REVERIFY_SEC", 2.0, float)
    max_embed_per_frame: int = _env("MAX_EMBED_PER_FRAME", 1, int)
    track_ttl: float = _env("TRACK_TTL", 1.0, float)

    # Quality gate
    min_face_px: int = _env("MIN_FACE_PX", 60, int)
    min_det_score: float = _env("MIN_DET_SCORE", 0.60, float)
    max_yaw: float = _env("MAX_YAW", 0.28, float)
    min_sharpness: float = _env("MIN_SHARPNESS", 25.0, float)

    # Enrollment
    enroll_min_frames: int = _env("ENROLL_MIN_FRAMES", 3, int)
    enroll_max_age: float = _env("ENROLL_MAX_AGE", 5.0, float)
    allow_duplicate_enroll: bool = _env("ALLOW_DUPLICATE_ENROLL", "0") == "1"

    # Logging to Supabase
    known_log_cooldown: float = _env("KNOWN_LOG_COOLDOWN", 60.0, float)
    unknown_log_cooldown: float = _env("UNKNOWN_LOG_COOLDOWN", 10.0, float)
    visit_cooldown_min: int = _env("VISIT_COOLDOWN_MIN", 30, int)   # same as CRM VisitService
    reload_interval: float = _env("RELOAD_INTERVAL", 120.0, float)
    gallery_since: str = _env("GALLERY_SINCE", "")              # e.g. 2026-09-12 to skip legacy embeddings

    # Stream
    stream_fps: float = _env("STREAM_FPS", 15.0, float)
    jpeg_quality: int = _env("JPEG_QUALITY", 70, int)


# Embeddings created before this date came from older pipelines
# (ResNet50 until early Sep 2026, then MobileFaceNet without alignment).
PIPELINE_SINCE = "2026-09-12"


# ── Model files ───────────────────────────────────────────────────────────────

BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
_MODEL_DIRS = [HERE / "models", Path.home() / ".insightface" / "models" / "buffalo_sc"]


def find_model(filename: str, override: str = "") -> str:
    """Locate an ONNX file: explicit path -> ./models -> ~/.insightface -> download buffalo_sc."""
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"Model not found: {override}")
        return override
    for d in _MODEL_DIRS:
        if (d / filename).exists():
            return str(d / filename)
    target = _MODEL_DIRS[0]
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "buffalo_sc.zip"
    log.info("Downloading buffalo_sc model pack (~15 MB) ...")
    urllib.request.urlretrieve(BUFFALO_SC_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = os.path.basename(member)
            if name.endswith(".onnx"):
                (target / name).write_bytes(z.read(member))
    zip_path.unlink(missing_ok=True)
    return str(target / filename)


def make_session(path: str, num_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3                                  # silence shape warnings
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # don't burn idle cores
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


# ── Alignment ─────────────────────────────────────────────────────────────────

# ArcFace 112x112 template: left eye, right eye, nose tip, left mouth, right mouth
REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama), same as insightface norm_crop."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    r = u @ np.diag(d) @ vt
    scale = (s * d).sum() / sc.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    m = similarity_transform(kps, REFERENCE_5PTS)
    return cv2.warpAffine(img, m, (112, 112), borderValue=0)


# ── SCRFD-0.5G detector ───────────────────────────────────────────────────────

class SCRFD:
    """InsightFace SCRFD (det_500m.onnx): boxes + 5 landmarks, strides 8/16/32, 2 anchors."""

    def __init__(self, model_path: str, num_threads: int, det_size: int = 320,
                 conf_thresh: float = 0.5, nms_thresh: float = 0.4):
        self.session = make_session(model_path, num_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_size = det_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = (8, 16, 32)
        self._anchor_cache: dict = {}

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._anchor_cache:
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            self._anchor_cache[key] = np.repeat(centers, 2, axis=0)
        return self._anchor_cache[key]

    def detect(self, img: np.ndarray):
        """Returns dets (N,5) [x1,y1,x2,y2,score] and kps (N,5,2), sorted by score."""
        ih, iw = img.shape[:2]
        # Input keeps the frame's aspect ratio (e.g. 640x480 -> 320x256) instead of a square pad.
        in_w = self.det_size
        in_h = int(np.ceil(self.det_size * ih / iw / 32.0) * 32)
        scale = min(in_w / iw, in_h / ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (in_w, in_h), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_l, boxes_l, kps_l = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            boxes = outs[i + 3].reshape(-1, 4) * stride
            kps = outs[i + 6].reshape(-1, 10) * stride
            anchors = self._anchors(in_h // stride, in_w // stride, stride)
            keep = np.where(scores >= self.conf_thresh)[0]
            if len(keep) == 0:
                continue
            a = anchors[keep]
            b = boxes[keep]
            boxes_l.append(np.stack([a[:, 0] - b[:, 0], a[:, 1] - b[:, 1],
                                     a[:, 0] + b[:, 2], a[:, 1] + b[:, 3]], axis=-1) / scale)
            k = kps[keep].reshape(-1, 5, 2)
            kps_l.append((k + a[:, None, :]) / scale)
            scores_l.append(scores[keep])

        if not scores_l:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)

        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l)
        kps = np.concatenate(kps_l)
        tlwh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
        idx = cv2.dnn.NMSBoxes(tlwh.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh)
        idx = np.array(idx, dtype=int).reshape(-1)
        idx = idx[np.argsort(-scores[idx])]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih - 1)
        dets = np.column_stack([boxes[idx], scores[idx]]).astype(np.float32)
        return dets, kps[idx].astype(np.float32)


# ── MobileFaceNet embedder ────────────────────────────────────────────────────

class MobileFaceNet:
    """InsightFace w600k_mbf.onnx: MobileFaceNet trained with ArcFace loss on WebFace600K."""

    def __init__(self, model_path: str, num_threads: int):
        self.session = make_session(model_path, num_threads)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]
        feat = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        return l2_normalize(feat)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-10)).astype(np.float32)


# ── Gallery (FAISS cosine search) ─────────────────────────────────────────────

class Gallery:
    """Enrolled embeddings. Vectors are L2-normalised, so inner product = cosine similarity."""

    def __init__(self):
        self._lock = threading.Lock()
        self.ids: list[str] = []
        self.names: dict[str, str] = {}
        self.vectors = np.zeros((0, EMBED_DIM), np.float32)
        self.index = None

    def load(self, rows: list[tuple[str, str, np.ndarray]]):
        ids = [r[0] for r in rows]
        names = {r[0]: r[1] for r in rows}
        vecs = (np.stack([l2_normalize(r[2]) for r in rows]) if rows
                else np.zeros((0, EMBED_DIM), np.float32))
        index = None
        if _HAS_FAISS and len(vecs):
            index = faiss.IndexFlatIP(EMBED_DIM)
            index.add(vecs)
        with self._lock:
            self.ids, self.names, self.vectors, self.index = ids, names, vecs, index

    def search(self, probe: np.ndarray) -> tuple[Optional[str], float]:
        """Best-matching customer and its cosine score (top-1 over all enrolled embeddings)."""
        with self._lock:
            if not self.ids:
                return None, 0.0
            if self.index is not None:
                d, i = self.index.search(probe.reshape(1, -1), 1)
                return self.ids[int(i[0, 0])], float(d[0, 0])
            sims = self.vectors @ probe
            j = int(np.argmax(sims))
            return self.ids[j], float(sims[j])

    def name_of(self, customer_id: Optional[str]) -> str:
        return self.names.get(customer_id, "") if customer_id else ""

    def stats(self) -> dict:
        with self._lock:
            return {"n_embeddings": len(self.ids), "n_customers": len(set(self.ids)),
                    "backend": "faiss.IndexFlatIP" if self.index is not None else "numpy"}


# ── Quality gate ──────────────────────────────────────────────────────────────

def pose_and_size(kps: np.ndarray, bbox: np.ndarray) -> tuple[float, float, float]:
    """Yaw proxy (nose offset / eye distance; 0 = frontal), pitch proxy (~0.5 frontal), face size px."""
    le, re, nose, lm, rm = kps
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    eye_dist = max(np.linalg.norm(re - le), 1e-3)
    yaw = float((nose[0] - eye_mid[0]) / eye_dist)
    pitch = float((nose[1] - eye_mid[1]) / max(mouth_mid[1] - eye_mid[1], 1e-3))
    size = float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return yaw, pitch, size


def sharpness(aligned: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def check_quality(cfg: Config, det: np.ndarray, kps: np.ndarray,
                  aligned: Optional[np.ndarray] = None) -> Optional[str]:
    """None if the face is good enough to recognise, else a short reason shown on screen."""
    yaw, pitch, size = pose_and_size(kps, det[:4])
    if size < cfg.min_face_px:
        return "move closer"
    if det[4] < cfg.min_det_score:
        return "low det score"
    if abs(yaw) > cfg.max_yaw or not (0.2 <= pitch <= 0.8):
        return "face the camera"
    if aligned is not None and sharpness(aligned) < cfg.min_sharpness:
        return "blurry"
    return None


# ── Tracking + temporal voting ────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    first_seen: float
    last_seen: float
    votes: collections.deque = field(default_factory=collections.deque)
    samples: collections.deque = field(default_factory=collections.deque)  # good embeddings (RAM only)
    state: str = "pending"            # pending | known | unknown
    customer_id: Optional[str] = None
    score: float = 0.0
    last_embed: float = 0.0
    quality_msg: Optional[str] = None
    logged_state: Optional[tuple] = None


class FaceTracker:
    """Greedy IoU tracker. A track is one face standing in front of the camera."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, dets: np.ndarray, kps: np.ndarray, now: float) -> list[Track]:
        # Drop stale tracks first so a new person at the same spot never inherits an old identity.
        for tid in [tid for tid, t in self.tracks.items() if now - t.last_seen > self.cfg.track_ttl]:
            del self.tracks[tid]
        pairs = sorted(((iou(t.bbox, d[:4]), tid, j)
                        for tid, t in self.tracks.items() for j, d in enumerate(dets)), reverse=True)
        used_t, used_d, updated = set(), set(), []
        for overlap, tid, j in pairs:
            if overlap < 0.3:
                break
            if tid in used_t or j in used_d:
                continue
            t = self.tracks[tid]
            t.bbox, t.kps, t.det_score, t.last_seen = dets[j][:4].copy(), kps[j].copy(), float(dets[j][4]), now
            used_t.add(tid)
            used_d.add(j)
            updated.append(t)
        for j, d in enumerate(dets):
            if j in used_d:
                continue
            t = Track(self._next_id, d[:4].copy(), kps[j].copy(), float(d[4]), now, now,
                      collections.deque(maxlen=self.cfg.vote_window),
                      collections.deque(maxlen=10))
            self.tracks[t.track_id] = t
            self._next_id += 1
            updated.append(t)
        return updated


def decide(track: Track, cfg: Config) -> tuple[str, Optional[str], float]:
    """Majority vote over the last VOTE_WINDOW single-frame results of one track."""
    counts = collections.Counter(cid for cid, _ in track.votes)
    best, n_best = None, 0
    for cid, n in counts.items():
        if cid is not None and n > n_best:
            best, n_best = cid, n
    if best is not None and n_best >= cfg.vote_min:
        return "known", best, float(np.mean([s for c, s in track.votes if c == best]))
    if counts.get(None, 0) >= cfg.vote_min:
        return "unknown", None, float(np.mean([s for c, s in track.votes if c is None]))
    return "pending", None, 0.0


# ── Recognition engine ────────────────────────────────────────────────────────

class FaceEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cv2.setNumThreads(1)  # leave the cores to ONNX Runtime
        self.detector = SCRFD(find_model("det_500m.onnx", cfg.det_model), cfg.num_threads,
                              cfg.det_size, cfg.det_thresh)
        self.embedder = MobileFaceNet(find_model("w600k_mbf.onnx", cfg.rec_model), cfg.num_threads)
        self.gallery = Gallery()
        self.tracker = FaceTracker(cfg)
        self.timing = {"det_ms": 0.0, "emb_ms": 0.0, "frame_ms": 0.0}

    def _ema(self, key: str, value: float):
        self.timing[key] = value if self.timing[key] == 0 else 0.9 * self.timing[key] + 0.1 * value

    def match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """Open-set decision for one embedding: (customer_id or None, cosine score)."""
        cid, score = self.gallery.search(emb)
        return (cid if cid is not None and score >= self.cfg.match_threshold else None), score

    def process(self, frame: np.ndarray, now: float) -> tuple[list[Track], list[Track]]:
        """Run one frame. Returns (visible tracks, tracks whose decision just changed)."""
        t0 = time.perf_counter()
        dets, kps = self.detector.detect(frame)
        self._ema("det_ms", (time.perf_counter() - t0) * 1000)
        visible = self.tracker.update(dets, kps, now)

        # Embed the faces that need it most: undecided first, then largest.
        def priority(t: Track):
            return (t.state != "pending", -(t.bbox[2] - t.bbox[0]))
        budget = self.cfg.max_embed_per_frame
        changed = []
        for t in sorted(visible, key=priority):
            # Re-check every frame while undecided or when the last vote disagreed
            # (e.g. two people swapped places), otherwise only every REVERIFY_SEC.
            disagrees = bool(t.votes) and t.votes[-1][0] != t.customer_id
            due = t.state == "pending" or disagrees or now - t.last_embed >= self.cfg.reverify_sec
            t.quality_msg = check_quality(self.cfg, np.append(t.bbox, t.det_score), t.kps)
            if budget <= 0 or not due or t.quality_msg:
                continue
            aligned = align_face(frame, t.kps)
            if sharpness(aligned) < self.cfg.min_sharpness:
                t.quality_msg = "blurry"
                continue
            t1 = time.perf_counter()
            emb = self.embedder.embed(aligned)
            self._ema("emb_ms", (time.perf_counter() - t1) * 1000)
            budget -= 1
            t.last_embed = now
            t.samples.append(emb)
            t.votes.append(self.match(emb))
            state, cid, score = decide(t, self.cfg)
            if state != "pending":
                if (state, cid) != (t.state, t.customer_id):
                    changed.append(t)
                t.state, t.customer_id, t.score = state, cid, score
        self._ema("frame_ms", (time.perf_counter() - t0) * 1000)
        return visible, changed

    def embed_image(self, img: np.ndarray) -> dict:
        """Stateless: largest good face in a still image -> embedding (for uploads)."""
        dets, kps = self.detector.detect(img)
        if len(dets) == 0:
            return {"ok": False, "message": "No face detected in the image."}
        j = int(np.argmax((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])))
        aligned = align_face(img, kps[j])
        reason = check_quality(self.cfg, dets[j], kps[j], aligned)
        if reason:
            return {"ok": False, "message": f"Face quality too low: {reason}."}
        return {"ok": True, "embedding": self.embedder.embed(aligned), "det_score": float(dets[j][4]),
                "bbox": [int(v) for v in dets[j][:4]]}


# ── Supabase worker (all network I/O off the camera loop) ─────────────────────

def parse_vector(v) -> np.ndarray:
    if isinstance(v, str):
        v = json.loads(v) if v.strip().startswith("[") else [float(x) for x in v.split(",")]
    return np.asarray(v, dtype=np.float32)


class Database(threading.Thread):
    def __init__(self, cfg: Config, gallery: Gallery):
        super().__init__(daemon=True, name="db")
        self.cfg = cfg
        self.gallery = gallery
        self.jobs: queue.Queue = queue.Queue(maxsize=500)
        self.conn = None
        self.online = False
        self.gallery_loaded = threading.Event()
        self._reload_at = 0.0

    def submit(self, kind: str, **kw):
        try:
            self.jobs.put_nowait((kind, kw))
        except queue.Full:
            log.warning("[DB] queue full, dropping %s", kind)

    def request_reload(self, delay: float = 0.0):
        self._reload_at = time.time() + delay

    def _connect(self):
        import psycopg2
        self.conn = psycopg2.connect(self.cfg.database_url, connect_timeout=10)
        self.conn.autocommit = True
        self.online = True
        log.info("[DB] Connected to Supabase")

    def _load_gallery(self):
        sql = """SELECT e.customer_id::text, c.name, e.embedding_vector::text, e.created_at
                 FROM embeddings e JOIN customers c ON c.customer_id = e.customer_id
                 WHERE c.is_active = TRUE"""
        params = ()
        if self.cfg.gallery_since:
            sql += " AND e.created_at >= %s"
            params = (self.cfg.gallery_since,)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        good, legacy = [], 0
        for cid, name, vec, created in rows:
            v = parse_vector(vec)
            if v.shape != (EMBED_DIM,):
                continue
            if created is not None and str(created)[:10] < PIPELINE_SINCE:
                legacy += 1
            good.append((cid, name or "", v))
        self.gallery.load(good)
        self.gallery_loaded.set()
        s = self.gallery.stats()
        log.info("[Gallery] %d embeddings / %d customers (%s)", s["n_embeddings"], s["n_customers"], s["backend"])
        if legacy:
            log.warning("[Gallery] %d embeddings were enrolled before %s with an older model/pipeline. "
                        "They will not match MobileFaceNet reliably: re-enroll those customers "
                        "(scripts/gallery_tool.py check).", legacy, PIPELINE_SINCE)

    def _execute(self, kind: str, kw: dict):
        with self.conn.cursor() as cur:
            if kind == "log":
                cur.execute("""INSERT INTO recognition_logs
                                   (customer_id, similarity_score, model_used, camera_id, recognized)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (kw["customer_id"], round(kw["score"], 6), MODEL_TAG,
                             self.cfg.camera_id, kw["customer_id"] is not None))
            elif kind == "visit":
                # One visit per customer per VISIT_COOLDOWN_MIN, same rule as the CRM VisitService.
                cur.execute("""INSERT INTO visits (customer_id, entry_time)
                               SELECT %s::uuid, NOW()
                               WHERE NOT EXISTS (
                                   SELECT 1 FROM visits WHERE customer_id = %s::uuid
                                   AND entry_time > NOW() - make_interval(mins => %s))""",
                            (kw["customer_id"], kw["customer_id"], self.cfg.visit_cooldown_min))

    def run(self):
        backoff = 2.0
        pending = None
        while True:
            try:
                if self.conn is None or self.conn.closed:
                    self._connect()
                    backoff = 2.0
                if time.time() >= self._reload_at:
                    self._load_gallery()
                    self._reload_at = time.time() + self.cfg.reload_interval
                if pending is None:
                    try:
                        pending = self.jobs.get(timeout=1.0)
                    except queue.Empty:
                        continue
                self._execute(*pending)
                pending = None
            except Exception as e:
                self.online = False
                log.warning("[DB] %s -- retrying in %.0fs (recognition keeps running)", e, backoff)
                try:
                    if self.conn is not None:
                        self.conn.close()
                except Exception:
                    pass
                self.conn = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)


# ── Camera (always hands out the newest frame -> no buffer lag) ──────────────

class Camera(threading.Thread):
    def __init__(self, cfg: Config):
        super().__init__(daemon=True, name="camera")
        self.cfg = cfg
        self.cap = None
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def _open(self) -> bool:
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        indices = [self.cfg.camera_index] if self.cfg.camera_index >= 0 else range(5)
        for i in indices:
            cap = cv2.VideoCapture(i, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # less USB bandwidth / CPU
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                log.info("[Camera] /dev/video%d opened at %dx%d", i, frame.shape[1], frame.shape[0])
                self.cap = cap
                return True
            cap.release()
        return False

    def run(self):
        fails = 0
        while True:
            if self.cap is None and not self._open():
                log.error("[Camera] No webcam found, retrying in 3s ...")
                time.sleep(3)
                continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    log.warning("[Camera] Lost camera, reopening")
                    self.cap.release()
                    self.cap = None
                    fails = 0
                time.sleep(0.01)
                continue
            fails = 0
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def latest(self, after_seq: int, timeout: float = 1.0):
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq, timeout=timeout)
            return self._frame, self._seq


# ── Shared state between the camera loop and the HTTP server ─────────────────

class AppState:
    def __init__(self, cfg: Config, engine: FaceEngine, db: Optional[Database]):
        self.cfg = cfg
        self.engine = engine
        self.db = db
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.jpeg_seq = 0
        self.stream_clients = 0
        self.enroll: Optional[dict] = None   # template of the main face currently in view
        self.fps = 0.0
        self.faces_in_view = 0
        self.started = time.time()


# ── Overlay ───────────────────────────────────────────────────────────────────

def draw(frame: np.ndarray, tracks: list[Track], engine: FaceEngine, fps: float) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        if t.state == "known":
            color, label = (70, 200, 70), f"{engine.gallery.name_of(t.customer_id) or t.customer_id[:8]} {t.score:.2f}"
        elif t.state == "unknown":
            color, label = (0, 165, 255), f"Unknown {t.score:.2f}"
        else:
            color, label = (200, 200, 200), t.quality_msg or "recognizing..."
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (20, 20, 20), 1, cv2.LINE_AA)
    tm = engine.timing
    hud = f"FPS {fps:.1f} | det {tm['det_ms']:.0f}ms | emb {tm['emb_ms']:.0f}ms"
    cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return out


# ── HTTP server ───────────────────────────────────────────────────────────────

def _read_upload(handler: BaseHTTPRequestHandler) -> Optional[bytes]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0 or length > 10 * 1024 * 1024:
        return None
    body = handler.rfile.read(length)
    ctype = handler.headers.get("Content-Type", "")
    if not ctype.startswith("multipart/form-data"):
        return body
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
    for part in msg.iter_parts():
        if part.get_filename() or part.get_param("name", header="content-disposition") == "file":
            return part.get_payload(decode=True)
    return None


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def _json(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/video_feed"):
                return self._stream()
            if path == "/latest-embedding":
                return self._latest_embedding()
            if path == "/health":
                return self._health()
            self._json(404, {"detail": "Not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/reload-gallery":
                if state.db is None:
                    return self._json(503, {"detail": "No database configured."})
                state.db.request_reload()
                return self._json(202, {"detail": "Gallery reload scheduled."})
            if path not in ("/extract-embedding", "/recognize"):
                return self._json(404, {"detail": "Not found"})
            raw = _read_upload(self)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            if img is None:
                return self._json(400, {"detail": "Invalid or missing image (multipart field 'file')."})
            t0 = time.perf_counter()
            res = state.engine.embed_image(img)
            if not res["ok"]:
                return self._json(400, {"detail": res["message"]})
            if path == "/extract-embedding":
                return self._json(200, {"embedding": res["embedding"].tolist(),
                                        "det_score": round(res["det_score"], 4), "bbox": res["bbox"]})
            cid, score = state.engine.match(res["embedding"])
            self._json(200, {"status": "known" if cid else "unknown", "customer_id": cid,
                             "name": state.engine.gallery.name_of(cid), "score": round(score, 4),
                             "threshold": state.cfg.match_threshold, "bbox": res["bbox"],
                             "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})

        def _latest_embedding(self):
            with state.lock:
                e = state.enroll
            if e is None or time.time() - e["time"] > state.cfg.enroll_max_age:
                return self._json(404, {"detail": "Belum ada wajah terdeteksi di kamera. "
                                                  "Minta pelanggan menghadap kamera."})
            if e["n_frames"] < state.cfg.enroll_min_frames:
                return self._json(404, {"detail": "Wajah belum cukup jelas, tahan posisi sebentar."})
            if e["state"] == "known" and not state.cfg.allow_duplicate_enroll:
                return self._json(409, {"detail": f"Wajah ini sudah terdaftar sebagai "
                                                  f"{e['name'] or e['customer_id']}."})
            # The CRM saves this embedding right after the call: pick it up a few seconds later
            # so the new customer is recognised immediately instead of after RELOAD_INTERVAL.
            if state.db is not None:
                state.db.request_reload(delay=5.0)
            self._json(200, {"embedding": e["embedding"], "n_frames": e["n_frames"],
                             "faces_in_view": e["faces_in_view"],
                             "recognition": {"status": e["state"], "customer_id": e["customer_id"],
                                             "score": e["score"]}})

        def _health(self):
            self._json(200, {
                "status": "ok", "camera_id": state.cfg.camera_id, "model": MODEL_TAG,
                "uptime_s": round(time.time() - state.started), "fps": round(state.fps, 1),
                "timing_ms": {k: round(v, 1) for k, v in state.engine.timing.items()},
                "faces_in_view": state.faces_in_view, "gallery": state.engine.gallery.stats(),
                "db_online": bool(state.db and state.db.online),
                "threshold": state.cfg.match_threshold,
                "voting": {"window": state.cfg.vote_window, "min_votes": state.cfg.vote_min},
                "quality": {"min_face_px": state.cfg.min_face_px, "max_yaw": state.cfg.max_yaw,
                            "min_sharpness": state.cfg.min_sharpness},
            })

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state.lock:
                state.stream_clients += 1
            last = 0
            try:
                while True:
                    with state.lock:
                        jpeg, seq = state.jpeg, state.jpeg_seq
                    if jpeg is not None and seq != last:
                        last = seq
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with state.lock:
                    state.stream_clients -= 1
    return Handler


# ── Main loop ─────────────────────────────────────────────────────────────────

def handle_events(state: AppState, changed: list[Track], now: float, last_known: dict, last_unknown: list):
    """Turn confirmed track decisions into Supabase rows (one per person, not per frame)."""
    for t in changed:
        key = (t.state, t.customer_id)
        if t.logged_state == key:
            continue
        t.logged_state = key
        name = state.engine.gallery.name_of(t.customer_id)
        if t.state == "known":
            if now - last_known.get(t.customer_id, 0) < state.cfg.known_log_cooldown:
                continue
            last_known[t.customer_id] = now
            log.info("[KNOWN]   %s (%s) score=%.3f track=%d", name, t.customer_id[:8], t.score, t.track_id)
            if state.db:
                state.db.submit("log", customer_id=t.customer_id, score=t.score)
                state.db.submit("visit", customer_id=t.customer_id)
        else:
            if now - last_unknown[0] < state.cfg.unknown_log_cooldown:
                continue
            last_unknown[0] = now
            log.info("[UNKNOWN] score=%.3f track=%d", t.score, t.track_id)
            if state.db:
                state.db.submit("log", customer_id=None, score=t.score)


def update_enrollment(state: AppState, visible: list[Track], now: float):
    """Keep an averaged template of the largest face in view for /latest-embedding."""
    if not visible:
        return
    main = max(visible, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]))
    if not main.samples:
        return
    template = l2_normalize(np.mean(np.stack(main.samples), axis=0))
    with state.lock:
        state.enroll = {"time": now, "embedding": template.tolist(), "n_frames": len(main.samples),
                        "faces_in_view": len(visible), "state": main.state,
                        "customer_id": main.customer_id, "name": state.engine.gallery.name_of(main.customer_id),
                        "score": round(main.score, 4)}


def main():
    cfg = Config()
    log.info("=" * 60)
    log.info(" Cafe FaceRec -- SCRFD-0.5G + MobileFaceNet + FAISS (UNO Q)")
    log.info("=" * 60)
    if not cfg.database_url:
        log.warning("DATABASE_URL not set: running offline (no gallery, nothing logged).")

    engine = FaceEngine(cfg)
    log.info("[Model] ready | threads=%d det_size=%d tau=%.3f vote=%d/%d",
             cfg.num_threads, cfg.det_size, cfg.match_threshold, cfg.vote_min, cfg.vote_window)

    db = None
    if cfg.database_url:
        db = Database(cfg, engine.gallery)
        db.start()
        if not db.gallery_loaded.wait(timeout=20):
            log.warning("[Gallery] Not loaded yet (DB unreachable?) -- starting anyway, will retry.")

    state = AppState(cfg, engine, db)
    camera = Camera(cfg)
    camera.start()

    server = ThreadingHTTPServer(("0.0.0.0", cfg.port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    log.info("[HTTP] http://<uno-q-ip>:%d/video_feed  /latest-embedding  /health", cfg.port)

    last_known: dict[str, float] = {}
    last_unknown = [0.0]
    seq, n_frames, fps_t0, last_jpeg = 0, 0, time.time(), 0.0
    try:
        while True:
            frame, seq = camera.latest(seq)
            if frame is None:
                continue
            now = time.time()
            visible, changed = engine.process(frame, now)
            handle_events(state, changed, now, last_known, last_unknown)
            update_enrollment(state, visible, now)
            state.faces_in_view = len(visible)

            n_frames += 1
            if now - fps_t0 >= 1.0:
                state.fps, n_frames, fps_t0 = n_frames / (now - fps_t0), 0, now

            # Encode the overlay only when someone is watching, capped at STREAM_FPS.
            if state.stream_clients > 0 and now - last_jpeg >= 1.0 / cfg.stream_fps:
                last_jpeg = now
                ok, buf = cv2.imencode(".jpg", draw(frame, visible, engine, state.fps),
                                       [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                if ok:
                    with state.lock:
                        state.jpeg, state.jpeg_seq = buf.tobytes(), state.jpeg_seq + 1
    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
