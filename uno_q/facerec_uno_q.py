"""
facerec_uno_q.py -- cashier face recognition for the Arduino UNO Q (Linux side) + USB webcam
=============================================================================================
Everything that runs on the UNO Q is in this one file.

Pipeline
    webcam (640x480, newest frame only)
      -> SCRFD-0.5G (det_500m)        face box + 5 landmarks, 320x320 input
      -> quality score                size, head pose, sharpness, brightness (bad frames dropped)
      -> 5-point similarity alignment  Umeyama, ArcFace 112x112 template (= insightface norm_crop)
      -> MobileFaceNet (w600k_mbf)     512-d embedding, L2-normalised
      -> face vault rotation           secret orthogonal matrix: cosine unchanged, raw vector never stored
      -> FAISS IndexFlatIP             1:N cosine search over the gallery from Supabase
      -> session decision              ~10 frames of one face (<= 2.5 s): mean of the 3 best frames
                                       >= tau, margin to the next customer >= m, and >= 60 % of the
                                       best frames vote for the same customer
      -> outputs                       recognition_logs + visit (Supabase), face event -> Raspberry Pi

Inference uses OpenCV DNN, not onnxruntime: the onnxruntime wheel stops with
"Illegal instruction" on the UNO Q's CPU (QRB2210, no ARMv8.2 dot-product).

Faces are enrolled ONLY from the CRM (customer consent first): the CRM writes an
enrollment_requests row, this program captures the face and stores the embeddings.

Run:  python3 facerec_uno_q.py      (settings in .env next to this file)
"""

from __future__ import annotations

import base64
import collections
import glob
import hashlib
import hmac
import json
import logging
import os
import queue
import socket
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

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

HERE = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(HERE / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("facerec")

VERSION = "2.0.0"
MODEL_TAG = "scrfd500m+mobilefacenet"   # recognition_logs.model_used
MODEL_NAME = "mbf-scrfd-v2"             # embeddings.model_name: only these rows are loaded
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
    device_id: str = _env("CAMERA_ID", "uno-q-001")         # = CRM Pengaturan > Perangkat > ID kamera kasir
    camera_source: str = _env("CAMERA_SOURCE", "")           # "" auto | 0 | /dev/v4l/by-id/... | http://...
    frame_w: int = _env("FRAME_WIDTH", 640, int)
    frame_h: int = _env("FRAME_HEIGHT", 480, int)
    threads: int = _env("NUM_THREADS", 4, int)

    det_size: int = _env("DET_SIZE", 320, int)               # 320: 45 ms on the UNO Q, 640: 174 ms
    det_thresh: float = _env("DET_THRESH", 0.50, float)

    match_threshold: float = _env("MATCH_THRESHOLD", 0.40, float)
    match_margin: float = _env("MATCH_MARGIN", 0.08, float)
    vote_ratio: float = _env("VOTE_RATIO", 0.60, float)
    session_frames: int = _env("SESSION_FRAMES", 10, int)
    session_max_sec: float = _env("SESSION_MAX_SEC", 2.5, float)
    session_min_frames: int = _env("SESSION_MIN_FRAMES", 3, int)

    enroll_frames: int = _env("ENROLL_FRAMES", 20, int)
    enroll_keep: int = _env("ENROLL_KEEP", 5, int)
    enroll_timeout: float = _env("ENROLL_TIMEOUT_SEC", 20.0, float)

    known_log_cooldown: float = _env("KNOWN_LOG_COOLDOWN", 30.0, float)   # < CRM Kasir window (45 s)
    unknown_log_cooldown: float = _env("UNKNOWN_LOG_COOLDOWN", 10.0, float)
    visit_max_open_hours: int = _env("VISIT_MAX_OPEN_HOURS", 6, int)
    visit_reopen_min: int = _env("VISIT_REOPEN_MIN", 30, int)
    reload_interval: float = _env("RELOAD_INTERVAL", 120.0, float)
    heartbeat_sec: float = _env("HEARTBEAT_SEC", 10.0, float)

    pi_face_url: str = _env("PI_FACE_URL", "")               # http://<pi-ip>:8765/wajah
    face_token: str = _env("FACE_TOKEN", "")                 # same as FACE_TOKEN on the Pi
    pi_cooldown: float = _env("PI_COOLDOWN", 2.0, float)
    area_camera: str = _env("AREA_CAMERA_ID", "CAM_1")       # Pi camera that sees the cashier

    vault_key: str = _env("VAULT_KEY", "")                   # 64 hex chars; KEEP A BACKUP
    local_host: str = _env("STREAM_HOST", "127.0.0.1")       # local preview/health only
    local_port: int = _env("STREAM_PORT", 5001, int)


# ── Small helpers ─────────────────────────────────────────────────────────────

def l2n(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return (v / (np.linalg.norm(v) + 1e-10)).astype(np.float32)


def utc_iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1000):03d}Z"


def list_cameras() -> list[dict]:
    """Cameras plugged into this board, by stable id (survives replugging)."""
    out = []
    for p in sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
        name = os.path.basename(p).replace("usb-", "").replace("-video-index0", "").replace("_", " ")
        out.append({"source": p, "name": name})
    return out


# ── Model files ───────────────────────────────────────────────────────────────

BUFFALO_SC_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
MODEL_DIRS = [HERE / "models", Path.home() / ".insightface" / "models" / "buffalo_sc"]


def find_model(filename: str) -> str:
    for d in MODEL_DIRS:
        if (d / filename).exists():
            return str(d / filename)
    target = MODEL_DIRS[0]
    target.mkdir(parents=True, exist_ok=True)
    log.info("Downloading buffalo_sc (SCRFD-0.5G + MobileFaceNet, ~15 MB) ...")
    zpath = target / "buffalo_sc.zip"
    urllib.request.urlretrieve(BUFFALO_SC_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if m.endswith(".onnx"):
                (target / os.path.basename(m)).write_bytes(z.read(m))
    zpath.unlink(missing_ok=True)
    if not (target / filename).exists():
        raise FileNotFoundError(filename)
    return str(target / filename)


def det_model_path(size: int) -> str:
    """SCRFD with a FIXED input size (models/det_500m_320.onnx, shipped with this program).
    The original det_500m.onnx has a dynamic input that OpenCV 4.x cannot import; OpenCV 5
    can, so the original is the fallback."""
    fixed = HERE / "models" / f"det_500m_{size}.onnx"
    if fixed.exists():
        return str(fixed)
    return find_model("det_500m.onnx")


# ── Face vault (cancelable template) ──────────────────────────────────────────

def _hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256 (RFC 5869) with an all-zero salt."""
    prk = hmac.new(b"\x00" * 32, key, hashlib.sha256).digest()
    out, block, i = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([i]), hashlib.sha256).digest()
        out += block
        i += 1
    return out[:length]


class Vault:
    """
    Secret orthogonal rotation R of the embedding space (cancelable biometrics,
    Ratha et al. 2001). Cosine similarity is unchanged, so matching is exactly the
    same, but the stored vectors are not MobileFaceNet outputs: they cannot be
    matched against other systems or inverted with public models. Changing the key
    revokes every stored template at once. R is derived only from SHA-256, so it is
    identical on every machine and every numpy version.
    """

    def __init__(self, key_hex: str):
        key = bytes.fromhex(key_hex)
        if len(key) != 32:
            raise ValueError("VAULT_KEY must be 64 hex characters")
        seed = _hkdf(key, b"facerec/rotation/v2")
        need = EMBED_DIM * EMBED_DIM
        stream = bytearray()
        counter = 0
        while len(stream) < need * 8:
            stream += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
            counter += 1
        u = (np.frombuffer(bytes(stream[:need * 8]), dtype=">u8").astype(np.float64) + 0.5) / 2.0 ** 64
        u1, u2 = u[: need // 2], u[need // 2:]
        r = np.sqrt(-2.0 * np.log(u1))
        g = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)]).reshape(EMBED_DIM, EMBED_DIM)
        q, rr = np.linalg.qr(g)
        self.R = (q * np.sign(np.diag(rr))).astype(np.float32)
        self.fingerprint = hashlib.sha256(b"fp" + seed).hexdigest()[:12]

    def protect(self, emb: np.ndarray) -> np.ndarray:
        return l2n(self.R @ l2n(emb))


def load_vault_key(cfg: Config) -> str:
    if cfg.vault_key:
        return cfg.vault_key
    path = HERE / "data" / "vault.key"
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32).hex()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    log.warning("[Vault] New face vault key created in %s. COPY IT to VAULT_KEY in .env and keep a "
                "backup: without it every enrolled face must be enrolled again.", path)
    return key


# ── Alignment ─────────────────────────────────────────────────────────────────

ARC_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (Umeyama 1991), same as insightface norm_crop."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    rot = u @ np.diag(d) @ vt
    scale = (s * d).sum() / sc.var(0).sum()
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]]).astype(np.float32)


def align(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(frame, similarity_transform(kps, ARC_TEMPLATE), (112, 112), borderValue=0)


# ── Models (OpenCV DNN) ───────────────────────────────────────────────────────

@dataclass
class Face:
    box: np.ndarray          # x1, y1, x2, y2
    score: float
    kps: np.ndarray          # 5x2: eye (image left), eye (image right), nose, mouth left, mouth right

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])


class SCRFD:
    """InsightFace SCRFD-0.5G: strides 8/16/32, 2 anchors per cell, square letterboxed input."""

    STRIDES = (8, 16, 32)

    def __init__(self, path: str, size: int = 320, thresh: float = 0.5, nms: float = 0.4):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.names = self.net.getUnconnectedOutLayersNames()
        self.size, self.thresh, self.nms = size, thresh, nms
        self._anchors = {}

    def _anchor(self, stride: int) -> np.ndarray:
        if stride not in self._anchors:
            n = self.size // stride
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).astype(np.float32) * stride
            self._anchors[stride] = np.repeat(c.reshape(-1, 2), 2, axis=0)
        return self._anchors[stride]

    def _split(self, outs) -> dict:
        """Map outputs to (stride, kind) by shape, independent of the output order."""
        by = {}
        for o in outs:
            o = o.reshape(-1, o.shape[-1]) if o.ndim > 1 else o.reshape(-1, 1)
            n, k = o.shape
            for s in self.STRIDES:
                if n == (self.size // s) ** 2 * 2:
                    by[(s, {1: "score", 4: "box", 10: "kps"}[k])] = o
        return by

    def detect(self, frame: np.ndarray) -> list[Face]:
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        canvas[:small.shape[0], :small.shape[1]] = small
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        by = self._split(self.net.forward(self.names))
        boxes, scores, kpss = [], [], []
        for s in self.STRIDES:
            sc = by[(s, "score")].reshape(-1)
            keep = np.where(sc >= self.thresh)[0]
            if not len(keep):
                continue
            c = self._anchor(s)[keep]
            d = by[(s, "box")][keep] * s
            k = by[(s, "kps")][keep].reshape(-1, 5, 2) * s + c[:, None, :]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
            kpss.append(k)
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        tlwh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        idx = np.array(cv2.dnn.NMSBoxes(tlwh, scores.tolist(), self.thresh, self.nms), dtype=int).reshape(-1)
        faces = [Face(boxes[i].astype(np.float32), float(scores[i]), kpss[i].astype(np.float32)) for i in idx]
        return sorted(faces, key=lambda f: -f.width)


class MobileFaceNet:
    """InsightFace w600k_mbf: MobileFaceNet trained with ArcFace loss on WebFace600K."""

    def __init__(self, path: str):
        self.net = cv2.dnn.readNetFromONNX(path)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        return l2n(self.net.forward())


# ── Quality (from the board's titik_wajah.py, unchanged thresholds) ───────────

MIN_WIDTH_REL = 0.10        # face width / image width
MIN_DET = 0.5
MAX_YAW = 0.45              # beyond this the face is practically in profile
MIN_SHARP = 6.0
BRIGHT = (35.0, 235.0)
EDGE = 0.01
MAIN_RATIO = 1.3            # the customer's face must be >= 1.3x wider than the next face


def quality(frame: np.ndarray, faces: list[Face]) -> tuple[Optional[Face], Optional[float], str]:
    """(main face, quality 0..1 or None if unusable, reason shown on screen)."""
    h, w = frame.shape[:2]
    if not faces:
        return None, None, "Menunggu wajah"
    f = faces[0]
    if len(faces) > 1 and faces[1].width * MAIN_RATIO > f.width:
        return f, None, "Beberapa wajah sama dekat"
    x1, y1, x2, y2 = f.box
    le, re, nose, lm, rm = f.kps
    eye_d = max(abs(re[0] - le[0]), 1.0)
    yaw = float((nose[0] - (le[0] + re[0]) / 2) / eye_d)
    eye_y, mouth_y = (le[1] + re[1]) / 2, (lm[1] + rm[1]) / 2
    pitch = float((nose[1] - eye_y) / max(mouth_y - eye_y, 1.0))
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    crop = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
    if crop.size == 0:
        return f, None, "Wajah terpotong tepi"
    gray = cv2.cvtColor(cv2.resize(crop, (160, max(1, int(160 * crop.shape[0] / max(crop.shape[1], 1))))),
                        cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    rel = f.width / w
    if f.score < MIN_DET:
        return f, None, "Wajah samar"
    if x1 < EDGE * w or y1 < EDGE * h or x2 > (1 - EDGE) * w or y2 > (1 - EDGE) * h:
        return f, None, "Wajah terpotong tepi"
    if rel < MIN_WIDTH_REL:
        return f, None, "Terlalu jauh"
    if abs(yaw) > MAX_YAW:
        return f, None, "Tampak samping"
    if not BRIGHT[0] <= bright <= BRIGHT[1]:
        return f, None, "Terlalu gelap/terang"
    if sharp < MIN_SHARP:
        return f, None, "Buram"
    facing = (max(0.0, 1 - abs(yaw) / MAX_YAW) * max(0.0, 1 - abs(pitch - 0.53) / 0.35)
              * max(0.0, 1 - abs(roll) / 35))
    q = (0.45 * facing + 0.25 * min(1.0, sharp / 80) + 0.15 * min(1.0, rel / 0.30)
         + 0.15 * (1 - min(1.0, abs(bright - 130) / 100)))
    return f, float(q), ""


# ── Gallery ───────────────────────────────────────────────────────────────────

class Gallery:
    """Enrolled (vault-protected) embeddings; inner product = cosine similarity."""

    def __init__(self):
        self.lock = threading.Lock()
        self.ids: list[str] = []
        self.names: dict[str, str] = {}
        self.vecs = np.zeros((0, EMBED_DIM), np.float32)
        self.index = None
        self.legacy = 0          # customers whose face was enrolled with an older pipeline

    def load(self, rows: list[tuple[str, str, np.ndarray]], legacy: int = 0):
        ids = [r[0] for r in rows]
        vecs = np.stack([l2n(r[2]) for r in rows]) if rows else np.zeros((0, EMBED_DIM), np.float32)
        index = None
        if HAS_FAISS and len(vecs):
            index = faiss.IndexFlatIP(EMBED_DIM)
            index.add(vecs)
        with self.lock:
            self.ids, self.vecs, self.index = ids, vecs, index
            self.names = {r[0]: r[1] for r in rows}
            self.legacy = legacy

    def top2(self, probe: np.ndarray, exclude: Optional[str] = None) -> tuple[Optional[str], float, float]:
        """Best customer, its cosine, and the best cosine of any OTHER customer (for the margin)."""
        with self.lock:
            if not self.ids:
                return None, 0.0, 0.0
            if self.index is not None:
                k = min(len(self.ids), 32)
                d, i = self.index.search(probe.reshape(1, -1), k)
                pairs = [(self.ids[int(j)], float(s)) for s, j in zip(d[0], i[0]) if j >= 0]
            else:
                sims = self.vecs @ probe
                order = np.argsort(-sims)[:32]
                pairs = [(self.ids[int(j)], float(sims[j])) for j in order]
        best: dict[str, float] = {}
        for cid, s in pairs:
            if cid != exclude and s > best.get(cid, -2.0):
                best[cid] = s
        if not best:
            return None, 0.0, 0.0
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][0], ranked[0][1], second

    def name(self, cid: Optional[str]) -> str:
        return self.names.get(cid, "") if cid else ""

    def stats(self) -> dict:
        with self.lock:
            return {"embeddings": len(self.ids), "customers": len(set(self.ids)), "legacy": self.legacy,
                    "backend": "faiss" if self.index is not None else "numpy"}


# ── Session: collect one face for a few frames, decide once ───────────────────

@dataclass
class Sample:
    q: float
    emb: np.ndarray          # vault-protected


class Session:
    def __init__(self):
        self.samples: list[Sample] = []
        self.t0 = 0.0
        self.t_last = 0.0

    def mean(self) -> np.ndarray:
        return l2n(np.mean([s.emb for s in self.samples], axis=0))

    def best(self, k: int) -> list[Sample]:
        """k highest-quality, mutually non-duplicate samples."""
        out: list[Sample] = []
        for s in sorted(self.samples, key=lambda s: -s.q):
            if all(float(s.emb @ o.emb) < 0.985 for o in out):
                out.append(s)
            if len(out) >= k:
                break
        return out


@dataclass
class Decision:
    state: str                     # known | unknown | uncertain
    customer_id: Optional[str]
    score: float
    margin: float
    votes: str
    n: int
    t: float = field(default_factory=time.time)


def decide(cfg: Config, gallery: Gallery, sess: Session) -> Decision:
    best = sess.best(5)
    template = l2n(np.mean([s.emb for s in best[:3]], axis=0))
    cid, s1, s2 = gallery.top2(template)
    votes = 0
    for s in best:
        c, v, _ = gallery.top2(s.emb)
        votes += int(c == cid and v >= cfg.match_threshold)
    need = int(np.ceil(cfg.vote_ratio * len(best)))
    vote_txt = f"{votes}/{len(best)}"
    if cid is not None and s1 >= cfg.match_threshold and s1 - s2 >= cfg.match_margin and votes >= need:
        return Decision("known", cid, s1, s1 - s2, vote_txt, len(sess.samples))
    if cid is None or s1 < cfg.match_threshold - 0.05:
        return Decision("unknown", None, s1, s1 - s2, vote_txt, len(sess.samples))
    return Decision("uncertain", None, s1, s1 - s2, vote_txt, len(sess.samples))


# ── Camera (newest frame only) ────────────────────────────────────────────────

class Camera(threading.Thread):
    def __init__(self, cfg: Config):
        super().__init__(daemon=True, name="camera")
        self.cfg = cfg
        self.source = cfg.camera_source
        self.cap = None
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.ok = False
        self.size = (0, 0)
        self.is_file = False
        self.file_fps = 0.0
        self._switch = threading.Event()

    def switch(self, source: str):
        if source != self.source:
            log.info("[Camera] switching to %s", source or "auto")
            self.source = source
            self._switch.set()

    def _open(self) -> bool:
        src = self.source
        candidates: list = []
        if src == "":
            candidates = [c["source"] for c in list_cameras()] or list(range(5))
        elif src.isdigit():
            candidates = [int(src)]
        else:
            candidates = [src]
        for c in candidates:
            is_url = isinstance(c, str) and "://" in c
            is_file = isinstance(c, str) and not is_url and os.path.isfile(c)   # recorded video (rehearsal)
            if is_url:
                cap = cv2.VideoCapture(c, cv2.CAP_FFMPEG)
            elif is_file:
                cap = cv2.VideoCapture(c)
            elif sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(c, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(c)
            if not cap.isOpened():
                cap.release()
                continue
            self.is_file = is_file
            self.file_fps = (cap.get(cv2.CAP_PROP_FPS) or 15.0) if is_file else 0.0
            if not (is_url or is_file):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, fr = cap.read()
            if ok and fr is not None:
                self.cap, self.ok, self.size = cap, True, (fr.shape[1], fr.shape[0])
                self.opened = str(c)
                log.info("[Camera] %s opened at %dx%d", c, fr.shape[1], fr.shape[0])
                return True
            cap.release()
        return False

    def run(self):
        fails = 0
        while True:
            if self._switch.is_set():
                self._switch.clear()
                if self.cap is not None:
                    self.cap.release()
                self.cap, self.ok = None, False
            if self.cap is None and not self._open():
                self.ok = False
                log.error("[Camera] no webcam (%s), retrying in 3 s", self.source or "auto")
                time.sleep(3)
                continue
            ok, fr = self.cap.read()
            if (not ok or fr is None) and getattr(self, "is_file", False):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # loop the recording
                continue
            if getattr(self, "is_file", False):
                time.sleep(1.0 / max(self.file_fps, 1.0))          # play at its own speed
            if not ok or fr is None:
                fails += 1
                if fails > 30:
                    log.warning("[Camera] lost, reopening")
                    self.cap.release()
                    self.cap, self.ok, fails = None, False, 0
                time.sleep(0.01)
                continue
            fails = 0
            with self.cond:
                self.frame, self.seq = fr, self.seq + 1
                self.cond.notify_all()

    def latest(self, after: int, timeout: float = 1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after, timeout=timeout)
            return self.frame, self.seq


# ── Shared state ──────────────────────────────────────────────────────────────

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.fps = 0.0
        self.det_ms = 0.0
        self.emb_ms = 0.0
        self.last_decision: Optional[Decision] = None
        self.view: dict = {}               # what is on screen right now (for the overlay)
        self.preview: Optional[np.ndarray] = None
        self.enroll: Optional[dict] = None  # active enrollment request
        self.pi_ok: Optional[bool] = None
        self.started = time.time()


# ── Raspberry Pi link (face event -> body at the cashier) ─────────────────────

class PiLink(threading.Thread):
    def __init__(self, cfg: Config, state: State):
        super().__init__(daemon=True, name="pi")
        self.cfg, self.state = cfg, state
        self.q: queue.Queue = queue.Queue(maxsize=20)
        self.last: dict[str, float] = {}

    def send(self, cid: str, score: float, name: str):
        now = time.time()
        if now - self.last.get(cid, 0.0) < self.cfg.pi_cooldown:
            return
        self.last[cid] = now
        try:
            self.q.put_nowait({"customer_id": cid, "score": round(float(score), 4), "label": name[:40],
                               "camera_id": self.cfg.area_camera, "zone_id": "TITIK_KASIR",
                               "face_camera": self.cfg.device_id})
        except queue.Full:
            pass

    def run(self):
        while True:
            body = json.dumps(self.q.get()).encode()
            headers = {"Content-Type": "application/json"}
            if self.cfg.face_token:
                headers["X-Auth-Token"] = self.cfg.face_token
            req = urllib.request.Request(self.cfg.pi_face_url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=2) as r:
                    r.read()
                self.state.pi_ok = True
            except Exception as e:
                self.state.pi_ok = False
                log.warning("[Pi] %s unreachable: %s", self.cfg.pi_face_url, e)


# ── Database thread (all network I/O off the camera loop) ─────────────────────

def parse_vec(v) -> np.ndarray:
    if isinstance(v, str):
        v = json.loads(v) if v.strip().startswith("[") else [float(x) for x in v.split(",")]
    return np.asarray(v, dtype=np.float32)


class Database(threading.Thread):
    def __init__(self, cfg: Config, gallery: Gallery, state: State, camera: Camera, engine: "Engine"):
        super().__init__(daemon=True, name="db")
        self.cfg, self.gallery, self.state, self.camera, self.engine = cfg, gallery, state, camera, engine
        self.jobs: queue.Queue = queue.Queue(maxsize=500)
        self.conn = None
        self.online = False
        self.loaded = threading.Event()
        self.reload_at = 0.0
        self.next = collections.defaultdict(float)
        self.off: set = set()              # optional features whose migration is missing
        self.has_visit_fn = True
        self.sources_seen = None
        self.restart_requested = False

    def submit(self, kind: str, **kw):
        try:
            self.jobs.put_nowait((kind, kw))
        except queue.Full:
            log.warning("[DB] queue full, dropping %s", kind)

    # connection
    def _connect(self):
        import psycopg2
        self.conn = psycopg2.connect(self.cfg.database_url, connect_timeout=10,
                                     keepalives=1, keepalives_idle=30, keepalives_interval=10)
        self.conn.autocommit = True
        self.online = True
        log.info("[DB] connected")

    def _q(self, sql: str, params=(), fetch: str = ""):
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return cur.rowcount

    def _optional(self, name: str, fn):
        if name in self.off:
            return
        try:
            fn()
        except Exception as e:
            msg = str(e)
            if "does not exist" in msg:
                self.off.add(name)
                log.warning("[DB] %s disabled: %s (run the CRM database migrations 004/005)",
                            name, msg.splitlines()[0])
            else:
                raise

    # gallery
    def load_gallery(self):
        rows = self._q("""SELECT e.customer_id::text, c.name, e.embedding_vector::text
                          FROM embeddings e JOIN customers c ON c.customer_id = e.customer_id
                          WHERE c.is_active = TRUE AND e.model_name = %s""", (MODEL_NAME,), "all")
        legacy = self._q("""SELECT COUNT(DISTINCT e.customer_id) FROM embeddings e
                            JOIN customers c ON c.customer_id = e.customer_id
                            WHERE c.is_active = TRUE AND e.model_name <> %s
                              AND NOT EXISTS (SELECT 1 FROM embeddings n WHERE n.customer_id = e.customer_id
                                              AND n.model_name = %s)""", (MODEL_NAME, MODEL_NAME), "one")[0]
        good = []
        for cid, name, vec in rows:
            v = parse_vec(vec)
            if v.shape == (EMBED_DIM,):
                good.append((cid, name or "", v))
        self.gallery.load(good, int(legacy or 0))
        self.loaded.set()
        st = self.gallery.stats()
        log.info("[Gallery] %d customers (%d embeddings, %s)%s", st["customers"], st["embeddings"],
                 st["backend"], f"; {st['legacy']} need re-enrollment (old pipeline)" if st["legacy"] else "")

    # jobs from the camera loop
    def _run_job(self, kind: str, kw: dict):
        if kind == "log":
            self._q("""INSERT INTO recognition_logs (customer_id, similarity_score, model_used, camera_id, recognized)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (kw["customer_id"], round(float(kw["score"]), 6), MODEL_TAG, self.cfg.device_id,
                     kw["customer_id"] is not None))
        elif kind == "visit":
            cid = kw["customer_id"]
            if self.has_visit_fn:
                try:
                    self._q("SELECT get_or_create_visit(%s::uuid, %s, %s, NOW())",
                            (cid, self.cfg.visit_max_open_hours, self.cfg.visit_reopen_min))
                    return
                except Exception as e:
                    if "get_or_create_visit" not in str(e):
                        raise
                    self.has_visit_fn = False
                    log.warning("[DB] get_or_create_visit() missing: run database/005_devices.sql")
            self._q("""INSERT INTO visits (customer_id, entry_time) SELECT %s::uuid, NOW()
                       WHERE NOT EXISTS (SELECT 1 FROM visits WHERE customer_id = %s::uuid AND (
                           (exit_time IS NULL AND entry_time > NOW() - make_interval(hours => %s))
                           OR exit_time > NOW() - make_interval(mins => %s)))""",
                    (cid, cid, self.cfg.visit_max_open_hours, self.cfg.visit_reopen_min))
        elif kind == "enroll_done":
            self._finish_enroll(**kw)

    # enrollment (CRM -> board)
    def _poll_enroll(self):
        if self.state.enroll is not None:
            return
        row = self._q("""UPDATE enrollment_requests SET status = 'RUNNING'
                         WHERE request_id = (SELECT request_id FROM enrollment_requests
                                             WHERE camera_id = %s AND status = 'PENDING'
                                               AND created_at > NOW() - INTERVAL '25 seconds'
                                             ORDER BY created_at LIMIT 1)
                           AND status = 'PENDING'
                         RETURNING request_id::text, customer_id::text""", (self.cfg.device_id,), "one")
        if row:
            log.info("[Enroll] request %s for customer %s", row[0][:8], row[1][:8])
            self.engine.start_enroll(row[0], row[1])

    def _finish_enroll(self, request_id: str, customer_id: str, ok: bool, message: str, embs=None):
        if ok:
            with self.conn.cursor() as cur:
                cur.execute("BEGIN")
                try:
                    cur.execute("DELETE FROM embeddings WHERE customer_id = %s::uuid", (customer_id,))
                    for i, e in enumerate(embs):
                        cur.execute("""INSERT INTO embeddings (customer_id, embedding_vector, model_name, is_primary)
                                       VALUES (%s::uuid, %s, %s, %s)""",
                                    (customer_id, "[" + ",".join(f"{x:.7f}" for x in e) + "]", MODEL_NAME, i == 0))
                    cur.execute("""UPDATE enrollment_requests SET status = 'DONE', message = 'OK', n_frames = %s,
                                   completed_at = NOW() WHERE request_id = %s::uuid""", (len(embs), request_id))
                    cur.execute("COMMIT")
                except Exception:
                    cur.execute("ROLLBACK")
                    raise
            log.info("[Enroll] customer %s enrolled (%d embeddings)", customer_id[:8], len(embs))
            self.reload_at = 0.0
        else:
            self._q("""UPDATE enrollment_requests SET status = 'FAILED', message = %s, completed_at = NOW()
                       WHERE request_id = %s::uuid AND status IN ('PENDING', 'RUNNING')""", (message, request_id))
            log.info("[Enroll] failed: %s", message)

    # device management
    def _heartbeat(self):
        st, cam = self.state, self.camera
        d = st.last_decision
        info = {
            "version": VERSION, "model": MODEL_TAG, "fps": round(st.fps, 1),
            "timing_ms": {"detect": round(st.det_ms, 1), "embed": round(st.emb_ms, 1)},
            "gallery": self.gallery.stats(), "threshold": self.cfg.match_threshold,
            "margin": self.cfg.match_margin,
            "camera": {"ok": cam.ok, "source": getattr(cam, "opened", cam.source) if cam.ok else cam.source,
                       "width": cam.size[0], "height": cam.size[1]},
            "cameras": [{**c, "role": "face"} for c in list_cameras()],
            "roles": ["face"],
            "pi": {"url": self.cfg.pi_face_url, "ok": st.pi_ok},
            "uptime_s": int(time.time() - st.started),
            "last_decision": None if d is None else {"state": d.state, "score": round(d.score, 3),
                                                     "at": utc_iso(d.t)},
        }
        self._q("""INSERT INTO device_status (device_id, kind, last_seen_at, info) VALUES (%s, 'facerec', NOW(), %s::jsonb)
                   ON CONFLICT (device_id) DO UPDATE SET kind = 'facerec', last_seen_at = NOW(), info = EXCLUDED.info""",
                (self.cfg.device_id, json.dumps(info)))

    def _settings(self):
        row = self._q("SELECT value FROM app_settings WHERE key = 'camera_sources'", (), "one")
        src = ""
        if row and isinstance(row[0], dict):
            src = str(((row[0].get(self.cfg.device_id) or {}).get("face") or {}).get("source") or "")
        src = src or self.cfg.camera_source
        if src != self.sources_seen:
            self.sources_seen = src
            self.camera.switch(src)

    def _commands(self):
        row = self._q("""UPDATE device_commands SET status = 'RUNNING', started_at = NOW()
                         WHERE command_id = (SELECT command_id FROM device_commands
                                             WHERE device_id = %s AND status = 'PENDING'
                                               AND created_at > NOW() - INTERVAL '2 minutes'
                                             ORDER BY created_at LIMIT 1)
                         RETURNING command_id::text, command, payload""", (self.cfg.device_id,), "one")
        if not row:
            return
        cid, cmd, _payload = row
        status, message, result = "DONE", "OK", {}
        try:
            if cmd == "self_check":
                result = self.self_check()
                bad = [c["name"] for c in result["checks"] if c["status"] == "fail"]
                message = "Semua pemeriksaan lulus." if not bad else "Ada masalah: " + ", ".join(bad)
            elif cmd == "reload_gallery":
                self.load_gallery()
                message = f"Galeri dimuat ulang ({self.gallery.stats()['customers']} pelanggan)."
            elif cmd == "restart":
                message = "Program dimulai ulang."
                self.restart_requested = True
            else:
                status, message = "FAILED", f"Perintah tidak dikenal: {cmd}"
        except Exception as e:
            status, message = "FAILED", str(e)[:300]
        self._q("""UPDATE device_commands SET status = %s, message = %s, result = %s::jsonb, finished_at = NOW()
                   WHERE command_id = %s::uuid""", (status, message, json.dumps(result), cid))
        if self.restart_requested:
            log.info("[Cmd] restart requested from the CRM")
            time.sleep(0.5)
            os._exit(0)          # systemd (Restart=always) starts the program again

    def self_check(self) -> dict:
        st, cam = self.state, self.camera
        checks = []

        def add(name, ok, detail, hint=""):
            checks.append({"name": name, "status": "ok" if ok is True else ("warn" if ok is None else "fail"),
                           "detail": detail, "hint": hint})

        add("Kamera wajah", cam.ok, f"{getattr(cam, 'opened', cam.source)} {cam.size[0]}x{cam.size[1]}" if cam.ok
            else "tidak ada gambar", "Colok webcam ke UNO Q (lewat hub USB-C), lalu pilih di Perangkat > Kamera.")
        add("Kecepatan", st.fps >= 3 or None, f"{st.fps:.1f} FPS, deteksi {st.det_ms:.0f} ms, ciri {st.emb_ms:.0f} ms",
            "Di bawah 3 FPS: kurangi aplikasi lain di papan.")
        g = self.gallery.stats()
        add("Wajah terdaftar", g["customers"] > 0 or None, f"{g['customers']} pelanggan",
            "Daftarkan wajah dari halaman Pelanggan (dengan persetujuan).")
        if g["legacy"]:
            add("Wajah model lama", None, f"{g['legacy']} pelanggan perlu didaftarkan ulang", "Buka Wajah Terdaftar.")
        add("Database", True, "terhubung")
        if self.cfg.pi_face_url:
            add("Raspberry Pi", st.pi_ok if st.pi_ok is not None else None,
                self.cfg.pi_face_url + (" (belum ada kiriman)" if st.pi_ok is None else ""),
                "Periksa PI_FACE_URL dan FACE_TOKEN di .env UNO Q dan Pi.")
        else:
            add("Raspberry Pi", None, "PI_FACE_URL belum diisi", "Tanpa ini nama tidak ikut pada pelacakan.")
        db_now = self._q("SELECT EXTRACT(EPOCH FROM NOW())", (), "one")[0]
        skew = abs(float(db_now) - time.time())
        add("Jam papan", skew < 5 or None, f"selisih {skew:.1f} detik dari server",
            "Aktifkan sinkron waktu: sudo timedatectl set-ntp true")
        add("Model", True, f"{MODEL_TAG} v{VERSION}, kunci brankas {self.engine.vault.fingerprint}")
        return {"checks": checks, "at": utc_iso(time.time())}

    def _frames(self):
        key = f"{self.cfg.device_id}/face"
        row = self._q("SELECT wanted_until > NOW() FROM device_frames WHERE camera_key = %s", (key,), "one")
        if not row or not row[0]:
            return
        with self.state.lock:
            img = None if self.state.preview is None else self.state.preview.copy()
            view = dict(self.state.view)
        if img is None:
            return
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            return
        self._q("""UPDATE device_frames SET device_id = %s, camera = 'face', jpeg_b64 = %s, width = %s, height = %s,
                   overlay = %s::jsonb, captured_at = NOW() WHERE camera_key = %s""",
                (self.cfg.device_id, base64.b64encode(buf.tobytes()).decode(), img.shape[1], img.shape[0],
                 json.dumps(view), key))

    def run(self):
        backoff, pending = 2.0, None
        while True:
            try:
                if self.conn is None or self.conn.closed:
                    self._connect()
                    backoff = 2.0
                now = time.time()
                if now >= self.reload_at:
                    self.load_gallery()
                    self.reload_at = now + self.cfg.reload_interval
                for name, every, fn in (("enrollment", 1.0, self._poll_enroll),
                                        ("commands", 1.5, self._commands),
                                        ("settings", 5.0, self._settings),
                                        ("frames", 1.0, self._frames),
                                        ("heartbeat", self.cfg.heartbeat_sec, self._heartbeat)):
                    if now >= self.next[name]:
                        self.next[name] = now + every
                        self._optional(name, fn)
                if pending is None:
                    try:
                        pending = self.jobs.get(timeout=0.2)
                    except queue.Empty:
                        continue
                self._run_job(*pending)
                pending = None
            except Exception as e:
                self.online = False
                log.warning("[DB] %s -- retrying in %.0f s (recognition keeps running)", str(e).splitlines()[0], backoff)
                try:
                    if self.conn is not None:
                        self.conn.close()
                except Exception:
                    pass
                self.conn = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    def __init__(self, cfg: Config, state: State):
        self.cfg, self.state = cfg, state
        cv2.setNumThreads(cfg.threads)
        self.det = SCRFD(det_model_path(cfg.det_size), cfg.det_size, cfg.det_thresh)
        self.rec = MobileFaceNet(find_model("w600k_mbf.onnx"))
        self.vault = Vault(load_vault_key(cfg))
        self.gallery = Gallery()
        self.sess = Session()
        self.enroll_sess: Optional[Session] = None
        self.enroll_req: Optional[tuple] = None
        self.enroll_t0 = 0.0
        self.db: Optional[Database] = None
        self.pi: Optional[PiLink] = None
        self.last_known: dict[str, float] = {}
        self.last_unknown = 0.0
        self.message = ""
        self.message_until = 0.0

    def start_enroll(self, request_id: str, customer_id: str):
        # Session first, request last: the camera loop reads enroll_req without a lock.
        self.enroll_sess = Session()
        self.enroll_t0 = time.time()
        self.enroll_req = (request_id, customer_id)
        with self.state.lock:
            self.state.enroll = {"request_id": request_id, "customer_id": customer_id}

    def _end_enroll(self, ok: bool, message: str, embs=None):
        rid, cid = self.enroll_req
        self.enroll_req, self.enroll_sess = None, None
        with self.state.lock:
            self.state.enroll = None
        self.flash(("Wajah tersimpan" if ok else "Gagal: " + message), 3.0)
        if self.db:
            self.db.submit("enroll_done", request_id=rid, customer_id=cid, ok=ok, message=message,
                           embs=[e.tolist() for e in embs] if embs else None)

    def flash(self, text: str, sec: float = 2.5):
        self.message, self.message_until = text, time.time() + sec

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        t = time.perf_counter()
        e = self.vault.protect(self.rec.embed(align(frame, face.kps)))
        self.state.emb_ms = 0.9 * self.state.emb_ms + 0.1 * (time.perf_counter() - t) * 1000
        return e

    def step(self, frame: np.ndarray, now: float):
        t = time.perf_counter()
        faces = self.det.detect(frame)
        self.state.det_ms = 0.9 * self.state.det_ms + 0.1 * (time.perf_counter() - t) * 1000
        face, q, reason = quality(frame, faces)
        if self.enroll_req is not None:
            self._step_enroll(frame, now, faces, face, q, reason)
        else:
            self._step_recognize(frame, now, face, q, reason)
        self._preview(frame, faces, face, q, reason, now)

    def _step_recognize(self, frame, now, face, q, reason):
        s = self.sess
        if q is None:
            if s.samples and now - s.t_last > 1.0:
                self._close_session(now)
            return
        e = self.embed(frame, face)
        if s.samples and float(e @ s.mean()) < 0.40:       # a different person: close the old session
            self._close_session(now)
            s = self.sess
        if not s.samples:
            s.t0 = now
        s.samples.append(Sample(q, e))
        s.t_last = now
        if len(s.samples) >= self.cfg.session_frames or now - s.t0 >= self.cfg.session_max_sec:
            self._close_session(now)

    def _close_session(self, now: float):
        s, self.sess = self.sess, Session()
        if len(s.samples) < self.cfg.session_min_frames or max(x.q for x in s.samples) < 0.30:
            return
        d = decide(self.cfg, self.gallery, s)
        self.state.last_decision = d
        name = self.gallery.name(d.customer_id)
        if d.state == "known":
            self.flash(f"{name or d.customer_id[:8]} ({d.score:.2f})")
            log.info("[KNOWN] %s score=%.3f margin=%.3f votes=%s frames=%d", name, d.score, d.margin, d.votes, d.n)
            if self.pi:
                self.pi.send(d.customer_id, d.score, name)
            if now - self.last_known.get(d.customer_id, 0.0) >= self.cfg.known_log_cooldown:
                self.last_known[d.customer_id] = now
                if self.db:
                    self.db.submit("log", customer_id=d.customer_id, score=d.score)
                    self.db.submit("visit", customer_id=d.customer_id)
        elif d.state == "unknown":
            self.flash("Belum terdaftar")
            if now - self.last_unknown >= self.cfg.unknown_log_cooldown:
                self.last_unknown = now
                log.info("[UNKNOWN] best=%.3f frames=%d", d.score, d.n)
                if self.db:
                    self.db.submit("log", customer_id=None, score=d.score)
        else:
            self.flash("Belum yakin, tahan sebentar")
            log.info("[UNCERTAIN] best=%.3f margin=%.3f votes=%s", d.score, d.margin, d.votes)

    def _step_enroll(self, frame, now, faces, face, q, reason):
        s = self.enroll_sess
        if now - self.enroll_t0 > self.cfg.enroll_timeout:
            msg = ("Ada lebih dari satu wajah di depan kamera." if reason == "Beberapa wajah sama dekat"
                   else "Wajah tidak terlihat jelas di kamera kasir. Minta pelanggan menghadap kamera.")
            return self._end_enroll(False, msg)
        if q is None:
            return
        s.samples.append(Sample(q, self.embed(frame, face)))
        if len(s.samples) < self.cfg.enroll_frames:
            return
        keep = s.best(self.cfg.enroll_keep)
        mean = l2n(np.mean([k.emb for k in keep], axis=0))
        if len(keep) < 3 or min(float(k.emb @ mean) for k in keep) < 0.50:
            return self._end_enroll(False, "Foto wajah tidak konsisten (orang berganti?). Ulangi.")
        rid, cid = self.enroll_req
        other, score, _ = self.gallery.top2(mean, exclude=cid)
        if other is not None and score >= self.cfg.match_threshold:
            return self._end_enroll(False, f"Wajah ini sudah terdaftar sebagai {self.gallery.name(other) or 'pelanggan lain'}.")
        self._end_enroll(True, "OK", [k.emb for k in keep])

    def _preview(self, frame, faces, face, q, reason, now):
        view = {"faces": [], "status": "", "enrolling": self.enroll_req is not None}
        h, w = frame.shape[:2]
        for f in faces[:3]:
            view["faces"].append({"box": [round(float(f.box[0]) / w, 4), round(float(f.box[1]) / h, 4),
                                          round(float(f.box[2]) / w, 4), round(float(f.box[3]) / h, 4)],
                                  "main": f is face})
        label = self.message if now < self.message_until else (
            "Mendaftarkan wajah..." if self.enroll_req else (reason or f"mutu {q:.2f}" if q is not None else reason))
        view["status"] = label
        img = frame.copy()
        for f in faces[:3]:
            x1, y1, x2, y2 = [int(v) for v in f.box]
            col = (60, 200, 60) if f is face and q is not None else (160, 160, 160)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, label.encode("ascii", "replace").decode(), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        with self.state.lock:
            self.state.preview, self.state.view = img, view


# ── Local HTTP (on the board only: preview + health) ──────────────────────────

def http_handler(state: State, engine: Engine, cfg: Config):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/health":
                body = json.dumps({"status": "ok", "device_id": cfg.device_id, "version": VERSION,
                                   "fps": round(state.fps, 1), "gallery": engine.gallery.stats(),
                                   "db_online": bool(engine.db and engine.db.online)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if p in ("/", "/video_feed"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with state.lock:
                            img = state.preview
                        if img is not None:
                            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ok:
                                b = buf.tobytes()
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            self.send_error(404)
    return H


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    log.info("Cafe FaceRec %s on %s -- SCRFD-0.5G + MobileFaceNet + FAISS (OpenCV DNN)", VERSION, socket.gethostname())
    state = State()
    engine = Engine(cfg, state)
    log.info("[Model] ready | tau=%.2f margin=%.2f vote>=%.0f%% | vault %s", cfg.match_threshold,
             cfg.match_margin, cfg.vote_ratio * 100, engine.vault.fingerprint)
    camera = Camera(cfg)
    if cfg.database_url:
        engine.db = Database(cfg, engine.gallery, state, camera, engine)
        engine.db.start()
        if not engine.db.loaded.wait(timeout=20):
            log.warning("[Gallery] not loaded yet (database unreachable?) -- starting anyway")
    else:
        log.warning("DATABASE_URL is empty: offline mode, nobody can be recognised or enrolled")
    if cfg.pi_face_url:
        engine.pi = PiLink(cfg, state)
        engine.pi.start()
    else:
        log.warning("PI_FACE_URL is empty: names will not be linked to the person tracking")
    camera.start()
    server = ThreadingHTTPServer((cfg.local_host, cfg.local_port), http_handler(state, engine, cfg))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    log.info("[HTTP] http://%s:%d/video_feed  /health", cfg.local_host, cfg.local_port)

    seq, n, t_fps = 0, 0, time.time()
    try:
        while True:
            frame, seq = camera.latest(seq)
            if frame is None:
                continue
            now = time.time()
            engine.step(frame, now)
            n += 1
            if now - t_fps >= 2.0:
                state.fps, n, t_fps = n / (now - t_fps), 0, now
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
