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
import csv
import glob
import hashlib
import hmac
import http.client
import json
import logging
import os
import queue
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
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
    # Early decision: after EARLY_FRAMES frames, decide at once if the evidence is already much
    # stronger than the normal rule requires (sequential decision, Wald 1945): every frame votes
    # for the same customer, score >= tau + 0.10 and margin >= 2 x MATCH_MARGIN.
    early_frames: int = _env("EARLY_FRAMES", 4, int)

    enroll_frames: int = _env("ENROLL_FRAMES", 20, int)
    enroll_keep: int = _env("ENROLL_KEEP", 5, int)
    enroll_timeout: float = _env("ENROLL_TIMEOUT_SEC", 20.0, float)

    known_log_cooldown: float = _env("KNOWN_LOG_COOLDOWN", 30.0, float)   # < CRM Kasir window (45 s)
    unknown_log_cooldown: float = _env("UNKNOWN_LOG_COOLDOWN", 10.0, float)
    visit_max_open_hours: int = _env("VISIT_MAX_OPEN_HOURS", 6, int)
    visit_reopen_min: int = _env("VISIT_REOPEN_MIN", 30, int)
    reload_interval: float = _env("RELOAD_INTERVAL", 120.0, float)
    heartbeat_sec: float = _env("HEARTBEAT_SEC", 10.0, float)
    stream_fps: float = _env("STREAM_FPS", 10.0, float)      # live pictures to the CRM while watched
    stream_monthly_mb: float = _env("STREAM_MONTHLY_MB", 300.0, float)  # fallback Realtime MB/month (0 = no cap)
    live_port: int = _env("LIVE_PORT", 8771, int)          # real-time view, 127.0.0.1 only
    live_fps: float = _env("LIVE_FPS", 15.0, float)
    live_width: int = _env("LIVE_WIDTH", 640, int)
    live_tunnel: bool = _env("LIVE_TUNNEL", 1, int) == 1   # cloudflared quick tunnel
    live_public_url: str = _env("LIVE_PUBLIC_URL", "")     # own fixed https address instead of a quick tunnel
    live_bind: str = _env("LIVE_BIND", "0.0.0.0")          # 0.0.0.0 = local recording view on the Wi-Fi too

    pi_face_url: str = _env("PI_FACE_URL", "")               # http://<pi-ip>:8765/wajah, or "auto"
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

# Face width / image width. Recognition: 0.075 (48 px of 640, about 1.6 m with a ~65 deg webcam).
# Measured on the dataset with probes shrunk to that size (noise, MJPEG, landmark error added):
# the person's own score stays >= 0.58 for 95 % of probes while the best other person never
# exceeds 0.38 at any size, so tau = 0.40 keeps its margin. Enrolment stays close (0.10, ~1.2 m):
# the stored templates are the best photos.
MIN_WIDTH_REL = 0.075
ENROLL_MIN_WIDTH_REL = 0.10
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
    geo: Optional[tuple] = None   # (centre x, width) of the face as fractions of the image width


def face_geo(frame: np.ndarray, face: "Face") -> tuple:
    w = frame.shape[1]
    x1, _, x2, _ = [float(v) for v in face.box]
    return ((x1 + x2) / 2.0 / w, max(x2 - x1, 1.0) / w)


def median_geo(samples: list) -> Optional[dict]:
    """Where the recognised face was (sent to the Pi to tell apart two people at the cashier)."""
    g = [x.geo for x in samples if x.geo]
    if not g:
        return None
    return {"cx": round(float(np.median([a for a, _ in g])), 4), "w": round(float(np.median([b for _, b in g])), 4)}


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
        while True:                            # a camera error must never end this thread
            try:
                self._run()
            except Exception as e:
                log.warning("[Camera] error (%s), reopening in 1 s", e)
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap, self.ok = None, False
                time.sleep(1)

    def _run(self):
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
        self.preview_seq = 0
        self.enroll: Optional[dict] = None  # active enrollment request
        self.pi_ok: Optional[bool] = None
        self.started = time.time()


# ── Raspberry Pi link (face event -> body at the cashier) ─────────────────────

class PiLink(threading.Thread):
    def __init__(self, cfg: Config, state: State):
        super().__init__(daemon=True, name="pi")
        self.cfg, self.state = cfg, state
        # Where face events go. Set from .env, then kept up to date from the CRM
        # (Perangkat -> Alamat Raspberry Pi: automatic from the Pi's heartbeat, or manual).
        self.url = "" if cfg.pi_face_url.lower() in ("", "auto") else cfg.pi_face_url
        self.mode = "env" if self.url else "auto"
        self.q: queue.Queue = queue.Queue(maxsize=20)
        self.last: dict[str, float] = {}

    def send(self, cid: str, score: float, name: str, face: Optional[dict] = None):
        if not self.url:
            return
        now = time.time()
        if now - self.last.get(cid, 0.0) < self.cfg.pi_cooldown:
            return
        self.last[cid] = now
        try:
            self.q.put_nowait({"customer_id": cid, "score": round(float(score), 4), "label": name[:40],
                               "camera_id": self.cfg.area_camera, "zone_id": "TITIK_KASIR",
                               "face_camera": self.cfg.device_id, **({"face": face} if face else {})})
        except queue.Full:
            pass

    def run(self):
        while True:
            body = json.dumps(self.q.get()).encode()
            headers = {"Content-Type": "application/json"}
            if self.cfg.face_token:
                headers["X-Auth-Token"] = self.cfg.face_token
            url = self.url
            if not url:
                continue
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=2) as r:
                    r.read()
                self.state.pi_ok = True
            except Exception as e:
                self.state.pi_ok = False
                log.warning("[Pi] %s unreachable: %s", url, e)


# ── Live pictures to the CRM over Supabase Realtime (same as the Pi's streamer.py) ──

class FrameStreamer(threading.Thread):
    IDLE_FPS = 2.0

    def __init__(self, source, fps: float = 8.0, max_width: int = 512, quality: int = 60,
                 monthly_mb: float = 300.0, budget_path: str | None = None):
        """source() -> list of (camera_key, image, overlay, frame_key) for the cameras being watched."""
        super().__init__(daemon=True, name="stream")
        self.source, self.fps, self.max_width, self.quality = source, max(1.0, fps), max_width, quality
        self.monthly_max, self.budget_path = int(monthly_mb * 1e6), budget_path
        self.month, self.used = self._load_budget()
        self.saved_used = self.used
        self.budget_logged = False
        self.last_sent: dict = {}
        self.conf = None
        self.lock = threading.Lock()
        self.conn = None
        self.ok_at = 0.0
        self.fail = 0
        self.pause_until = 0.0
        self.sent: dict = {}

    def configure(self, conf: dict | None):
        conf = conf if conf and conf.get("url") and conf.get("key") and conf.get("topic") else None
        with self.lock:
            if conf != self.conf:
                self.conf = conf
                self._close()
                if conf:
                    log.info("[Stream] live pictures over Realtime (%s)", urlparse(conf["url"]).hostname)

    @staticmethod
    def _this_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _load_budget(self):
        try:
            with open(self.budget_path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("month") == self._this_month():
                return d["month"], int(d.get("bytes", 0))
        except Exception:
            pass
        return self._this_month(), 0

    def _save_budget(self):
        if not self.budget_path or self.used == self.saved_used:
            return
        try:
            os.makedirs(os.path.dirname(self.budget_path) or ".", exist_ok=True)
            tmp = self.budget_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"month": self.month, "bytes": self.used}, f)
            os.replace(tmp, self.budget_path)
            self.saved_used = self.used
        except Exception:
            pass

    def _budget_left(self) -> bool:
        m = self._this_month()
        if m != self.month:                       # new month: the quota starts again
            self.month, self.used, self.budget_logged = m, 0, False
        if self.monthly_max > 0 and self.used >= self.monthly_max:
            if not self.budget_logged:
                log.warning("[Stream] monthly Realtime budget used (%.0f MB); the fallback view uses stored "
                            "frames until next month (STREAM_MONTHLY_MB)", self.used / 1e6)
                self.budget_logged = True
                self._save_budget()
            return False
        return True

    def stats(self) -> dict:
        return {"month": self.month, "mb": round(self.used / 1e6, 1), "max_mb": round(self.monthly_max / 1e6),
                "active": self.active}

    @property
    def active(self) -> bool:
        """True while pictures are reaching Realtime (the stored-frame fallback can slow down)."""
        return time.time() - self.ok_at < 3.0

    def _close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None

    def _post(self, conf: dict, payload: dict):
        u = urlparse(conf["url"])
        if self.conn is None:
            cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
            self.conn = cls(u.netloc, timeout=5)
        body = json.dumps({"messages": [{"topic": conf["topic"], "event": "frame", "payload": payload}]})
        self.conn.request("POST", "/realtime/v1/api/broadcast", body=body, headers={
            "apikey": conf["key"], "Authorization": f"Bearer {conf['key']}", "Content-Type": "application/json"})
        self.used += len(body)
        r = self.conn.getresponse()
        r.read()
        if r.status >= 300:
            raise RuntimeError(f"HTTP {r.status}")

    def run(self):
        period = 1.0 / self.fps
        last_save = time.time()
        while True:
            t0 = time.time()
            with self.lock:
                conf = self.conf
            if t0 - last_save > 30:
                self._save_budget()
                last_save = t0
            if conf and t0 >= self.pause_until and self._budget_left():
                try:
                    items = self.source() or []
                except Exception:
                    items = []
                for key, img, overlay, frame_key in items:
                    if self.sent.get(key) == frame_key:
                        continue
                    busy = bool((overlay or {}).get("people") or (overlay or {}).get("faces"))
                    if not busy and t0 - self.last_sent.get(key, 0.0) < 1.0 / self.IDLE_FPS:
                        continue                  # empty scene: a few pictures a second are enough
                    self.sent[key] = frame_key
                    self.last_sent[key] = t0
                    h, w = img.shape[:2]
                    if w > self.max_width:
                        img = cv2.resize(img, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    if not ok:
                        continue
                    payload = {"k": key, "img": base64.b64encode(buf.tobytes()).decode(), "w": w, "h": h,
                               "o": overlay, "t": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
                    try:
                        self._post(conf, json.loads(json.dumps(payload, default=str)))
                        self.ok_at, self.fail = time.time(), 0
                    except Exception as e:
                        self._close()
                        self.fail += 1
                        if self.fail in (1, 10) or self.fail % 100 == 0:
                            log.warning("[Stream] Realtime send failed (%s); the CRM falls back to stored frames", e)
                        if self.fail >= 3:
                            self.pause_until = time.time() + 10      # back off, then try again
                        break
            time.sleep(max(0.0, period - (time.time() - t0)))


# ── Real-time camera view straight to the CRM browser (same as the Pi's livecast.py) ──
# WebSocket through a cloudflared quick tunnel; no Supabase traffic. See livecast.py for details.

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_IN_FLIGHT = 2          # pictures sent but not yet shown by the browser


class _WS:
    """Minimal server side of RFC 6455 (binary/text out, ack/ping/close in)."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.send_lock = threading.Lock()
        self.closed = threading.Event()
        self.acked = 0
        self.ack_at = time.time()
        self.ack_cond = threading.Condition()

    def send(self, opcode: int, data: bytes):
        n = len(data)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        with self.send_lock:
            self.sock.sendall(head + data)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    def reader(self):
        try:
            while not self.closed.is_set():
                b1, b2 = self._recv_exact(2)
                opcode, masked, n = b1 & 0x0F, b2 & 0x80, b2 & 0x7F
                if n == 126:
                    n = struct.unpack("!H", self._recv_exact(2))[0]
                elif n == 127:
                    n = struct.unpack("!Q", self._recv_exact(8))[0]
                if n > 1 << 16:
                    raise ConnectionError("message too large")
                mask = self._recv_exact(4) if masked else b"\0\0\0\0"
                data = bytes(c ^ mask[i % 4] for i, c in enumerate(self._recv_exact(n)))
                if opcode == 0x8:                      # close
                    break
                if opcode == 0x9:                      # ping
                    self.send(0xA, data)
                elif opcode == 0x1:                    # text: acknowledgements
                    with self.ack_cond:
                        self.acked += data.count(b"a")
                        self.ack_at = time.time()
                        self.ack_cond.notify_all()
        except Exception:
            pass
        finally:
            self.closed.set()
            with self.ack_cond:
                self.ack_cond.notify_all()


class Tunnel(threading.Thread):
    """Keeps a cloudflared quick tunnel to 127.0.0.1:port running; .url is its HTTPS address."""

    URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    def __init__(self, port: int, binary: str = ""):
        super().__init__(daemon=True, name="tunnel")
        self.port, self.binary = port, binary
        self.url: str | None = None
        self.state = "starting"
        self.proc: subprocess.Popen | None = None
        self.found_at = 0.0

    def find(self) -> str | None:
        home = os.path.expanduser("~")
        for c in (self.binary, os.environ.get("CLOUDFLARED", ""), shutil.which("cloudflared") or "",
                  os.path.join(home, "cloudflared"), os.path.join(home, ".local", "bin", "cloudflared"),
                  "/usr/local/bin/cloudflared", "/usr/bin/cloudflared"):
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return None

    def run(self):
        threading.Thread(target=self._watchdog, daemon=True, name="tunnel-check").start()
        backoff, warned = 5, False
        while True:
            exe = self.find()
            if not exe:
                self.state = "cloudflared belum dipasang"
                if not warned:
                    log.warning("[Live] cloudflared not installed: the CRM uses the slower camera view "
                                "(see README, 'Kamera real-time')")
                    warned = True
                time.sleep(60)
                continue
            try:
                self.proc = subprocess.Popen(
                    [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                self.state = "menghubungkan"
                for line in self.proc.stdout:              # also drains the log so it never blocks
                    m = self.URL_RE.search(line)
                    if m and m.group(0) != self.url:
                        self.url, self.state, self.found_at, backoff = m.group(0), "ok", time.time(), 5
                        log.info("[Live] camera view online at %s", self.url)
                self.proc.wait()
            except Exception as e:
                log.warning("[Live] cloudflared failed: %s", e)
            self.url, self.state = None, "terputus, mencoba lagi"
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    def _watchdog(self):
        """A new address needs up to a minute to work; after that it must answer, or restart."""
        fails = 0
        while True:
            time.sleep(60)
            url, proc = self.url, self.proc
            if not url or proc is None or time.time() - self.found_at < 90:
                fails = 0
                continue
            try:
                with urllib.request.urlopen(url + "/ping", timeout=10) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            fails = 0 if ok else fails + 1
            if fails >= 3:
                log.warning("[Live] %s does not answer, restarting cloudflared", url)
                fails = 0
                try:
                    proc.terminate()
                except Exception:
                    pass


class LiveCast:
    def __init__(self, source, port: int = 8770, fps: float = 15.0, max_width: int = 640,
                 tunnel: bool = True, max_clients: int = 6, public_url: str = "", bind: str = "0.0.0.0",
                 lan_ip=None):
        """source(camera) -> (raw_image, frame_number, overlay) | None. Called per frame per viewer."""
        self.source, self.port = source, port
        self.fps, self.max_width, self.max_clients = max(1.0, fps), max_width, max_clients
        self.token = secrets.token_urlsafe(24)
        self.bind, self.lan_ip = bind, lan_ip
        self.public_url = public_url.rstrip("/")   # own fixed address (e.g. a named tunnel): no quick tunnel
        self.tunnel = Tunnel(port) if tunnel and not self.public_url else None
        self.clients = 0
        self.lock = threading.Lock()
        self.cache: dict = {}                  # camera -> (key, jpeg bytes, w, h)
        self.sent = 0
        self.bytes = 0

    # ── public ──────────────────────────────────────────────────────────────
    def start(self):
        try:
            srv = ThreadingHTTPServer((self.bind, self.port), self._handler())
        except OSError as e:
            log.error("[Live] port %d busy (%s): real-time camera view off", self.port, e)
            return self
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True, name="live-http").start()
        if self.tunnel:
            self.tunnel.start()
        return self

    def info(self) -> dict:
        """For the heartbeat: the CRM gives url + token to logged-in users only."""
        url = self.public_url or (self.tunnel.url if self.tunnel else None)
        state = "ok" if self.public_url else (self.tunnel.state if self.tunnel else "off")
        ip = self.lan_ip() if (self.lan_ip and self.bind != "127.0.0.1") else ""
        lan = f"http://{ip}:{self.port}" if ip else None
        return {"url": url, "lan": lan, "token": self.token if (url or lan) else None, "state": state,
                "viewers": self.clients, "fps_max": self.fps, "sent": self.sent, "mb": round(self.bytes / 1e6, 1)}

    # ── internals ───────────────────────────────────────────────────────────
    def _jpeg(self, camera: str, img, seq, width: int, quality: int):
        key = (seq, width, quality)
        with self.lock:
            hit = self.cache.get(camera)
            if hit and hit[0] == key:
                return hit[1], hit[2], hit[3]
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None, 0, 0
        out = (buf.tobytes(), img.shape[1], img.shape[0])
        with self.lock:
            self.cache[camera] = (key, *out)
        return out

    def _serve(self, ws: _WS, camera: str, fps: float, width: int, quality: int):
        period, last_seq, last_sent, sent, idle_ping = 1.0 / fps, None, 0.0, 0, time.time()
        while not ws.closed.is_set():
            with ws.ack_cond:                  # at most MAX_IN_FLIGHT pictures on the way
                if sent - ws.acked >= MAX_IN_FLIGHT:
                    ws.ack_cond.wait(timeout=0.5)
                    if sent - ws.acked >= MAX_IN_FLIGHT:
                        if time.time() - ws.ack_at > 10:
                            break               # the browser stopped answering
                        continue
            wait = period - (time.time() - last_sent)
            if wait > 0:
                time.sleep(wait)
            try:
                got = self.source(camera)
            except Exception:
                got = None
            if got is None or got[1] == last_seq:
                if time.time() - idle_ping > 15:               # keep proxies from closing the link
                    ws.send(0x9, b"")
                    idle_ping = time.time()
                time.sleep(0.01)
                continue
            img, seq, overlay = got
            jpg, w, h = self._jpeg(camera, img, seq, width, quality)
            if jpg is None:
                continue
            head = json.dumps({"o": overlay, "w": w, "h": h, "s": seq, "t": int(time.time() * 1000)},
                              default=str, separators=(",", ":")).encode()
            ws.send(0x2, struct.pack("!I", len(head)) + head + jpg)
            last_seq, last_sent, idle_ping = seq, time.time(), time.time()
            sent += 1
            self.sent += 1
            self.bytes += len(jpg)

    def _handler(self):
        cast = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _plain(self, code: int, text: str):
                b = text.encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/ping":
                    return self._plain(200, "ok")
                if u.path in ("/", "/lihat"):
                    b = VIEWER_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(b)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(b)
                    return
                if u.path != "/live":
                    return self._plain(404, "not found")
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if not hmac.compare_digest(q.get("token", ""), cast.token):
                    return self._plain(403, "forbidden")
                key = self.headers.get("Sec-WebSocket-Key", "")
                if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
                    return self._plain(400, "websocket required")
                if cast.clients >= cast.max_clients:
                    return self._plain(503, "too many viewers")
                camera = q.get("cam", "")
                try:
                    fps = min(cast.fps, max(1.0, float(q.get("fps", cast.fps))))
                    width = min(cast.max_width, max(240, int(q.get("w", cast.max_width))))
                    quality = min(85, max(30, int(q.get("q", 60))))
                except ValueError:
                    return self._plain(400, "bad parameters")
                accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                sock = self.connection
                sock.settimeout(None)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                ws = _WS(sock)
                with cast.lock:
                    cast.clients += 1
                log.info("[Live] viewer connected (%s, %.0f fps, %d px)", camera, fps, width)
                threading.Thread(target=ws.reader, daemon=True, name="live-read").start()
                try:
                    cast._serve(ws, camera, fps, width, quality)
                except Exception:
                    pass
                finally:
                    ws.closed.set()
                    with cast.lock:
                        cast.clients -= 1
                    try:
                        ws.send(0x8, b"")
                    except Exception:
                        pass
                    self.close_connection = True
                    log.info("[Live] viewer left (%s)", camera)

        return H


VIEWER_HTML = r"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kamera</title>
<style>
html,body{margin:0;height:100%;background:#000;overflow:hidden;font-family:system-ui,Segoe UI,Arial,sans-serif}
canvas{display:block;width:100vw;height:100vh;cursor:pointer}
</style></head><body><canvas id="c" title="Klik dua kali: penuhi jendela / tampil utuh"></canvas>
<script>
// Local camera view for recording: straight from the device over the cafe Wi-Fi (no internet data).
// Light by default (480 px, quality 55, 12 fps) so a weak Wi-Fi link or a busy device still keeps up.
// Everything drawn (labels, bars) scales with the window, so small side-by-side windows stay readable.
const q = new URLSearchParams(location.search);
const cam = q.get("cam") || "", token = q.get("token") || "", title = q.get("name") || cam;
const fps = q.get("fps") || "12", width = q.get("w") || "480", quality = q.get("q") || "55";
document.title = title || "Kamera";
const cv = document.getElementById("c"), ctx = cv.getContext("2d");
let since = performance.now(), shown = [], got = [], status = "Menghubungkan...", lastHead = null, lastImg = null;
let cover = localStorage.getItem("lihat-cover") === "1";     // fill the window (edges cut) or show all
const pad = (n, k = 2) => String(n).padStart(k, "0");
const clock = (ms) => { const d = new Date(ms); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`; };
function fit() { cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio; if (lastImg) draw(lastImg, lastHead); else message(status); }
addEventListener("resize", fit);
cv.addEventListener("dblclick", () => { cover = !cover; localStorage.setItem("lihat-cover", cover ? "1" : "0"); fit(); });
function label(x, y, text, bg, u) {
  ctx.font = `bold ${Math.round(u)}px system-ui,Arial`;
  const w = ctx.measureText(text).width + u * 0.6, h = u * 1.4;
  y = Math.max(y, h);
  ctx.fillStyle = bg; ctx.fillRect(x, y - h, w, h);
  ctx.fillStyle = "#fff"; ctx.fillText(text, x + u * 0.3, y - u * 0.35);
}
function draw(img, head) {
  lastImg = img; lastHead = head;
  const W = cv.width, H = cv.height;
  const s = cover ? Math.max(W / img.width, H / img.height) : Math.min(W / img.width, H / img.height);
  const iw = img.width * s, ih = img.height * s, ox = (W - iw) / 2, oy = (H - ih) / 2;
  // visible part of the picture (differs from the picture when it is cut to fill the window)
  const vx = Math.max(ox, 0), vy = Math.max(oy, 0), vw = Math.min(iw, W), vh = Math.min(ih, H);
  const u = Math.max(9 * devicePixelRatio, Math.min(16 * devicePixelRatio, vw / 42, vh / 22));   // text size
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  ctx.drawImage(img, ox, oy, iw, ih);
  const X = (f) => ox + f * iw, Y = (f) => oy + f * ih, o = (head && head.o) || {}, L = o.layout || {};
  const lw = Math.max(1, u / 7);
  ctx.lineWidth = lw;
  const rect = (r, stroke, fill) => { ctx.fillStyle = fill; ctx.strokeStyle = stroke;
    ctx.fillRect(X(r.x1), Y(r.y1), X(r.x2) - X(r.x1), Y(r.y2) - Y(r.y1)); ctx.strokeRect(X(r.x1), Y(r.y1), X(r.x2) - X(r.x1), Y(r.y2) - Y(r.y1)); };
  (L.staff || []).forEach((r) => rect(r, "#f59e0b", "rgba(245,158,11,.18)"));
  (L.exit || []).forEach((r) => rect(r, "#0ea5e9", "rgba(14,165,233,.18)"));
  (L.transition || []).forEach((r) => rect(r, "#a3e635", "rgba(163,230,53,.12)"));
  if (L.cashier) {
    ctx.strokeStyle = "#22c55e"; ctx.fillStyle = "rgba(34,197,94,.15)";
    ctx.beginPath(); ctx.arc(X(L.cashier.x), Y(L.cashier.y), L.cashier.r * iw, 0, 7); ctx.fill(); ctx.stroke();
  }
  if (L.facecam) {
    const k = u * 0.4;
    ctx.fillStyle = "#8b5cf6"; ctx.fillRect(X(L.facecam.x) - k, Y(L.facecam.y) - k, 2 * k, 2 * k);
    if (L.cashier) { ctx.strokeStyle = "#8b5cf6"; ctx.setLineDash([u * 0.5, u * 0.4]); ctx.beginPath(); ctx.moveTo(X(L.facecam.x), Y(L.facecam.y)); ctx.lineTo(X(L.cashier.x), Y(L.cashier.y)); ctx.stroke(); ctx.setLineDash([]); }
  }
  for (const p of o.people || []) {
    const [x1, y1, x2, y2] = p.box, col = p.staff ? "#f59e0b" : p.customer_id ? "#22c55e" : "#3b82f6";
    ctx.strokeStyle = col; ctx.lineWidth = lw * 1.5; ctx.strokeRect(X(x1), Y(y1), X(x2) - X(x1), Y(y2) - Y(y1));
    const id = String(p.id || "").split(":").pop();
    const who = p.staff ? "Staf" : p.customer_id ? (p.name || "Pelanggan") + (p.mode === "MANUAL" ? " (manual)" : "") : "?";
    label(X(x1), Y(y1), `${who} · ${id}`, col, u);
  }
  for (const f of o.faces || []) {
    const [x1, y1, x2, y2] = f.box; ctx.strokeStyle = f.main ? "#22c55e" : "#9ca3af"; ctx.lineWidth = lw * 1.5;
    ctx.strokeRect(X(x1), Y(y1), X(x2) - X(x1), Y(y2) - Y(y1));
  }
  const bar = u * 1.7;
  if (o.status !== undefined && o.status !== "") {
    ctx.font = `${Math.round(u * 1.05)}px system-ui,Arial`;
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(vx, vy, vw, bar);
    ctx.fillStyle = "#fff"; ctx.fillText(o.status, vx + u * 0.6, vy + bar - u * 0.5, vw - u * 1.2);
  }
  const now = performance.now(); shown.push(now); while (shown.length && now - shown[0] > 2000) shown.shift();
  while (got.length && now - got[0][0] > 2000) got.shift();
  const span = Math.max(1000, Math.min(2000, now - since));          // ms counted (shorter just after connecting)
  const rate = (shown.length * 1000 / span).toFixed(1);
  const kbs = Math.round(got.reduce((a, g) => a + g[1], 0) * 1000 / span / 1024);
  ctx.font = `bold ${Math.round(u)}px Consolas,monospace`;
  const t = head ? clock(head.t) : "-";
  let info = `${title} · ${rate} fps · ${kbs} KB/s · ${t}`;
  if (ctx.measureText(info).width + u > vw) info = `${rate} fps · ${kbs} KB/s · ${t}`;
  if (ctx.measureText(info).width + u > vw) info = `${rate} fps · ${t}`;
  const bw = Math.min(vw, ctx.measureText(info).width + u);
  ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(vx, vy + vh - bar, bw, bar);
  ctx.fillStyle = "#fff"; ctx.fillText(info, vx + u * 0.5, vy + vh - u * 0.5);
}
function message(text) {
  status = text; if (lastImg) return;
  const u = Math.max(10 * devicePixelRatio, Math.min(18 * devicePixelRatio, cv.width / 30));
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#fff"; ctx.font = `${Math.round(u)}px system-ui,Arial`; ctx.fillText(`${title}: ${text}`, u, u * 2, cv.width - 2 * u);
}
fit();
function connect() {
  if (!token) { message("Token tidak ada. Buka dari Perangkat."); return; }
  const ws = new WebSocket(`ws://${location.host}/live?cam=${encodeURIComponent(cam)}&token=${encodeURIComponent(token)}&fps=${fps}&w=${width}&q=${quality}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { since = performance.now(); shown = []; got = []; message("Menunggu gambar..."); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") return;
    got.push([performance.now(), ev.data.byteLength]);
    try {
      const dv = new DataView(ev.data), n = dv.getUint32(0);
      const head = JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data, 4, n)));
      const bmp = await createImageBitmap(new Blob([new Uint8Array(ev.data, 4 + n)], { type: "image/jpeg" }));
      if (lastImg && lastImg.close) lastImg.close();
      draw(bmp, head);
    } catch (e) { /* skip a broken picture */ }
    if (ws.readyState === 1) ws.send("a");          // shown: the device may send the next one
  };
  ws.onclose = (e) => {
    const why = e.code === 1006 && !lastImg ? "Tidak tersambung. Buka lagi dari Perangkat." : "Terputus, menyambung lagi...";
    lastImg = null; message(why); setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}
connect();
</script></body></html>
"""


# ── Test-day logs (data/decisions.csv, data/metrics.csv): evidence without writing by hand ──

class CsvLog:
    def __init__(self, path, header: list):
        self.path, self.header, self.lock = str(path), header, threading.Lock()

    def write(self, row: list):
        try:
            with self.lock:
                new = not os.path.exists(self.path)
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(self.header)
                    w.writerow([datetime.now().astimezone().isoformat(timespec="milliseconds")] + row)
        except Exception:
            pass                                    # logging must never stop recognition


def lan_ip() -> str:
    """This board's address on the current Wi-Fi (asks the routing table, sends nothing)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return ""


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 1)
    except Exception:
        return ""


DECISIONS = CsvLog(HERE / "data" / "decisions.csv",
                   ["waktu", "hasil", "pelanggan", "nama", "skor", "margin", "suara", "frame", "keputusan_dini",
                    "lama_sesi_s", "wajah_cx", "wajah_lebar"])
METRICS = CsvLog(HERE / "data" / "metrics.csv", ["waktu", "fps", "deteksi_ms", "embedding_ms", "suhu_c", "penonton_live"])


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
            # "at" = when the face was recognised, so a write delayed by a network outage keeps its time
            self._q("""INSERT INTO recognition_logs (customer_id, similarity_score, model_used, camera_id, recognized,
                                                     created_at)
                       VALUES (%s, %s, %s, %s, %s, LEAST(COALESCE(%s::timestamptz, NOW()), NOW()))""",
                    (kw["customer_id"], round(float(kw["score"]), 6), MODEL_TAG, self.cfg.device_id,
                     kw["customer_id"] is not None, kw.get("at")))
        elif kind == "visit":
            cid = kw["customer_id"]
            if self.has_visit_fn:
                try:
                    self._q("SELECT get_or_create_visit(%s::uuid, %s, %s, LEAST(COALESCE(%s::timestamptz, NOW()), NOW()))",
                            (cid, self.cfg.visit_max_open_hours, self.cfg.visit_reopen_min, kw.get("at")))
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
            "pi": {"url": self.engine.pi.url if self.engine.pi else "", "mode": self.engine.pi.mode if self.engine.pi else "",
                   "ok": st.pi_ok},
            "uptime_s": int(time.time() - st.started),
            "live": self.engine.live.info() if getattr(self.engine, "live", None) else None,
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
        vs = self._q("SELECT value FROM app_settings WHERE key = 'visits'", (), "one")
        if vs and isinstance(vs[0], dict):                 # CRM -> Pengaturan -> Lantai & kunjungan
            try:
                if vs[0].get("reopen_minutes"):
                    self.cfg.visit_reopen_min = int(vs[0]["reopen_minutes"])
                if vs[0].get("max_open_hours"):
                    self.cfg.visit_max_open_hours = int(vs[0]["max_open_hours"])
            except (TypeError, ValueError):
                pass
        st = self._q("SELECT value FROM app_settings WHERE key = 'stream'", (), "one")
        if self.engine.streamer is not None:
            self.engine.streamer.configure(st[0] if st and isinstance(st[0], dict) else None)
        src = src or self.cfg.camera_source
        if src != self.sources_seen:
            self.sources_seen = src
            self.camera.switch(src)
        self._pi_address()

    def _pi_address(self):
        """
        Where to send face events. CRM setting `network` (Perangkat -> Alamat Raspberry Pi):
          auto   : the address the Pi itself reports in its heartbeat (follows Wi-Fi/IP changes)
          manual : the URL typed in the CRM
          env    : PI_FACE_URL from .env (default when nothing is set in the CRM)
        """
        pi = self.engine.pi
        if pi is None:
            return
        row = self._q("SELECT value FROM app_settings WHERE key = 'network'", (), "one")
        net = row[0] if row and isinstance(row[0], dict) else {}
        env_url = "" if self.cfg.pi_face_url.lower() in ("", "auto") else self.cfg.pi_face_url
        mode = net.get("pi_face_mode") or ("env" if env_url else "auto")
        url = pi.url
        if mode == "manual" and str(net.get("pi_face_url") or "").startswith("http"):
            url = str(net["pi_face_url"]).strip()
        elif mode == "env" and env_url:
            url = env_url
        elif mode == "auto":
            r = self._q("""SELECT info->'network'->>'lan_ip', info->'network'->>'face_port' FROM device_status
                           WHERE kind = 'persontrack' AND last_seen_at > NOW() - INTERVAL '60 seconds'
                           ORDER BY last_seen_at DESC LIMIT 1""", (), "one")
            if r and r[0]:
                url = f"http://{r[0]}:{r[1] or 8765}/wajah"
        if url != pi.url or mode != pi.mode:
            log.info("[Pi] face events -> %s (%s)", url or "-", mode)
            pi.url, pi.mode = url, mode
            self.state.pi_ok = None

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
        pi_url = self.engine.pi.url if self.engine.pi else ""
        if pi_url:
            add("Raspberry Pi", st.pi_ok if st.pi_ok is not None else None,
                pi_url + (" (belum ada kiriman)" if st.pi_ok is None else ""),
                "Periksa Alamat Raspberry Pi (Perangkat) dan FACE_TOKEN yang sama di UNO Q dan Pi.")
        else:
            add("Raspberry Pi", None, "alamat belum diketahui",
                "Nyalakan Raspberry Pi; atau isi Alamat Raspberry Pi di halaman Perangkat.")
        db_now = self._q("SELECT EXTRACT(EPOCH FROM NOW())", (), "one")[0]
        skew = abs(float(db_now) - time.time())
        add("Jam papan", skew < 5 or None, f"selisih {skew:.1f} detik dari server",
            "Aktifkan sinkron waktu: sudo timedatectl set-ntp true")
        add("Model", True, f"{MODEL_TAG} v{VERSION}, kunci brankas {self.engine.vault.fingerprint}")
        return {"checks": checks, "at": utc_iso(time.time())}

    def _frames(self):
        key = f"{self.cfg.device_id}/face"
        now = time.time()
        if now >= self.next["watch_check"]:            # is a CRM page watching? (asked once a second)
            self.next["watch_check"] = now + 1.0
            row = self._q("SELECT GREATEST(0, EXTRACT(EPOCH FROM wanted_until - NOW())) FROM device_frames "
                          "WHERE camera_key = %s", (key,), "one")
            self.watch_until = now + float(row[0]) if row and row[0] is not None else 0.0
        if now >= getattr(self, "watch_until", 0.0):
            return
        if self.engine.streamer is not None and self.engine.streamer.active:
            if now - getattr(self, "db_frame_at", 0.0) < 1.0:
                return                            # live pictures go over Realtime; keep a 1/s fallback
        self.db_frame_at = now
        with self.state.lock:
            img = None if self.state.preview is None else self.state.preview.copy()
            view = dict(self.state.view)
        if img is None:
            return
        if img.shape[1] > 480:                         # preview only: small frames upload fast on a hotspot
            img = cv2.resize(img, (480, int(img.shape[0] * 480 / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
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
                                        ("frames", 0.25, self._frames),
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
        self.streamer: Optional[FrameStreamer] = None
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
        DECISIONS.write(["daftar_ok" if ok else "daftar_gagal", cid, "", "", "", "", len(embs or []), 0, "", "", message])
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
        s.samples.append(Sample(q, e, face_geo(frame, face)))
        s.t_last = now
        if 0 < self.cfg.early_frames <= len(s.samples) < self.cfg.session_frames and self._strong(s):
            self._close_session(now, early=True)
            return
        if len(s.samples) >= self.cfg.session_frames or now - s.t0 >= self.cfg.session_max_sec:
            self._close_session(now)

    def _strong(self, s: Session) -> bool:
        d = decide(self.cfg, self.gallery, s)
        if d.state != "known":
            return False
        agree, total = (int(x) for x in d.votes.split("/"))
        # at least 3 DISTINCT frames (a perfectly still face gives near-identical frames, which best() merges)
        return (total >= 3 and agree == total and d.score >= self.cfg.match_threshold + 0.10
                and d.margin >= 2 * self.cfg.match_margin)

    def _close_session(self, now: float, early: bool = False):
        s, self.sess = self.sess, Session()
        if len(s.samples) < self.cfg.session_min_frames or max(x.q for x in s.samples) < 0.30:
            return
        d = decide(self.cfg, self.gallery, s)
        self.state.last_decision = d
        name = self.gallery.name(d.customer_id)
        geo = median_geo(s.samples) or {}
        DECISIONS.write([d.state, d.customer_id or "", name, round(d.score, 4), round(d.margin, 4), d.votes, d.n,
                         int(early), round(now - s.t0, 2), geo.get("cx", ""), geo.get("w", "")])
        if d.state == "known":
            self.flash(f"{name or d.customer_id[:8]} ({d.score:.2f})")
            log.info("[KNOWN] %s score=%.3f margin=%.3f votes=%s frames=%d", name, d.score, d.margin, d.votes, d.n)
            if self.pi:
                self.pi.send(d.customer_id, d.score, name, median_geo(s.samples))
            if now - self.last_known.get(d.customer_id, 0.0) >= self.cfg.known_log_cooldown:
                self.last_known[d.customer_id] = now
                if self.db:
                    self.db.submit("log", customer_id=d.customer_id, score=d.score, at=utc_iso(now))
                    self.db.submit("visit", customer_id=d.customer_id, at=utc_iso(now))
        elif d.state == "unknown":
            self.flash("Belum terdaftar")
            if now - self.last_unknown >= self.cfg.unknown_log_cooldown:
                self.last_unknown = now
                log.info("[UNKNOWN] best=%.3f frames=%d", d.score, d.n)
                if self.db:
                    self.db.submit("log", customer_id=None, score=d.score, at=utc_iso(now))
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
        if face_geo(frame, face)[1] < ENROLL_MIN_WIDTH_REL:
            self.flash("Mendekat sedikit ke kamera", 1.0)
            return
        s.samples.append(Sample(q, self.embed(frame, face), face_geo(frame, face)))
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
        if self.pi:
            # The new customer is standing in front of the camera right now: tell the Pi at once,
            # so the name is attached to their body without waiting for the next recognition.
            self.pi.send(cid, 1.0, "", median_geo(s.samples[-10:]))

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
            self.state.preview_seq += 1


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
    engine.pi = PiLink(cfg, state)       # address may also come from the CRM / the Pi's heartbeat
    engine.pi.start()

    def stream_source():
        db = engine.db
        if db is None or time.time() >= getattr(db, "watch_until", 0.0):
            return []
        with state.lock:
            img, view, seq = state.preview, dict(state.view), state.preview_seq
        return [] if img is None else [(f"{cfg.device_id}/face", img, view, seq)]
    engine.streamer = FrameStreamer(stream_source, fps=cfg.stream_fps, max_width=480, quality=60,
                                    monthly_mb=cfg.stream_monthly_mb,
                                    budget_path=str(HERE / "data" / "stream_budget.json"))
    engine.streamer.start()

    def live_source(_camera: str):
        """Real-time view: newest RAW camera frame + the newest face boxes (drawn by the browser)."""
        with camera.cond:
            fr, fseq = camera.frame, camera.seq
        if fr is None:
            return None
        with state.lock:
            view = dict(state.view)
        view["cam"] = "face"
        return fr, fseq, view
    engine.live = LiveCast(live_source, port=cfg.live_port, fps=cfg.live_fps, max_width=cfg.live_width,
                           tunnel=cfg.live_tunnel, public_url=cfg.live_public_url, bind=cfg.live_bind,
                           lan_ip=lan_ip).start()
    camera.start()
    server = ThreadingHTTPServer((cfg.local_host, cfg.local_port), http_handler(state, engine, cfg))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    log.info("[HTTP] http://%s:%d/video_feed  /health", cfg.local_host, cfg.local_port)

    seq, n, t_fps, errors = 0, 0, time.time(), 0
    t_metrics = time.time()
    try:
        while True:
            frame, new_seq = camera.latest(seq)
            if frame is None or new_seq == seq:   # no new picture (camera reconnecting)
                continue
            seq = new_seq
            now = time.time()
            try:
                engine.step(frame, now)
            except Exception:                  # one bad frame must not stop the cashier camera
                errors += 1
                if errors <= 5 or errors % 100 == 0:
                    log.exception("[Engine] frame failed (%d so far), continuing", errors)
                time.sleep(0.05)
                continue
            n += 1
            if now - t_fps >= 2.0:
                state.fps, n, t_fps = n / (now - t_fps), 0, now
            if now - t_metrics >= 60:
                t_metrics = now
                METRICS.write([round(state.fps, 1), round(state.det_ms, 1), round(state.emb_ms, 1), cpu_temp(),
                               engine.live.clients if getattr(engine, "live", None) else 0])
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
