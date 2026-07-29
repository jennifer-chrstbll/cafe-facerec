"""
models.py
---------
Unified wrapper for all 5 face recognition models used in the benchmark.
Each model exposes:
    get_embedding(bgr_image) -> np.ndarray (512-d, L2-normalised float32)

Detector (shared, not in this file):
    SCRFD-10GF — buffalo_l/det_10g.onnx via InsightFace FaceAnalysis.
    "SCRFD" = Sample and Computation Redistribution for Efficient Face Detection
    (Guo et al. 2021). This is what InsightFace buffalo_l actually uses —
    NOT RetinaFace, despite historical comments in older versions of this file.

Recognition models (benchmark — all 5 kept for evaluate.py / Bab IV):
    1. ArcFace      — R50,  WebFace600K,  ArcFace loss     (buffalo_l/w600k_r50.onnx)
    2. FaceNet-VGG  — InceptionResnetV1,  VGGFace2, Triplet (facenet-pytorch)
    3. FaceNet-CASIA— InceptionResnetV1,  CASIA-WebFace, Triplet (facenet-pytorch)
    4. AdaFace      — R100, Glint360K,    adaptive margin   (antelopev2/glintr100.onnx)
    5. MagFace      — MobileFaceNet,      WebFace600K       (buffalo_sc/w600k_mbf.onnx)

Production model (Fase 1+):
    ArcFace (Model 1) — best EER (1.11%) in benchmark, used by FaceRecognitionModule.

Benchmark narrative (Bab IV):
    "Does dataset size matter more than loss function? Does depth beat speed?"
"""

import numpy as np
import cv2
import os
import zipfile
import urllib.request
import warnings
warnings.filterwarnings("ignore")

INSIGHTFACE_ROOT = os.path.join(os.path.expanduser("~"), ".insightface", "models")

PACK_URLS = {
    "buffalo_l":  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "buffalo_sc": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
    "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-10)

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def prep_112(bgr_img: np.ndarray) -> np.ndarray:
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Empty image passed to prep_112")
    return cv2.resize(bgr_img, (112, 112))

def _find_onnx(pack_name: str, filename: str) -> str:
    """Find filename anywhere under pack folder — handles nested zip extraction."""
    base = os.path.join(INSIGHTFACE_ROOT, pack_name)
    for root, dirs, files in os.walk(base):
        if filename in files:
            return os.path.join(root, filename)
    all_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            all_files.append(os.path.relpath(os.path.join(root, f), INSIGHTFACE_ROOT))
    raise FileNotFoundError(
        f"'{filename}' not found under {base}\nAll files: {all_files}"
    )

def _download_pack(pack_name: str):
    pack_dir = os.path.join(INSIGHTFACE_ROOT, pack_name)
    zip_path = pack_dir + ".zip"
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    if os.path.isdir(pack_dir) and os.listdir(pack_dir):
        print(f"  [OK] {pack_name} already downloaded.")
        return
    url = PACK_URLS[pack_name]
    print(f"  [DL] Downloading {pack_name} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  [DL] Extracting ...")
    os.makedirs(pack_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        # Flatten extraction — always puts files directly in pack_dir
        for member in z.namelist():
            fname = os.path.basename(member)
            if not fname:
                continue
            with z.open(member) as src, open(os.path.join(pack_dir, fname), "wb") as dst:
                dst.write(src.read())
    os.remove(zip_path)
    print(f"  [DL] Done → {pack_dir}")

def _load_onnx(pack_name: str, filename: str):
    """Load an InsightFace ONNX model via model_zoo (for SCRFD detector + non-ArcFace models)."""
    _download_pack(pack_name)
    onnx_path = _find_onnx(pack_name, filename)
    print(f"  [MODEL] {onnx_path}")
    from insightface.model_zoo import get_model
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    model = get_model(onnx_path, session_options=so, providers=["CPUExecutionProvider"])
    model.prepare(-1)
    return model


class _DirectArcFaceONNX:
    """
    Direct ONNXRuntime wrapper for ArcFace (buffalo_l / w600k_r50.onnx).

    WHY this exists:
    InsightFace's ArcFaceONNX.__init__ calls onnx.load(model_file) to parse the
    174 MB protobuf BEFORE creating the session. When the SCRFD detector is
    already loaded in the same process, the protobuf arena allocator fails with
    "bad allocation". This class bypasses onnx.load() entirely — it creates the
    ORT InferenceSession directly with enable_mem_pattern=False, which works.

    Produces identical embeddings to ArcFaceONNX (same preprocessing: BGR→RGB,
    normalize to [-1, 1], NCHW blob, run session, return (1, 512) float32).
    """

    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        self._session    = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        # buffalo_l w600k_r50.onnx: standard InsightFace normalization
        self._input_mean = 127.5
        self._input_std  = 127.5

    def get_feat(self, bgr_img_112: np.ndarray) -> np.ndarray:
        """Run ArcFace on a 112×112 BGR image. Returns (1, 512) float32."""
        rgb  = cv2.cvtColor(bgr_img_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - self._input_mean) / self._input_std
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]   # → (1, 3, 112, 112)
        return self._session.run(None, {self._input_name: blob})[0]

def _load_facenet(pretrained: str):
    from facenet_pytorch import InceptionResnetV1
    print(f"  [MODEL] FaceNet pretrained={pretrained}")
    return InceptionResnetV1(pretrained=pretrained).eval()

def _facenet_embed(model, bgr_img: np.ndarray) -> np.ndarray:
    import torch
    img    = prep_112(bgr_img)
    arr    = bgr_to_rgb(img).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor).squeeze().numpy()
    return l2_normalize(emb).astype(np.float32)


# ── Model 1: ArcFace — R50, WebFace600K ──────────────────────────────────────

class ArcFaceModel:
    """
    buffalo_l/w600k_r50.onnx — R50, WebFace600K, ArcFace loss.
    Uses _DirectArcFaceONNX to bypass InsightFace's onnx.load() parse
    so it can co-exist with the SCRFD detector in the same process.
    """
    name      = "ArcFace (R50, WebFace600K)"
    embed_dim = 512

    def __init__(self):
        _download_pack("buffalo_l")
        onnx_path = _find_onnx("buffalo_l", "w600k_r50.onnx")
        print(f"  [MODEL] {onnx_path}")
        self.model = _DirectArcFaceONNX(onnx_path)

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        emb = self.model.get_feat(prep_112(bgr_img)).flatten()
        return l2_normalize(emb).astype(np.float32)


# ── Model 2: FaceNet-VGGFace2 ─────────────────────────────────────────────────

class FaceNetVGGModel:
    """InceptionResnetV1 pretrained on VGGFace2 — Triplet loss, 3.3M celebrity images."""
    name      = "FaceNet (VGGFace2)"
    embed_dim = 512

    def __init__(self):
        self.model = _load_facenet("vggface2")

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        return _facenet_embed(self.model, bgr_img)


# ── Model 3: FaceNet-CASIA ────────────────────────────────────────────────────

class FaceNetCASIAModel:
    """
    InceptionResnetV1 pretrained on CASIA-WebFace — same architecture as Model 2
    but different training data (494k images, 10k identities, more ethnically diverse).
    Triplet loss. Directly comparable to Model 2 to isolate dataset effect.
    """
    name      = "FaceNet (CASIA-WebFace)"
    embed_dim = 512

    def __init__(self):
        self.model = _load_facenet("casia-webface")

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        return _facenet_embed(self.model, bgr_img)


# ── Model 4: AdaFace — R100, Glint360K ───────────────────────────────────────

class AdaFaceModel:
    """antelopev2/glintr100.onnx — R100, Glint360K, adaptive margin loss."""
    name      = "AdaFace (R100, Glint360K)"
    embed_dim = 512

    def __init__(self):
        self.model = _load_onnx("antelopev2", "glintr100.onnx")

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        emb = self.model.get_feat(prep_112(bgr_img)).flatten()
        return l2_normalize(emb).astype(np.float32)


# ── Model 5: MagFace — MobileFaceNet, WebFace600K ────────────────────────────

class MagFaceModel:
    """buffalo_sc/w600k_mbf.onnx — MobileFaceNet, WebFace600K. Fastest, smallest."""
    name      = "MagFace (MobileFaceNet, WebFace600K)"
    embed_dim = 512

    def __init__(self):
        self.model = _load_onnx("buffalo_sc", "w600k_mbf.onnx")

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        emb = self.model.get_feat(prep_112(bgr_img)).flatten()
        return l2_normalize(emb).astype(np.float32)


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_MODELS = {
    "arcface":       ArcFaceModel,
    "facenet_vgg":   FaceNetVGGModel,
    "facenet_casia": FaceNetCASIAModel,
    "adaface":       AdaFaceModel,
    "magface":       MagFaceModel,
}

def load_model(model_key: str):
    key = model_key.lower()
    if key not in ALL_MODELS:
        raise ValueError(f"Unknown model '{key}'. Choose from: {list(ALL_MODELS.keys())}")
    print(f"[LOAD] Loading {ALL_MODELS[key].name} ...")
    return ALL_MODELS[key]()

def load_all_models():
    models = {}
    for key in ALL_MODELS:
        try:
            models[key] = load_model(key)
        except Exception as e:
            print(f"[WARN] Could not load {key}: {e}")
    return models