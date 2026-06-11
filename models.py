"""
models.py
---------
Unified wrapper for all 5 face recognition models.
Each model exposes:
    get_embedding(bgr_image) -> np.ndarray (512-d, L2-normalised float32)

Models (confirmed available, no extra downloads needed):
    1. ArcFace      — R50,  WebFace600K,  ArcFace loss     (buffalo_l/w600k_r50.onnx)
    2. FaceNet-VGG  — InceptionResnetV1,  VGGFace2, Triplet (facenet-pytorch)
    3. FaceNet-CASIA— InceptionResnetV1,  CASIA-WebFace, Triplet (facenet-pytorch)
    4. AdaFace      — R100, Glint360K,    adaptive margin   (antelopev2/glintr100.onnx)
    5. MagFace      — MobileFaceNet,      WebFace600K       (buffalo_sc/w600k_mbf.onnx)

Why these are genuinely distinct:
    Model 1: ArcFace loss + margin-based angular softmax
    Model 2: Triplet loss + VGGFace2 (celebrity-heavy, 3.3M images)
    Model 3: Same architecture as Model 2 but CASIA-WebFace (10k identities,
             500k images, more diverse) — measurably different embeddings
    Model 4: R100 (deepest) + Glint360K (360M images, largest dataset here)
    Model 5: MobileFaceNet (fastest, smallest) — IoT edge deployment candidate

    This comparison has a clear thesis narrative:
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
    _download_pack(pack_name)
    onnx_path = _find_onnx(pack_name, filename)
    print(f"  [MODEL] {onnx_path}")
    from insightface.model_zoo import get_model
    model = get_model(onnx_path, providers=["CPUExecutionProvider"])
    model.prepare(-1)
    return model

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
    """buffalo_l/w600k_r50.onnx — R50, WebFace600K, ArcFace loss."""
    name      = "ArcFace (R50, WebFace600K)"
    embed_dim = 512

    def __init__(self):
        self.model = _load_onnx("buffalo_l", "w600k_r50.onnx")

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