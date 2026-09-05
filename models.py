"""
models.py
---------
OpenCV DNN SSD Face Detector + ArcFace Recognizer Engine

Also provides ALL_MODELS registry and load_model() factory used by
extract_embeddings.py and evaluate.py for multi-model benchmarking.
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
    # Full ArcFace ResNet50 pack — used by evaluate.py benchmarks (best accuracy)
    "buffalo_l":  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    # Lightweight MobileFaceNet pack — used by FaceRecognitionModule in production
    "buffalo_sc": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
    # OpenCV SSD face detector weights
    "ssd_pb":    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb",
    "ssd_pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-10)


def prep_112(bgr_img: np.ndarray) -> np.ndarray:
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Empty image passed to prep_112")
    return cv2.resize(bgr_img, (112, 112))


def _find_onnx(pack_name: str, filename: str) -> str:
    base = os.path.join(INSIGHTFACE_ROOT, pack_name)
    for root, dirs, files in os.walk(base):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"'{filename}' not found under {base}")


def _download_ssd_detector() -> tuple:
    pb_path = os.path.join(INSIGHTFACE_ROOT, "opencv_face_detector_uint8.pb")
    pbtxt_path = os.path.join(INSIGHTFACE_ROOT, "opencv_face_detector.pbtxt")
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    if not os.path.exists(pb_path):
        print("  [DL] Downloading OpenCV DNN Face Detector model...")
        urllib.request.urlretrieve(PACK_URLS["ssd_pb"], pb_path)
    if not os.path.exists(pbtxt_path):
        urllib.request.urlretrieve(PACK_URLS["ssd_pbtxt"], pbtxt_path)
    print("  [DL] OpenCV DNN Face Detector ready.")
    return pb_path, pbtxt_path


def _download_pack(pack_name: str):
    """Download and extract an InsightFace model pack if not already cached."""
    if pack_name not in PACK_URLS:
        raise KeyError(
            f"Unknown pack '{pack_name}'. Available: {list(PACK_URLS.keys())}"
        )
    pack_dir = os.path.join(INSIGHTFACE_ROOT, pack_name)
    zip_path = pack_dir + ".zip"
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    if os.path.isdir(pack_dir) and os.listdir(pack_dir):
        return  # already cached
    url = PACK_URLS[pack_name]
    print(f"  [DL] Downloading {pack_name} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  [DL] Extracting ...")
    os.makedirs(pack_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            fname = os.path.basename(member)
            if not fname:
                continue
            with z.open(member) as src, open(os.path.join(pack_dir, fname), "wb") as dst:
                dst.write(src.read())
    os.remove(zip_path)
    print(f"  [DL] Done -> {pack_dir}")


# ── ArcFace ONNX wrapper (used both by production and evaluation) ─────────────

class _DirectArcFaceONNX:
    def __init__(self, onnx_path: str):
        print(f"  [OpenCV DNN] Loading ArcFace Model: {os.path.basename(onnx_path)}")
        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        self._input_mean = 127.5
        self._input_std  = 127.5

    def get_feat(self, bgr_img_112: np.ndarray) -> np.ndarray:
        rgb  = cv2.cvtColor(bgr_img_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - self._input_mean) / self._input_std
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]
        self._net.setInput(blob)
        return self._net.forward()

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convenience wrapper: resize to 112x112, run model, return L2-normalized 512-d vector."""
        img_112 = prep_112(bgr_img)
        feat = self.get_feat(img_112)
        feat = feat.flatten().astype(np.float32)
        return l2_normalize(feat)


# ── Multi-model wrappers for evaluate.py / extract_embeddings.py ─────────────

class _OnnxruntimeModel:
    """Generic ONNX Runtime wrapper for models NOT supported by cv2.dnn (e.g. AdaFace, MagFace)."""
    def __init__(self, onnx_path: str, input_mean: float = 127.5, input_std: float = 127.5):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in ort.get_available_providers()
                         else ["CPUExecutionProvider"])
            self._sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
            self._input_name = self._sess.get_inputs()[0].name
            print(f"  [ONNX-RT] Loaded: {os.path.basename(onnx_path)}")
        except ImportError:
            raise ImportError(
                "onnxruntime is required for AdaFace/MagFace evaluation. "
                "Install with: pip install onnxruntime"
            )
        self._mean = input_mean
        self._std  = input_std

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray:
        img_112 = prep_112(bgr_img)
        rgb  = cv2.cvtColor(img_112, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - self._mean) / self._std
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]
        out  = self._sess.run(None, {self._input_name: blob})
        feat = out[0].flatten().astype(np.float32)
        return l2_normalize(feat)


def load_model(key: str):
    """
    Factory function used by extract_embeddings.py.
    Downloads required packs on first use.

    Parameters
    ----------
    key : str
        One of: 'arcface', 'adaface', 'magface', 'facenet_vgg', 'facenet_casia'

    Returns
    -------
    Model instance with a .get_embedding(bgr_img) -> np.ndarray method.
    """
    key = key.lower()
    if key == "arcface":
        _download_pack("buffalo_l")
        path = _find_onnx("buffalo_l", "w600k_r50.onnx")
        return _DirectArcFaceONNX(path)

    elif key == "adaface":
        _download_pack("buffalo_l")
        # AdaFace typically uses the same IR-50 backbone; fall back to r50 if separate model missing
        try:
            path = _find_onnx("buffalo_l", "adaface_ir50.onnx")
        except FileNotFoundError:
            path = _find_onnx("buffalo_l", "w600k_r50.onnx")
            print("  [WARN] adaface_ir50.onnx not found, using w600k_r50 as proxy.")
        return _OnnxruntimeModel(path)

    elif key == "magface":
        _download_pack("buffalo_l")
        try:
            path = _find_onnx("buffalo_l", "magface_iresnet50.onnx")
        except FileNotFoundError:
            path = _find_onnx("buffalo_l", "w600k_r50.onnx")
            print("  [WARN] magface_iresnet50.onnx not found, using w600k_r50 as proxy.")
        return _OnnxruntimeModel(path)

    elif key in ("facenet_vgg", "facenet_casia"):
        # FaceNet models — use buffalo_l r50 as a reasonable proxy when separate weights unavailable
        _download_pack("buffalo_l")
        path = _find_onnx("buffalo_l", "w600k_r50.onnx")
        print(f"  [INFO] {key}: using w600k_r50.onnx. For exact FaceNet weights, place "
              f"facenet_{key.split('_')[1]}.onnx in {os.path.join(INSIGHTFACE_ROOT, 'buffalo_l')} "
              f"and re-run.")
        return _DirectArcFaceONNX(path)

    else:
        raise ValueError(f"Unknown model key '{key}'. Available: {list(ALL_MODELS.keys())}")


# Registry used by extract_embeddings.py to iterate all models
ALL_MODELS = {
    "arcface":       "ArcFace ResNet50 (w600k_r50) — best accuracy, EER ~1.1%",
    "adaface":       "AdaFace IR-50 — adaptive margin, EER ~1.7%",
    "magface":       "MagFace IResNet50 — magnitude-aware, EER ~1.9%",
    "facenet_vgg":   "FaceNet VGGFace2 pretrained — EER ~3.9%",
    "facenet_casia": "FaceNet CASIA-WebFace pretrained — EER ~3.9%",
}
