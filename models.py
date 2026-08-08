"""
models.py
---------
OpenCV DNN SSD Face Detector + ArcFace Recognizer Engine
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
    "buffalo_l": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "ssd_pb":    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb",
    "ssd_pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
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

def _download_ssd_detector() -> tuple[str, str]:
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
    pack_dir = os.path.join(INSIGHTFACE_ROOT, pack_name)
    zip_path = pack_dir + ".zip"
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    if os.path.isdir(pack_dir) and os.listdir(pack_dir):
        return
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
