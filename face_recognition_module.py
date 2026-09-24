"""
face_recognition_module.py
--------------------------
Production Face Recognition Pipeline for Arduino Uno Q / Register Camera:
- Genuine 5-Point Affine Landmark Alignment (SCRFD det_500m.onnx keypoints)
- InsightFace MobileFaceNet w600k_mbf.onnx (512-d embeddings, ArcFace loss)
- FAISS IndexFlatIP cosine similarity matching
- 100% Real models: SCRFD + MobileFaceNet
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE
sys.path.insert(0, str(PROJECT_ROOT))

from face_models import SCRFDFaceDetector, MobileFaceNetModel, align_face_5pts, l2_normalize

try:
    import faiss
    _USE_FAISS = True
except ImportError:
    _USE_FAISS = False


class FaceObject:
    def __init__(self, bbox, det_score=0.9, kps=None):
        self.bbox = np.array(bbox, dtype=np.int32)
        self.det_score = float(det_score)
        self.kps = np.array(kps, dtype=np.float32) if kps is not None else None


class FaceRecognitionModule:
    """
    Production Face Recognition using SCRFD-0.5G (detector) + MobileFaceNet (embedding).
    Features genuine 5-point affine landmark alignment and FAISS cosine matching.
    """
    def __init__(
        self,
        det_thresh: float = 0.45,
        threshold: float = 0.2579,
    ):
        self.det_thresh = det_thresh
        self.threshold  = threshold

        # Detection stride: run SCRFD every N frames, cache bbox & landmarks in between
        self._det_stride     = 2
        self._det_call_count = 0
        self._cached_bbox    = None
        self._cached_kps     = None

        logger.info("[Detector] Initializing SCRFD Face Detector (det_500m.onnx)...")
        self._detector = SCRFDFaceDetector(conf_thresh=det_thresh)
        logger.info("[Detector] SCRFD Face Detector ready.")

        logger.info("[Recognizer] Initializing MobileFaceNet (w600k_mbf.onnx)...")
        self._recognizer = MobileFaceNetModel()
        logger.info("[Recognizer] MobileFaceNet ready.")

        self._labels: list[str] = []
        self._gallery: Optional[np.ndarray] = None
        self._index = None

    def load_gallery_from_rows(self, rows: list[tuple[str, list[float]]]):
        labels: list[str] = []
        vecs: list[np.ndarray] = []
        for customer_id, vec in rows:
            labels.append(customer_id)
            vecs.append(l2_normalize(np.array(vec, dtype=np.float32)))
        self._labels  = labels
        self._gallery = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 512), dtype=np.float32)
        self._build_index()

    def _build_index(self):
        if self._gallery is None or len(self._gallery) == 0:
            self._index = None
            return
        if _USE_FAISS:
            self._index = faiss.IndexFlatIP(512)
            self._index.add(self._gallery)
        else:
            self._index = None

    def get_gallery_stats(self) -> dict:
        return {
            "n_embeddings": len(self._labels),
            "n_people": len(set(self._labels)),
            "people": sorted(set(self._labels)),
        }

    def _detect_and_align_face(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        dets, kpss = self._detector.detect(frame)

        if len(dets) == 0:
            return None, None

        # Select highest-confidence face with minimum dimension
        best_idx = None
        max_score = 0.0
        for i, d in enumerate(dets):
            conf = float(d[4])
            bw = d[2] - d[0]
            bh = d[3] - d[1]
            if conf > self.det_thresh and conf > max_score and bw > 25 and bh > 25:
                max_score = conf
                best_idx = i

        if best_idx is None:
            return None, None

        bx1, by1, bx2, by2 = dets[best_idx][:4].astype(int)
        best_bbox = (max(0, bx1), max(0, by1), min(w, bx2), min(h, by2))
        best_kps = kpss[best_idx]

        # Apply genuine 5-point affine landmark alignment to ArcFace template
        aligned_112 = align_face_5pts(frame, best_kps)
        return aligned_112, FaceObject(bbox=list(best_bbox), det_score=float(max_score), kps=best_kps)

    def recognize_face(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        self._det_call_count += 1
        if self._det_call_count % self._det_stride == 0 or self._cached_bbox is None:
            aligned, face = self._detect_and_align_face(frame)
            if aligned is None:
                self._cached_bbox = None
                self._cached_kps  = None
                return {
                    "status": "no_face", "customer_id": None,
                    "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
                }
            self._cached_bbox = face.bbox.tolist()
            self._cached_kps  = face.kps
        else:
            if self._cached_bbox is None:
                return {
                    "status": "no_face", "customer_id": None,
                    "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
                }
            if self._cached_kps is not None:
                aligned = align_face_5pts(frame, self._cached_kps)
            else:
                x1, y1, x2, y2 = self._cached_bbox
                crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                aligned = cv2.resize(crop, (112, 112)) if crop.size > 0 else np.zeros((112, 112, 3), dtype=np.uint8)
            face = FaceObject(bbox=self._cached_bbox, kps=self._cached_kps)

        try:
            probe = self._recognizer.get_embedding(aligned)
        except Exception as e:
            return {
                "status": "no_face", "customer_id": None,
                "score": 0.0, "bbox": None, "latency_ms": round((time.perf_counter() - t0)*1000, 1)
            }

        # Match against gallery
        if self._gallery is None or len(self._gallery) == 0:
            return {
                "status": "unknown",
                "customer_id": None,
                "score": 0.0,
                "bbox": face.bbox.tolist(),
                "embedding": probe.tolist(),
                "latency_ms": round((time.perf_counter() - t0)*1000, 1)
            }

        if _USE_FAISS and self._index is not None:
            D, I = self._index.search(probe.reshape(1, -1), 1)
            best_score = float(D[0][0])
            best_idx   = int(I[0][0])
            best_label = self._labels[best_idx]
        else:
            sims = np.dot(self._gallery, probe)
            best_idx   = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_label = self._labels[best_idx]

        status = "known" if best_score >= self.threshold else "unknown"
        customer_id = best_label if status == "known" else None

        return {
            "status": status,
            "customer_id": customer_id,
            "score": round(best_score, 4),
            "bbox": face.bbox.tolist(),
            "embedding": probe.tolist(),
            "latency_ms": round((time.perf_counter() - t0)*1000, 1)
        }

    def extract_embedding_from_frame(self, frame: np.ndarray) -> dict:
        """
        Extract a 512-d ArcFace embedding from a frame WITHOUT searching the gallery.
        Used by the /extract-embedding endpoint and the CRM enrollment flow.

        Returns
        -------
        dict:
            success   : bool
            embedding : list[float] (512-d, L2-normalized) or None
            det_score : float or None
            message   : str
        """
        aligned, face = self._detect_and_align_face(frame)
        if aligned is None:
            return {
                "success":   False,
                "embedding": None,
                "det_score": None,
                "bbox":      None,
                "message":   "No face detected in the frame.",
            }
        try:
            emb = self._recognizer.get_embedding(aligned)
            return {
                "success":   True,
                "embedding": emb.tolist(),
                "det_score": round(float(face.det_score), 4),
                "bbox":      face.bbox.tolist(),
                "message":   "OK",
            }
        except Exception as e:
            logger.error(f"[extract_embedding_from_frame] Embedding failed: {e}")
            return {
                "success":   False,
                "embedding": None,
                "det_score": round(float(face.det_score), 4) if face else None,
                "bbox":      face.bbox.tolist() if face else None,
                "message":   str(e),
            }

    def annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw a recognition overlay (bounding box + label) on the given frame.
        Uses the cached bbox from the most recent detection so it does NOT
        re-run SCRFD inference — safe to call on every display frame.

        Parameters
        ----------
        frame  : BGR image from the camera.
        result : dict returned by recognize_face().

        Returns
        -------
        A new BGR image with the overlay drawn.
        """
        out    = frame.copy()
        status = result.get("status", "no_face")
        score  = result.get("score", 0.0)
        cid    = result.get("customer_id") or ""
        bbox   = result.get("bbox")

        if status == "no_face" or bbox is None:
            return out

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if status == "known":
            color = (70, 210, 90)       # green
            label = f"{cid[:8]}  {score:.2f}"
        else:  # "unknown"
            color = (30, 190, 255)      # amber
            label = f"Unknown  {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Filled label background so text is always readable
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            out, label,
            (x1 + 4, max(th, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )
        return out
