"""
benchmark_edge_facerec.py
--------------------------
REAL benchmark for the Cashier Face Recognition Pipeline.

This script loads and runs the actual FaceRecognitionModule with its actual
ONNX models (OpenCV DNN SSD detector + MobileFaceNet w600k_mbf), measures
real inference latency, and evaluates PASS/FAIL against an explicit target.

NOTE: Previous version of this script had hardcoded constants (+15.2, +24.5 ms)
and never loaded any model. That has been completely replaced here.

Targets (based on cashier UX requirement: recognition within 300ms):
  - Detection + Embedding: <= 250 ms  (leaves headroom for gallery search)
  - Gallery search (cosine, numpy): negligible
  - Total: <= 300 ms per recognition event

Usage:
    python scripts/benchmark_edge_facerec.py
    python scripts/benchmark_edge_facerec.py --runs 100
    python scripts/benchmark_edge_facerec.py --image path/to/face.jpg
"""
import os
import sys
import time
import json
import argparse
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Latency targets (ms)
TARGET_TOTAL_MS = 300.0   # max acceptable end-to-end latency per face
TARGET_DET_MS   = 120.0   # max for detection alone (SSD is heavy on CPU)
TARGET_EMB_MS   = 150.0   # max for embedding alone (MobileFaceNet on CPU)


def _make_test_frame(image_path: str = None) -> np.ndarray:
    """
    Return a realistic test frame.
    - If an image path is provided, load it.
    - Otherwise generate a synthetic face-like pattern (gradient + circle for head shape)
      so the detector at least sees a plausible region to process.
    """
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        if frame is not None:
            print(f"[Benchmark] Using real image: {image_path}")
            return frame

    print("[Benchmark] No real image provided — using synthetic face-like test frame.")
    print("            Tip: run with --image <path_to_face.jpg> for accurate detection latency.")
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    for i in range(h):
        frame[i, :] = [int(i * 60 / h), int(i * 40 / h), int(i * 80 / h)]
    # Approximate face oval
    cx, cy = w // 2, h // 2
    cv2.ellipse(frame, (cx, cy), (90, 110), 0, 0, 360, (200, 175, 155), -1)
    # Eyes
    cv2.circle(frame, (cx - 35, cy - 25), 12, (50, 40, 30), -1)
    cv2.circle(frame, (cx + 35, cy - 25), 12, (50, 40, 30), -1)
    # Mouth
    cv2.ellipse(frame, (cx, cy + 40), (30, 12), 0, 0, 180, (120, 80, 80), -1)
    return frame


def benchmark_facerec_pipeline(num_runs: int = 50, image_path: str = None):
    print("=" * 60)
    print("  EDGE BENCHMARK: CASHIER FACE RECOGNITION PIPELINE")
    print("  (Real model inference — no hardcoded constants)")
    print("=" * 60)

    # ── Load the actual FaceRecognitionModule ──────────────────────
    print("\n[Init] Loading FaceRecognitionModule (SSD + MobileFaceNet)...")
    t_init_start = time.time()
    try:
        from face_recognition_module import FaceRecognitionModule
        frm = FaceRecognitionModule()
    except Exception as e:
        print(f"[ERROR] Failed to load FaceRecognitionModule: {e}")
        print("        Make sure models are downloaded (run live_demo.py once to trigger download).")
        sys.exit(1)
    t_init_ms = (time.time() - t_init_start) * 1000
    print(f"[Init] Module loaded in {t_init_ms:.0f} ms\n")

    # ── Prepare test frame ─────────────────────────────────────────
    test_frame = _make_test_frame(image_path)
    print(f"[Benchmark] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print(f"[Benchmark] Running {num_runs} iterations each for detection + embedding...\n")

    # ── Benchmark 1: End-to-end detect_and_recognize ──────────────
    # Warm-up (model cold-start JIT compilation can skew first few runs)
    for _ in range(3):
        frm._detect_and_align_face(test_frame)

    e2e_times = []
    det_times  = []
    emb_times  = []

    for i in range(num_runs):
        # Detection timing
        t0 = time.time()
        result = frm._detect_and_align_face(test_frame)
        t_det = (time.time() - t0) * 1000
        det_times.append(t_det)

        # Embedding timing (use aligned crop if detection succeeded, else resize)
        if result is not None:
            aligned_face = result
        else:
            aligned_face = cv2.resize(test_frame, (112, 112))

        t1 = time.time()
        _ = frm._recognizer.get_feat(aligned_face)
        t_emb = (time.time() - t1) * 1000
        emb_times.append(t_emb)

        e2e_times.append(t_det + t_emb)

    # ── Benchmark 2: Gallery cosine search (numpy, 100 enrolled) ──
    gallery_sim = np.random.randn(100, 512).astype(np.float32)
    gallery_sim /= np.linalg.norm(gallery_sim, axis=1, keepdims=True)
    probe = np.random.randn(512).astype(np.float32)
    probe /= np.linalg.norm(probe)

    search_times = []
    for _ in range(num_runs):
        t0 = time.time()
        sims = np.dot(gallery_sim, probe)
        _ = int(np.argmax(sims))
        search_times.append((time.time() - t0) * 1000)

    # ── Results ────────────────────────────────────────────────────
    avg_det    = float(np.mean(det_times))
    std_det    = float(np.std(det_times))
    avg_emb    = float(np.mean(emb_times))
    std_emb    = float(np.std(emb_times))
    avg_e2e    = float(np.mean(e2e_times))
    avg_search = float(np.mean(search_times))
    total_lat  = avg_e2e + avg_search

    det_pass    = avg_det  <= TARGET_DET_MS
    emb_pass    = avg_emb  <= TARGET_EMB_MS
    total_pass  = total_lat <= TARGET_TOTAL_MS

    print("─" * 60)
    print("  BENCHMARK RESULTS")
    print("─" * 60)
    print(f"  SSD Detection     : {avg_det:.1f} ms ± {std_det:.1f} ms  "
          f"{'✓ PASS' if det_pass else '✗ NEEDS_OPTIMIZATION'} (target ≤ {TARGET_DET_MS:.0f} ms)")
    print(f"  MobileFaceNet Emb : {avg_emb:.1f} ms ± {std_emb:.1f} ms  "
          f"{'✓ PASS' if emb_pass else '✗ NEEDS_OPTIMIZATION'} (target ≤ {TARGET_EMB_MS:.0f} ms)")
    print(f"  Numpy Gallery (N=100): {avg_search:.3f} ms")
    print(f"  Total per face    : {total_lat:.1f} ms  "
          f"{'✓ PASS' if total_pass else '✗ NEEDS_OPTIMIZATION'} (target ≤ {TARGET_TOTAL_MS:.0f} ms)")
    print(f"  Throughput        : {1000.0 / max(total_lat, 1e-5):.1f} faces/sec")
    print("─" * 60)

    if not total_pass:
        print("\n  Optimization suggestions:")
        if avg_det > TARGET_DET_MS:
            print("    - SSD is slow: consider replacing with SCRFD (faster + outputs real landmarks)")
        if avg_emb > TARGET_EMB_MS:
            print("    - MobileFaceNet is slow: check CPU thread count, or use ONNX Runtime instead of cv2.dnn")

    overall_status = "PASS" if total_pass else "NEEDS_OPTIMIZATION"
    results = {
        "benchmark_environment": sys.platform,
        "model_detector": "OpenCV DNN SSD (opencv_face_detector_uint8.pb)",
        "model_embedding": "MobileFaceNet (w600k_mbf.onnx) via buffalo_sc",
        "note": "All latencies are REAL measurements from actual model inference — no hardcoded constants.",
        "ssd_detection_avg_ms":    round(avg_det,    2),
        "ssd_detection_std_ms":    round(std_det,    2),
        "mobilefacenet_emb_avg_ms": round(avg_emb,   2),
        "mobilefacenet_emb_std_ms": round(std_emb,   2),
        "gallery_search_avg_ms":   round(avg_search, 3),
        "total_latency_avg_ms":    round(total_lat,  2),
        "throughput_faces_per_sec": round(1000.0 / max(total_lat, 1e-5), 1),
        "target_total_ms":         TARGET_TOTAL_MS,
        "overall_status":          overall_status,
    }

    out_file = os.path.join(RESULTS_DIR, "edge_facerec_benchmark.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_file}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real cashier face recognition benchmark")
    parser.add_argument("--runs",  type=int, default=50, help="Number of benchmark iterations (default: 50)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a face image for realistic detection benchmark")
    args = parser.parse_args()
    benchmark_facerec_pipeline(num_runs=args.runs, image_path=args.image)
