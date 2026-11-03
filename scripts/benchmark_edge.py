"""
benchmark_edge.py
-----------------
Latency benchmark of the real cashier pipeline. Run it ON THE ARDUINO UNO Q
(copy the repo, or the uno_q/ folder plus this script) so the numbers in the thesis
come from the target device, not a laptop.

    python3 scripts/benchmark_edge.py                     # synthetic frame with a dataset face
    python3 scripts/benchmark_edge.py --camera            # real webcam frames
    python3 scripts/benchmark_edge.py --int8 uno_q/models/w600k_mbf_int8.onnx

Reports mean / p95 ms for: SCRFD at several input sizes, alignment, MobileFaceNet
(FP32 and optional INT8), FAISS search with a 50-person gallery, the full per-frame
step, and the expected time-to-decision (VOTE_MIN frames) vs the proposal target
(end-to-end <= 3 s). Saves results/edge_benchmark_<host>.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uno_q"))
import facerec_uno_q as fr  # noqa: E402


def timeit(fn, n: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(ts)), 2), "p95_ms": round(float(np.percentile(ts, 95)), 2)}


def test_frame(cfg: fr.Config, use_camera: bool) -> np.ndarray:
    if use_camera:
        cam = fr.Camera(cfg)
        if not cam._open():
            sys.exit("No webcam found.")
        for _ in range(15):
            ok, frame = cam.cap.read()
        cam.cap.release()
        return frame
    face = sorted((ROOT / "dataset").glob("*/*.jpg"))
    frame = np.full((cfg.frame_h, cfg.frame_w, 3), 90, np.uint8)
    if face:
        frame[120:320, 220:420] = cv2.resize(cv2.imread(str(face[2])), (200, 200))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--int8", default="", help="path to INT8 MobileFaceNet to compare")
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    cfg = fr.Config()
    engine = fr.FaceEngine(cfg)
    frame = test_frame(cfg, args.camera)
    dets, kps = engine.detector.detect(frame)
    if len(dets) == 0:
        sys.exit("No face in the test frame (with --camera, sit in front of the webcam).")
    aligned = fr.align_face(frame, kps[0])

    rng = np.random.default_rng(0)
    gallery = rng.normal(size=(150, fr.EMBED_DIM)).astype(np.float32)
    engine.gallery.load([(f"c{i // 3}", "", v) for i, v in enumerate(gallery)])  # 50 people x 3
    probe = engine.embedder.embed(aligned)

    report = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version(),
              "threads": cfg.num_threads, "frame": f"{frame.shape[1]}x{frame.shape[0]}", "results": {}}
    res = report["results"]
    for size in (320, 480, 640):
        det = fr.SCRFD(fr.find_model("det_500m.onnx", cfg.det_model), cfg.num_threads, size, cfg.det_thresh)
        res[f"scrfd_{size}"] = timeit(lambda: det.detect(frame), args.n)
    res["align"] = timeit(lambda: fr.align_face(frame, kps[0]), args.n)
    res["mobilefacenet_fp32"] = timeit(lambda: engine.embedder.embed(aligned), args.n)
    if args.int8:
        q = fr.MobileFaceNet(args.int8, cfg.num_threads)
        res["mobilefacenet_int8"] = timeit(lambda: q.embed(aligned), args.n)
    res["faiss_search_150"] = timeit(lambda: engine.gallery.search(probe), args.n)

    # Per-frame cost on the device: detection + one embedding (worst case while recognising)
    def step():
        d, k = engine.detector.detect(frame)
        engine.embedder.embed(fr.align_face(frame, k[0]))
    res["frame_det_plus_embed"] = timeit(step, args.n)
    res["jpeg_encode"] = timeit(lambda: cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]), args.n)

    frame_ms = res["frame_det_plus_embed"]["mean_ms"]
    decision_s = cfg.vote_min * frame_ms / 1000
    report["time_to_decision_s"] = round(decision_s, 3)
    report["fps_while_recognising"] = round(1000 / frame_ms, 1)
    report["fps_tracking_only"] = round(1000 / res[f"scrfd_{cfg.det_size}"]["mean_ms"], 1)

    print(f"\n{report['host']} ({report['machine']}), {cfg.num_threads} threads, frame {report['frame']}")
    for k, v in res.items():
        print(f"  {k:24s} {v['mean_ms']:8.2f} ms  (p95 {v['p95_ms']:.2f})")
    print(f"\n  FPS while recognising : {report['fps_while_recognising']}")
    print(f"  FPS when identity known: {report['fps_tracking_only']}  (embedding only every REVERIFY_SEC)")
    print(f"  Time to decision      : {decision_s:.2f} s for {cfg.vote_min} votes "
          f"(+ camera/network; proposal target end-to-end <= 3 s)")

    out = ROOT / "results" / f"edge_benchmark_{report['host']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
