"""
benchmark_edge.py
-----------------
Latency benchmark of the real cashier pipeline. Run it ON THE ARDUINO UNO Q so the numbers
in the thesis come from the target device, not a laptop:

    python3 benchmark_edge.py               # (copied next to facerec_uno_q.py) synthetic frame
    python3 benchmark_edge.py --camera      # real webcam frames (sit in front of the camera)

Reports mean / p95 ms for SCRFD-0.5G (320), alignment, MobileFaceNet, the vault rotation,
FAISS search over a 50-person gallery (x5 embeddings), the per-frame step, and the expected
time to a decision (one session of SESSION_FRAMES frames) vs the proposal target (<= 3 s).
Saves results/edge_benchmark_<host>.json.
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

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "uno_q"):
    if (p / "facerec_uno_q.py").exists():
        sys.path.insert(0, str(p))
import facerec_uno_q as fr  # noqa: E402

ROOT = HERE.parent if (HERE.parent / "dataset").exists() else HERE


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
    faces = sorted((ROOT / "dataset").glob("*/*.jpg"))
    frame = np.full((cfg.frame_h, cfg.frame_w, 3), 90, np.uint8)
    if faces:
        frame[110:370, 190:450] = cv2.resize(cv2.imread(str(faces[2])), (260, 260))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    cfg = fr.Config()
    cv2.setNumThreads(cfg.threads)
    det = fr.SCRFD(fr.det_model_path(cfg.det_size), cfg.det_size, cfg.det_thresh)
    rec = fr.MobileFaceNet(fr.find_model("w600k_mbf.onnx"))
    vault = fr.Vault("00" * 32)
    frame = test_frame(cfg, args.camera)
    faces = det.detect(frame)
    if not faces:
        sys.exit("No face in the test frame (with --camera, sit in front of the webcam).")
    face = faces[0]
    aligned = fr.align(frame, face.kps)

    rng = np.random.default_rng(0)
    gallery = fr.Gallery()
    gallery.load([(f"c{i // 5}", "", v) for i, v in enumerate(rng.normal(size=(250, fr.EMBED_DIM)).astype(np.float32))])
    emb = rec.embed(aligned)
    probe = vault.protect(emb)

    report = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version(),
              "opencv": cv2.__version__, "threads": cfg.threads, "frame": f"{frame.shape[1]}x{frame.shape[0]}",
              "results": {}}
    res = report["results"]
    res[f"scrfd_{cfg.det_size}"] = timeit(lambda: det.detect(frame), args.n)
    res["quality"] = timeit(lambda: fr.quality(frame, faces), args.n)
    res["align"] = timeit(lambda: fr.align(frame, face.kps), args.n)
    res["mobilefacenet"] = timeit(lambda: rec.embed(aligned), args.n)
    res["vault_rotation"] = timeit(lambda: vault.protect(emb), args.n)
    res["faiss_search_250"] = timeit(lambda: gallery.top2(probe), args.n)

    def step():
        fs = det.detect(frame)
        f, q, _ = fr.quality(frame, fs)
        vault.protect(rec.embed(fr.align(frame, fs[0].kps)))
    res["frame_total"] = timeit(step, args.n)

    frame_ms = res["frame_total"]["mean_ms"]
    frames = min(cfg.session_frames, max(cfg.session_min_frames, int(cfg.session_max_sec * 1000 / frame_ms)))
    decision_s = min(cfg.session_max_sec, cfg.session_frames * frame_ms / 1000)
    report["fps"] = round(1000 / frame_ms, 1)
    report["session_frames"] = frames
    report["time_to_decision_s"] = round(decision_s, 2)

    print(f"\n{report['host']} ({report['machine']}), OpenCV {cv2.__version__}, frame {report['frame']}")
    for k, v in res.items():
        print(f"  {k:20s} {v['mean_ms']:8.2f} ms  (p95 {v['p95_ms']:.2f})")
    print(f"\n  FPS while a face is in view : {report['fps']}")
    print(f"  Time to decision           : {decision_s:.2f} s ({frames} frames; proposal target <= 3 s)")

    out = ROOT / "results" / f"edge_benchmark_{report['host']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
