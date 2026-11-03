"""
evaluate.py
-----------
Offline evaluation of the deployed recognizer (MobileFaceNet w600k_mbf) on dataset/<person>/*.jpg.
The photos are aligned 112x112 crops made by scripts/collect_dataset.py.

What it measures (the face-recognition metrics in the proposal):
  1. Verification (1:1, all pairs)        genuine/impostor scores, EER, ROC, TAR@FAR
  2. Open-set identification (1:N)        exactly how the cashier camera works:
       - each person is enrolled with an averaged template of N_ENROLL photos
         (same as /latest-embedding), remaining photos are probes
       - Rank-1  : top-1 gallery match is the right person (closed-set)
       - FRR     : enrolled person NOT accepted as themself at threshold tau
       - FAR     : person removed from the gallery (unknown) but accepted as someone
                   (leave-one-person-out, so every person acts as an impostor once)
  3. Track-level voting                   the same FAR/FRR after the 3-of-5 vote used on
                                          the device (frames sampled from the person's probes;
                                          an approximation, real video frames are correlated)
  4. Recommended tau                      smallest tau with open-set FAR <= --target-far

Usage
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/w600k_mbf_int8.onnx   # evaluate the INT8 model
    python scripts/evaluate.py --threshold 0.36 --enroll 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uno_q"))
import facerec_uno_q as fr  # noqa: E402

DATASET = ROOT / "dataset"
RESULTS = ROOT / "results"


def load_embeddings(model_path: str):
    model = fr.MobileFaceNet(fr.find_model("w600k_mbf.onnx", model_path), os.cpu_count() or 4)
    names, embs = [], []
    for person in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        for img_path in sorted(person.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))
            names.append(person.name)
            embs.append(model.embed(img))
    return np.array(names), np.stack(embs)


def verification(labels, X):
    sims = X @ X.T
    iu = np.triu_indices(len(X), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    s = sims[iu]
    return s[same], s[~same]


def eer(genuine, impostor):
    ts = np.linspace(-0.2, 1.0, 2401)
    far = np.array([(impostor >= t).mean() for t in ts])
    frr = np.array([(genuine < t).mean() for t in ts])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i]), ts, far, frr


def open_set_scores(labels, X, n_enroll, seed):
    """Per probe: (true person, best id with own template, its score, best score with own template removed)."""
    rng = np.random.default_rng(seed)
    people = sorted(set(labels))
    templates, probes = {}, []
    for p in people:
        idx = rng.permutation(np.where(labels == p)[0])
        templates[p] = fr.l2_normalize(X[idx[:n_enroll]].mean(0))
        probes += [(p, j) for j in idx[n_enroll:]]
    T = np.stack([templates[p] for p in people])
    out = []
    for p, j in probes:
        s = T @ X[j]
        own = people.index(p)
        best = int(np.argmax(s))
        s_wo = s.copy()
        s_wo[own] = -1
        others = np.delete(s, own)
        out.append((p, people[best], float(s[best]), float(s_wo.max()), others))
    return out


def rates(results, tau):
    n = len(results)
    rank1 = sum(r[0] == r[1] for r in results) / n
    frr = sum(not (r[0] == r[1] and r[2] >= tau) for r in results) / n
    far = sum(r[3] >= tau for r in results) / n
    return rank1, far, frr


def voting_rates(results, tau, window, need, trials, seed):
    """Simulate tracks: draw `window` frames of one person, decide with >= `need` agreeing votes."""
    rng = np.random.default_rng(seed)
    by_person: dict = {}
    for r in results:
        by_person.setdefault(r[0], []).append(r)
    fa = fr_ = total = 0
    for p, rs in by_person.items():
        for _ in range(trials):
            pick = [rs[i] for i in rng.integers(0, len(rs), window)]
            # Enrolled: count votes for the correct id at tau
            correct = sum(r[1] == p and r[2] >= tau for r in pick)
            fr_ += correct < need
            # Same frames as an unknown person (own template removed)
            fa += sum(r[3] >= tau for r in pick) >= need
            total += 1
    return fa / total, fr_ / total


def far_vs_gallery_size(results, tau, sizes, seed):
    """Open-set FAR when only k other people are enrolled: FAR grows with gallery size."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in sizes:
        hits = 0
        for r in results:
            others = r[4]
            pick = rng.choice(len(others), size=min(k, len(others)), replace=False)
            hits += others[pick].max() >= tau
        out[k] = hits / len(results)
    return out


def plot(genuine, impostor, ts, far, frr, tau, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(-0.3, 1.0, 66)
    ax[0].hist(impostor, bins, alpha=0.6, density=True, label="impostor (different people)", color="#d1495b")
    ax[0].hist(genuine, bins, alpha=0.6, density=True, label="genuine (same person)", color="#2e86ab")
    ax[0].axvline(tau, color="k", ls="--", label=f"tau = {tau:.3f}")
    ax[0].set_xlabel("cosine similarity")
    ax[0].set_ylabel("density")
    ax[0].set_title("MobileFaceNet score distributions")
    ax[0].legend()
    ax[1].plot(far * 100, (1 - frr) * 100, color="#2e86ab")
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.01, 100)
    ax[1].set_xlabel("FAR (%)")
    ax[1].set_ylabel("TAR (%)")
    ax[1].set_title("ROC (1:1 verification)")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="ONNX recognizer (default: w600k_mbf.onnx FP32)")
    ap.add_argument("--threshold", type=float, default=float(os.getenv("MATCH_THRESHOLD", 0.40)))
    ap.add_argument("--enroll", type=int, default=3, help="photos averaged into one enrollment template")
    ap.add_argument("--target-far", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--splits", type=int, default=10)
    args = ap.parse_args()

    labels, X = load_embeddings(args.model)
    tag = Path(args.model).stem if args.model else "w600k_mbf_fp32"
    print(f"\nModel: {tag} | {len(X)} photos | {len(set(labels))} people")

    genuine, impostor = verification(labels, X)
    e, e_t, ts, far_c, frr_c = eer(genuine, impostor)
    tar = {f: float((genuine >= np.quantile(impostor, 1 - f)).mean()) for f in (0.01, 0.001)}

    results = [r for s in range(args.splits) for r in open_set_scores(labels, X, args.enroll, s)]
    sweep = []
    for tau in np.round(np.arange(0.20, 0.61, 0.01), 2):
        r1, far, frr = rates(results, tau)
        vfar, vfrr = voting_rates(results, tau, args.window, args.votes, 200, 0)
        sweep.append({"tau": float(tau), "rank1": r1, "far": far, "frr": frr,
                      "track_far": vfar, "track_frr": vfrr})
    ok = [s for s in sweep if s["far"] <= args.target_far]
    rec = ok[0]["tau"] if ok else sweep[-1]["tau"]

    r1, far, frr = rates(results, args.threshold)
    vfar, vfrr = voting_rates(results, args.threshold, args.window, args.votes, 200, 0)

    print("\n1:1 verification")
    print(f"  genuine mean {genuine.mean():.3f} | impostor mean {impostor.mean():.3f} | max impostor {impostor.max():.3f}")
    print(f"  EER {e*100:.2f}% at tau={e_t:.3f} | TAR@FAR=1% {tar[0.01]*100:.1f}% | TAR@FAR=0.1% {tar[0.001]*100:.1f}%")
    print(f"\n1:N open-set (enroll={args.enroll} photos, {args.splits} random splits, {len(results)} probes)")
    print(f"  {'tau':>5} {'Rank-1':>7} {'FAR':>7} {'FRR':>7} | {'vote FAR':>8} {'vote FRR':>8}")
    for s in sweep:
        if s["tau"] in (0.26, 0.30, 0.33, 0.36, 0.40, 0.45) or s["tau"] == rec:
            mark = " <- recommended" if s["tau"] == rec else ""
            print(f"  {s['tau']:5.2f} {s['rank1']*100:6.1f}% {s['far']*100:6.2f}% {s['frr']*100:6.2f}% |"
                  f" {s['track_far']*100:7.2f}% {s['track_frr']*100:7.2f}%{mark}")
    n_others = len(set(labels)) - 1
    sizes = sorted({k for k in (1, 5, 10, 15, n_others) if k <= n_others})
    scaling = far_vs_gallery_size(results, args.threshold, sizes, 0)
    print(f"\nOpen-set FAR vs number of enrolled people (tau={args.threshold:.2f}):")
    print("  " + "  ".join(f"{k} people: {v*100:.2f}%" for k, v in scaling.items()))

    print(f"\nAt tau={args.threshold:.2f} (MATCH_THRESHOLD) vs proposal targets:")
    print(f"  Rank-1 {r1*100:.1f}%  (target >= 90%)  {'OK' if r1 >= 0.90 else 'MISS'}")
    print(f"  FAR    {far*100:.2f}%  (target <= 1%)   {'OK' if far <= 0.01 else 'MISS'}   after voting {vfar*100:.2f}%")
    print(f"  FRR    {frr*100:.2f}%  (target <= 10%)  {'OK' if frr <= 0.10 else 'MISS'}   after voting {vfrr*100:.2f}%")

    RESULTS.mkdir(exist_ok=True)
    report = {
        "model": tag, "n_photos": int(len(X)), "n_people": int(len(set(labels))),
        "verification": {"genuine_mean": float(genuine.mean()), "impostor_mean": float(impostor.mean()),
                         "eer": e, "eer_threshold": e_t, "tar_at_far_1pct": tar[0.01],
                         "tar_at_far_0_1pct": tar[0.001]},
        "open_set": {"enroll_photos": args.enroll, "splits": args.splits, "threshold": args.threshold,
                     "rank1": r1, "far": far, "frr": frr, "track_far": vfar, "track_frr": vfrr,
                     "vote": f"{args.votes}/{args.window}",
                     "recommended_threshold_for_far_target": rec, "target_far": args.target_far,
                     "far_vs_gallery_size": {str(k): v for k, v in scaling.items()}},
        "sweep": sweep,
    }
    out = RESULTS / f"evaluation_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    plot(genuine, impostor, ts, far_c, frr_c, args.threshold, RESULTS / f"scores_roc_{tag}.png")
    print(f"\nSaved {out.relative_to(ROOT)} and results/scores_roc_{tag}.png")


if __name__ == "__main__":
    main()
