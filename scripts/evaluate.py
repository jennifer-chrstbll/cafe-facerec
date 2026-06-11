"""
evaluate.py
-----------
Evaluates all 5 models using embeddings already extracted by extract_embeddings.py.

Protocol: Leave-One-Out (LOO)
    For each person, one embedding is the probe. The rest (from all people) form
    the gallery. We ask: does the top-1 FAISS match return the correct person?

Metrics reported per model:
    - Rank-1 Accuracy       : correct top-1 match rate
    - TAR @ FAR=0.01        : True Accept Rate when False Accept Rate ≤ 1%
    - TAR @ FAR=0.001       : stricter threshold
    - EER                   : Equal Error Rate (where FAR == FRR)
    - Avg inference latency : from extract_embeddings summary (ms)
    - Threshold @ EER       : cosine similarity threshold to use in live demo

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model arcface
"""

import os
import sys
import argparse
import numpy as np
from itertools import combinations
from scipy.optimize import brentq
from scipy.interpolate import interp1d

PROJECT_ROOT   = r"D:\Projects\cafe_facerec"
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
sys.path.insert(0, PROJECT_ROOT)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # embeddings are already L2-normalised


# ── Load embeddings for one model ─────────────────────────────────────────────

def load_embeddings(model_key: str):
    """
    Returns:
        labels : list of str  — person name per embedding
        embs   : np.ndarray   — shape (N, D)
    """
    model_dir = os.path.join(EMBEDDINGS_DIR, model_key)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No embeddings found for '{model_key}' at {model_dir}")

    labels, embs = [], []
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".npy"):
            continue
        person = fname[:-4]
        arr    = np.load(os.path.join(model_dir, fname))   # (n_photos, D)
        for emb in arr:
            labels.append(person)
            embs.append(emb)

    return labels, np.stack(embs).astype(np.float32)


# ── Build genuine / impostor score pairs ─────────────────────────────────────

def build_score_pairs(labels, embs):
    """
    Genuine pairs  : same person, different photo
    Impostor pairs : different people
    Returns two lists of cosine similarity scores.
    """
    n          = len(labels)
    genuine    = []
    impostor   = []

    for i in range(n):
        for j in range(i + 1, n):
            score = cosine_sim(embs[i], embs[j])
            if labels[i] == labels[j]:
                genuine.append(score)
            else:
                impostor.append(score)

    return np.array(genuine), np.array(impostor)


# ── Rank-1 accuracy (Leave-One-Out) ──────────────────────────────────────────

def rank1_loo(labels, embs):
    """
    For each embedding as probe, compare against all others (gallery).
    Gallery includes all embeddings EXCEPT the probe itself.
    """
    n       = len(labels)
    correct = 0

    for i in range(n):
        probe       = embs[i]
        gallery_emb = np.concatenate([embs[:i], embs[i+1:]], axis=0)
        gallery_lbl = labels[:i] + labels[i+1:]

        sims       = gallery_emb @ probe           # (n-1,) cosine similarities
        top1_idx   = int(np.argmax(sims))
        if gallery_lbl[top1_idx] == labels[i]:
            correct += 1

    return correct / n


# ── TAR @ FAR, EER ───────────────────────────────────────────────────────────

def compute_tar_at_far(genuine, impostor, target_far=0.01):
    """Compute TAR when FAR is at or below target_far."""
    thresholds = np.linspace(
        min(genuine.min(), impostor.min()),
        max(genuine.max(), impostor.max()),
        10000
    )
    best_tar = 0.0
    for thresh in thresholds:
        far = np.mean(impostor >= thresh)
        tar = np.mean(genuine  >= thresh)
        if far <= target_far:
            best_tar = max(best_tar, tar)
    return best_tar


def compute_eer(genuine, impostor):
    """
    EER: threshold where FAR == FRR.
    Returns (eer_value, threshold_at_eer).
    """
    thresholds = np.linspace(
        min(genuine.min(), impostor.min()),
        max(genuine.max(), impostor.max()),
        10000
    )
    far_curve = np.array([np.mean(impostor >= t) for t in thresholds])
    frr_curve = np.array([np.mean(genuine  <  t) for t in thresholds])

    # Find EER via interpolation
    try:
        eer_thresh = brentq(
            lambda t: (np.mean(impostor >= t) - np.mean(genuine < t)),
            thresholds[0], thresholds[-1]
        )
        eer = np.mean(impostor >= eer_thresh)
    except ValueError:
        # fallback: find where |FAR - FRR| is minimised
        diffs      = np.abs(far_curve - frr_curve)
        idx        = np.argmin(diffs)
        eer        = (far_curve[idx] + frr_curve[idx]) / 2
        eer_thresh = thresholds[idx]

    return float(eer), float(eer_thresh)


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model_key: str):
    print(f"\n{'='*55}")
    print(f"  Evaluating: {model_key.upper()}")
    print(f"{'='*55}")

    labels, embs = load_embeddings(model_key)
    n_people     = len(set(labels))
    n_embs       = len(labels)
    print(f"  Loaded {n_embs} embeddings across {n_people} people.")

    if n_people < 2:
        print("  [SKIP] Need at least 2 people to evaluate.")
        return None

    genuine, impostor = build_score_pairs(labels, embs)
    print(f"  Genuine pairs : {len(genuine)}")
    print(f"  Impostor pairs: {len(impostor)}")

    rank1          = rank1_loo(labels, list(labels), embs) if False else rank1_loo(list(labels), embs)
    tar_at_far01   = compute_tar_at_far(genuine, impostor, target_far=0.01)
    tar_at_far001  = compute_tar_at_far(genuine, impostor, target_far=0.001)
    eer, eer_thresh = compute_eer(genuine, impostor)

    results = {
        "model":          model_key,
        "n_people":       n_people,
        "n_embeddings":   n_embs,
        "rank1":          rank1,
        "tar_at_far_1pct":  tar_at_far01,
        "tar_at_far_01pct": tar_at_far001,
        "eer":            eer,
        "eer_threshold":  eer_thresh,
        "genuine_mean":   float(genuine.mean()),
        "impostor_mean":  float(impostor.mean()),
    }

    print(f"\n  Results:")
    print(f"    Rank-1 Accuracy          : {rank1*100:.2f}%")
    print(f"    TAR @ FAR=1%             : {tar_at_far01*100:.2f}%")
    print(f"    TAR @ FAR=0.1%           : {tar_at_far001*100:.2f}%")
    print(f"    EER                      : {eer*100:.2f}%")
    print(f"    Threshold @ EER          : {eer_thresh:.4f}  ← use this in live_demo.py")
    print(f"    Genuine  similarity mean : {genuine.mean():.4f}")
    print(f"    Impostor similarity mean : {impostor.mean():.4f}")

    # Save per-model results
    out_path = os.path.join(RESULTS_DIR, f"{model_key}_results.txt")
    with open(out_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Saved to: {out_path}")

    return results


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: list):
    valid = [r for r in all_results if r is not None]
    if not valid:
        print("\n[WARN] No results to summarise.")
        return

    header = f"\n{'='*80}\n  FINAL COMPARISON TABLE\n{'='*80}"
    row_fmt = "  {:<18} {:>10} {:>12} {:>13} {:>8} {:>12}"
    divider = "  " + "-"*78

    lines = [
        header,
        row_fmt.format("Model", "Rank-1 %", "TAR@FAR1% %", "TAR@FAR.1% %", "EER %", "Thresh@EER"),
        divider,
    ]

    # Sort by Rank-1 descending
    for r in sorted(valid, key=lambda x: x["rank1"], reverse=True):
        lines.append(row_fmt.format(
            r["model"],
            f"{r['rank1']*100:.2f}",
            f"{r['tar_at_far_1pct']*100:.2f}",
            f"{r['tar_at_far_01pct']*100:.2f}",
            f"{r['eer']*100:.2f}",
            f"{r['eer_threshold']:.4f}",
        ))

    lines.append(divider)
    # Highlight winner
    winner = sorted(valid, key=lambda x: x["rank1"], reverse=True)[0]
    lines.append(f"\n  Best model (Rank-1): {winner['model'].upper()}")
    lines.append(f"  Recommended threshold for live demo: {winner['eer_threshold']:.4f}\n")

    output = "\n".join(lines)
    print(output)

    # Save summary CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=valid[0].keys())
        writer.writeheader()
        writer.writerows(valid)
    print(f"  Summary CSV saved to: {csv_path}")
    print(f"  Open this in Excel for your thesis table.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(target_model=None):
    from models import ALL_MODELS

    model_keys = [target_model] if target_model else list(ALL_MODELS.keys())

    all_results = []
    for key in model_keys:
        try:
            result = evaluate_model(key)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {key}: {e}")
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {key}: {e}")
            traceback.print_exc()

    if len(all_results) > 1:
        print_summary(all_results)
    elif all_results and all_results[0]:
        r = all_results[0]
        print(f"\n  Recommended threshold for live demo: {r['eer_threshold']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Evaluate only one model. Options: arcface, facenet, insightface, adaface, magface")
    args = parser.parse_args()
    main(target_model=args.model)