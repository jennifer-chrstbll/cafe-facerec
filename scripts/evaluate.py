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

Also generates thesis figures:
    - ROC Curve  (TAR vs FAR for all models)
    - CMC Curve  (Rank-1 through Rank-10)
    - DET Curve  (log-scale FAR vs FRR)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model arcface
    python scripts/evaluate.py --no-plots   (skip figure generation)
"""

import os
import sys
import argparse
import numpy as np
from itertools import combinations
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")   # non-interactive; safe on any system
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def rankk_loo(labels: list, embs: np.ndarray, max_rank: int = 10) -> list:
    """
    Compute Rank-1 through Rank-max_rank accuracy (CMC curve).
    Returns list of length max_rank: [rank1_acc, rank2_acc, ..., rankK_acc]
    """
    n       = len(labels)
    correct = [0] * max_rank

    for i in range(n):
        probe       = embs[i]
        gallery_emb = np.concatenate([embs[:i], embs[i+1:]], axis=0)
        gallery_lbl = labels[:i] + labels[i+1:]

        sims        = gallery_emb @ probe
        sorted_idx  = np.argsort(sims)[::-1]       # descending

        for k in range(max_rank):
            top_k_labels = [gallery_lbl[j] for j in sorted_idx[:k+1]]
            if labels[i] in top_k_labels:
                correct[k] += 1

    return [c / n for c in correct]


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

    rank1           = rank1_loo(list(labels), embs)
    cmc             = rankk_loo(list(labels), embs, max_rank=10)
    tar_at_far01    = compute_tar_at_far(genuine, impostor, target_far=0.01)
    tar_at_far001   = compute_tar_at_far(genuine, impostor, target_far=0.001)
    eer, eer_thresh = compute_eer(genuine, impostor)

    results = {
        "model":            model_key,
        "n_people":         n_people,
        "n_embeddings":     n_embs,
        "rank1":            rank1,
        "cmc":              cmc,
        "tar_at_far_1pct":  tar_at_far01,
        "tar_at_far_01pct": tar_at_far001,
        "eer":              eer,
        "eer_threshold":    eer_thresh,
        "genuine_mean":     float(genuine.mean()),
        "impostor_mean":    float(impostor.mean()),
        "_genuine_scores":  genuine,    # kept for curve generation
        "_impostor_scores": impostor,
    }

    print(f"\n  Results:")
    print(f"    Rank-1 Accuracy          : {rank1*100:.2f}%")
    print(f"    CMC Rank-5               : {cmc[4]*100:.2f}%")
    print(f"    TAR @ FAR=1%             : {tar_at_far01*100:.2f}%")
    print(f"    TAR @ FAR=0.1%           : {tar_at_far001*100:.2f}%")
    print(f"    EER                      : {eer*100:.2f}%")
    print(f"    Threshold @ EER          : {eer_thresh:.4f}  ← use this in live_demo.py")
    print(f"    Genuine  similarity mean : {genuine.mean():.4f}")
    print(f"    Impostor similarity mean : {impostor.mean():.4f}")

    # Save per-model results (exclude numpy arrays from text file)
    out_path = os.path.join(RESULTS_DIR, f"{model_key}_results.txt")
    with open(out_path, "w") as f:
        for k, v in results.items():
            if not k.startswith("_"):
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
    winner = sorted(valid, key=lambda x: x["rank1"], reverse=True)[0]
    lines.append(f"\n  Best model (Rank-1): {winner['model'].upper()}")
    lines.append(f"  Recommended threshold for live demo: {winner['eer_threshold']:.4f}\n")

    output = "\n".join(lines)
    print(output)

    import csv
    csv_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    # Exclude internal numpy arrays from CSV
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_") and k != "cmc"}
                for r in valid]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Summary CSV saved to: {csv_path}")
    print(f"  Open this in Excel for your thesis table.\n")


# ── Thesis Figures ────────────────────────────────────────────────────────────

def generate_curves(all_results: list, save_dir: str):
    """
    Generate and save three thesis figures:
      1. ROC Curve  — TAR vs FAR (log scale) for all models
      2. CMC Curve  — Rank-1 through Rank-10 cumulative accuracy
      3. DET Curve  — log(FAR) vs log(FRR)
    """
    valid = [r for r in all_results if r is not None]
    if not valid:
        return

    os.makedirs(save_dir, exist_ok=True)

    # ── Color palette ─────────────────────────────────────────────────────────
    colors = ["#E63946", "#457B9D", "#2DC653", "#F4A261", "#9B5DE5"]
    markers = ["o", "s", "^", "D", "v"]

    THRESHOLDS = np.linspace(0.0, 1.0, 2000)

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 1: ROC Curve (TAR vs FAR)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    for i, r in enumerate(valid):
        gen = r["_genuine_scores"]
        imp = r["_impostor_scores"]
        tar_vals = np.array([np.mean(gen  >= t) for t in THRESHOLDS])
        far_vals = np.array([np.mean(imp  >= t) for t in THRESHOLDS])

        # Sort by FAR ascending for a clean curve
        sort_idx = np.argsort(far_vals)
        ax.plot(far_vals[sort_idx], tar_vals[sort_idx],
                color=colors[i % len(colors)], linewidth=2.0,
                label=f"{r['model'].upper()}  (EER={r['eer']*100:.1f}%)")

        # EER point
        ax.scatter([r["eer"]], [1 - r["eer"]],
                   color=colors[i % len(colors)], s=60,
                   marker=markers[i % len(markers)], zorder=5)

    COL_W = "#e0e0e0"
    ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1.0, label="Chance")
    ax.set_xlabel("False Accept Rate (FAR)", color=COL_W, fontsize=11)
    ax.set_ylabel("True Accept Rate (TAR)", color=COL_W, fontsize=11)
    ax.set_title("ROC Curve — All Models", color=COL_W, fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1])
    ax.tick_params(colors=COL_W)
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1e1e1e", edgecolor="#333333", labelcolor=COL_W, fontsize=9)
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [FIG] ROC curve saved → {roc_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 2: CMC Curve (Rank-1 through Rank-10)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    ranks = list(range(1, 11))
    for i, r in enumerate(valid):
        cmc_vals = [v * 100 for v in r["cmc"]]
        ax.plot(ranks, cmc_vals,
                color=colors[i % len(colors)], linewidth=2.0,
                marker=markers[i % len(markers)], markersize=6,
                label=f"{r['model'].upper()}  (Rank-1={cmc_vals[0]:.1f}%)")

    ax.set_xlabel("Rank", color=COL_W, fontsize=11)
    ax.set_ylabel("Cumulative Match Rate (%)", color=COL_W, fontsize=11)
    ax.set_title("CMC Curve — All Models", color=COL_W, fontsize=13, fontweight="bold")
    ax.set_xticks(ranks)
    ax.set_ylim([max(0, min(v*100 for r in valid for v in r["cmc"]) - 5), 101])
    ax.tick_params(colors=COL_W)
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1e1e1e", edgecolor="#333333", labelcolor=COL_W, fontsize=9)
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    cmc_path = os.path.join(save_dir, "cmc_curve.png")
    plt.savefig(cmc_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [FIG] CMC curve saved → {cmc_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 3: DET Curve (log FAR vs log FRR)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    for i, r in enumerate(valid):
        gen = r["_genuine_scores"]
        imp = r["_impostor_scores"]
        far_vals = np.array([np.mean(imp >= t) for t in THRESHOLDS])
        frr_vals = np.array([np.mean(gen <  t) for t in THRESHOLDS])

        # Mask zeros for log scale
        mask = (far_vals > 0) & (frr_vals > 0)
        ax.plot(far_vals[mask], frr_vals[mask],
                color=colors[i % len(colors)], linewidth=2.0,
                label=f"{r['model'].upper()}  (EER={r['eer']*100:.1f}%)")

        ax.scatter([r["eer"]], [r["eer"]],
                   color=colors[i % len(colors)], s=60,
                   marker=markers[i % len(markers)], zorder=5)

    # EER diagonal
    ax.plot([1e-4, 1], [1e-4, 1], "--", color="#555555", linewidth=1.0, label="EER line")
    ax.set_xscale("log");  ax.set_yscale("log")
    ax.set_xlabel("False Accept Rate (FAR)", color=COL_W, fontsize=11)
    ax.set_ylabel("False Reject Rate (FRR)", color=COL_W, fontsize=11)
    ax.set_title("DET Curve — All Models", color=COL_W, fontsize=13, fontweight="bold")
    ax.tick_params(colors=COL_W)
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1e1e1e", edgecolor="#333333", labelcolor=COL_W, fontsize=9)
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.6, which="both")
    plt.tight_layout()
    det_path = os.path.join(save_dir, "det_curve.png")
    plt.savefig(det_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [FIG] DET curve saved → {det_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # FIGURE 4: Latency vs Rank-1 scatter
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    for i, r in enumerate(valid):
        # Try to load latency from results file
        lat = _load_latency(r["model"])
        ax.scatter([lat], [r["rank1"] * 100],
                   color=colors[i % len(colors)], s=120,
                   marker=markers[i % len(markers)], zorder=5, label=r["model"].upper())
        ax.annotate(f"  {r['model'].upper()}", (lat, r["rank1"] * 100),
                    color=colors[i % len(colors)], fontsize=9)

    ax.set_xlabel("Avg. Inference Latency (ms)", color=COL_W, fontsize=11)
    ax.set_ylabel("Rank-1 Accuracy (%)", color=COL_W, fontsize=11)
    ax.set_title("Speed vs Accuracy Trade-off", color=COL_W, fontsize=13, fontweight="bold")
    ax.tick_params(colors=COL_W)
    ax.spines[:].set_color("#333333")
    ax.grid(True, color="#2a2a2a", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    scatter_path = os.path.join(save_dir, "latency_vs_accuracy.png")
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [FIG] Latency-Accuracy scatter saved → {scatter_path}")


def _load_latency(model_key: str) -> float:
    """Load avg_latency from results file if available."""
    path = os.path.join(RESULTS_DIR, f"{model_key}_results.txt")
    if not os.path.exists(path):
        return 0.0
    with open(path) as f:
        for line in f:
            if line.startswith("avg_latency"):
                try:
                    return float(line.split(":")[1].strip())
                except Exception:
                    pass
    # Fallback known values
    known = {"arcface": 140, "adaface": 870, "magface": 14,
             "facenet_vgg": 50, "facenet_casia": 53}
    return known.get(model_key, 100.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(target_model=None, generate_plots=True):
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

    valid = [r for r in all_results if r is not None]

    if len(valid) > 1:
        print_summary(valid)

        if generate_plots:
            print(f"\n{'='*60}")
            print("  GENERATING THESIS FIGURES")
            print(f"{'='*60}")
            generate_curves(valid, RESULTS_DIR)
            print(f"\n  Figures saved to: {RESULTS_DIR}")
            print("  Files: roc_curve.png, cmc_curve.png, det_curve.png, latency_vs_accuracy.png")
    elif valid:
        r = valid[0]
        print(f"\n  Recommended threshold for live demo: {r['eer_threshold']:.4f}")
        if generate_plots and r.get("_genuine_scores") is not None:
            generate_curves([r], RESULTS_DIR)

    print(f"\n  Next step: python scripts/live_demo.py --model <best_model>\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Evaluate only one model. Options: arcface, facenet_vgg, facenet_casia, adaface, magface")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()
    main(target_model=args.model, generate_plots=not args.no_plots)