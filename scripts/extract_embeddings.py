"""
extract_embeddings.py
---------------------
Reads every photo in dataset/<person>/ folders,
runs them through all 5 models, and saves embeddings to embeddings/.

Output structure:
    embeddings/
        arcface/
            Jennifer.npy      <- shape (10, 512)
            Teo.npy
        facenet/
            Jennifer.npy
        ...

Usage:
    python scripts/extract_embeddings.py
    python scripts/extract_embeddings.py --model arcface   (single model)
"""

import os
import sys
import argparse
import time
import traceback
import numpy as np
import cv2
from tqdm import tqdm

# ── Make sure models.py in project root is importable ─────────────────────────
PROJECT_ROOT = r"D:\Projects\cafe_facerec"
sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR    = os.path.join(PROJECT_ROOT, "dataset")
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")

# ──────────────────────────────────────────────────────────────────────────────

def get_person_images(person_name: str):
    """Return list of BGR images for a person."""
    folder = os.path.join(DATASET_DIR, person_name)
    images = []
    # NOTE: images in dataset/ are expected to be pre-aligned 112x112
    # crops from collect_dataset.py (norm_crop). Do not mix with raw photos.
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, fname)
            img  = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images


def extract_for_model(model_key: str, model_instance, people: list):
    """
    Extract embeddings for all people using one model.
    Saves embeddings/<model_key>/<person>.npy  (shape: N x 512)
    """
    out_dir = os.path.join(EMBEDDINGS_DIR, model_key)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for person in tqdm(people, desc=f"  {model_key}", ncols=70):
        images = get_person_images(person)
        if not images:
            print(f"    [WARN] No images found for {person}, skipping.")
            continue

        embs   = []
        failed = 0
        times  = []

        for img in images:
            try:
                t0  = time.perf_counter()
                emb = model_instance.get_embedding(img)
                t1  = time.perf_counter()
                if emb is not None:
                    embs.append(emb)
                    times.append((t1 - t0) * 1000)
                else:
                    failed += 1
            except Exception as img_err:
                print(f"    [IMG ERROR] {img_err}")
                traceback.print_exc()
                failed += 1

        if embs:
            arr      = np.stack(embs)
            out_path = os.path.join(out_dir, f"{person}.npy")
            np.save(out_path, arr)
            avg_ms   = np.mean(times)
            summary[person] = {
                "saved":  len(embs),
                "failed": failed,
                "avg_ms": avg_ms,
            }
        else:
            print(f"    [ERROR] All embeddings failed for {person} with {model_key}.")
            summary[person] = {"saved": 0, "failed": failed, "avg_ms": 0}

    return summary


def main(target_model: str = None):
    people = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not people:
        print("[ERROR] No person folders found in dataset/. Collect photos first.")
        return

    print(f"\n[INFO] Found {len(people)} person(s): {people}")
    print(f"[INFO] Saving embeddings to: {EMBEDDINGS_DIR}\n")

    from models import ALL_MODELS, load_model

    model_keys = [target_model] if target_model else list(ALL_MODELS.keys())

    all_results = {}

    for key in model_keys:
        print(f"\n{'='*50}")
        print(f"  Model: {key.upper()}")
        print(f"{'='*50}")
        try:
            model   = load_model(key)
            summary = extract_for_model(key, model, people)
            all_results[key] = summary
        except Exception as e:
            print(f"  [ERROR] Failed to load or run {key}: {e}")
            traceback.print_exc()
            continue

        print(f"\n  Results for {key}:")
        for person, info in summary.items():
            print(f"    {person:20s}  saved={info['saved']}  "
                  f"failed={info['failed']}  avg_latency={info['avg_ms']:.1f}ms")

    print(f"\n\n{'='*60}")
    print("  EXTRACTION COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'Total saved':>12} {'Avg latency (ms)':>18}")
    print(f"  {'-'*15} {'-'*12} {'-'*18}")

    for key, summary in all_results.items():
        total_saved = sum(v["saved"] for v in summary.values())
        all_times   = [v["avg_ms"] for v in summary.values() if v["avg_ms"] > 0]
        avg_lat     = np.mean(all_times) if all_times else 0
        print(f"  {key:<15} {total_saved:>12} {avg_lat:>17.1f}ms")

    print(f"\n  Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"  Next step: run  python scripts/evaluate.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        help="Run only one model. Options: arcface, facenet, insightface, adaface, magface"
    )
    args = parser.parse_args()
    main(target_model=args.model)