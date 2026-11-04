"""
gallery_tool.py
---------------
Read-only report on the face gallery in Supabase (table `embeddings`).

    python scripts/gallery_tool.py check

Shows:
  * customers enrolled with the current device pipeline (model_name = mbf-scrfd-v2, vault-protected)
  * customers whose only face data comes from an OLD pipeline (ResNet50, MobileFaceNet without
    alignment, or without the vault): the UNO Q ignores these rows, so the customer must be
    enrolled again from the CRM (Pelanggan -> Daftarkan wajah, or Wajah Terdaftar)
  * possible duplicate enrollments: two customers whose faces match each other (cosine >= tau)
  * active customers without any face

Enrolling and deleting faces is done in the CRM (with the customer's consent), not here.
DATABASE_URL is read from the environment or uno_q/.env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uno_q"))
import facerec_uno_q as fr  # noqa: E402  (also loads uno_q/.env)


def connect():
    import psycopg2
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set (environment or uno_q/.env).")
    return psycopg2.connect(url, connect_timeout=15)


def check(_args):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT c.customer_id::text, c.name, c.is_active, e.model_name, e.embedding_vector::text
                       FROM customers c LEFT JOIN embeddings e ON e.customer_id = c.customer_id
                       WHERE c.customer_id <> '00000000-0000-0000-0000-000000000001'""")
        rows = cur.fetchall()
    conn.close()

    current: dict[str, list] = {}
    names: dict[str, str] = {}
    old_only: dict[str, set] = {}
    no_face = []
    for cid, name, active, model, vec in rows:
        names[cid] = name or "-"
        if vec is None:
            if active:
                no_face.append(cid)
            continue
        if model == fr.MODEL_NAME:
            v = fr.parse_vec(vec)
            if v.shape == (fr.EMBED_DIM,):
                current.setdefault(cid, []).append(fr.l2n(v))
        else:
            old_only.setdefault(cid, set()).add(model or "?")
    for cid in list(old_only):
        if cid in current:
            del old_only[cid]

    print(f"\nCustomers: {len(names)}")
    print(f"  enrolled with the current pipeline ({fr.MODEL_NAME}) : {len(current)}")
    print(f"  only OLD face data -> enroll again from the CRM    : {len(old_only)}")
    for cid, models in sorted(old_only.items(), key=lambda kv: names[kv[0]]):
        print(f"    {names[cid]:24s} {cid}  ({', '.join(sorted(models))})")
    print(f"  active, no face                                    : {len(no_face)}")

    tau = fr.Config().match_threshold
    ids = sorted(current)
    dup = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = max(float(a @ b) for a in current[ids[i]] for b in current[ids[j]])
            if s >= tau:
                dup.append((s, ids[i], ids[j]))
    if dup:
        print(f"\nPossible duplicate enrollments (cosine >= tau = {tau:.2f}); delete one of each pair in the CRM:")
        for s, a, b in sorted(dup, reverse=True):
            print(f"  {s:.3f}  {names[a]} ({a[:8]})  <->  {names[b]} ({b[:8]})")
    else:
        print("\nNo duplicate enrollments.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    args = ap.parse_args()
    {"check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
