"""
gallery_tool.py
---------------
Inspect and repair the face gallery in Supabase (table `embeddings`).

    python3 scripts/gallery_tool.py check
        Read-only report: which embeddings were made by an older pipeline (ResNet50 before
        Sep 2026, or MobileFaceNet without alignment before 2026-09-12), wrong dimensions,
        customers that look enrolled twice (two customer_ids with the same face), and
        customers without any face.

    python3 scripts/gallery_tool.py reenroll --customer-id <uuid> [--agent http://192.168.18.80:5001]
        Customer stands in front of the cashier camera; their old embeddings are replaced by
        a fresh MobileFaceNet template from the agent's /latest-embedding, then the agent
        reloads its gallery.

DATABASE_URL is read from the environment or uno_q/.env. Run it where the database host is
reachable (e.g. on the UNO Q itself).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
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


def check(args):
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""SELECT c.customer_id::text, c.name, c.is_active, e.model_name, e.created_at,
                              e.embedding_vector::text
                       FROM customers c LEFT JOIN embeddings e ON e.customer_id = c.customer_id
                       ORDER BY e.created_at NULLS FIRST""")
        rows = cur.fetchall()
    conn.close()

    no_face, legacy, bad_dim, good = [], [], [], []
    for cid, name, active, model, created, vec in rows:
        if vec is None:
            no_face.append((cid, name))
            continue
        v = fr.parse_vector(vec)
        day = str(created)[:10] if created else "unknown"
        if v.shape != (fr.EMBED_DIM,):
            bad_dim.append((cid, name, v.shape))
        elif day == "unknown" or day < fr.PIPELINE_SINCE:
            legacy.append((cid, name, day, active))
        else:
            good.append((cid, name, fr.l2_normalize(v), active))

    print(f"\nCustomers: {len({r[0] for r in rows})} | embeddings: {sum(r[5] is not None for r in rows)}")
    print(f"  OK (MobileFaceNet + alignment, since {fr.PIPELINE_SINCE}) : {len(good)}")
    print(f"  LEGACY (older model/pipeline, re-enroll)               : {len(legacy)}")
    print(f"  wrong dimension                                        : {len(bad_dim)}")
    print(f"  customers without a face                               : {len(no_face)}")
    for cid, name, day, active in legacy:
        print(f"    re-enroll: {name or '-':20s} {cid}  enrolled {day}{'' if active else '  (inactive)'}")

    # Two different customer_ids with the same face -> the gallery would flip between them.
    tau = fr.Config().match_threshold
    dup = []
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            if good[i][0] != good[j][0]:
                s = float(good[i][2] @ good[j][2])
                if s >= tau:
                    dup.append((s, good[i][1], good[i][0], good[j][1], good[j][0]))
    if dup:
        print(f"\nPossible duplicate enrollments (cosine >= tau={tau:.2f}):")
        for s, n1, c1, n2, c2 in sorted(dup, reverse=True):
            print(f"  {s:.3f}  {n1} ({c1[:8]})  <->  {n2} ({c2[:8]})")
    print("\nTip: set GALLERY_SINCE=" + fr.PIPELINE_SINCE + " in uno_q/.env to ignore legacy rows "
          "until they are re-enrolled.")


def http_json(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def reenroll(args):
    status, data = http_json(f"{args.agent}/latest-embedding")
    if status != 200:
        sys.exit(f"Agent returned {status}: {data.get('detail')}")
    emb = data["embedding"]
    if len(emb) != fr.EMBED_DIM:
        sys.exit("Unexpected embedding size from agent.")
    print(f"Got template from {data.get('n_frames')} frames, faces in view: {data.get('faces_in_view')}")
    if data.get("faces_in_view", 1) > 1:
        print("WARNING: more than one face in view; the largest face was used.")

    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM customers WHERE customer_id = %s::uuid", (args.customer_id,))
        row = cur.fetchone()
        if not row:
            sys.exit("Customer not found.")
        cur.execute("SELECT count(*) FROM embeddings WHERE customer_id = %s::uuid", (args.customer_id,))
        n_old = cur.fetchone()[0]
        if not args.yes:
            ans = input(f"Replace {n_old} embedding(s) of '{row[0]}' with the new template? [y/N] ")
            if ans.strip().lower() != "y":
                sys.exit("Cancelled.")
        cur.execute("DELETE FROM embeddings WHERE customer_id = %s::uuid", (args.customer_id,))
        cur.execute("""INSERT INTO embeddings (embedding_id, customer_id, embedding_vector, model_name,
                                               is_primary, created_at)
                       VALUES (gen_random_uuid(), %s::uuid, %s, %s, TRUE, NOW())""",
                    (args.customer_id, json.dumps(emb), "mobilefacenet"))
    conn.commit()
    conn.close()
    http_json(f"{args.agent}/reload-gallery", "POST")
    print(f"Re-enrolled '{row[0]}'. Agent gallery reload requested.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("reenroll")
    r.add_argument("--customer-id", required=True)
    r.add_argument("--agent", default="http://localhost:5001")
    r.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    {"check": check, "reenroll": reenroll}[args.cmd](args)


if __name__ == "__main__":
    main()
