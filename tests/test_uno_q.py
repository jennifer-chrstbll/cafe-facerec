"""Unit tests for the UNO Q program (uno_q/facerec_uno_q.py). Run: python -m pytest tests -q"""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uno_q"))
os.environ.setdefault("VAULT_KEY", "ab" * 32)
sys.argv = ["test"]
import facerec_uno_q as f  # noqa: E402

RS = np.random.RandomState(0)


def vec(seed=None):
    r = np.random.RandomState(seed) if seed is not None else RS
    return f.l2n(r.randn(f.EMBED_DIM).astype(np.float32))


def near(base, noise=0.5, seed=None):
    """Another photo of the same face: cosine ~0.9 to `base`. Noise seeds are kept apart from face seeds."""
    r = np.random.RandomState(1000 + seed) if seed is not None else RS
    return f.l2n(base + noise * r.randn(f.EMBED_DIM).astype(np.float32) / np.sqrt(f.EMBED_DIM))


def face(x1, y1, x2, y2, score=0.9, yaw=0.0):
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    kps = np.array([[cx - w * 0.18, y1 + h * 0.4], [cx + w * 0.18, y1 + h * 0.4], [cx + yaw * w * 0.36, y1 + h * 0.58],
                    [cx - w * 0.14, y1 + h * 0.75], [cx + w * 0.14, y1 + h * 0.75]], np.float32)
    return f.Face(np.array([x1, y1, x2, y2], np.float32), score, kps)


def textured(h=480, w=640, level=130):
    img = np.full((h, w, 3), level, np.uint8)
    noise = np.random.RandomState(1).randint(-60, 60, (h, w, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ── quality gate ─────────────────────────────────────────────────────────────
def test_quality_accepts_a_clear_frontal_face():
    fr = textured()
    main, q, reason = f.quality(fr, [face(250, 120, 390, 300)])
    assert q is not None and reason == "" and 0 < q <= 1


def test_quality_reasons():
    fr = textured()
    assert f.quality(fr, [])[2] == "Menunggu wajah"
    assert f.quality(fr, [face(250, 120, 390, 300), face(420, 120, 550, 290)])[2] == "Beberapa wajah sama dekat"
    assert f.quality(fr, [face(300, 200, 330, 240)])[2] == "Terlalu jauh"                    # 30 px of 640
    assert f.quality(fr, [face(250, 120, 390, 300, yaw=0.9)])[2] == "Tampak samping"
    assert f.quality(textured(level=10), [face(250, 120, 390, 300)])[2] == "Terlalu gelap/terang"
    assert f.quality(fr, [face(0, 120, 140, 300)])[2] == "Wajah terpotong tepi"


def test_far_limit_is_48_px_of_640():
    fr = textured()
    assert f.quality(fr, [face(290, 200, 290 + 50, 260)])[2] != "Terlalu jauh"
    assert f.quality(fr, [face(290, 200, 290 + 44, 255)])[2] == "Terlalu jauh"


# ── decision rule (tau 0.40, margin 0.08, 60 % votes) ─────────────────────────
def session_of(embs):
    s = f.Session()
    for e in embs:
        s.samples.append(f.Sample(0.8, e, (0.5, 0.2)))
    return s


def gallery_of(people):
    g = f.Gallery()
    g.load([(cid, name, e) for cid, (name, e) in people.items()])
    return g


def test_known_person_is_recognised():
    a, b = vec(1), vec(2)
    g = gallery_of({"A": ("Andi", a), "B": ("Budi", b)})
    d = f.decide(f.Config(), g, session_of([near(a, seed=i) for i in range(6)]))
    assert d.state == "known" and d.customer_id == "A" and d.score >= 0.40


def test_stranger_is_unknown():
    g = gallery_of({"A": ("Andi", vec(1)), "B": ("Budi", vec(2))})
    d = f.decide(f.Config(), g, session_of([near(vec(9), seed=i) for i in range(6)]))
    assert d.state == "unknown" and d.customer_id is None


def test_face_between_two_people_is_not_given_a_name():
    a, b = vec(1), vec(2)
    mid = f.l2n(a + b)                                         # equally like both: margin ~0
    g = gallery_of({"A": ("Andi", a), "B": ("Budi", b)})
    d = f.decide(f.Config(), g, session_of([mid] * 6))
    assert d.state != "known"


def test_empty_gallery_is_unknown():
    d = f.decide(f.Config(), f.Gallery(), session_of([vec(1)] * 5))
    assert d.state == "unknown"


# ── face vault (cancelable template) ─────────────────────────────────────────
def test_vault_keeps_similarity_and_changes_the_vector():
    v = f.Vault("ab" * 32)
    a, b = vec(1), near(vec(1), seed=3)
    pa, pb = v.protect(a), v.protect(b)
    assert abs(float(a @ b) - float(pa @ pb)) < 1e-4            # rotation keeps cosine similarity
    assert float(a @ pa) < 0.5                                   # stored vector is not the raw one


def test_vaults_with_different_keys_do_not_match():
    a = vec(1)
    assert abs(float(f.Vault("ab" * 32).protect(a) @ f.Vault("cd" * 32).protect(a))) < 0.3


# ── face position sent to the Pi ─────────────────────────────────────────────
def test_face_geo_and_median():
    fr = textured()
    cx, w = f.face_geo(fr, face(320, 100, 448, 260))
    assert abs(cx - 0.6) < 1e-6 and abs(w - 0.2) < 1e-6
    s = [f.Sample(0.8, vec(1), (0.4, 0.2)), f.Sample(0.8, vec(1), (0.6, 0.3)), f.Sample(0.8, vec(1), (0.5, 0.25))]
    assert f.median_geo(s) == {"cx": 0.5, "w": 0.25}
    assert f.median_geo([f.Sample(0.8, vec(1))]) is None


def test_pi_link_sends_face_position():
    cfg = f.Config()
    link = f.PiLink(cfg, f.State())
    link.url = "http://127.0.0.1:9/wajah"
    link.send("cust", 0.77, "Andi", {"cx": 0.4, "w": 0.2})
    item = link.q.get_nowait()
    assert item["customer_id"] == "cust" and item["face"] == {"cx": 0.4, "w": 0.2}
    link.send("cust", 0.77, "Andi")                              # within the cooldown: not sent twice
    assert link.q.empty()


# ── enrolment sends an immediate event ───────────────────────────────────────
def test_successful_enrolment_tells_the_pi_at_once():
    eng = f.Engine(f.Config(), f.State())
    sent = []
    eng.pi = types.SimpleNamespace(send=lambda *a: sent.append(a))
    base = vec(5)
    rs = np.random.RandomState(7)
    eng.embed = lambda frame, fc: f.l2n(base + 0.03 * rs.randn(f.EMBED_DIM).astype(np.float32))
    eng.enroll_req, eng.enroll_t0, eng.enroll_sess = ("req", "cust-new"), __import__("time").time(), f.Session()
    fr = textured()
    fc = face(200, 100, 328, 260)
    for i in range(eng.cfg.enroll_frames):
        eng._step_enroll(fr, __import__("time").time(), [fc], fc, 0.5 + 0.02 * i, "")
    assert sent and sent[0][0] == "cust-new" and sent[0][3]["w"] == pytest.approx(0.2)


def test_enrolment_needs_a_close_face():
    eng = f.Engine(f.Config(), f.State())
    eng.pi = types.SimpleNamespace(send=lambda *a: None)
    eng.embed = lambda frame, fc: vec(5)
    eng.enroll_req, eng.enroll_t0, eng.enroll_sess = ("req", "c"), __import__("time").time(), f.Session()
    far = face(300, 200, 300 + 55, 270)                         # 55 px: recognisable, too far to enrol
    eng._step_enroll(textured(), __import__("time").time(), [far], far, 0.7, "")
    assert len(eng.enroll_sess.samples) == 0


# ── CSV log ─────────────────────────────────────────────────────────────────
def test_csv_log_writes_header_once(tmp_path):
    log = f.CsvLog(tmp_path / "x.csv", ["waktu", "a"])
    log.write([1])
    log.write([2])
    lines = (tmp_path / "x.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "waktu,a" and len(lines) == 3 and "+" in lines[1].split(",")[0][-6:]   # time zone kept
