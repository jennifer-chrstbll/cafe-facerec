# cafe-facerec: cashier face recognition

The face-recognition part of the thesis *Perancangan Sistem Personalisasi Layanan Pelanggan Kafe
Berbasis Person Tracking dan Face Recognition secara Real-Time*. It runs on an **Arduino UNO Q**
with a USB webcam at the cashier, recognises enrolled customers passively, and writes
recognitions and visits to Supabase. The CRM dashboard (`cafe-crm`) and person tracking
(`cafe-persontrack`) use those rows.

```
webcam ─▶ SCRFD-0.5G ─▶ IoU tracker ─▶ quality gate ─▶ 5-pt alignment ─▶ MobileFaceNet ─▶ FAISS (cosine)
                                                                                         │
             Supabase ◀── recognition_logs + visits ◀── 3-of-5 vote ◀── τ = 0.40 ◀──────┘
                ▲                                                         
CRM ── /latest-embedding (enrollment), /video_feed (live view) ── UNO Q :5001
```

## Layout

| Path | What it is |
|---|---|
| `uno_q/` | **Deployable device code** (single file `facerec_uno_q.py`), see `uno_q/README.md` |
| `scripts/collect_dataset.py` | Capture aligned 112×112 photos of a participant (with consent) |
| `scripts/evaluate.py` | Rank-1, FAR, FRR, EER, ROC, threshold sweep, voting, FAR vs gallery size |
| `scripts/benchmark_edge.py` | Latency of each stage **on the UNO Q** |
| `scripts/quantize_int8.py` | Build and compare an INT8 MobileFaceNet |
| `scripts/gallery_tool.py` | Check / re-enroll embeddings stored in Supabase |
| `results/` | Outputs of the scripts above |
| `dataset/` | Participant photos, **local only** (git-ignored, biometric data) |
| `legacy/` | Superseded code and invalid old results, local only, safe to delete |

## Quick start (PC)

```bash
pip install -r requirements.txt
python scripts/evaluate.py                 # metrics on dataset/
python uno_q/facerec_uno_q.py              # run the agent on the PC webcam (reads uno_q/.env)
```

See `docs/THESIS_NOTES.md` for how every part works and the numbers to quote.
