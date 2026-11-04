# cafe-facerec: cashier face recognition

The face-recognition part of the thesis *Perancangan Sistem Personalisasi Layanan Pelanggan Kafe
Berbasis Person Tracking dan Face Recognition secara Real-Time*. It runs on an **Arduino UNO Q**
with a USB webcam at the cashier, recognises enrolled customers passively, and writes
recognitions and visits to Supabase. The CRM dashboard (`cafe-crm`) and person tracking
(`cafe-persontrack`) use those rows.

```
webcam ─▶ SCRFD-0.5G ─▶ quality score ─▶ 5-pt alignment ─▶ MobileFaceNet ─▶ face vault ─▶ FAISS (cosine)
                                                                                          │
   session of ≤10 frames: mean of 3 best ≥ τ 0.40, margin ≥ 0.08, ≥60 % votes ◀────────────┘
        │                                   │
        ├─▶ Supabase: recognition_logs, visits, embeddings (enrollment from the CRM, with consent)
        └─▶ Raspberry Pi :8765  (Customer ID → the body at the cashier, person tracking)
```

Inference runs on OpenCV DNN (onnxruntime crashes with "Illegal instruction" on the UNO Q CPU).
The whole system (UNO Q + Raspberry Pi + CRM) is described in `cafe-crm/docs/INTEGRATION.md`.

## Layout

| Path | What it is |
|---|---|
| `uno_q/` | **Deployable device code** (single file `facerec_uno_q.py`), see `uno_q/README.md` |
| `scripts/collect_dataset.py` | Capture aligned 112×112 photos of a participant (with consent) |
| `scripts/evaluate.py` | Rank-1, FAR, FRR, EER, ROC, threshold sweep, FAR vs gallery size, and the device's exact session rule |
| `scripts/benchmark_edge.py` | Latency of each stage **on the UNO Q** |
| `scripts/quantize_int8.py` | Build and compare an INT8 MobileFaceNet |
| `scripts/gallery_tool.py` | Report on the face gallery in Supabase (old-model faces, duplicates) |
| `results/` | Outputs of the scripts above |
| `dataset/` | Participant photos, **local only** (git-ignored, biometric data) |
| `legacy/` | Superseded code and invalid old results, local only, safe to delete |

## Quick start (PC)

```bash
pip install -r requirements.txt
python scripts/evaluate.py                 # metrics on dataset/
python uno_q/facerec_uno_q.py              # run the device program on the PC webcam (reads uno_q/.env)
```

See `docs/THESIS_NOTES.md` for how every part works and the numbers to quote.
