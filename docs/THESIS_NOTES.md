# Thesis notes: face recognition, person tracking and identity association (for the defense)

Numbers come from `scripts/evaluate.py` on `dataset/` (20 people × 10 aligned photos) unless marked
otherwise. Latency numbers for Bab IV must be measured on the devices (section 7).

---

## 1. The system in one picture

```
CASHIER: face webcam ─ Arduino UNO Q        CEILING (~3 m): 1–2 area cameras ─ Raspberry Pi 5
  SCRFD-0.5G → align → MobileFaceNet          YOLO11n → ByteTrack + OSNet x0_5 → global ID
  → face vault → FAISS → session decision      → identity association (face → body)
        │ Customer ID, LAN :8765 (token) ──────────────▶ │
        └────────────── Supabase ◀────────────────────────┘ ◀── CRM (Vercel): Kasir, Kamera, Perangkat, …
```

- **UNO Q** (`uno_q/facerec_uno_q.py`): who is at the cashier → `recognition_logs`, opens the visit,
  sends the Customer ID to the Pi.
- **Raspberry Pi** (`DEPLOY/RaspberryPi5/cafe_pi.py`): tracks bodies, decides which body owns the
  recognised face, keeps the name on that body through occlusions and camera changes, and records
  entry/exit → **visit duration**.
- **CRM**: everything the staff and owner see and configure, including manual correction.

## 2. Models and why

| Stage | Model | Key facts | Why |
|---|---|---|---|
| Face detection | **SCRFD-0.5G** (`det_500m`, InsightFace) | box + score + **5 landmarks**; input 320×320 | fast on CPU (≈45 ms on the UNO Q at 320 vs 174 ms at 640), landmarks needed for alignment |
| Face embedding | **MobileFaceNet** (`w600k_mbf`) | 112×112 → 512-d, ArcFace loss, WebFace600K, 13.6 MB | built for mobile CPUs |
| Matching | **FAISS `IndexFlatIP`** | exact inner product on unit vectors = cosine | exact is right for tens–hundreds of people |
| Person detection | **YOLO11n** (ONNX, opset 17) | COCO class 0, **letterboxed** 640 input | smallest current YOLO; real-time on the Pi 5 CPU |
| Tracking | **ByteTrack** + appearance (BoT-SORT-style gating) | two-stage association (high + low scores) | keeps people through partial occlusion; no camera-motion compensation needed (fixed camera) |
| Re-identification | **OSNet x0_5** (dynamic batch) | 512-d appearance | measured best separation vs x0_25 / AIN x1_0 at this resolution |

**Why cosine:** ArcFace separates identities by angle on a hypersphere; cosine is what was optimised
(‖a−b‖² = 2 − 2cos θ on unit vectors).

**Why not ResNet50:** EER 1.11 % vs 1.89 % but 13× the size and ~13× the latency (174 MB, ~186 ms per
face on a PC). MobileFaceNet already meets every target.

**Runtime on the UNO Q: OpenCV DNN, not ONNX Runtime.** The onnxruntime aarch64 wheel stops with
"Illegal instruction" on the UNO Q (QRB2210, Cortex-A53 without ARMv8.2 dot-product; tested on the
board, exit code 132). OpenCV DNN gives **identical embeddings** (cosine 1.000000 vs onnxruntime on all
198 dataset faces). SCRFD is shipped with a fixed 320×320 input (`models/det_500m_320.onnx`) because
OpenCV 4.x cannot import its dynamic-shape graph.

## 3. Face pipeline, step by step

1. **Capture**: 640×480 MJPEG, a thread keeps only the newest frame (no buffer lag).
2. **Detect** with SCRFD (letterbox into 320×320, NMS 0.4, score ≥ 0.5).
3. **Main face**: the largest face; if a second face is almost as large (< 1.3× smaller) the frame is
   skipped ("several faces equally close").
4. **Quality** (unusable frames are dropped, the rest get a score 0..1): face width ≥ 10 % of the image,
   not cut off at the edge, |yaw| ≤ 0.45, brightness 35–235, Laplacian sharpness ≥ 6.
   Score = 0.45·facing + 0.25·sharpness + 0.15·size + 0.15·exposure.
5. **Align**: least-squares similarity transform (Umeyama) from the 5 landmarks to the ArcFace
   112×112 template (= insightface `norm_crop`).
6. **Embed** with MobileFaceNet, L2-normalise.
7. **Face vault**: multiply by a secret orthogonal matrix *R* (section 8). Cosine is unchanged.
8. **Session**: frames of the same face (cosine to the session mean ≥ 0.40) are collected until
   10 frames or 2.5 s, or the face is gone for 1 s.
9. **Decision** (one per session):
   - template = mean of the 3 best-quality frames; FAISS top-1 customer *c* with score *s₁*, best
     other customer *s₂*;
   - **known(c)** if *s₁* ≥ τ = 0.40 **and** *s₁* − *s₂* ≥ 0.08 **and** ≥ 60 % of the 5 best frames
     individually vote *c* (≥ τ);
   - **unknown** if *s₁* < 0.35; otherwise **not sure** (nothing recorded, next session decides).
10. **Outputs**: `recognition_logs` (known: at most every 30 s per customer, so the Kasir screen stays
    current; unknown: at most every 10 s), `get_or_create_visit()`, face event to the Pi.

Why sessions + margin + vote: averaging the best frames removes random errors (blur, motion, a bad
landmark); the **margin** rejects faces that are close to two customers (look-alikes); the **vote**
requires the frames to agree. The threshold handles strangers.

## 4. Threshold τ (likely question)

1:N open-set: a stranger is compared with *everyone*, so FAR grows with the gallery.

| τ | 1 enrolled | 5 | 10 | 15 | 19 enrolled |
|---|---|---|---|---|---|
| 0.26 (old ResNet50 value) | 3.6 % | 14.5 % | 27.6 % | 40.9 % | **47.8 %** |
| 0.33 | 0.6 % | 2.6 % | 4.9 % | 7.9 % | 9.5 % |
| 0.40 (deployed) | 0 % | 0 % | 0 % | 0 % | **0 %** |

Smallest τ with FAR ≤ 1 % on this data: 0.38; **0.40** is deployed for margin. Re-run with the
field-test participants (`python scripts/evaluate.py`).

## 5. Results (offline, `results/evaluation_w600k_mbf_fp32.json`)

| Metric | Single photo (τ 0.40) | **Device rule** (session, margin, vote) | Target |
|---|---|---|---|
| Rank-1 | 100 % | – | ≥ 90 % ✔ |
| Enrolled customer recognised correctly | 98.9 % | **99.85 %** | – |
| Wrong customer | 0 % | **0.00 %** | – |
| FRR | 1.14 % | **0.15 %** (all "not sure", none wrong) | ≤ 10 % ✔ |
| FAR (stranger accepted) | 0.00 % | **0.00 %** | ≤ 1 % ✔ |

Device-rule protocol: 5 embeddings enrolled per person (as the device stores), sessions of 3–5 of the
person's other photos, 10 splits × 20 sessions × 20 people = 4,000 sessions; strangers = same sessions
with that person removed from the gallery (leave-one-person-out).
1:1 verification: genuine mean 0.829, impostor mean 0.068, EER 1.89 %, TAR@FAR=0.1 % 97.1 %.

End-to-end test on a laptop with the real programs and a Postgres-protocol database (25 + 11 checks,
all passed): consent enrollment, duplicate refused, unknown face never accepted, recognition → visit →
face event to the Pi, self-check, live frames, manual assign/unassign, visit kept or removed correctly.

**Limits (say them):** 20 people, one session, good light, aligned photos. Field numbers will be
lower; that is what the field test measures (Log Wajah: the cashier's ✓/✗ is the ground truth).

## 6. Face → body (identity association, on the Pi)

For each body *j* in the camera that sees the cashier, at the frame of the face event:
foot point *p_j* (bottom centre of the box), body height *h_j*,
*d_j* = |*p_j* − *Q*|/*h_j*, *v_j* = speed/*h_j* (perspective normalisation: thresholds in body
heights work anywhere in the image, no calibration).

- Gate: inside the cashier circle drawn in the CRM, not staff, not carrying another customer.
- Likelihood *L_j* = exp(−*d_j*²/2σ_d²)·exp(−*v_j*²/2σ_v²), σ_d = 0.25, σ_v = 0.6.
- Posterior *P_j* = *L_j* / (Σ*L_k* + *L₀*), *L₀* = exp(−2.5²/2): the "customer not visible"
  hypothesis of probabilistic data association (Bar-Shalom & Fortmann, 1988). Staff and bodies of
  other customers are excluded from the normaliser because they are known not to be this customer.
- Bind if *P* ≥ 0.85 and the same body wins **2** face events in a row; else the cashier taps the
  person on the CRM *Kamera* page.

Worked example: customer at the counter (d ≈ 0.1) with the next one queuing ~0.9 m behind (d ≈ 0.55)
→ P ≈ 0.88 → bound. Two people within ~0.6 m → ambiguous → asked. Wrong identities cost a customer's
history; a question costs one tap.

The proposal's weighted sum S = 0.5·spatial + 0.3·kinematic + 0.2·temporal is the same idea; the
product-of-Gaussians + normaliser makes it a probability with an explicit "nobody" option and lets the
decision depend on the *other* people present (a queue), which a fixed θ on S cannot.
References: Bar-Shalom & Fortmann (1988); Hartley & Zisserman (2003) for foot points; Javed et al.
(ICCV 2003) for space-time + appearance across cameras; Kuhn (1955) Hungarian assignment (ByteTrack).

Keeping the name: strict mode re-checks the body's appearance every frame against the appearance at
binding time and drops the name rather than let it drift to someone else; recovery after an ID split
needs appearance + position + size consistency; staff areas and "one person, one place" across
cameras. Manual "Bukan dia" removes a wrong name immediately.

## 7. Latency (proposal ≤ 3 s) — measure on the devices

- UNO Q: `python3 benchmark_edge.py` (copied next to `facerec_uno_q.py`) → `results/edge_benchmark_<host>.json`.
  Board measurements so far: SCRFD@320 ≈ 45 ms. Time to decision = one session (≤ 2.5 s by design) +
  LAN to the Pi (ms) + Kasir screen refresh (2 s poll).
- Pi: `scripts/benchmark_edge_performance.py`. Earlier: YOLO11n 640 ≈ 93 ms, OSNet ≈ 13 ms (1 person).

**INT8 (the proposal says "kuantisasi INT8").** Static PTQ (`scripts/quantize_int8.py`): EER 1.89 % →
2.89 %, open-set FAR at τ 0.40: 0 % → 1.79 %, file 13.6 → 8.1 MB, and *slower* on x86 (18 → 36 ms).
OpenCV DNN on the UNO Q runs the FP32 model. Wording for the thesis: "FP32 deployed; INT8 was evaluated
and not adopted because accuracy dropped without a speed gain". INT8 and FP32 embeddings cannot be mixed.

## 8. Enrollment and privacy (UU PDP No. 27/2022)

- Enrollment **only from the CRM after explicit consent** (18+, checkbox). The old board program
  enrolled every unknown face automatically ("Pelanggan 0001") — removed: no consent, and it created
  duplicate identities when a known customer scored just under τ.
- The UNO Q captures 20 good frames, keeps the 5 best non-duplicates, checks they are one person
  (each ≥ 0.50 to their mean) and not already enrolled as someone else (≥ τ), stores 5 embeddings.
- **Face vault (cancelable biometrics, Ratha, Connell & Bolle 2001):** stored vectors are *R·f* with
  *R* a secret 512×512 orthogonal matrix derived from `VAULT_KEY` (HKDF-SHA256 → SHA-256 stream →
  Gaussian → QR). Cosine is preserved exactly, so accuracy is unchanged, but the stored numbers are not
  MobileFaceNet outputs: they cannot be matched with other systems or inverted with public models;
  changing the key revokes all templates. No photos are stored anywhere.
- Unknown faces live only in RAM for the session. Live pictures reach the website only while a page
  is open and are overwritten, never recorded.
- *Wajah Terdaftar* (owner): list, re-enroll, delete selected / all faces; retention (*Pengaturan →
  Privasi*); full erasure on the customer profile. Every deletion makes the UNO Q reload its gallery.

## 9. Likely questions

- **Why SCRFD, not MTCNN/RetinaFace?** Faster at similar accuracy, landmarks in one pass.
- **Why align?** The model was trained on aligned faces; the old guessed-landmark alignment gave the
  same face only ~0.26 similarity.
- **Why sessions instead of single frames?** Customers do not pose; the best frames of ~2 s are much
  better than any single frame, and the decision is made once, not flickering.
- **What if two customers look alike?** The margin rule refuses (not sure); `gallery_tool.py check`
  lists enrolled pairs ≥ τ.
- **ByteTrack or BoT-SORT?** ByteTrack association with BoT-SORT-style appearance gating; BoT-SORT's
  camera-motion compensation is unnecessary for fixed cameras and costs CPU.
- **Why is ReID run every frame?** Strict identity mode checks every frame that the body still looks
  like the person who was recognised; skipping frames would let a name drift during occlusions.
- **Internet drops?** Recognition and tracking keep running; visit events wait in the Pi's on-disk
  outbox and are sent in order later.

## 10. Before the field test

1. Rotate the Supabase password and purge the public repos' history (see the CRM README).
2. Run `database/005_devices.sql` in Supabase.
3. Deploy `DEPLOY/ArduinoUnoQ` and `DEPLOY/RaspberryPi5` (each has a README), back up `VAULT_KEY`.
4. In the CRM: *Perangkat* → choose cameras, *Cek perangkat*; *Tata Letak Kamera* → cashier point,
   staff area, door; *Wajah Terdaftar* → re-enroll customers marked "Perlu daftar ulang".
5. Measure latency on both devices; record ground truth (Log Wajah ✓/✗, stopwatch visit durations).
6. Raw photos: keep `dataset/` only for consenting participants' evaluation; delete on schedule.
