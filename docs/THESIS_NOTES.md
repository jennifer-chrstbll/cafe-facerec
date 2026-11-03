# Thesis notes: cashier face recognition (for the defense)

Covers the face-recognition node of the system (Arduino UNO Q at the cashier). All numbers come
from `scripts/evaluate.py` on `dataset/` (20 people × 10 aligned photos) unless marked otherwise.

---

## 1. Where this module sits in the system

- **Cafe-area cameras** (Raspberry Pi 5): YOLO11 → ByteTrack → OSNet ReID → anonymous **Track ID**.
- **Cashier camera** (this repo, Arduino UNO Q): SCRFD → MobileFaceNet → FAISS → **Customer ID**.
- **Identity association** (CRM backend): links Track ID ↔ Customer ID with
  S = 0.5·spatial + 0.3·kinematic + 0.2·temporal; below θ = 0.60 the cashier confirms manually.
- The UNO Q writes `recognition_logs` (every decision) and `visits` (at most one per customer per
  30 min). The dashboard shows profile, history and top-3 recommendations.

## 2. The models and why they were chosen

| Stage | Model | Key facts | Why |
|---|---|---|---|
| Face detection | **SCRFD-0.5G** (`det_500m.onnx`, InsightFace) [8] | ~0.5 GFLOPs, 2.5 MB; predicts box + score + **5 landmarks** (eyes, nose, mouth corners) at strides 8/16/32 with 2 anchors per location | Fast on CPU, and the landmarks are needed for alignment (a plain box detector like OpenCV SSD has none) |
| Face embedding | **MobileFaceNet** (`w600k_mbf.onnx`, InsightFace *buffalo_sc*) [9] | 112×112 RGB in, **512-d** vector out; trained with **ArcFace loss** (additive angular margin) on WebFace600K; 13.6 MB | Designed for mobile CPUs, so it fits the UNO Q latency budget |
| Matching | **FAISS `IndexFlatIP`** [10] | Exact inner-product search; on L2-normalized vectors, inner product = **cosine similarity** | Exact (not approximate) is correct for a gallery of tens to hundreds of people; ≈0.03 ms per search |

**Why cosine similarity:** ArcFace training places identities on a hypersphere and separates them by
*angle*. Cosine of the angle is exactly what the network was optimised for; Euclidean distance on
normalised vectors is equivalent (‖a−b‖² = 2 − 2cos θ).

**Why not ResNet50 (more accurate)?** A genuine comparison on the same data (FP32, PC):

| Model | EER | TAR@FAR=0.1% | Size | Latency per face (PC) |
|---|---|---|---|---|
| MobileFaceNet (deployed) | 1.89% | 97.1% | 13.6 MB | ~14 ms |
| ArcFace ResNet50 (`w600k_r50`) | 1.11% | 96.1% | 174 MB | ~186 ms |

ResNet50 gains 0.8 percentage points of EER for 13× the size and ~13× the latency. On the UNO Q's
Cortex-A53 cores that would be well over a second per face. MobileFaceNet already meets every
target, so a full model comparison isn't needed for this thesis (the novelty is the integration).
This one table is enough to answer "why MobileFaceNet".

> The old `results/` 5-model comparison (ArcFace / AdaFace / MagFace / FaceNet ×2) was **not valid**:
> four of the five "models" silently loaded the same ResNet50 file, and MobileFaceNet wasn't
> in it. It's in `legacy/`; don't use it in the thesis.

## 3. The pipeline, step by step

1. **Capture.** 640×480 MJPEG from the webcam. A camera thread always keeps only the newest frame,
   so processing never works on a stale, buffered frame (this was the main "lag" before).
2. **Detection.** The frame is resized to 320×256 (keeps the 4:3 aspect, multiple of 32) and
   normalised ((x−127.5)/128, BGR→RGB). SCRFD outputs scores, box distances and landmark offsets
   relative to anchor centres; boxes with score ≥ 0.5 go through NMS (IoU 0.4).
3. **Tracking.** Greedy IoU matching (IoU ≥ 0.3) keeps the same **track** for the same face across
   frames; a track dies after 1 s without a detection. Detection runs every frame (cheap at 320), so
   the landmarks are always fresh.
4. **Quality gate.** A face is only recognised if:
   - its size is ≥ 60 px;
   - the detector score is ≥ 0.6;
   - it's roughly frontal: |yaw| ≤ 0.28 (nose offset from the eye midpoint divided by eye distance,
     ≈ 40° head turn) and pitch ratio between 0.2 and 0.8;
   - it's sharp: Laplacian variance of the aligned crop ≥ 25.

   Bad frames are skipped instead of producing wrong votes.
5. **Alignment.** A similarity transform (scale, rotation, translation; 4 degrees of freedom) is
   estimated by least squares (Umeyama) from the 5 detected landmarks to the fixed ArcFace 112×112
   template, then `warpAffine`. This is the same method as InsightFace's `norm_crop`, which the
   model was trained with.
6. **Embedding.** MobileFaceNet produces f ∈ ℝ⁵¹², then f ← f/‖f‖.
7. **1:N search.** FAISS returns the best gallery vector and its cosine score s.
8. **Open-set decision.** If s ≥ τ = 0.40 the vote is "customer X", otherwise "unknown". This is
   what lets the system reject people who never enrolled.
9. **Temporal voting.** Each track keeps its last **5** single-frame votes. A track becomes
   **known(X)** when ≥ **3** votes agree on X, or **unknown** when ≥ 3 votes are unknown.
   Undecided tracks are embedded every frame. Decided tracks are re-checked every 2 s, or
   immediately if a vote disagrees (catches two people swapping places).
10. **Logging.** When a track's decision changes: one `recognition_logs` row (customer_id or NULL,
    similarity score, `model_used = scrfd500m+mobilefacenet`, camera_id, recognized). For a known
    customer, also one `visits` row, unless the customer already has a visit in the last 30 min.
    Database work runs in a background thread with a queue and auto-reconnect, so a slow or offline
    database never freezes the camera.

## 4. Threshold τ: how it was chosen (likely examiner question)

- In **1:1 verification** (two photos, same person?) the EER point is τ = 0.274 (EER 1.89%).
- At the cashier it's **1:N open-set identification**. A stranger is compared against *every*
  enrolled person, and it's enough that *one* of them scores above τ. So **FAR grows with gallery
  size**:

| τ | 1 enrolled | 5 | 10 | 15 | 19 enrolled |
|---|---|---|---|---|---|
| 0.26 (old value) | 3.6% | 14.5% | 27.6% | 40.9% | **47.8%** |
| 0.33 | 0.6% | 2.6% | 4.9% | 7.9% | 9.5% |
| 0.40 (deployed) | 0% | 0% | 0% | 0% | **0%** |

- The old τ = 0.2579 was the *ResNet50* EER threshold. It was copied into the MobileFaceNet
  system and would have accepted about half of all strangers as some customer.
- τ is chosen as the smallest value with open-set FAR ≤ 1% (the proposal target): **0.38** on this
  data. **0.40** is deployed for margin, because the field test will have 30–50 participants.
  **Re-run `evaluate.py` with the field-test participants and recalibrate** (proposal step 3).

## 5. Results on the dataset (offline)

Protocol: each person is enrolled with an averaged template of 3 photos (like `/latest-embedding`);
the other 7 are probes; 10 random splits; 1,400 probes. FAR uses leave-one-person-out, so each
person acts as an unknown visitor once.

| Metric (τ = 0.40) | Single frame | After 3-of-5 vote | Proposal target |
|---|---|---|---|
| Rank-1 accuracy | 100% | – | ≥ 90% ✔ |
| FAR (stranger accepted) | 0.00% | 0.00% | ≤ 1% ✔ |
| FRR (customer rejected) | 1.14% | 0.07% | ≤ 10% ✔ |

1:1 verification: genuine mean 0.829, impostor mean 0.068, EER 1.89%, TAR@FAR=1% 98.0%,
TAR@FAR=0.1% 97.1%.

**Be honest about limits:** 20 people, photos captured in one session in good light, already
aligned. Field numbers (lighting, queues, partial occlusion, people walking) *will* be lower. That
is exactly what the field test measures.

**Voting nuance (good answer if asked):** if frame errors were independent, 3-of-5 voting would cut
a 2.7% per-frame FAR to ~0.02% (binomial). The simulation only reaches 0.9% at τ = 0.36, because
errors are *person-dependent*: a stranger who resembles a customer resembles them in every frame.
Voting removes **random** errors (blur, motion, a bad landmark). The **threshold** handles
**look-alikes**. The two are complementary.

## 6. Latency (proposal: end-to-end ≤ 3 s)

PC reference: SCRFD@320 ≈ 7 ms, MobileFaceNet ≈ 14–18 ms, FAISS ≈ 0.03 ms, one full recognition
step ≈ 26 ms. **The thesis must quote numbers measured on the UNO Q**
(`scripts/benchmark_edge.py --camera`). Expect several times slower on its Cortex-A53 CPU.
Time to a decision is about 3 recognition frames (the 3 votes) plus network time to Supabase and
the dashboard refresh, which should still be well inside 3 s.

Speed choices to explain:
- detector input 320 instead of 640 (~2–3× faster);
- detection every frame but embedding only for undecided faces;
- newest-frame camera thread with a 1-frame buffer and MJPEG;
- ONNX Runtime with 4 threads and no busy-spinning;
- database I/O off the main loop;
- JPEG encoding only while someone watches the stream.

## 7. INT8 quantization (the proposal says "kuantisasi INT8")

`scripts/quantize_int8.py`: static post-training quantization (ONNX Runtime, QOperator, INT8
weights / UINT8 activations), calibrated on the dataset, embedding head kept in FP32.

| | FP32 | INT8 |
|---|---|---|
| EER | 1.89% | 2.89% |
| Open-set FAR / FRR at τ = 0.40 | 0.00% / 1.14% | 1.79% / 1.43% |
| Recommended τ | 0.38 | 0.41 |
| File size | 13.6 MB | 8.1 MB |
| Speed on PC (x86) | ~18 ms | ~36 ms (slower) |
| cos(FP32, INT8) same face | – | 0.88 |

**Conclusion so far:** INT8 costs accuracy and isn't faster on x86. Decide using the **UNO Q
benchmark** (`benchmark_edge.py --int8 ...`). If it's not clearly faster there, keep FP32 and
change the proposal/thesis wording to "FP32; INT8 was evaluated and not adopted because…".
INT8 and FP32 embeddings **can't be mixed**: the whole gallery must be enrolled with the model the
device runs.

## 8. Enrollment and privacy (UU PDP)

- Enrollment: the cashier asks; the CRM calls `GET /latest-embedding`. The device returns the
  **average of up to 10 good-quality frames** of the largest face in view (≥ 3 frames, seen within
  the last 5 s). If that face is already recognised as a customer it returns **409**, which
  prevents duplicate customer IDs for one person.
- **Only the 512-d embedding** is stored (Supabase `embeddings`). No image is stored by the device.
- Unknown faces exist only in RAM, for the life of the track plus 5 s for enrollment. Nothing about
  them is written except a log row with `customer_id = NULL` and a score.
- An embedding can't be turned back into a photo, but it is still biometric personal data.
- Deleting a customer deletes their embeddings (FK `ON DELETE CASCADE`). The device reloads the
  gallery every 120 s, and 5 s after each enrollment.

## 9. API on the UNO Q (port 5001)

| Endpoint | Use |
|---|---|
| `GET /video_feed` | MJPEG live view with boxes: green = known (name + score), orange = unknown, grey = recognizing / quality message |
| `GET /latest-embedding` | Enrollment template (404 no good face, 409 already registered) |
| `POST /extract-embedding` | Embedding from an uploaded photo |
| `POST /recognize` | 1:N result for an uploaded photo |
| `POST /reload-gallery` | Reload embeddings now |
| `GET /health` | FPS, timings, gallery size, τ, voting and quality settings |

## 10. Likely questions

- **Why SCRFD and not MTCNN/RetinaFace?** Much faster at similar accuracy on small faces, and it
  outputs the 5 landmarks needed for alignment in one pass.
- **Why align?** The model was trained on aligned faces. Without alignment, rotation and scale
  changes lower genuine scores (the pre-September version of this code had no alignment).
- **Why FAISS for so few people?** It's exact (`IndexFlatIP`), so results equal brute-force cosine
  and it scales if the gallery grows; a numpy fallback gives identical results.
- **What if two customers look alike?** The threshold and voting reduce it. Below the association
  threshold the cashier confirms manually. `gallery_tool.py check` flags pairs of enrolled
  customers whose similarity is ≥ τ.
- **What about twins, masks, heavy occlusion?** Known limitation (roadmap year 2).
- **What if the internet drops?** Recognition keeps running. Database writes are queued and
  retried; the gallery stays in memory.
- **Why 3 of 5?** It's a trade-off between speed (3 frames ≈ well under 1 s) and stability. It's a
  parameter that can be calibrated on site (`VOTE_WINDOW`, `VOTE_MIN`).

## 11. To do before the field test and defense

1. **Rotate the Supabase database password now.** It was committed in `.env.uniq`, and the GitHub
   repo is public. Then purge history: add `.env.uniq` and `legacy/` to the existing
   `purge_cafe-facerec_history.sh` and run it.
2. **Re-enroll** every customer enrolled before 2026-09-12 (`gallery_tool.py check`). Those
   embeddings came from ResNet50 or from un-aligned crops and won't match.
3. Deploy `uno_q/` to the board, measure `benchmark_edge.py` there, and put those numbers in Bab IV.
4. CRM integration (done in `cafe-crm`): enrollment now goes through `enrollment_requests`
   (the UNO Q writes `model_name = 'mobilefacenet'`), consent is an explicit required step,
   duplicates are refused, and after each confirmed customer the UNO Q calls the CRM's
   identity-association endpoint. Contract: `cafe-crm/docs/INTEGRATION.md`.
5. Set `CRM_URL` and `DEVICE_API_KEY` in `uno_q/.env` (same key as in Vercel).
6. Raw photos: the proposal says raw face images are deleted within 24 h of extraction. Keep
   `dataset/` for consenting participants' offline evaluation only, and delete it on schedule.
