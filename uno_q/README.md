# Arduino UNO Q — cashier face-recognition agent

Everything that runs on the UNO Q is in this folder:

| File | Purpose |
|---|---|
| `facerec_uno_q.py` | The agent (detector, alignment, recognizer, FAISS, voting, Supabase, HTTP server) |
| `.env.example` | Configuration template → copy to `.env` |
| `requirements.txt` | Python packages |
| `facerec.service` | systemd unit (auto-start at boot) |
| `models/` | Optional: put `det_500m.onnx` + `w600k_mbf.onnx` here for offline install |

It replaces the old `camera_agent.py`. Same port (5001) and the same endpoints the CRM already
calls (`/video_feed`, `/latest-embedding`, `/extract-embedding`), so the CRM needs no change.

## Install

From the PC (in the repo root):

```bash
scp -r uno_q arduino@192.168.18.80:~/facerec
# optional offline models (otherwise downloaded automatically on first start, ~15 MB):
scp ~/.insightface/models/buffalo_sc/det_500m.onnx ~/.insightface/models/buffalo_sc/w600k_mbf.onnx \
    arduino@192.168.18.80:~/facerec/models/
```

On the UNO Q (`ssh arduino@192.168.18.80`):

```bash
cd ~/facerec
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env          # set DATABASE_URL
pkill -f camera_agent.py || true           # stop the old agent (port 5001)
venv/bin/python facerec_uno_q.py           # test run, Ctrl+C to stop
```

Check `http://192.168.18.80:5001/video_feed` (live overlay) and `http://192.168.18.80:5001/health`.

Auto-start at boot:

```bash
sudo cp facerec.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now facerec
journalctl -u facerec -f
```

## How it talks to the rest of the system

| What | How |
|---|---|
| Recognitions, visits | rows in Supabase `recognition_logs`, `visits` |
| Face enrollment from the CRM | CRM inserts `enrollment_requests`; this agent polls it every 1.5 s, captures the face in front of the camera and writes `embeddings` itself (the cloud website cannot reach the board on the cafe LAN) |
| Status light in the CRM | heartbeat row in `device_status` every 30 s (FPS, gallery size) |
| Identity association | after each confirmed customer: `POST {CRM_URL}/api/association/bind` with header `x-device-key` |
| Live video | `http://<uno-q-ip>:5001/video_feed`, only reachable on the cafe Wi-Fi |

Requires `cafe-crm/database/004_integration.sql` to be applied; without it enrollment and
heartbeat switch themselves off with a warning and recognition keeps working.

## Before the field test

1. **Gallery check:** `venv/bin/python ../scripts/gallery_tool.py check` (copy `scripts/` too, or
   run it on any machine that reaches the database). Customers enrolled before 2026-09-12 were
   made with an older model and must be re-enrolled:
   `gallery_tool.py reenroll --customer-id <uuid> --agent http://localhost:5001`.
2. **Latency on the device:** `scripts/benchmark_edge.py --camera` → `results/edge_benchmark_<host>.json`.
3. **Calibration on site** (proposal step 3):
   - `MATCH_THRESHOLD`: run `scripts/evaluate.py` on the participants' photos; use the
     recommended τ (open-set FAR ≤ 1%). Re-run when the gallery grows, since FAR rises with the
     number of enrolled people.
   - `MIN_FACE_PX`: stand where customers pay; faces there should be ≥ this size
     (see the box on `/video_feed`, grey "move closer" means too small).
   - `MIN_SHARPNESS` / `MAX_YAW`: lower them if faces at the counter stay on
     "blurry"/"face the camera" in the cafe lighting.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not translate host name db.<ref>.supabase.co` | The direct host is IPv6-only. Use the *Session pooler* connection string from Supabase → Project Settings → Database. |
| `Address already in use` | Old `camera_agent.py` or a second copy is running: `pkill -f camera_agent.py`. |
| Low FPS | Keep `DET_SIZE=320`, `NUM_THREADS=4`; close the stream when not needed (encoding only runs while someone watches). |
| Everyone "Unknown" | Gallery empty or legacy embeddings → `gallery_tool.py check`; check `/health` → `gallery`. |
