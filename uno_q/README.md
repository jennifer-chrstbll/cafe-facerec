# Arduino UNO Q — cashier face recognition

Everything that runs on the UNO Q is in this folder.

| File | Purpose |
|---|---|
| `facerec_uno_q.py` | the program: SCRFD-0.5G + 5-point alignment + MobileFaceNet + face vault + FAISS + session decision; talks to Supabase and the Raspberry Pi |
| `models/det_500m_320.onnx` | SCRFD with a fixed 320×320 input (OpenCV 4.x cannot load the original). MobileFaceNet (`w600k_mbf.onnx`) is taken from `~/.insightface/models/buffalo_sc/` (already on the board) or downloaded once |
| `.env.example` | settings template → copy to `.env` |
| `requirements.txt` | Python packages (no onnxruntime: it crashes with "Illegal instruction" on this CPU) |
| `facerec.service` | systemd unit (start at boot, restart after a crash or "Mulai ulang" in the CRM) |

## Hardware

Face webcam → **powered USB‑C hub** → UNO Q. The UNO Q joins the cafe Wi‑Fi (set up once in
Arduino App Lab). It does not need a cable to the Raspberry Pi.

## Install (once)

From the laptop, with the board on USB (Arduino App Lab closed):

```powershell
$adb = "C:\Users\USER\AppData\Local\Android\Sdk\platform-tools\adb.exe"
& $adb shell "mkdir -p /home/arduino/facerec/old && cd /home/arduino/facerec && mv -f camera_agent.py* daftar_wajah.py* penyimpanan_lokal.py* titik_wajah.py* face_recognition_module.py models.py old/ 2>/dev/null; true"
& $adb push D:\Projects\DEPLOY\ArduinoUnoQ\. /home/arduino/facerec/
```

On the board (`& $adb shell`, or SSH over Wi‑Fi):

```bash
cd ~/facerec
~/facerec-env/bin/pip install -r requirements.txt     # the existing venv already has most of it
cp .env.example .env && nano .env                     # DATABASE_URL, VAULT_KEY, PI_FACE_URL, FACE_TOKEN
sudo timedatectl set-ntp true                         # correct clock (needs Wi-Fi)
~/facerec-env/bin/python facerec_uno_q.py             # test run; Ctrl+C to stop
sudo cp facerec.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now facerec
```

The Raspberry Pi used to start the old `camera_agent.py` over USB. Switch that off on the Pi
(see the Pi README): `sudo systemctl disable --now cafe-wajah-unoq cafe-jembatan-usb`.

## What it does

| Event | Result |
|---|---|
| A face is in front of the cashier | a *session* of up to 10 frames (≤ 2.5 s) of the same face; frames that are cut off, too small, in profile, too dark/bright or blurry are ignored |
| End of the session | **known** when the mean of the 3 best frames ≥ τ = 0.40, it beats the next customer by ≥ 0.08, and ≥ 60 % of the best frames vote for the same customer. **unknown** when the best score < 0.35. Otherwise **not sure** (nothing is recorded) |
| Known | `recognition_logs` row (at most every 30 s per customer), the customer's visit is opened (`get_or_create_visit`), and the face event is sent to the Pi (`PI_FACE_URL`), which links it to the body at the cashier |
| CRM "Daftarkan wajah" (after consent) | the board captures 20 good frames, keeps the 5 best, checks they are the same person and not already enrolled as someone else, stores them (vault‑protected) |
| CRM "Perangkat" | heartbeat every 10 s, self-check, reload gallery, restart, choose the webcam, live preview (only while someone watches) |

Measured on the project dataset with this exact rule (`scripts/evaluate.py`):
enrolled customers 99.85 % correct, 0 % wrong person; unknown faces 0 % accepted.

## Face vault

Stored embeddings are multiplied by a secret orthogonal matrix derived from `VAULT_KEY`
(cancelable biometrics). Cosine similarity — and therefore recognition — is unchanged, but the
stored numbers are not MobileFaceNet outputs and cannot be matched against other systems.
**Back up `VAULT_KEY`.** If it is lost, every customer must be enrolled again.
Faces enrolled with an older pipeline (`model_name` ≠ `mbf-scrfd-v2`) are ignored; the CRM
page *Wajah Terdaftar* lists them for re-enrollment.

## Troubleshooting

| Symptom (CRM → Perangkat → Cek perangkat) | Fix |
|---|---|
| Kamera wajah: tidak ada gambar | replug the webcam (hub must be powered), pick it again in *Perangkat* |
| Raspberry Pi: gagal | `PI_FACE_URL` must be the Pi's IP; same `FACE_TOKEN` on both |
| Jam papan: selisih | `sudo timedatectl set-ntp true` (needs Wi‑Fi) |
| Customer always "not sure" | re-enroll in good light; check the camera is at face height, 0.5–1.5 m |
| Log: `Illegal instruction` | something imported onnxruntime; this program does not — use `facerec_uno_q.py` |
