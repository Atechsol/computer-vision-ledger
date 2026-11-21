# Credit Watchlist — face-alert prototype

A local, single-machine prototype for a shop owner. Credit buyers are enrolled from a photo together with a ledger of what they owe. When the camera sees an enrolled buyer whose outstanding balance exceeds a threshold, the dashboard raises an alert showing the enrolled photo, a live snapshot, the name and the amount due.

Everything runs locally: no cloud calls, no Docker, no external services after the one-time model download.

> **Status:** prototype. The API, pipeline and UI are implemented and smoke-tested, but recognition accuracy has not yet been validated on real faces, lighting conditions or the shop's entrance camera. See [Limitations](#limitations-and-known-gaps).

---

## Contents

1. [Architecture](#architecture)
2. [Runtime model (threads and data flow)](#runtime-model)
3. [Recognition pipeline](#recognition-pipeline)
4. [Alert decision logic](#alert-decision-logic)
5. [Data model](#data-model)
6. [Design decisions](#design-decisions)
7. [Methodology](#methodology)
8. [API](#api)
9. [Configuration](#configuration)
10. [Setup and run](#setup-and-run)
11. [Project layout](#project-layout)
12. [Limitations and known gaps](#limitations-and-known-gaps)
13. [Privacy](#privacy)

---

## Architecture

```
                        ┌──────────────────────────── one Python process (uvicorn) ───────────────────────────┐
                        │                                                                                      │
 Webcam / RTSP ──────►  │  Camera thread ──► latest frame (lock-guarded, overwritten, never queued)            │
 cv2.VideoCapture       │        │                                                                             │
                        │        ├────────────────────────┐                                                    │
                        │        ▼                        ▼                                                    │
                        │  Pipeline thread (~8 fps)   MJPEG generator (≤25 fps, asyncio)                       │
                        │   detect → embed → match      latest frame + latest boxes → JPEG → /video            │
                        │   → alert rules                                                                      │
                        │        │  boxes (shared)                                                             │
                        │        ├────────────────────────►┘                                                   │
                        │        │ fire                                                                        │
                        │        ▼                                                                             │
                        │  SQLite  ◄── FastAPI routes (REST) ◄──────────── Browser dashboard (vanilla JS)      │
                        │  (buyers, embeddings, ledger,        ▲                    ▲                          │
                        │   alerts, settings)                  │                    │                          │
                        │        └─► run_coroutine_threadsafe ─► WebSocket hub ─────┘  (alert push)            │
                        └──────────────────────────────────────────────────────────────────────────────────────┘

 Enrollment (photo upload) ──► FastAPI ──► same FaceEngine ──► embedding(s) ──► SQLite
```

| Module | Responsibility |
|---|---|
| `app/camera.py` | Capture thread; always holds only the newest frame; auto-reconnects. |
| `app/face.py` | `FaceEngine`: `detect(frame)` and `embed(frame, face)`. The only module that knows about YuNet/SFace. |
| `app/matcher.py` | Cosine match of one embedding against all enrolled embeddings. |
| `app/alerts.py` | N-of-M frame confirmation and per-buyer cooldown. |
| `app/pipeline.py` | The live loop; owns the per-frame box overlay shown in the video stream. |
| `app/enroll.py` | Photo → validated embedding(s) → storage. |
| `app/db.py` | Schema, queries, settings cache, embedding-version counter. |
| `app/main.py` | FastAPI app, MJPEG stream, WebSocket hub, lifecycle. |
| `static/` | One HTML page with a sidebar and four hash-routed views (Dashboard, Register buyer, Buyers, Settings). Vanilla JS, no build step. The MJPEG stream is only connected while the Dashboard view is visible. |

## Runtime model

Three concurrent actors share one process:

- **Camera thread** reads frames as fast as the device delivers them and overwrites a single slot. Slow consumers therefore always see the newest frame and latency does not accumulate (no queue to back up).
- **Pipeline thread** takes the latest frame at most `TARGET_FPS` (8) times a second, runs detection, embedding, matching and the alert rules, and publishes the face boxes to a shared list.
- **asyncio event loop** (uvicorn) serves REST, the MJPEG stream and the WebSocket. The pipeline thread pushes alerts to it with `asyncio.run_coroutine_threadsafe`.

The video stream is **decoupled from recognition**: the MJPEG generator draws the *most recent boxes* onto the *most recent frame* at up to ~25 fps. Recognition runs at ~8 fps, but the video stays smooth; boxes lag by at most one recognition tick.

## Recognition pipeline

Models come from [`opencv_zoo`](https://github.com/opencv/opencv_zoo) and run through OpenCV's DNN module on CPU:

| Stage | Model | Notes |
|---|---|---|
| Detection | YuNet `face_detection_yunet_2023mar.onnx` via `cv2.FaceDetectorYN` | Score threshold 0.8, NMS 0.3. Returns box, 5 landmarks, score. |
| Alignment | `FaceRecognizerSF.alignCrop` | Similarity transform from the 5 landmarks to a canonical 112×112 crop. |
| Embedding | SFace `face_recognition_sface_2021dec.onnx` via `cv2.FaceRecognizerSF` | 128-d vector, L2-normalised and stored as float32. |
| Matching | Cosine similarity (dot product of unit vectors) | Default threshold **0.363**, the OpenCV-documented operating point for SFace. |

Per frame:

1. Downscale to 640 px wide and **detect on the small image** (speed).
2. Scale boxes and landmarks back to full resolution and **embed from the full-resolution frame** (accuracy: alignment on a downscaled crop would throw away detail the recogniser needs).
3. Skip faces under 50 px (embeddings of tiny faces are unreliable).
4. Match each face against **every enrolled embedding**, taking the single best score; accept if it clears the threshold.

Enrollment (`app/enroll.py`) validates each uploaded photo before storing an embedding: it must be decodable, contain a face, have a face at least 80 px wide and tall, and contain only one prominent face (a second face larger than 40 % of the biggest is rejected). The first accepted photo is stored as the buyer's display photo. Several photos per buyer are encouraged (angles, glasses on/off).

## Alert decision logic

A match alone never raises an alert. For each frame, `AlertEngine.update` applies, in order:

1. **Dues check, read fresh every frame:** `outstanding > min_due` (default ₹1,000). Because dues are queried per frame from the ledger, recording a payment stops alerts immediately, without a restart or cache invalidation.
2. **Temporal confirmation (N-of-M):** the buyer must have matched in at least `N` of the last `M` processed frames (default 3 of 5, roughly 0.4 s at 8 fps). This suppresses single-frame false positives from motion blur or a passing look-alike.
3. **Cooldown:** at most one alert per buyer per `cooldown` seconds (default 60).

Only if all three pass does the pipeline save an annotated snapshot, insert an `alerts` row and broadcast over the WebSocket. The dashboard shows a modal that must be acknowledged; multiple pending alerts queue, and unacknowledged alerts reappear after a page reload.

The confirmation history stores *who matched* regardless of dues, and dues are evaluated only at fire time. A buyer who has just paid therefore cannot trigger an alert from stale history.

## Data model

SQLite, created on first run (`data/app.db`).

```
buyers(id, name, phone, photo_path, created_at)
embeddings(id, buyer_id → buyers ON DELETE CASCADE, vector BLOB[128 × float32])
ledger(id, buyer_id → buyers ON DELETE CASCADE, kind IN ('credit','payment'), amount, note, ts)
alerts(id, buyer_id → buyers ON DELETE CASCADE, score, outstanding, snapshot_path, ts, acknowledged)
settings(key, value)
```

`outstanding = Σ credit − Σ payment` is **derived**, never stored. Deleting a buyer cascades to their embeddings, ledger and alerts, which is the "delete face data when an account closes" requirement.

## Design decisions

| Decision | Choice | Why / trade-off |
|---|---|---|
| Face stack | OpenCV YuNet + SFace (ONNX) | Native in OpenCV, CPU-only, no Docker or extra Python ML runtime, fast enough on Apple Silicon. Less accurate than large GPU models (ArcFace-class); the engine sits behind a two-method interface so it can be swapped. |
| Engine interface | `detect(frame)`, `embed(frame, face)` in `face.py` | Isolates the vendor; CompreFace or a GPU model can replace it without touching the matcher, alerts or API. |
| Matching target | All enrolled buyers, alert only if on watchlist | If only debtors were enrolled for matching, a paid-up customer who resembles a debtor could score above threshold against that debtor and trigger a false alert. With everyone enrolled in the match set, the customer's own embedding scores higher and wins; dues are checked only afterwards. |
| Per-buyer score | Best single embedding (max), not a mean/centroid | Robust to pose and lighting variety across enrolled photos; a centroid blurs distinct conditions together. |
| Matching algorithm | Brute-force `matrix @ vector` in NumPy | Exact, trivial, and fast for the hundreds of embeddings a shop will have. An ANN index would add complexity with no benefit at this scale. |
| False-positive control | N-of-M frames + cooldown, not just a higher threshold | Raising the threshold trades recall for precision globally; temporal voting removes transient errors without hurting steady-state recognition. |
| Dues evaluation | Per frame, from the DB | Payment takes effect instantly; no cache-coherence bug class. Costs one cheap SQLite query per tick. |
| Ledger | Append-only credit/payment rows; balance derived | Auditable history, no drift between a stored total and its entries. |
| Concurrency | Threads for blocking CV/camera I/O, asyncio for network I/O | OpenCV calls block and release the GIL; keeping them off the event loop keeps the API and stream responsive. |
| Frame handoff | Single overwritten "latest frame" slot | Real-time systems should drop stale frames rather than queue them; bounded latency by construction. |
| Detect small, embed large | 640 px detection, full-res embedding | Best speed/accuracy balance. |
| Shared OpenCV models | One lock in `FaceEngine` | DNN objects are not thread-safe, and both the live loop and upload enrollment use the same instance. |
| Video transport | MJPEG over HTTP | Works in a plain `<img>` tag, no JS player, no WebRTC signalling. Higher bandwidth than H.264, irrelevant on localhost. |
| Alert transport | WebSocket push | Immediate delivery; the page reconnects automatically and reloads unacknowledged alerts after a gap. |
| Enrollment source | Uploaded photo | Chosen by the product owner: buyers are registered from a photo, and the webcam is used only for live matching. |
| Persistence | SQLite, settings cached in-process | Zero setup. Each call opens a short-lived connection (SQLite serialises writers itself), and the settings cache is invalidated on write. |
| UI | Vanilla HTML/CSS/JS, no build step | No toolchain to maintain; dynamic content is built with DOM APIs (`textContent`), not string HTML, so buyer names cannot inject markup. |
| Camera permission | Open the camera once on the main thread at startup | macOS ties the Camera prompt to the opening process; the status flag turns healthy only after real frames arrive, because on macOS a denied camera can open successfully and return zero frames. |

## Methodology

- **Incremental, milestone-driven build.** The project followed a 7-milestone plan: scaffold and models, camera and detection stream, database and ledger, enrollment, matching and alerts, dashboard, tuning. Each layer was kept runnable on its own.
- **Interface-first for the volatile part.** The face engine, the component most likely to change, was isolated behind a minimal interface before anything depended on it.
- **Separation of concerns by change rate.** Recognition (`face`, `matcher`), policy (`alerts`) and I/O (`camera`, `main`) are separate modules, so policy tuning needs no recognition changes.
- **Derive, don't duplicate state.** Balances are computed from the ledger; the matcher reloads embeddings only when a version counter changes.
- **Acceptance-test oriented.** The behaviours to verify are defined up front (see below), and the thresholds and N-of-M are exposed as runtime settings so tuning needs no code change.
- **Commit history by concern.** Scaffold, DB, face engine, camera, matcher and alerts, pipeline, API, UI, seed script, styling, each as its own commit.

### Acceptance checklist (manual)

- [ ] Enrolled buyer with dues is recognised within ~2 s of facing the camera
- [ ] Enrolled buyer with zero dues triggers no alert
- [ ] Unknown face triggers no alert
- [ ] The same person does not re-alert inside the cooldown
- [ ] After a payment brings dues to zero, alerts stop immediately
- [ ] Works with glasses on and off, and in normal and dim light
- [ ] Several faces in one frame are handled

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/video` | MJPEG stream with face boxes |
| GET | `/status` | Camera health and processing rate |
| GET | `/buyers` | Buyers with outstanding balance and embedding count |
| POST | `/buyers` | Create buyer (`name`, `phone`, `opening_credit`) |
| DELETE | `/buyers/{id}` | Delete buyer, ledger, alerts and face data |
| POST | `/buyers/{id}/enroll` | Multipart `photos[]`; returns `enrolled` count and per-file `errors` |
| POST | `/buyers/{id}/ledger` | `{kind: credit\|payment, amount, note}` |
| GET | `/alerts` | Recent alerts (`?unacked=true`, `?limit=`) |
| POST | `/alerts/{id}/ack` | Mark handled |
| GET / POST | `/settings` | Read / update runtime settings |
| WS | `/ws` | Pushes `{"type":"alert","alert":{…}}` |
| GET | `/photos/...` | Enrolled photos and alert snapshots |

Request bodies are validated with Pydantic (non-negative opening credit, positive ledger amounts, bounded N/M).

## Configuration

Runtime settings (editable in the dashboard, stored in SQLite):

| Setting | Default | Meaning |
|---|---|---|
| `min_due` | 1000 | Watchlist if outstanding is above this (₹) |
| `threshold` | 0.363 | Minimum cosine similarity to accept a match |
| `cooldown` | 60 | Seconds between alerts for the same buyer |
| `confirm_n` / `confirm_m` | 3 / 5 | Must match in N of the last M frames |

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Device index, or an `rtsp://…` URL for a shop camera |

Code constants worth knowing: `TARGET_FPS`, `DETECT_WIDTH`, `MIN_FACE_PX` in `app/pipeline.py`; detector score threshold in `app/face.py`.

## Setup and run

Requires Python 3.11+ and macOS, Linux or Windows with a camera.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py        # fetches the two .onnx models (~38 MB)
.venv/bin/uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>.

**macOS camera permission:** grant Camera access to your terminal app in *System Settings → Privacy & Security → Camera*, then restart the terminal. If the status pill reads "camera unavailable", this is the usual cause.

Optional demo data:

```bash
.venv/bin/python scripts/seed_demo.py path/to/your_photo.jpg
```

### Suggested 3-minute demo

1. Open **Register buyer**, add "Demo Merchant" with ₹25,000 due and upload a clear photo of your face.
2. Step out of frame and back in. An alert modal appears with the enrolled photo, the snapshot and the amount.
3. Record a ₹25,000 payment in *Ledger*. Walk in again: no alert.
4. Have a colleague walk in: unknown, no alert.

## Project layout

```
app/        backend (camera, face, matcher, alerts, pipeline, enroll, db, main)
static/     index.html + style.css (dashboard)
scripts/    download_models.py, seed_demo.py
models/     downloaded .onnx files (git-ignored)
data/       app.db, photos, alert snapshots (git-ignored)
```

## Limitations and known gaps

- **Accuracy is unvalidated.** The 0.363 threshold is the library default, not tuned for this population, lighting or camera. Tune `threshold` against real tests (a lower value finds more people but raises false matches).
- **SFace is a small model.** It is weaker than large GPU models on low light, extreme angles, masks and low-resolution CCTV. Dim-light and entrance-camera performance must be tested before any real use.
- **No liveness detection.** A printed photo or a phone screen held to the camera can match.
- **No authentication.** The dashboard and API are open to anyone who can reach the port; run on localhost or a trusted network only.
- **Single camera, single process.** There is no multi-camera support, and in-memory alert state resets on restart (the cooldown restarts).
- **No automated tests.** Verification is the manual checklist above plus smoke tests.
- **Identity is probabilistic.** A match is a signal for a human to act on, not proof. Treat alerts as prompts for a polite conversation, not accusations.

## Privacy

Face embeddings and photos are biometric personal data.

- Enroll only people who have agreed. Demo with yourself and teammates.
- `data/` and `models/*.onnx` are git-ignored; never commit face data.
- Deleting a buyer removes their embeddings, ledger entries and alerts. Delete the stored photo files too when closing an account (they live under `data/photos/`).
- For any production use: obtain consent (for example in the credit agreement), post camera signage, restrict access to the dashboard, store no government ID numbers, and review obligations under India's DPDP Act, whose main obligations apply from 13 May 2027.
