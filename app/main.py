"""FastAPI app: REST + MJPEG + WebSocket. Run: uvicorn app.main:app"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .camera import Camera
from .enroll import enroll_photos
from .face import FaceEngine
from .pipeline import Pipeline

STATIC = Path(__file__).resolve().parent.parent / "static"


class Hub:
    """Tracks dashboard websockets; lets the pipeline thread broadcast."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def _send(self, msg: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    def push(self, alert: dict):  # called from the pipeline thread
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._send({"type": "alert", "alert": alert}), self.loop)


hub = Hub()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    hub.loop = asyncio.get_running_loop()
    engine = FaceEngine()
    camera = Camera()
    camera.open()  # on the main thread so macOS shows the Camera permission prompt here
    camera.start()
    pipeline = Pipeline(camera, engine, hub.push)
    pipeline.start()
    state.update(engine=engine, camera=camera, pipeline=pipeline)
    yield
    pipeline.stop()
    camera.stop()


app = FastAPI(title="Credit Watchlist", lifespan=lifespan)


class BuyerIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(default="", max_length=30)
    opening_credit: float = Field(default=0, ge=0)


class LedgerIn(BaseModel):
    kind: Literal["credit", "payment"]
    amount: float = Field(gt=0)
    note: str = Field(default="", max_length=200)


class SettingsIn(BaseModel):
    min_due: float | None = Field(default=None, ge=0)
    threshold: float | None = Field(default=None, ge=0, le=1)
    cooldown: float | None = Field(default=None, ge=0)
    confirm_n: int | None = Field(default=None, ge=1, le=20)
    confirm_m: int | None = Field(default=None, ge=1, le=20)


def _need_buyer(buyer_id: int) -> dict:
    b = db.get_buyer(buyer_id)
    if not b:
        raise HTTPException(404, "buyer not found")
    return b


# ---- video -------------------------------------------------------------
async def _mjpeg():
    pipeline: Pipeline = state["pipeline"]
    last = -1
    while True:
        frame, seq = await asyncio.to_thread(pipeline.annotated)
        if seq != last or seq == 0:
            last = seq
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        await asyncio.sleep(0.04)


@app.get("/video")
async def video():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    return {"camera": state["camera"].ok, "fps": round(state["pipeline"].fps, 1)}


# ---- buyers ------------------------------------------------------------
@app.get("/buyers")
def list_buyers():
    return db.list_buyers()


@app.post("/buyers")
def create_buyer(body: BuyerIn):
    bid = db.create_buyer(body.name.strip(), body.phone.strip(), body.opening_credit)
    return db.get_buyer(bid)


@app.delete("/buyers/{buyer_id}")
def delete_buyer(buyer_id: int):
    _need_buyer(buyer_id)
    db.delete_buyer(buyer_id)
    return {"ok": True}


@app.post("/buyers/{buyer_id}/enroll")
async def enroll(buyer_id: int, photos: list[UploadFile] = File(...)):
    _need_buyer(buyer_id)
    blobs = [(p.filename or "photo", await p.read()) for p in photos]
    result = await asyncio.to_thread(enroll_photos, state["engine"], buyer_id, blobs)
    result["buyer"] = db.get_buyer(buyer_id)
    return result


@app.post("/buyers/{buyer_id}/ledger")
def add_ledger(buyer_id: int, body: LedgerIn):
    _need_buyer(buyer_id)
    db.add_ledger(buyer_id, body.kind, body.amount, body.note)
    return db.get_buyer(buyer_id)


# ---- alerts / settings -------------------------------------------------
@app.get("/alerts")
def list_alerts(limit: int = 50, unacked: bool = False):
    return db.list_alerts(limit, unacked)


@app.post("/alerts/{alert_id}/ack")
def ack(alert_id: int):
    if not db.ack_alert(alert_id):
        raise HTTPException(404, "alert not found")
    return {"ok": True}


@app.get("/settings")
def get_settings():
    return db.get_settings()


@app.post("/settings")
def set_settings(body: SettingsIn):
    return db.set_settings({k: v for k, v in body.model_dump().items() if v is not None})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive pings from the page
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(websocket)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


db.PHOTOS.mkdir(parents=True, exist_ok=True)
app.mount("/photos", StaticFiles(directory=db.PHOTOS), name="photos")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
