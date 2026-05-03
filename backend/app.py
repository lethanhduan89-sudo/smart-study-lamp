import os
import re
import uuid
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio_cache"

UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart Study Lamp Clean Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


device_status: Dict[str, Any] = {
    "power": False,
    "brightness": 0,
    "ambient_lux": None,
    "distance_cm": None,
    "present": False,
    "posture_bad": False,
    "online": False,
    "last_seen": None,
}

pending_command: Dict[str, Any] = {
    "command": "none",
    "value": -1,
    "reply": "",
    "audio_url": None,
}


class AskBody(BaseModel):
    text: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_audio_url(filename: str, request: Request) -> str:
    return f"{str(request.base_url).rstrip('/')}/audio/{filename}"


def make_tts_audio(reply: str, request: Request) -> str | None:
    if client is None or not reply:
        return None

    try:
        filename = f"{uuid.uuid4().hex}.wav"
        output_path = AUDIO_DIR / filename

        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply,
            response_format="wav",
        )

        output_path.write_bytes(speech.content)
        return make_audio_url(filename, request)

    except Exception:
        return None


def parse_command(text: str) -> Dict[str, Any]:
    t = normalize_text(text)

    match_percent = re.search(r"(\d{1,3})\s*(%|phan tram)?", t)

    if any(x in t for x in ["bat den", "mo den"]):
        return {
            "command": "lamp_on",
            "value": -1,
            "reply": "Đèn đã bật.",
        }

    if any(x in t for x in ["tat den", "dong den"]):
        return {
            "command": "lamp_off",
            "value": -1,
            "reply": "Đèn đã tắt.",
        }

    if "sang" in t and match_percent:
        value = clamp(int(match_percent.group(1)), 0, 100)

        return {
            "command": "set_brightness",
            "value": value,
            "reply": f"Đã chỉnh độ sáng {value} phần trăm.",
        }

    if any(x in t for x in ["tang sang", "sang hon"]):
        return {
            "command": "brighter",
            "value": -1,
            "reply": "Đã tăng độ sáng.",
        }

    if any(x in t for x in ["giam sang", "toi hon"]):
        return {
            "command": "dimmer",
            "value": -1,
            "reply": "Đã giảm độ sáng.",
        }

    if any(x in t for x in ["day la gi", "ban la ai", "gioi thieu"]):
        return {
            "command": "none",
            "value": -1,
            "reply": (
                "Mình là đèn học thông minh AI. "
                "Mình có thể điều khiển đèn bằng giọng nói, "
                "đo ánh sáng, phát hiện sai tư thế và tự tắt khi không có người."
            ),
        }

    return {
        "command": "none",
        "value": -1,
        "reply": "Mình chưa hiểu lệnh này.",
    }


def queue_command(command: str, value: int, reply: str, audio_url: str | None):
    global pending_command

    pending_command = {
        "command": command,
        "value": value,
        "reply": reply,
        "audio_url": audio_url,
    }


def handle_text(text: str, request: Request) -> Dict[str, Any]:
    result = parse_command(text)

    command = result["command"]
    value = int(result["value"])
    reply = result["reply"]
    audio_url = make_tts_audio(reply, request)

    if command != "none":
        queue_command(command, value, reply, audio_url)

    return {
        "heard_text": text,
        "command": command,
        "value": value,
        "reply": reply,
        "audio_url": audio_url,
        "status": device_status,
    }


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "smart-study-lamp-clean-backend",
        "status": device_status,
    }


@app.post("/ask")
def ask(body: AskBody, request: Request):
    return handle_text(body.text, request)


@app.post("/voice")
async def voice(request: Request, file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY chưa được cấu hình.",
        )

    suffix = Path(file.filename or "voice.wav").suffix or ".wav"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()
        temp_path.write_bytes(content)

        with temp_path.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
            )

        heard_text = getattr(transcript, "text", "").strip()

        if not heard_text:
            reply = "Mình chưa nghe rõ."
            audio_url = make_tts_audio(reply, request)

            return {
                "heard_text": "",
                "command": "none",
                "value": -1,
                "reply": reply,
                "audio_url": audio_url,
                "status": device_status,
            }

        return handle_text(heard_text, request)

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = AUDIO_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    return FileResponse(path, media_type="audio/wav")


@app.get("/device/pull")
def device_pull():
    global pending_command

    cmd = dict(pending_command)

    pending_command = {
        "command": "none",
        "value": -1,
        "reply": "",
        "audio_url": None,
    }

    return cmd


@app.post("/device/report")
def device_report(payload: Dict[str, Any]):
    device_status.update(payload)
    device_status["online"] = True
    device_status["last_seen"] = now_iso()

    return {
        "ok": True,
        "status": device_status,
    }


@app.post("/device/alert")
def device_alert(payload: Dict[str, Any], request: Request):
    alert_type = payload.get("type", "")

    if alert_type == "wrong_posture":
        reply = "Bạn đang cúi quá thấp, hãy ngồi lại đúng tư thế."
    elif alert_type == "no_person":
        reply = "Không có người, đèn đã tắt."
    else:
        reply = "Có cảnh báo từ đèn học."

    audio_url = make_tts_audio(reply, request)

    return {
        "reply": reply,
        "audio_url": audio_url,
    }
