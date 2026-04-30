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

app = FastAPI(title="Smart Study Lamp Backend Audio Version")

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
    "auto_mode": True,
    "mode": "auto",
    "ambient_lux": None,
    "distance_cm": None,
    "present": False,
    "posture_bad": False,
    "alert": "none",
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


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def compute_public_status() -> Dict[str, Any]:
    s = dict(device_status)
    last_seen = s.get("last_seen")

    if last_seen:
        try:
            dt = datetime.fromisoformat(last_seen)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            age_s = (datetime.now(timezone.utc) - dt).total_seconds()

            if age_s > 15:
                s["online"] = False
        except Exception:
            pass

    return s


def status_reply() -> str:
    s = compute_public_status()

    power_text = "đang bật" if s.get("power") else "đang tắt"
    brightness = s.get("brightness", 0)
    auto_text = "bật" if s.get("auto_mode") else "tắt"

    lux = s.get("ambient_lux")
    lux_text = "-" if lux is None else f"{float(lux):.0f} lux"

    distance = s.get("distance_cm")
    distance_text = "-" if distance is None else f"{float(distance):.0f} cm"

    posture_text = "sai tư thế" if s.get("posture_bad") else "bình thường"

    return (
        f"Đèn {power_text}, độ sáng {brightness} phần trăm, "
        f"chế độ tự động {auto_text}, ánh sáng môi trường {lux_text}, "
        f"khoảng cách {distance_text}, tư thế {posture_text}."
    )


def parse_text_command(text: str) -> Dict[str, Any]:
    t = normalize_text(text)

    if any(x in t for x in ["gioi thieu", "ban la ai", "san pham gi", "day la gi", "day la cai gi"]):
        return {
            "command": "introduce",
            "value": -1,
            "reply": (
                "Mình là đèn học thông minh AI. "
                "Mình có thể tự chỉnh ánh sáng, phát hiện sai tư thế "
                "và hỗ trợ điều khiển bằng giọng nói."
            ),
        }

    if any(x in t for x in ["trang thai", "hien tai", "do sang bao nhieu", "den dang"]):
        return {
            "command": "status",
            "value": -1,
            "reply": status_reply(),
        }

    match_percent = re.search(r"(\d{1,3})\s*(%|phan tram)", t)
    if match_percent:
        value = clamp(int(match_percent.group(1)), 0, 100)

        return {
            "command": "set_brightness",
            "value": value,
            "reply": f"Đã chỉnh độ sáng {value} phần trăm.",
        }

    if any(x in t for x in ["bat den", "mo den"]):
        return {
            "command": "lamp_on",
            "value": -1,
            "reply": "Đã bật đèn học.",
        }

    if any(x in t for x in ["tat den", "dong den"]):
        return {
            "command": "lamp_off",
            "value": -1,
            "reply": "Đã tắt đèn học.",
        }

    if any(x in t for x in ["tang sang", "sang hon", "tang do sang"]):
        return {
            "command": "brighter",
            "value": -1,
            "reply": "Đã tăng độ sáng.",
        }

    if any(x in t for x in ["giam sang", "toi hon", "giam do sang"]):
        return {
            "command": "dimmer",
            "value": -1,
            "reply": "Đã giảm độ sáng.",
        }

    if any(x in t for x in ["tu dong", "auto"]):
        return {
            "command": "auto_mode",
            "value": -1,
            "reply": "Đã bật chế độ tự động.",
        }

    if any(x in t for x in ["thu cong", "manual"]):
        return {
            "command": "manual_mode",
            "value": -1,
            "reply": "Đã chuyển sang chế độ thủ công.",
        }

    if client is None:
        return {
            "command": "none",
            "value": -1,
            "reply": "Mình chưa hiểu rõ lệnh này.",
        }

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=(
                "Bạn là trợ lý của đèn học thông minh AI. "
                "Nếu người dùng hỏi kiến thức chung, hãy trả lời ngắn gọn bằng tiếng Việt. "
                "Nếu không phải lệnh điều khiển đèn, command là none.\n\n"
                f"Người dùng hỏi: {text}"
            ),
        )

        reply = response.output_text.strip()

        if not reply:
            reply = "Mình chưa có câu trả lời phù hợp."

        return {
            "command": "none",
            "value": -1,
            "reply": reply,
        }

    except Exception:
        return {
            "command": "none",
            "value": -1,
            "reply": "Mình chưa hiểu rõ lệnh này.",
        }


def apply_shadow_command(command: str, value: int) -> None:
    if command == "lamp_on":
        device_status["power"] = True

        if int(device_status.get("brightness", 0)) == 0:
            device_status["brightness"] = 50

    elif command == "lamp_off":
        device_status["power"] = False

    elif command == "brighter":
        device_status["power"] = True
        device_status["auto_mode"] = False
        device_status["mode"] = "manual"
        device_status["brightness"] = clamp(
            int(device_status.get("brightness", 0)) + 10,
            0,
            100,
        )

    elif command == "dimmer":
        device_status["auto_mode"] = False
        device_status["mode"] = "manual"
        device_status["brightness"] = clamp(
            int(device_status.get("brightness", 0)) - 10,
            0,
            100,
        )

        if int(device_status["brightness"]) == 0:
            device_status["power"] = False

    elif command == "set_brightness":
        b = clamp(int(value), 0, 100)
        device_status["brightness"] = b
        device_status["power"] = b > 0
        device_status["auto_mode"] = False
        device_status["mode"] = "manual"

    elif command == "auto_mode":
        device_status["power"] = True
        device_status["auto_mode"] = True
        device_status["mode"] = "auto"

    elif command == "manual_mode":
        device_status["power"] = True
        device_status["auto_mode"] = False
        device_status["mode"] = "manual"


def make_audio_url(filename: str, request: Request) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/audio/{filename}"


def make_tts_audio(reply: str, request: Request) -> str | None:
    if client is None:
        return None

    if not reply:
        return None

    try:
        filename = f"{uuid.uuid4().hex}.mp3"
        output_path = AUDIO_DIR / filename

        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply,
            response_format="mp3",
        )

        output_path.write_bytes(speech.content)

        return make_audio_url(filename, request)

    except Exception:
        return None


def queue_command(command: str, value: int, reply: str, audio_url: str | None) -> None:
    global pending_command

    pending_command = {
        "command": command,
        "value": value,
        "reply": reply,
        "audio_url": audio_url,
    }

    apply_shadow_command(command, value)


def handle_text(text: str, request: Request) -> Dict[str, Any]:
    result = parse_text_command(text)

    command = result["command"]
    value = int(result["value"])
    reply = result["reply"]

    audio_url = make_tts_audio(reply, request)

    if command in {
        "lamp_on",
        "lamp_off",
        "brighter",
        "dimmer",
        "set_brightness",
        "auto_mode",
        "manual_mode",
    }:
        queue_command(command, value, reply, audio_url)

    return {
        "heard_text": text,
        "command": command,
        "value": value,
        "reply": reply,
        "audio_url": audio_url,
        "status": compute_public_status(),
    }


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "smart-study-lamp-backend-audio",
        "status": compute_public_status(),
    }


@app.post("/ask")
def ask(body: AskBody, request: Request):
    return handle_text(body.text, request)


@app.post("/voice")
async def voice(request: Request, file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY chưa được cấu hình trên Render.",
        )

    suffix = Path(file.filename or "voice.webm").suffix or ".webm"
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
                "status": compute_public_status(),
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

    return FileResponse(path)


@app.get("/device/status")
def get_status():
    return compute_public_status()


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
    lux = payload.get("ambient_lux", payload.get("lux"))
    distance = payload.get("distance_cm", payload.get("distanceCm"))
    posture_bad = bool(payload.get("posture_bad", payload.get("postureBad", False)))

    if "power" in payload:
        device_status["power"] = bool(payload["power"])

    if "brightness" in payload:
        device_status["brightness"] = clamp(int(payload["brightness"]), 0, 100)

    if "auto_mode" in payload:
        device_status["auto_mode"] = bool(payload["auto_mode"])

    if "mode" in payload:
        device_status["mode"] = str(payload["mode"])

    if lux is not None:
        device_status["ambient_lux"] = lux

    if distance is not None:
        device_status["distance_cm"] = distance

    if "present" in payload:
        device_status["present"] = bool(payload["present"])

    device_status["posture_bad"] = posture_bad
    device_status["alert"] = "wrong_posture" if posture_bad else "none"
    device_status["online"] = True
    device_status["last_seen"] = now_iso()

    return {
        "ok": True,
        "device_status": compute_public_status(),
    }
