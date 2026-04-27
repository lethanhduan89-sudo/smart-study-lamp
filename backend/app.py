from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import uuid
from pathlib import Path

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://smart-study-lamp.onrender.com")

AUDIO_DIR = Path("audio_cache")
UPLOAD_DIR = Path("uploads")

AUDIO_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/audio_cache", StaticFiles(directory="audio_cache"), name="audio_cache")

latest_command = {
    "command": None,
    "value": None,
    "reply": None,
    "audio_url": None,
}

device_status = {
    "brightness": 0,
    "auto_mode": True,
    "ambient_lux": None,
    "distance_cm": None,
    "mic_level": None,
    "online": False,
}


class UserInput(BaseModel):
    text: str


class DeviceReport(BaseModel):
    brightness: int | None = None
    auto_mode: bool | None = None
    ambient_lux: float | None = None
    distance_cm: float | None = None
    mic_level: float | None = None


def make_tts_wav(text: str) -> str:
    file_id = str(uuid.uuid4())
    out_path = AUDIO_DIR / f"{file_id}.wav"

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav",
    ) as response:
        response.stream_to_file(out_path)

    return f"{BASE_URL}/audio_cache/{file_id}.wav"


def run_ai_from_text(user_text: str):
    global latest_command

    prompt = f"""
Bạn là trợ lý điều khiển đèn học thông minh AI.
Hãy trả về JSON hợp lệ duy nhất.

Các command hợp lệ:
- lamp_on
- lamp_off
- brighter
- dimmer
- set_brightness
- auto_mode
- manual_mode
- status
- introduce
- none

Quy tắc:
- reply phải là tiếng Việt tự nhiên, thân thiện, ngắn gọn.
- Nếu người dùng hỏi "đây là cái gì", "đây là sản phẩm gì", "bạn là ai", "giới thiệu về bạn", hãy dùng command = introduce.
- Khi command = introduce, reply phải là: "Chào bạn, mình là sản phẩm đèn học thông minh AI. Mình có thể tự điều chỉnh ánh sáng, nhắc bạn ngồi đúng tư thế và hỗ trợ điều khiển bằng giọng nói."
- Nếu người dùng yêu cầu bật đèn, dùng command = lamp_on.
- Nếu người dùng yêu cầu tắt đèn, dùng command = lamp_off.
- Nếu người dùng yêu cầu tăng sáng, dùng command = brighter.
- Nếu người dùng yêu cầu giảm sáng, dùng command = dimmer.
- Nếu người dùng yêu cầu đặt độ sáng cụ thể, dùng command = set_brightness và value từ 0 đến 100.
- Nếu người dùng yêu cầu chế độ tự động, dùng command = auto_mode.
- Nếu người dùng yêu cầu chế độ thủ công, dùng command = manual_mode.
- Nếu người dùng hỏi trạng thái đèn, dùng command = status.
- Nếu không phải lệnh điều khiển hoặc giới thiệu sản phẩm, dùng command = none.
- Chỉ trả về JSON, không thêm giải thích ngoài JSON.

Trạng thái thiết bị hiện tại:
{json.dumps(device_status, ensure_ascii=False)}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
    )

    raw = response.output_text.strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {
            "reply": "Mình chưa hiểu rõ, bạn nói lại giúp mình nhé.",
            "command": "none",
        }

    if parsed.get("command") == "status":
        parsed["reply"] = (
            f"Đèn đang ở {device_status['brightness']} phần trăm, "
            f"chế độ tự động là {'bật' if device_status['auto_mode'] else 'tắt'}."
        )

    if parsed.get("command") == "introduce":
        parsed["reply"] = (
            "Chào bạn, mình là sản phẩm đèn học thông minh AI. "
            "Mình có thể tự điều chỉnh ánh sáng, nhắc bạn ngồi đúng tư thế "
            "và hỗ trợ điều khiển bằng giọng nói."
        )

    reply_text = parsed.get("reply", "Mình đã nhận lệnh rồi nhé.")
    parsed["audio_url"] = make_tts_wav(reply_text)

    latest_command = parsed
    return parsed


@app.get("/")
def root():
    return {"ok": True, "message": "backend running"}


@app.post("/ask")
def ask_ai(data: UserInput):
    return run_ai_from_text(data.text)


@app.post("/voice")
async def ask_voice(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() or ".wav"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

    content = await file.read()
    upload_path.write_bytes(content)

    with open(upload_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )

    heard_text = transcript.text
    result = run_ai_from_text(heard_text)
    result["heard_text"] = heard_text
    return result


@app.get("/device/pull")
def device_pull():
    global latest_command

    cmd = latest_command.copy()

    latest_command = {
        "command": None,
        "value": None,
        "reply": None,
        "audio_url": None,
    }

    return cmd


@app.post("/device/report")
def report_device(data: DeviceReport):
    global device_status

    if data.brightness is not None:
        device_status["brightness"] = data.brightness
    if data.auto_mode is not None:
        device_status["auto_mode"] = data.auto_mode
    if data.ambient_lux is not None:
        device_status["ambient_lux"] = data.ambient_lux
    if data.distance_cm is not None:
        device_status["distance_cm"] = data.distance_cm
    if data.mic_level is not None:
        device_status["mic_level"] = data.mic_level

    device_status["online"] = True
    return {"ok": True}


@app.get("/device/status")
def get_status():
    return device_status
